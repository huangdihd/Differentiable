import { type Backend, type Command, type Handle, type Shape, size } from './backend';
import type { Gpu, GpuBuffer, GpuDevice, GpuEncoder, GpuPipeline } from './gpu-api';
import { kernel } from './webgpu-kernels';
import { optimizerKernel } from './webgpu-optimizer';
import { limitDispatch } from './webgpu-dispatch';
import { BufferPool } from './buffer-pool';

export class BackendError extends Error {
  constructor(
    readonly key: string,
    readonly params: Record<string, string | number> = {},
  ) {
    super(key);
  }
}
const STORAGE = 128,
  COPY_SRC = 4,
  COPY_DST = 8,
  MAP_READ = 1;
/** Ordered eager commands share encoders. Only completed submissions can recycle storage. */
export class WebGpuBackend implements Backend {
  private next = 0;
  private live = new Map<Handle, GpuBuffer>();
  private lengths = new Map<Handle, number>();
  private pool = new BufferPool<GpuBuffer>(undefined, (bytes) => {
    this.allocatedBytes -= bytes;
  });
  private retired: GpuBuffer[] = [];
  private encoder?: GpuEncoder;
  private commands = 0;
  private pipelines = new Map<string, GpuPipeline>();
  private pending: Promise<void> = Promise.resolve();
  private fault?: BackendError;
  private closed = false;
  private allocatedBytes = 0;
  private readonly errorBuffer: GpuBuffer;
  private constructor(
    private readonly device: GpuDevice,
    readonly adapterInfo: unknown,
  ) {
    this.errorBuffer = device.createBuffer({ size: 4, usage: STORAGE | COPY_SRC | COPY_DST });
    device.lost.then((info) => {
      if (!this.closed)
        this.fault = new BackendError('engine.gpu.deviceLost', {
          reason: info.reason,
          message: info.message,
        });
    });
    device.addEventListener('uncapturederror', (event) => {
      this.fault = new BackendError('engine.gpu.validation', { message: event.error.message });
    });
  }
  static async create(): Promise<WebGpuBackend> {
    const gpu = (globalThis.navigator as unknown as { gpu?: Gpu } | undefined)?.gpu;
    if (!gpu) throw new BackendError('engine.gpu.unavailable');
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new BackendError('engine.gpu.noAdapter');
    try {
      // GPUAdapterInfo exposes its fields as prototype getters, which JSON.stringify skips.
      const info = (adapter.info ?? {}) as Record<string, unknown>;
      const adapterInfo = Object.fromEntries(
        ['vendor', 'architecture', 'device', 'description'].map((key) => [key, info[key] ?? '']),
      );
      return new WebGpuBackend(await adapter.requestDevice(), adapterInfo);
    } catch (error) {
      throw new BackendError('engine.gpu.device', { message: String(error) });
    }
  }
  get stats() {
    return {
      liveBuffers: this.live.size,
      pooledBuffers: this.pool.count,
      pooledBytes: this.pool.bytes,
      allocatedBytes: this.allocatedBytes,
    };
  }
  private check(): void {
    if (this.fault) throw this.fault;
    if (this.closed) throw new BackendError('engine.gpu.disposed');
  }
  private get(handle: Handle): GpuBuffer {
    this.check();
    const buffer = this.live.get(handle);
    if (!buffer) throw new BackendError('engine.tensor.disposed');
    return buffer;
  }
  private encode(): GpuEncoder {
    this.check();
    if (!this.encoder) {
      this.device.pushErrorScope('validation');
      this.encoder = this.device.createCommandEncoder();
    }
    return this.encoder;
  }
  allocate(shape: Shape, data?: ArrayLike<number>): Handle {
    this.check();
    const length = size(shape),
      bytes = Math.max(4, length * 4);
    if (data && data.length !== length) throw new BackendError('engine.data.length');
    if (
      !Number.isSafeInteger(bytes) ||
      bytes >
        Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize)
    )
      throw new BackendError('engine.gpu.bufferLimit', { bytes });
    let buffer = this.pool.take(bytes);
    if (!buffer) {
      buffer = this.device.createBuffer({ size: bytes, usage: STORAGE | COPY_SRC | COPY_DST });
      this.allocatedBytes += bytes;
    }
    if (data && length) this.device.queue.writeBuffer(buffer, 0, Float32Array.from(data));
    else this.encode().clearBuffer(buffer);
    const handle = { id: this.next++ };
    this.live.set(handle, buffer);
    this.lengths.set(handle, length);
    return handle;
  }
  release(handle: Handle): void {
    const buffer = this.live.get(handle);
    if (buffer) {
      this.live.delete(handle);
      this.lengths.delete(handle);
      this.retired.push(buffer);
    }
  }
  submit(commands: readonly Command[]): void {
    for (const command of commands) {
      this.check();
      if (
        !size(command.output.shape) &&
        !['gather', 'scatter_add'].includes(command.op) &&
        !(command.op === 'vjp' && ['gather', 'scatter_add'].includes(String(command.attrs.op)))
      )
        continue;
      const logical = optimizerKernel(command) ?? kernel(command);
      let spec;
      try {
        spec = limitDispatch(logical, this.device.limits.maxComputeWorkgroupsPerDimension);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw new BackendError('engine.gpu.dispatchLimit', {
          workgroups: logical.workgroups.join(','),
        });
      }
      const buffers = [
        ...command.inputs.map((v) => this.get(v.handle)),
        this.get(command.output.handle),
        this.errorBuffer,
      ];
      if (buffers.length > this.device.limits.maxStorageBuffersPerShaderStage)
        throw new BackendError('engine.gpu.bindingLimit');
      const encoder = this.encode();
      let pipeline = this.pipelines.get(spec.code);
      if (!pipeline) {
        const layout = this.device.createBindGroupLayout({
          entries: buffers.map((_, binding) => ({
            binding,
            visibility: 4,
            buffer: { type: binding < command.inputs.length ? 'read-only-storage' : 'storage' },
          })),
        });
        pipeline = this.device.createComputePipeline({
          layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
          compute: {
            module: this.device.createShaderModule({ code: spec.code }),
            entryPoint: 'main',
          },
        });
        if (this.pipelines.size >= 256) this.pipelines.delete(this.pipelines.keys().next().value!);
        this.pipelines.set(spec.code, pipeline);
      }
      const group = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...spec.workgroups);
      pass.end();
      // Bound command retention even when reporting is infrequent. This is not a CPU fence.
      if (++this.commands >= 512) this.submitPending();
    }
  }
  private submitPending(): void {
    const retired = this.retired.splice(0);
    let validation: Promise<{ message: string } | null> = Promise.resolve(null);
    if (this.encoder) {
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = undefined;
      this.commands = 0;
      validation = this.device.popErrorScope();
    }
    const done = this.device.queue.onSubmittedWorkDone();
    this.pending = Promise.all([this.pending, done, validation])
      .then(([, , error]) => {
        if (error)
          this.fault = new BackendError('engine.gpu.validation', { message: error.message });
        for (const buffer of retired) {
          if (this.closed) buffer.destroy();
          else this.pool.put(buffer);
        }
      })
      .catch((error: unknown) => {
        this.fault = new BackendError('engine.gpu.execution', { message: String(error) });
      });
  }
  async read(handle: Handle): Promise<Float32Array> {
    const buffer = this.get(handle);
    const length = this.lengths.get(handle)!;
    const staging = this.device.createBuffer({ size: buffer.size + 4, usage: COPY_DST | MAP_READ });
    try {
      const encoder = this.encode();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      encoder.copyBufferToBuffer(this.errorBuffer, 0, staging, buffer.size, 4);
      this.submitPending();
      await this.pending;
      this.check();
      await staging.mapAsync(MAP_READ);
      const mapped = staging.getMappedRange();
      if (new Uint32Array(mapped, buffer.size, 1)[0]) throw this.gatherFault();
      return new Float32Array(mapped.slice(0, length * 4));
    } finally {
      staging.destroy();
    }
  }
  /** Report an out-of-range gather once, then clear the flag: it is sticky on the device, so
   * without the reset one bad graph poisoned every later read, even of a corrected graph. */
  private gatherFault(): BackendError {
    this.device.queue.writeBuffer(this.errorBuffer, 0, new Uint32Array(1));
    return new BackendError('engine.gather.index');
  }
  fence(): Promise<void> {
    this.check();
    this.submitPending();
    return this.pending;
  }
  async flush(): Promise<void> {
    this.check();
    const staging = this.device.createBuffer({ size: 4, usage: COPY_DST | MAP_READ });
    try {
      this.encode().copyBufferToBuffer(this.errorBuffer, 0, staging, 0, 4);
      this.submitPending();
      await this.pending;
      this.check();
      await staging.mapAsync(MAP_READ);
      if (new Uint32Array(staging.getMappedRange())[0]) throw this.gatherFault();
    } finally {
      staging.destroy();
    }
  }
  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    for (const buffer of [...this.live.values(), ...this.retired]) buffer.destroy();
    this.live.clear();
    this.lengths.clear();
    this.pool.clear();
    this.retired = [];
    this.pipelines.clear();
    this.errorBuffer.destroy();
    this.device.destroy();
  }
}

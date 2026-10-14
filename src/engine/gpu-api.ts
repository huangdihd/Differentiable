/** Minimal structural WebGPU surface used here; avoids a runtime dependency. */
export interface GpuBuffer {
  readonly size: number;
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}
export interface GpuPipeline {
  getBindGroupLayout(index: number): unknown;
}
export interface GpuEncoder {
  clearBuffer(buffer: GpuBuffer): void;
  copyBufferToBuffer(a: GpuBuffer, ao: number, b: GpuBuffer, bo: number, size: number): void;
  beginComputePass(): {
    setPipeline(pipeline: GpuPipeline): void;
    setBindGroup(index: number, group: unknown): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  };
  finish(): unknown;
}
export interface GpuDevice {
  limits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
    maxStorageBuffersPerShaderStage: number;
  };
  lost: Promise<{ reason: string; message: string }>;
  addEventListener(name: string, listener: (event: { error: { message: string } }) => void): void;
  pushErrorScope(filter: string): void;
  popErrorScope(): Promise<{ message: string } | null>;
  createBuffer(descriptor: { size: number; usage: number }): GpuBuffer;
  createCommandEncoder(): GpuEncoder;
  createShaderModule(descriptor: { code: string }): unknown;
  createBindGroupLayout(descriptor: {
    entries: { binding: number; visibility: number; buffer: { type: string } }[];
  }): unknown;
  createPipelineLayout(descriptor: { bindGroupLayouts: unknown[] }): unknown;
  createComputePipeline(descriptor: {
    layout: unknown;
    compute: { module: unknown; entryPoint: string };
  }): GpuPipeline;
  createBindGroup(descriptor: {
    layout: unknown;
    entries: { binding: number; resource: { buffer: GpuBuffer } }[];
  }): unknown;
  queue: {
    writeBuffer(buffer: GpuBuffer, offset: number, data: Float32Array | Uint32Array): void;
    submit(commands: unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  };
  destroy(): void;
}
export interface GpuAdapter {
  info?: { vendor: string; architecture: string; device: string; description: string };
  requestDevice(): Promise<GpuDevice>;
}
export interface Gpu {
  requestAdapter(): Promise<GpuAdapter | null>;
}

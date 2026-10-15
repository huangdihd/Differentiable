/** Free storage only: callers must wait for GPU completion before returning buffers. */
export const FREE_BUFFER_BUDGET = 64 * 1024 * 1024;

export class BufferPool<T extends { size: number; destroy(): void }> {
  private free = new Map<T, number>();
  bytes = 0;
  constructor(
    readonly budget = FREE_BUFFER_BUDGET,
    private readonly destroyed: (bytes: number) => void = () => {},
  ) {}
  get count() {
    return this.free.size;
  }
  take(size: number): T | undefined {
    for (const [buffer, bytes] of this.free) {
      if (bytes !== size) continue;
      this.free.delete(buffer);
      this.bytes -= bytes;
      return buffer;
    }
  }
  put(buffer: T): void {
    if (this.free.has(buffer)) return;
    if (buffer.size > this.budget) {
      this.destroy(buffer);
      return;
    }
    while (this.bytes + buffer.size > this.budget) {
      const oldest = this.free.keys().next().value!;
      this.free.delete(oldest);
      this.bytes -= oldest.size;
      this.destroy(oldest);
    }
    this.free.set(buffer, buffer.size);
    this.bytes += buffer.size;
  }
  private destroy(buffer: T) {
    buffer.destroy();
    this.destroyed(buffer.size);
  }
  clear(): void {
    for (const buffer of this.free.keys()) this.destroy(buffer);
    this.free.clear();
    this.bytes = 0;
  }
}

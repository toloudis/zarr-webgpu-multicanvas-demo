/// <reference types="vite/client" />

declare const GPUTextureUsage: {
  readonly COPY_DST: GPUTextureUsageFlags;
  readonly TEXTURE_BINDING: GPUTextureUsageFlags;
};

declare const GPUBufferUsage: {
  readonly COPY_DST: GPUBufferUsageFlags;
  readonly UNIFORM: GPUBufferUsageFlags;
};

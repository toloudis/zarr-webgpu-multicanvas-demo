const SHADER = /* wgsl */ `
struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );

  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0, -1.0)
  );

  var out: VertexOut;
  out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  out.uv = uvs[vertexIndex];
  return out;
}

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;

@fragment
fn fragmentMain(in: VertexOut) -> @location(0) vec4f {
  return textureSample(imageTexture, imageSampler, in.uv);
}
`;

export async function createImageGridRenderer(gridElement, onStats) {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  return new ImageGridRenderer(device, gridElement, onStats);
}

class ImageGridRenderer {
  constructor(device, gridElement, onStats) {
    this.device = device;
    this.gridElement = gridElement;
    this.onStats = onStats;
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.tiles = new Map();
    this.canvasToTile = new Map();
    this.visibleCanvases = new Set();
    this.nextTileId = 1;
    this.rafId = 0;

    this.pipeline = device.createRenderPipeline({
      label: "image blit pipeline",
      layout: "auto",
      vertex: {
        module: device.createShaderModule({ label: "image vertex shader", code: SHADER }),
        entryPoint: "vertexMain",
      },
      fragment: {
        module: device.createShaderModule({ label: "image fragment shader", code: SHADER }),
        entryPoint: "fragmentMain",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.sampler = device.createSampler({
      label: "image sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.renderPassDescriptor = {
      label: "canvas image pass",
      colorAttachments: [
        {
          clearValue: [0.04, 0.045, 0.052, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    };

    this.resizeObserver = new ResizeObserver((entries) => this.onResize(entries));
    this.intersectionObserver = new IntersectionObserver((entries) => this.onIntersection(entries), {
      root: null,
      threshold: 0,
    });
  }

  addTile({ title, subtitle }) {
    const id = this.nextTileId++;
    const tileElement = document.createElement("article");
    tileElement.className = "image-tile is-loading";

    const canvas = document.createElement("canvas");
    const header = document.createElement("div");
    header.className = "tile-header";

    const titleElement = document.createElement("h2");
    titleElement.textContent = title;

    const subtitleElement = document.createElement("p");
    subtitleElement.textContent = subtitle;

    const stateElement = document.createElement("span");
    stateElement.className = "tile-state";
    stateElement.textContent = "Loading";

    header.append(titleElement, subtitleElement, stateElement);
    tileElement.append(canvas, header);
    this.gridElement.append(tileElement);

    const context = canvas.getContext("webgpu");
    context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "opaque",
    });

    const tile = {
      id,
      element: tileElement,
      canvas,
      context,
      titleElement,
      subtitleElement,
      stateElement,
      bindGroup: null,
      texture: null,
      ready: false,
    };

    this.tiles.set(id, tile);
    this.canvasToTile.set(canvas, tile);
    this.resizeObserver.observe(canvas);
    this.intersectionObserver.observe(canvas);
    this.requestRender();

    return { id };
  }

  updateTile(id, { title, subtitle }) {
    const tile = this.tiles.get(id);
    if (!tile) return;
    if (title) tile.titleElement.textContent = title;
    if (subtitle) tile.subtitleElement.textContent = subtitle;
  }

  uploadTile(id, image) {
    const tile = this.tiles.get(id);
    if (!tile) return;

    tile.texture?.destroy();
    tile.texture = this.device.createTexture({
      label: `slice ${id} texture`,
      size: [image.width, image.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const bytesPerRow = alignTo(image.width * 4, 256);
    const upload = bytesPerRow === image.width * 4
      ? image.rgba
      : copyWithAlignedRows(image.rgba, image.width, image.height, bytesPerRow);

    this.device.queue.writeTexture(
      { texture: tile.texture },
      upload,
      { bytesPerRow, rowsPerImage: image.height },
      { width: image.width, height: image.height },
    );

    tile.bindGroup = this.device.createBindGroup({
      label: `slice ${id} bind group`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tile.texture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });

    tile.ready = true;
    tile.element.classList.remove("is-loading", "has-error");
    tile.stateElement.textContent = "Ready";
    tile.canvas.style.aspectRatio = `${image.width} / ${image.height}`;
    this.requestRender();
  }

  setTileError(id, message) {
    const tile = this.tiles.get(id);
    if (!tile) return;
    tile.ready = false;
    tile.element.classList.remove("is-loading");
    tile.element.classList.add("has-error");
    tile.stateElement.textContent = "Error";
    tile.subtitleElement.textContent = message;
  }

  clear() {
    for (const tile of this.tiles.values()) {
      this.resizeObserver.unobserve(tile.canvas);
      this.intersectionObserver.unobserve(tile.canvas);
      tile.texture?.destroy();
    }

    this.tiles.clear();
    this.canvasToTile.clear();
    this.visibleCanvases.clear();
    this.gridElement.textContent = "";
    this.requestRender();
  }

  onResize(entries) {
    const ratio = window.devicePixelRatio || 1;
    for (const entry of entries) {
      const canvas = entry.target;
      const box = entry.contentBoxSize?.[0];
      const cssWidth = box?.inlineSize ?? canvas.clientWidth;
      const cssHeight = box?.blockSize ?? canvas.clientHeight;
      const width = Math.max(1, Math.min(Math.round(cssWidth * ratio), this.device.limits.maxTextureDimension2D));
      const height = Math.max(1, Math.min(Math.round(cssHeight * ratio), this.device.limits.maxTextureDimension2D));

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }
    this.requestRender();
  }

  onIntersection(entries) {
    for (const { target, isIntersecting } of entries) {
      if (isIntersecting) {
        this.visibleCanvases.add(target);
      } else {
        this.visibleCanvases.delete(target);
      }
    }
    this.requestRender();
  }

  requestRender() {
    if (this.rafId) return;
    this.rafId = requestAnimationFrame(() => this.render());
  }

  render() {
    this.rafId = 0;
    let rendered = 0;

    if (this.visibleCanvases.size === 0) {
      this.onStats?.({ rendered, visible: 0 });
      return;
    }

    const encoder = this.device.createCommandEncoder({ label: "multi-canvas encoder" });

    for (const canvas of this.visibleCanvases) {
      const tile = this.canvasToTile.get(canvas);
      if (!tile?.ready || !tile.bindGroup || canvas.width === 0 || canvas.height === 0) continue;

      this.renderPassDescriptor.colorAttachments[0].view = tile.context
        .getCurrentTexture()
        .createView();

      const pass = encoder.beginRenderPass(this.renderPassDescriptor);
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, tile.bindGroup);
      pass.draw(3);
      pass.end();
      rendered++;
    }

    if (rendered > 0) {
      this.device.queue.submit([encoder.finish()]);
    }

    this.onStats?.({ rendered, visible: this.visibleCanvases.size });
  }
}

function alignTo(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

function copyWithAlignedRows(source, width, height, bytesPerRow) {
  const tightBytesPerRow = width * 4;
  const aligned = new Uint8Array(bytesPerRow * height);

  for (let y = 0; y < height; y++) {
    const sourceStart = y * tightBytesPerRow;
    const targetStart = y * bytesPerRow;
    aligned.set(source.subarray(sourceStart, sourceStart + tightBytesPerRow), targetStart);
  }

  return aligned;
}

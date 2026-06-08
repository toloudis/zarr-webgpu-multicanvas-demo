import type { ChannelRenderSettings, LoadedPlaneSet, RenderStats } from "./types";

const MAX_GPU_CHANNELS = 32;
const CHANNEL_SETTINGS_FLOATS = (MAX_GPU_CHANNELS + 1) * 4;

const SHADER = /* wgsl */ `
const MAX_CHANNELS = ${MAX_GPU_CHANNELS}u;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct ChannelSettings {
  colors: array<vec4f, ${MAX_GPU_CHANNELS}>,
  details: vec4f,
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

@group(0) @binding(0) var channelTexture: texture_2d_array<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var<uniform> channelSettings: ChannelSettings;

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  var color = vec3f(0.0);
  let channelCount = min(u32(channelSettings.details.x), MAX_CHANNELS);

  for (var channelIndex = 0u; channelIndex < channelCount; channelIndex = channelIndex + 1u) {
    let tint = channelSettings.colors[channelIndex];
    if (tint.a > 0.0) {
      let value = textureSample(channelTexture, imageSampler, input.uv, i32(channelIndex)).r;
      color = color + (value * tint.rgb * tint.a);
    }
  }

  return vec4f(min(color, vec3f(1.0)), 1.0);
}
`;

type StatsCallback = (stats: RenderStats) => void;
type TileClickCallback = (tileId: number) => void;

interface TileText {
  title: string;
  subtitle: string;
}

interface TileHandle {
  id: number;
}

interface ImageTile {
  id: number;
  element: HTMLElement;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  titleElement: HTMLHeadingElement;
  subtitleElement: HTMLParagraphElement;
  stateElement: HTMLSpanElement;
  bindGroup: GPUBindGroup | null;
  channelTexture: GPUTexture | null;
  settingsBuffer: GPUBuffer | null;
  channelCount: number;
  ready: boolean;
}

export async function createImageGridRenderer(
  gridElement: HTMLElement,
  onStats?: StatsCallback,
  onTileClick?: TileClickCallback,
): Promise<ImageGridRenderer> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  return new ImageGridRenderer(device, gridElement, onStats, onTileClick);
}

export class ImageGridRenderer {
  private readonly device: GPUDevice;
  private readonly gridElement: HTMLElement;
  private readonly onStats?: StatsCallback;
  private readonly onTileClick?: TileClickCallback;
  private readonly presentationFormat: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly resizeObserver: ResizeObserver;
  private readonly intersectionObserver: IntersectionObserver;
  private readonly tiles = new Map<number, ImageTile>();
  private readonly canvasToTile = new Map<HTMLCanvasElement, ImageTile>();
  private readonly visibleCanvases = new Set<HTMLCanvasElement>();
  private nextTileId = 1;
  private rafId = 0;

  constructor(
    device: GPUDevice,
    gridElement: HTMLElement,
    onStats?: StatsCallback,
    onTileClick?: TileClickCallback,
  ) {
    this.device = device;
    this.gridElement = gridElement;
    this.onStats = onStats;
    this.onTileClick = onTileClick;
    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();

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

    this.resizeObserver = new ResizeObserver((entries) => this.onResize(entries));
    this.intersectionObserver = new IntersectionObserver((entries) => this.onIntersection(entries), {
      root: null,
      threshold: 0,
    });
  }

  addTile({ title, subtitle }: TileText): TileHandle {
    const id = this.nextTileId++;
    const tileElement = document.createElement("article");
    tileElement.className = "image-tile is-loading";

    const canvas = document.createElement("canvas");
    canvas.title = "Open in Vol-E";
    canvas.addEventListener("click", () => this.onTileClick?.(id));

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

    const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!context) {
      throw new Error("Could not create a WebGPU canvas context.");
    }

    context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "opaque",
    });

    const tile: ImageTile = {
      id,
      element: tileElement,
      canvas,
      context,
      titleElement,
      subtitleElement,
      stateElement,
      bindGroup: null,
      channelTexture: null,
      settingsBuffer: null,
      channelCount: 0,
      ready: false,
    };

    this.tiles.set(id, tile);
    this.canvasToTile.set(canvas, tile);
    this.resizeObserver.observe(canvas);
    this.intersectionObserver.observe(canvas);
    this.requestRender();

    return { id };
  }

  updateTile(id: number, { title, subtitle }: Partial<TileText>): void {
    const tile = this.tiles.get(id);
    if (!tile) return;
    if (title) tile.titleElement.textContent = title;
    if (subtitle) tile.subtitleElement.textContent = subtitle;
  }

  setTileLoading(id: number, subtitle?: string): void {
    const tile = this.tiles.get(id);
    if (!tile) return;
    tile.ready = false;
    tile.element.classList.add("is-loading");
    tile.element.classList.remove("has-error");
    tile.stateElement.textContent = "Loading";
    if (subtitle) tile.subtitleElement.textContent = subtitle;
  }

  uploadChannelPlanes(
    id: number,
    planeSet: LoadedPlaneSet,
    channels: readonly ChannelRenderSettings[],
  ): void {
    const tile = this.tiles.get(id);
    if (!tile) return;

    const channelPlanes = planeSet.channelPlanes.slice(0, MAX_GPU_CHANNELS);
    if (channelPlanes.length === 0) {
      this.setTileError(id, "No channel planes were loaded.");
      return;
    }

    tile.channelTexture?.destroy();
    tile.settingsBuffer?.destroy();
    tile.channelTexture = this.device.createTexture({
      label: `slice ${id} channel texture array`,
      size: {
        width: planeSet.width,
        height: planeSet.height,
        depthOrArrayLayers: channelPlanes.length,
      },
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const bytesPerRow = alignTo(planeSet.width, 256);
    for (let layer = 0; layer < channelPlanes.length; layer++) {
      const plane = channelPlanes[layer];
      const upload = bytesPerRow === planeSet.width
        ? plane.pixels
        : copySingleChannelWithAlignedRows(plane.pixels, planeSet.width, planeSet.height, bytesPerRow);

      this.device.queue.writeTexture(
        {
          texture: tile.channelTexture,
          origin: { x: 0, y: 0, z: layer },
        },
        upload,
        { bytesPerRow, rowsPerImage: planeSet.height },
        { width: planeSet.width, height: planeSet.height, depthOrArrayLayers: 1 },
      );
    }

    tile.settingsBuffer = this.device.createBuffer({
      label: `slice ${id} channel settings`,
      size: CHANNEL_SETTINGS_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    tile.channelCount = channelPlanes.length;
    this.writeChannelSettings(tile, channels);

    tile.bindGroup = this.device.createBindGroup({
      label: `slice ${id} bind group`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tile.channelTexture.createView({ dimension: "2d-array" }) },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: tile.settingsBuffer } },
      ],
    });

    tile.ready = true;
    tile.element.classList.remove("is-loading", "has-error");
    tile.stateElement.textContent = "Ready";
    tile.canvas.style.aspectRatio = `${planeSet.width} / ${planeSet.height}`;
    this.requestRender();
  }

  updateChannelSettings(id: number, channels: readonly ChannelRenderSettings[]): void {
    const tile = this.tiles.get(id);
    if (!tile?.settingsBuffer) return;

    this.writeChannelSettings(tile, channels);
    this.requestRender();
  }

  private writeChannelSettings(tile: ImageTile, channels: readonly ChannelRenderSettings[]): void {
    if (!tile.settingsBuffer) return;

    const data = createChannelSettingsData(tile.channelCount, channels);
    this.device.queue.writeBuffer(tile.settingsBuffer, 0, data);
  }

  setTileError(id: number, message: string): void {
    const tile = this.tiles.get(id);
    if (!tile) return;
    tile.ready = false;
    tile.element.classList.remove("is-loading");
    tile.element.classList.add("has-error");
    tile.stateElement.textContent = "Error";
    tile.subtitleElement.textContent = message;
  }

  clear(): void {
    for (const tile of this.tiles.values()) {
      this.resizeObserver.unobserve(tile.canvas);
      this.intersectionObserver.unobserve(tile.canvas);
      tile.channelTexture?.destroy();
      tile.settingsBuffer?.destroy();
    }

    this.tiles.clear();
    this.canvasToTile.clear();
    this.visibleCanvases.clear();
    this.gridElement.textContent = "";
    this.requestRender();
  }

  private onResize(entries: ResizeObserverEntry[]): void {
    const ratio = window.devicePixelRatio || 1;
    for (const entry of entries) {
      const canvas = entry.target;
      if (!(canvas instanceof HTMLCanvasElement)) continue;

      const box = entry.contentBoxSize[0];
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

  private onIntersection(entries: IntersectionObserverEntry[]): void {
    for (const { target, isIntersecting } of entries) {
      if (!(target instanceof HTMLCanvasElement)) continue;

      if (isIntersecting) {
        this.visibleCanvases.add(target);
      } else {
        this.visibleCanvases.delete(target);
      }
    }
    this.requestRender();
  }

  private requestRender(): void {
    if (this.rafId) return;
    this.rafId = requestAnimationFrame(() => this.render());
  }

  private render(): void {
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

      const pass = encoder.beginRenderPass({
        label: "canvas image pass",
        colorAttachments: [
          {
            view: tile.context.getCurrentTexture().createView(),
            clearValue: [0.04, 0.045, 0.052, 1],
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
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

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function copySingleChannelWithAlignedRows(
  source: Uint8Array,
  width: number,
  height: number,
  bytesPerRow: number,
): Uint8Array {
  const tightBytesPerRow = width;
  const aligned = new Uint8Array(bytesPerRow * height);

  for (let y = 0; y < height; y++) {
    const sourceStart = y * tightBytesPerRow;
    const targetStart = y * bytesPerRow;
    aligned.set(source.subarray(sourceStart, sourceStart + tightBytesPerRow), targetStart);
  }

  return aligned;
}

function createChannelSettingsData(
  channelCount: number,
  channels: readonly ChannelRenderSettings[],
): Float32Array {
  const data = new Float32Array(CHANNEL_SETTINGS_FLOATS);

  for (let index = 0; index < Math.min(channelCount, MAX_GPU_CHANNELS); index++) {
    const channel = channels[index];
    const color = parseHexColor(channel?.color);
    const offset = index * 4;
    data[offset] = color[0];
    data[offset + 1] = color[1];
    data[offset + 2] = color[2];
    data[offset + 3] = channel?.enabled ? 1 : 0;
  }

  const detailsOffset = MAX_GPU_CHANNELS * 4;
  data[detailsOffset] = Math.min(channelCount, MAX_GPU_CHANNELS);
  return data;
}

function parseHexColor(color = "#ffffff"): [number, number, number] {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);
  if (!match) return [1, 1, 1];

  return [
    Number.parseInt(match[1], 16) / 255,
    Number.parseInt(match[2], 16) / 255,
    Number.parseInt(match[3], 16) / 255,
  ];
}

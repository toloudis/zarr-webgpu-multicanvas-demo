import type { ChannelRenderSettings, LoadedChannelPlane, LoadedPlaneSet, RenderStats } from "./types";

const MAX_GPU_CHANNELS = 32;
const COLOR_SETTINGS_OFFSET = 0;
const THRESHOLD_SETTINGS_OFFSET = MAX_GPU_CHANNELS * 4;
const DETAIL_SETTINGS_OFFSET = MAX_GPU_CHANNELS * 2 * 4;
const CHANNEL_SETTINGS_FLOATS = (MAX_GPU_CHANNELS * 2 + 1) * 4;

const SHADER = /* wgsl */ `
const MAX_CHANNELS = ${MAX_GPU_CHANNELS}u;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct ChannelSettings {
  colors: array<vec4f, ${MAX_GPU_CHANNELS}>,
  thresholds: array<vec4f, ${MAX_GPU_CHANNELS}>,
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

@group(0) @binding(0) var channelTexture: texture_2d_array<u32>;
@group(0) @binding(1) var<uniform> channelSettings: ChannelSettings;

fn loadChannelTexel(channelIndex: u32, pixel: vec2i, dimensions: vec2u) -> f32 {
  let maxPixel = vec2i(i32(dimensions.x) - 1, i32(dimensions.y) - 1);
  let clampedPixel = clamp(pixel, vec2i(0, 0), maxPixel);
  return f32(textureLoad(channelTexture, clampedPixel, i32(channelIndex), 0).r);
}

fn sampleChannelBilinear(channelIndex: u32, uv: vec2f, dimensions: vec2u) -> f32 {
  let clampedUv = clamp(uv, vec2f(0.0), vec2f(1.0));
  let sourcePosition = clampedUv * vec2f(f32(dimensions.x), f32(dimensions.y)) - vec2f(0.5);
  let base = floor(sourcePosition);
  let basePixel = vec2i(i32(base.x), i32(base.y));
  let weight = sourcePosition - base;

  let v00 = loadChannelTexel(channelIndex, basePixel, dimensions);
  let v10 = loadChannelTexel(channelIndex, basePixel + vec2i(1, 0), dimensions);
  let v01 = loadChannelTexel(channelIndex, basePixel + vec2i(0, 1), dimensions);
  let v11 = loadChannelTexel(channelIndex, basePixel + vec2i(1, 1), dimensions);

  return mix(
    mix(v00, v10, weight.x),
    mix(v01, v11, weight.x),
    weight.y
  );
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  var color = vec3f(0.0);
  let channelCount = min(u32(channelSettings.details.x), MAX_CHANNELS);
  let dimensions = textureDimensions(channelTexture);

  for (var channelIndex = 0u; channelIndex < channelCount; channelIndex = channelIndex + 1u) {
    let tint = channelSettings.colors[channelIndex];
    if (tint.a > 0.0) {
      let thresholds = channelSettings.thresholds[channelIndex];
      let rawValue = sampleChannelBilinear(channelIndex, input.uv, dimensions);
      let value = clamp((rawValue - thresholds.x) / max(thresholds.y - thresholds.x, 0.000001), 0.0, 1.0);
      color = color + (value * tint.rgb * tint.a);
    }
  }

  return vec4f(min(color, vec3f(1.0)), 1.0);
}
`;

type StatsCallback = (stats: RenderStats) => void;
type TileClickCallback = (tileId: number) => void;
type TileSettingsClickCallback = (tileId: number, anchor: HTMLElement) => void;

interface TileText {
  title: string;
  subtitle: string;
  placement?: TilePlacement;
}

interface TileHandle {
  id: number;
}

interface FigureTrack {
  fr?: number;
}

export interface FigureGridLayout {
  rows: readonly FigureTrack[];
  cols: readonly FigureTrack[];
  rowLabels: readonly string[];
  colLabels: readonly string[];
}

export interface TilePlacement {
  row: number;
  col: number;
  rowSpan: number;
  colSpan: number;
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
  channelRanges: TileChannelRange[];
  ready: boolean;
}

interface TileChannelRange {
  channelIndex: number;
  min: number;
  max: number;
  autoMin: number;
  autoMax: number;
}

interface NativeTextureUploadInfo {
  format: GPUTextureFormat;
  bytesPerSample: number;
  typeLabel: string;
}

export async function createImageGridRenderer(
  gridElement: HTMLElement,
  onStats?: StatsCallback,
  onTileClick?: TileClickCallback,
  onTileSettingsClick?: TileSettingsClickCallback,
): Promise<ImageGridRenderer> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  return new ImageGridRenderer(device, gridElement, onStats, onTileClick, onTileSettingsClick);
}

export class ImageGridRenderer {
  private readonly device: GPUDevice;
  private readonly gridElement: HTMLElement;
  private readonly onStats?: StatsCallback;
  private readonly onTileClick?: TileClickCallback;
  private readonly onTileSettingsClick?: TileSettingsClickCallback;
  private readonly presentationFormat: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
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
    onTileSettingsClick?: TileSettingsClickCallback,
  ) {
    this.device = device;
    this.gridElement = gridElement;
    this.onStats = onStats;
    this.onTileClick = onTileClick;
    this.onTileSettingsClick = onTileSettingsClick;
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

    this.resizeObserver = new ResizeObserver((entries) => this.onResize(entries));
    this.intersectionObserver = new IntersectionObserver((entries) => this.onIntersection(entries), {
      root: null,
      threshold: 0,
    });
  }

  setAutoGridLayout(): void {
    this.gridElement.classList.remove("figure-grid");
    this.gridElement.style.removeProperty("grid-template-columns");
    this.gridElement.style.removeProperty("grid-template-rows");
  }

  setFigureGridLayout({ rows, cols, rowLabels, colLabels }: FigureGridLayout): void {
    this.gridElement.classList.add("figure-grid");
    this.gridElement.style.gridTemplateColumns = [
      "max-content",
      ...cols.map((col) => `minmax(180px, ${formatFr(col.fr)}fr)`),
    ].join(" ");
    this.gridElement.style.gridTemplateRows = [
      "auto",
      ...rows.map((row) => `minmax(0, ${formatFr(row.fr)}fr)`),
    ].join(" ");

    const corner = document.createElement("div");
    corner.className = "figure-axis-label figure-corner-label";
    corner.style.gridRow = "1";
    corner.style.gridColumn = "1";
    this.gridElement.append(corner);

    for (let index = 0; index < colLabels.length; index++) {
      this.gridElement.append(createFigureLabel("figure-col-label", colLabels[index], {
        gridRow: 1,
        gridColumn: index + 2,
      }));
    }

    for (let index = 0; index < rowLabels.length; index++) {
      this.gridElement.append(createFigureLabel("figure-row-label", rowLabels[index], {
        gridRow: index + 2,
        gridColumn: 1,
      }));
    }
  }

  addTile({ title, subtitle, placement }: TileText): TileHandle {
    const id = this.nextTileId++;
    const tileElement = document.createElement("article");
    tileElement.className = "image-tile is-loading";
    if (placement) {
      tileElement.style.gridRow = `${placement.row + 2} / span ${placement.rowSpan}`;
      tileElement.style.gridColumn = `${placement.col + 2} / span ${placement.colSpan}`;
    }

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

    const settingsButton = document.createElement("button");
    settingsButton.type = "button";
    settingsButton.className = "tile-settings-button";
    settingsButton.textContent = "Settings";
    settingsButton.addEventListener("click", () => this.onTileSettingsClick?.(id, settingsButton));

    const actions = document.createElement("div");
    actions.className = "tile-actions";
    actions.append(settingsButton, stateElement);

    header.append(titleElement, subtitleElement, actions);
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
      channelRanges: [],
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

    const uploadInfo = getNativeTextureUploadInfo(channelPlanes);
    tile.channelTexture?.destroy();
    tile.settingsBuffer?.destroy();
    tile.channelTexture = this.device.createTexture({
      label: `slice ${id} channel texture array`,
      size: {
        width: planeSet.width,
        height: planeSet.height,
        depthOrArrayLayers: channelPlanes.length,
      },
      format: uploadInfo.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    const tightBytesPerRow = planeSet.width * uploadInfo.bytesPerSample;
    const bytesPerRow = alignTo(tightBytesPerRow, 256);
    for (let layer = 0; layer < channelPlanes.length; layer++) {
      const plane = channelPlanes[layer];
      const sourceBytes = getTypedArrayBytes(plane.nativePixels);
      const upload = bytesPerRow === tightBytesPerRow
        ? sourceBytes
        : copyNativePlaneWithAlignedRows(
          sourceBytes,
          planeSet.width,
          planeSet.height,
          uploadInfo.bytesPerSample,
          bytesPerRow,
        );

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
    tile.channelRanges = channelPlanes.map((plane) => ({
      channelIndex: plane.channelIndex,
      min: plane.min,
      max: plane.max,
      autoMin: plane.autoMin,
      autoMax: plane.autoMax,
    }));
    this.writeChannelSettings(tile, channels);

    tile.bindGroup = this.device.createBindGroup({
      label: `slice ${id} bind group`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: tile.channelTexture.createView({ dimension: "2d-array" }) },
        { binding: 1, resource: { buffer: tile.settingsBuffer } },
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

    const data = createChannelSettingsData(tile.channelRanges, channels);
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

function copyNativePlaneWithAlignedRows(
  source: Uint8Array,
  width: number,
  height: number,
  bytesPerSample: number,
  bytesPerRow: number,
): Uint8Array {
  const tightBytesPerRow = width * bytesPerSample;
  const aligned = new Uint8Array(bytesPerRow * height);

  for (let y = 0; y < height; y++) {
    const sourceStart = y * tightBytesPerRow;
    const targetStart = y * bytesPerRow;
    aligned.set(source.subarray(sourceStart, sourceStart + tightBytesPerRow), targetStart);
  }

  return aligned;
}

function getNativeTextureUploadInfo(channelPlanes: readonly LoadedChannelPlane[]): NativeTextureUploadInfo {
  const firstInfo = getPlaneTextureUploadInfo(channelPlanes[0]);

  for (const plane of channelPlanes.slice(1)) {
    const info = getPlaneTextureUploadInfo(plane);
    if (info.format !== firstInfo.format) {
      throw new Error(
        `Mixed channel plane types are not supported in one GPU texture array. Got ${firstInfo.typeLabel} and ${info.typeLabel}.`,
      );
    }
  }

  return firstInfo;
}

function getPlaneTextureUploadInfo(plane: LoadedChannelPlane): NativeTextureUploadInfo {
  const pixels = plane.nativePixels;
  if (pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray) {
    return { format: "r8uint", bytesPerSample: 1, typeLabel: pixels.constructor.name };
  }

  if (pixels instanceof Uint16Array) {
    return { format: "r16uint", bytesPerSample: 2, typeLabel: pixels.constructor.name };
  }

  throw new Error(
    `Unsupported native image type ${pixels.constructor.name}; this renderer currently supports Uint8 and Uint16 planes.`,
  );
}

function getTypedArrayBytes(source: LoadedChannelPlane["nativePixels"]): Uint8Array {
  return new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
}

function createChannelSettingsData(
  channelRanges: readonly TileChannelRange[],
  channels: readonly ChannelRenderSettings[],
): Float32Array {
  const data = new Float32Array(CHANNEL_SETTINGS_FLOATS);
  const channelCount = Math.min(channelRanges.length, MAX_GPU_CHANNELS);

  for (let index = 0; index < channelCount; index++) {
    const channelRange = channelRanges[index];
    const channel = channels[channelRange.channelIndex];
    const color = parseHexColor(channel?.color);
    const colorOffset = COLOR_SETTINGS_OFFSET + index * 4;
    data[colorOffset] = color[0];
    data[colorOffset + 1] = color[1];
    data[colorOffset + 2] = color[2];
    data[colorOffset + 3] = channel?.enabled ? 1 : 0;

    const [thresholdMin, thresholdMax] = resolveThresholdRange(channel, channelRange);
    const thresholdOffset = THRESHOLD_SETTINGS_OFFSET + index * 4;
    data[thresholdOffset] = thresholdMin;
    data[thresholdOffset + 1] = thresholdMax;
  }

  data[DETAIL_SETTINGS_OFFSET] = channelCount;
  return data;
}

function resolveThresholdRange(
  channel: ChannelRenderSettings | undefined,
  channelRange: TileChannelRange,
): [number, number] {
  const autoMin = getFiniteNumber(channelRange.autoMin, channelRange.min);
  const autoMax = getFiniteNumber(channelRange.autoMax, channelRange.max);
  const thresholdMin = getFiniteNumber(channel?.min, autoMin);
  let thresholdMax = getFiniteNumber(channel?.max, autoMax);

  if (thresholdMax <= thresholdMin) {
    thresholdMax = thresholdMin + 1;
  }

  return [thresholdMin, thresholdMax];
}

function getFiniteNumber(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : fallback;
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

function createFigureLabel(
  className: string,
  text: string,
  placement: { gridRow: number; gridColumn: number },
): HTMLElement {
  const label = document.createElement("div");
  label.className = `figure-axis-label ${className}`;
  label.textContent = text;
  label.style.gridRow = String(placement.gridRow);
  label.style.gridColumn = String(placement.gridColumn);
  return label;
}

function formatFr(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value) || value <= 0) return "1";
  return String(value);
}

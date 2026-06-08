import "./styles.css";
import { loadBlendedSlice, loadImageMetadata } from "./zarrLoader";
import { ZARR_IMAGE_SOURCES } from "./zarrSources";
import type { ChannelRenderSettings, ZarrImageMetadata, ZarrImageSource } from "./types";
import { createImageGridRenderer, type ImageGridRenderer } from "./webgpuImageRenderer";

const DEFAULT_CHANNEL_COLORS = [
  "#ffffff",
  "#ff375f",
  "#32d74b",
  "#0a84ff",
  "#ffd60a",
  "#bf5af2",
  "#64d2ff",
  "#ff9f0a",
];

interface LoadedImageState {
  source: ZarrImageSource;
  tileId: number;
  metadata?: ZarrImageMetadata;
}

interface AppState {
  images: LoadedImageState[];
  currentT: number;
  currentZ: number;
  maxTimeIndex: number;
  maxZIndex: number;
  channels: ChannelRenderSettings[];
  axesInitialized: boolean;
}

const loadButton = requireElement<HTMLButtonElement>("#load-button");
const statusText = requireElement<HTMLElement>("#status-text");
const renderStats = requireElement<HTMLElement>("#render-stats");
const grid = requireElement<HTMLElement>("#canvas-grid");
const timeSlider = requireElement<HTMLInputElement>("#time-slider");
const timeValue = requireElement<HTMLOutputElement>("#time-value");
const zSlider = requireElement<HTMLInputElement>("#z-slider");
const zValue = requireElement<HTMLOutputElement>("#z-value");
const channelControls = requireElement<HTMLElement>("#channel-controls");

const appState: AppState = {
  images: [],
  currentT: 0,
  currentZ: 0,
  maxTimeIndex: 0,
  maxZIndex: 0,
  channels: [],
  axesInitialized: false,
};

let renderer: ImageGridRenderer | undefined;
let currentLoadAbortController: AbortController | undefined;
let currentRenderAbortController: AbortController | undefined;

init().catch((error) => {
  console.error(error);
  statusText.textContent = getErrorMessage(error);
});

loadButton.addEventListener("click", () => loadSources());
timeSlider.addEventListener("input", () => {
  appState.currentT = Number(timeSlider.value);
  renderControls();
  void renderLoadedImages();
});
zSlider.addEventListener("input", () => {
  appState.currentZ = Number(zSlider.value);
  renderControls();
  void renderLoadedImages();
});

async function init(): Promise<void> {
  renderer = await createImageGridRenderer(grid, (stats) => {
    renderStats.textContent = `${stats.rendered} rendered / ${stats.visible} visible`;
  });
  statusText.textContent = "WebGPU ready.";
  renderControls();
  await loadSources();
}

async function loadSources(): Promise<void> {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  currentLoadAbortController?.abort();
  currentRenderAbortController?.abort();
  const abortController = new AbortController();
  currentLoadAbortController = abortController;
  const sources = ZARR_IMAGE_SOURCES;

  activeRenderer.clear();
  appState.images = [];
  renderControls();

  if (sources.length === 0) {
    statusText.textContent = "No configured Zarr sources.";
    return;
  }

  statusText.textContent = `Reading metadata for ${sources.length} image${sources.length === 1 ? "" : "s"}.`;

  appState.images = sources.map((source, index) => {
    const tile = activeRenderer.addTile({
      title: source.label || `Image ${index + 1}`,
      subtitle: source.url,
    });
    return { source, tileId: tile.id };
  });

  const tasks = appState.images.map((imageState) => async (): Promise<void> => {
    activeRenderer.setTileLoading(imageState.tileId, "Reading metadata");

    try {
      imageState.metadata = await loadImageMetadata({
        source: imageState.source,
        signal: abortController.signal,
      });
      activeRenderer.updateTile(imageState.tileId, {
        subtitle: formatMetadataSubtitle(imageState.metadata),
      });
    } catch (error) {
      if (abortController.signal.aborted) return;
      console.error(error);
      activeRenderer.setTileError(imageState.tileId, getErrorMessage(error));
      throw error;
    }
  });

  const results = await runLimited(tasks, 4);
  if (abortController.signal.aborted) return;

  const failed = results.filter((result) => result.status === "rejected").length;
  const loaded = appState.images.filter((imageState) => imageState.metadata).length;

  if (loaded === 0) {
    statusText.textContent = `Loaded 0 metadata records; ${failed} failed.`;
    return;
  }

  configureGlobalStateFromImages();
  renderControls();
  statusText.textContent = `Loaded ${loaded} metadata record${loaded === 1 ? "" : "s"}.`;
  await renderLoadedImages();
}

async function renderLoadedImages(): Promise<void> {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  const imagesWithMetadata = appState.images.filter(hasMetadata);
  if (imagesWithMetadata.length === 0) return;

  currentRenderAbortController?.abort();
  const abortController = new AbortController();
  currentRenderAbortController = abortController;

  const selectedChannels = appState.channels.filter((channel) => channel.enabled).length;
  statusText.textContent = `Rendering T${appState.currentT} Z${appState.currentZ} with ${selectedChannels} channel${selectedChannels === 1 ? "" : "s"}.`;

  const tasks = imagesWithMetadata.map((imageState) => async (): Promise<void> => {
    const { metadata } = imageState;
    activeRenderer.setTileLoading(imageState.tileId, formatRenderSubtitle(metadata, "Loading"));

    try {
      const slice = await loadBlendedSlice({
        metadata,
        timeIndex: appState.currentT,
        zIndex: appState.currentZ,
        channels: appState.channels,
        signal: abortController.signal,
      });

      activeRenderer.uploadTile(imageState.tileId, slice);
      activeRenderer.updateTile(imageState.tileId, {
        subtitle: formatSliceSubtitle(slice),
      });
    } catch (error) {
      if (abortController.signal.aborted) return;
      console.error(error);
      activeRenderer.setTileError(imageState.tileId, getErrorMessage(error));
      throw error;
    }
  });

  const results = await runLimited(tasks, 4);
  if (abortController.signal.aborted) return;

  const failed = results.filter((result) => result.status === "rejected").length;
  const rendered = imagesWithMetadata.length - failed;
  statusText.textContent = failed
    ? `Rendered ${rendered}; ${failed} failed.`
    : `Rendered ${rendered} image${rendered === 1 ? "" : "s"}.`;
}

function configureGlobalStateFromImages(): void {
  const metadata = appState.images.flatMap((imageState) => imageState.metadata ? [imageState.metadata] : []);
  const maxT = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.t));
  const maxZ = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.z));
  const maxChannels = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.c));

  appState.maxTimeIndex = maxT - 1;
  appState.maxZIndex = maxZ - 1;

  if (appState.axesInitialized) {
    appState.currentT = clampIndex(appState.currentT, maxT);
    appState.currentZ = clampIndex(appState.currentZ, maxZ);
  } else {
    appState.currentT = 0;
    appState.currentZ = Math.floor(appState.maxZIndex / 2);
    appState.axesInitialized = true;
  }

  reconcileChannels(maxChannels);
}

function reconcileChannels(maxChannels: number): void {
  const previousChannels = appState.channels;
  appState.channels = Array.from({ length: maxChannels }, (_, index) => {
    const previous = previousChannels[index];
    return {
      index,
      enabled: previous?.enabled ?? true,
      color: previous?.color ?? DEFAULT_CHANNEL_COLORS[index % DEFAULT_CHANNEL_COLORS.length],
    };
  });
}

function renderControls(): void {
  timeSlider.max = String(appState.maxTimeIndex);
  timeSlider.value = String(appState.currentT);
  timeSlider.disabled = appState.maxTimeIndex === 0;
  timeValue.value = `T${appState.currentT} / T${appState.maxTimeIndex}`;

  zSlider.max = String(appState.maxZIndex);
  zSlider.value = String(appState.currentZ);
  zSlider.disabled = appState.maxZIndex === 0;
  zValue.value = `Z${appState.currentZ} / Z${appState.maxZIndex}`;

  channelControls.textContent = "";
  for (const channel of appState.channels) {
    channelControls.append(createChannelControl(channel));
  }
}

function createChannelControl(channel: ChannelRenderSettings): HTMLElement {
  const label = document.createElement("label");
  label.className = "channel-control";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = channel.enabled;
  checkbox.addEventListener("change", () => {
    channel.enabled = checkbox.checked;
    renderControls();
    void renderLoadedImages();
  });

  const color = document.createElement("input");
  color.type = "color";
  color.value = channel.color;
  color.disabled = !channel.enabled;
  color.addEventListener("input", () => {
    channel.color = color.value;
    void renderLoadedImages();
  });

  const name = document.createElement("span");
  name.textContent = `C${channel.index}`;

  label.append(checkbox, color, name);
  return label;
}

async function runLimited<T>(
  tasks: Array<() => Promise<T>>,
  limit: number,
): Promise<Array<PromiseSettledResult<T>>> {
  const results = new Array<PromiseSettledResult<T>>(tasks.length);
  let next = 0;

  async function worker(): Promise<void> {
    while (next < tasks.length) {
      const index = next++;
      try {
        const value = await tasks[index]();
        results[index] = { status: "fulfilled", value };
      } catch (error) {
        results[index] = { status: "rejected", reason: error };
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

function formatMetadataSubtitle(metadata: ZarrImageMetadata): string {
  return [
    `level ${metadata.multiresolutionLevel}`,
    `path ${metadata.arrayPath || "/"}`,
    `TCZYX [${formatTCZYX(metadata.shapeTCZYX)}]`,
    metadata.dtype,
  ].join("  ");
}

function formatRenderSubtitle(metadata: ZarrImageMetadata, state: string): string {
  return [
    state,
    `level ${metadata.multiresolutionLevel}`,
    `TCZYX [${formatTCZYX(metadata.shapeTCZYX)}]`,
  ].join("  ");
}

function formatSliceSubtitle(slice: {
  width: number;
  height: number;
  timeIndex: number;
  zIndex: number;
  channelIndices: number[];
  channelRanges: Array<{ channelIndex: number; min: number; max: number }>;
  shapeTCZYX: ZarrImageMetadata["shapeTCZYX"];
  multiresolutionLevel: number;
  arrayPath: string;
}): string {
  const channels = slice.channelIndices.length
    ? slice.channelIndices.map((index) => `C${index}`).join(", ")
    : "none";
  const ranges = slice.channelRanges.length
    ? slice.channelRanges
      .map((range) => `C${range.channelIndex} ${formatNumber(range.min)}-${formatNumber(range.max)}`)
      .join("; ")
    : "no channel ranges";

  return [
    `${slice.width} x ${slice.height}`,
    `T${slice.timeIndex} Z${slice.zIndex}`,
    `channels ${channels}`,
    `level ${slice.multiresolutionLevel}`,
    `path ${slice.arrayPath || "/"}`,
    `TCZYX [${formatTCZYX(slice.shapeTCZYX)}]`,
    ranges,
  ].join("  ");
}

function formatTCZYX(shape: ZarrImageMetadata["shapeTCZYX"]): string {
  return `${shape.t}, ${shape.c}, ${shape.z}, ${shape.y}, ${shape.x}`;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toPrecision(4);
}

function clampIndex(value: number, size: number): number {
  if (size <= 1) return 0;
  return Math.max(0, Math.min(Math.round(value), size - 1));
}

function hasMetadata(imageState: LoadedImageState): imageState is LoadedImageState & {
  metadata: ZarrImageMetadata;
} {
  return imageState.metadata !== undefined;
}

function requireElement<TElement extends Element>(selector: string): TElement {
  const element = document.querySelector<TElement>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

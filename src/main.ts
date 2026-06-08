import "./styles.css";
import { loadChannelPlaneSet, loadImageMetadata } from "./zarrLoader";
import { orchestraFigureLayout } from "./orchestraFigureLayout";
import { ZARR_IMAGE_SOURCES } from "./zarrSources";
import type { ChannelRenderSettings, LoadedPlaneSet, ZarrImageMetadata, ZarrImageSource } from "./types";
import {
  createImageGridRenderer,
  type FigureGridLayout,
  type ImageGridRenderer,
  type TilePlacement,
} from "./webgpuImageRenderer";

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
const VOLE_VIEWER_URL = "https://vole.allencell.org/viewer";
const USE_ORCHESTRA_FIGURE_LAYOUT = true;

interface LoadedImageState {
  source: ZarrImageSource;
  tileId: number;
  timeIndexOverride?: number;
  placement?: TilePlacement;
  metadata?: ZarrImageMetadata;
  planeSet?: LoadedPlaneSet;
}

interface ImageLoadEntry {
  source: ZarrImageSource;
  timeIndexOverride?: number;
  placement?: TilePlacement;
}

interface AppState {
  images: LoadedImageState[];
  currentT: number;
  currentZ: number;
  resolutionTarget: number;
  maxTimeIndex: number;
  maxZIndex: number;
  channels: ChannelRenderSettings[];
  axesInitialized: boolean;
  usingFigureLayout: boolean;
}

const loadButton = requireElement<HTMLButtonElement>("#load-button");
const statusText = requireElement<HTMLElement>("#status-text");
const renderStats = requireElement<HTMLElement>("#render-stats");
const grid = requireElement<HTMLElement>("#canvas-grid");
const timeSlider = requireElement<HTMLInputElement>("#time-slider");
const timeValue = requireElement<HTMLOutputElement>("#time-value");
const zSlider = requireElement<HTMLInputElement>("#z-slider");
const zValue = requireElement<HTMLOutputElement>("#z-value");
const resolutionSlider = requireElement<HTMLInputElement>("#resolution-slider");
const resolutionValue = requireElement<HTMLOutputElement>("#resolution-value");
const channelControls = requireElement<HTMLElement>("#channel-controls");

const appState: AppState = {
  images: [],
  currentT: 0,
  currentZ: 0,
  resolutionTarget: Number(resolutionSlider.value),
  maxTimeIndex: 0,
  maxZIndex: 0,
  channels: [],
  axesInitialized: false,
  usingFigureLayout: USE_ORCHESTRA_FIGURE_LAYOUT,
};

let renderer: ImageGridRenderer | undefined;
let currentLoadAbortController: AbortController | undefined;
let currentRenderAbortController: AbortController | undefined;
let resolutionReloadTimer = 0;

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
resolutionSlider.addEventListener("input", () => {
  appState.resolutionTarget = Number(resolutionSlider.value);
  renderControls();
  scheduleResolutionReload();
});

async function init(): Promise<void> {
  renderer = await createImageGridRenderer(
    grid,
    (stats) => {
      renderStats.textContent = `${stats.rendered} rendered / ${stats.visible} visible`;
    },
    (tileId) => openTileInVole(tileId),
  );
  statusText.textContent = "WebGPU ready.";
  renderControls();
  await loadSources();
}

async function loadSources(): Promise<void> {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  window.clearTimeout(resolutionReloadTimer);
  currentLoadAbortController?.abort();
  currentRenderAbortController?.abort();
  const abortController = new AbortController();
  currentLoadAbortController = abortController;
  const imageEntries = getImageLoadEntries();

  activeRenderer.clear();
  if (appState.usingFigureLayout) {
    activeRenderer.setFigureGridLayout(getFigureGridLayout());
  } else {
    activeRenderer.setAutoGridLayout();
  }
  appState.images = [];
  renderControls();

  if (imageEntries.length === 0) {
    statusText.textContent = "No configured Zarr sources.";
    return;
  }

  statusText.textContent = `Reading metadata for ${imageEntries.length} image${imageEntries.length === 1 ? "" : "s"}.`;

  appState.images = imageEntries.map((entry, index) => {
    const tile = activeRenderer.addTile({
      title: entry.source.label || `Image ${index + 1}`,
      subtitle: entry.source.url,
      placement: entry.placement,
    });
    return {
      source: entry.source,
      tileId: tile.id,
      timeIndexOverride: entry.timeIndexOverride,
      placement: entry.placement,
    };
  });

  const tasks = appState.images.map((imageState) => async (): Promise<void> => {
    activeRenderer.setTileLoading(imageState.tileId, "Reading metadata");

    try {
      imageState.metadata = await loadImageMetadata({
        source: imageState.source,
        resolutionTarget: appState.resolutionTarget,
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
  statusText.textContent = appState.usingFigureLayout
    ? `Rendering figure layout at Z${appState.currentZ} with ${selectedChannels} channel${selectedChannels === 1 ? "" : "s"}.`
    : `Rendering T${appState.currentT} Z${appState.currentZ} with ${selectedChannels} channel${selectedChannels === 1 ? "" : "s"}.`;

  const tasks = imagesWithMetadata.map((imageState) => async (): Promise<void> => {
    const { metadata } = imageState;
    activeRenderer.setTileLoading(imageState.tileId, formatRenderSubtitle(metadata, "Loading"));

    try {
      imageState.planeSet = await loadChannelPlaneSet({
        metadata,
        timeIndex: getImageTimeIndex(imageState),
        zIndex: appState.currentZ,
        signal: abortController.signal,
      });

      activeRenderer.uploadChannelPlanes(imageState.tileId, imageState.planeSet, appState.channels);
      activeRenderer.updateTile(imageState.tileId, {
        subtitle: formatPlaneSetSubtitle(imageState.planeSet, appState.channels),
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

function updateCachedChannelRendering(): void {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  const imagesWithPlanes = appState.images.filter(hasCurrentPlaneSet);
  if (imagesWithPlanes.length === 0) {
    statusText.textContent = "Channel settings will apply after image data finishes loading.";
    return;
  }

  for (const imageState of imagesWithPlanes) {
    activeRenderer.updateChannelSettings(imageState.tileId, appState.channels);
    activeRenderer.updateTile(imageState.tileId, {
      subtitle: formatPlaneSetSubtitle(imageState.planeSet, appState.channels),
    });
  }

  statusText.textContent = `Updated channel shader settings for ${imagesWithPlanes.length} image${imagesWithPlanes.length === 1 ? "" : "s"}.`;
}

function configureGlobalStateFromImages(): void {
  const metadata = appState.images.flatMap((imageState) => imageState.metadata ? [imageState.metadata] : []);
  const maxT = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.t));
  const maxZ = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.z));
  const maxChannels = Math.max(1, ...metadata.map((item) => item.shapeTCZYX.c));

  appState.maxTimeIndex = maxT - 1;
  appState.maxZIndex = maxZ - 1;

  if (appState.usingFigureLayout) {
    appState.currentT = 0;
    appState.currentZ = appState.axesInitialized
      ? clampIndex(appState.currentZ, maxZ)
      : Math.floor(appState.maxZIndex / 2);
    appState.axesInitialized = true;
  } else if (appState.axesInitialized) {
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
    const layoutChannel = appState.usingFigureLayout
      ? getFigureLayoutChannelSettings(index)
      : undefined;
    return {
      index,
      enabled: previous?.enabled ?? layoutChannel?.enabled ?? true,
      color: previous?.color ?? layoutChannel?.color ?? DEFAULT_CHANNEL_COLORS[index % DEFAULT_CHANNEL_COLORS.length],
      min: previous ? previous.min : layoutChannel?.min ?? null,
      max: previous ? previous.max : layoutChannel?.max ?? null,
    };
  });
}

function renderControls(): void {
  timeSlider.max = String(appState.maxTimeIndex);
  timeSlider.value = String(appState.currentT);
  timeSlider.disabled = appState.usingFigureLayout || appState.maxTimeIndex === 0;
  timeValue.value = appState.usingFigureLayout
    ? "layout T"
    : `T${appState.currentT} / T${appState.maxTimeIndex}`;

  zSlider.max = String(appState.maxZIndex);
  zSlider.value = String(appState.currentZ);
  zSlider.disabled = appState.maxZIndex === 0;
  zValue.value = `Z${appState.currentZ} / Z${appState.maxZIndex}`;

  resolutionSlider.value = String(appState.resolutionTarget);
  resolutionValue.value = `${appState.resolutionTarget} px`;

  channelControls.textContent = "";
  for (const channel of appState.channels) {
    channelControls.append(createChannelControl(channel));
  }
}

function scheduleResolutionReload(): void {
  window.clearTimeout(resolutionReloadTimer);
  resolutionReloadTimer = window.setTimeout(() => {
    void loadSources();
  }, 350);
}

function openTileInVole(tileId: number): void {
  const imageState = appState.images.find((item) => item.tileId === tileId);
  if (!imageState || !hasMetadata(imageState)) {
    statusText.textContent = "Image metadata is not ready yet.";
    return;
  }

  const voleUrl = buildVoleUrl(imageState);
  const openedWindow = window.open(voleUrl, "_blank");
  if (!openedWindow) {
    statusText.textContent = "Could not open Vol-E. Allow pop-ups for this page and try again.";
    return;
  }

  openedWindow.opener = null;
  statusText.textContent = "Opened image in Vol-E.";
}

function buildVoleUrl(imageState: LoadedImageState & { metadata: ZarrImageMetadata }): string {
  const { metadata } = imageState;
  const url = new URL(VOLE_VIEWER_URL);
  const zIndex = clampIndex(appState.currentZ, metadata.shapeTCZYX.z);
  const timeIndex = clampIndex(getImageTimeIndex(imageState), metadata.shapeTCZYX.t);
  const zSlice = metadata.shapeTCZYX.z <= 1 ? 0.5 : zIndex / (metadata.shapeTCZYX.z - 1);

  url.searchParams.set("url", metadata.source.url);
  url.searchParams.set("view", "Z");
  url.searchParams.set("t", String(timeIndex));
  url.searchParams.set("slice", `0.5,0.5,${formatUnitInterval(zSlice)}`);

  for (let index = 0; index < metadata.shapeTCZYX.c; index++) {
    const channel = appState.channels[index];
    if (!channel?.enabled) {
      url.searchParams.set(`c${index}`, "ven:0");
      continue;
    }

    url.searchParams.set(`c${index}`, `ven:1,col:${stripHexPrefix(channel.color)}`);
  }

  return url.toString();
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
    updateCachedChannelRendering();
  });

  const color = document.createElement("input");
  color.type = "color";
  color.value = channel.color;
  color.disabled = !channel.enabled;
  color.addEventListener("input", () => {
    channel.color = color.value;
    updateCachedChannelRendering();
  });

  const name = document.createElement("span");
  name.textContent = `C${channel.index}`;

  const min = createThresholdInput(channel, "min");
  const max = createThresholdInput(channel, "max");

  label.append(checkbox, color, name, min, max);
  return label;
}

function createThresholdInput(
  channel: ChannelRenderSettings,
  key: "min" | "max",
): HTMLInputElement {
  const input = document.createElement("input");
  input.type = "number";
  input.inputMode = "decimal";
  input.step = "1";
  input.placeholder = `auto ${key}`;
  input.value = formatThresholdInputValue(channel[key]);
  input.disabled = !channel.enabled;
  input.title = `C${channel.index} ${key} threshold`;
  input.setAttribute("aria-label", `C${channel.index} ${key} threshold`);
  input.addEventListener("input", () => {
    channel[key] = parseOptionalNumber(input.value);
    updateCachedChannelRendering();
  });
  return input;
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
    `target ${metadata.resolutionTarget}px`,
    `TCZYX [${formatTCZYX(metadata.shapeTCZYX)}]`,
    metadata.dtype,
  ].join("  ");
}

function formatRenderSubtitle(metadata: ZarrImageMetadata, state: string): string {
  return [
    state,
    `level ${metadata.multiresolutionLevel}`,
    `target ${metadata.resolutionTarget}px`,
    `TCZYX [${formatTCZYX(metadata.shapeTCZYX)}]`,
  ].join("  ");
}

function formatPlaneSetSubtitle(
  planeSet: LoadedPlaneSet,
  channels: readonly ChannelRenderSettings[],
): string {
  const enabledRanges = planeSet.channelPlanes.filter((plane) => channels[plane.channelIndex]?.enabled);
  const renderedChannels = enabledRanges.length
    ? enabledRanges.map((plane) => `C${plane.channelIndex}`).join(", ")
    : "none";
  const ranges = enabledRanges.length
    ? enabledRanges
      .map((range) => {
        const [thresholdMin, thresholdMax] = resolveChannelThresholdRange(channels[range.channelIndex], range);
        return `C${range.channelIndex} data ${formatNumber(range.min)}-${formatNumber(range.max)} thr ${formatNumber(thresholdMin)}-${formatNumber(thresholdMax)}`;
      })
      .join("; ")
    : "no channel ranges";

  return [
    `${planeSet.width} x ${planeSet.height}`,
    `T${planeSet.timeIndex} Z${planeSet.zIndex}`,
    `channels ${renderedChannels}`,
    `level ${planeSet.multiresolutionLevel}`,
    `target ${planeSet.resolutionTarget}px`,
    `path ${planeSet.arrayPath || "/"}`,
    `TCZYX [${formatTCZYX(planeSet.shapeTCZYX)}]`,
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

function formatThresholdInputValue(value: number | null): string {
  return value === null ? "" : String(value);
}

function parseOptionalNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;

  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function resolveChannelThresholdRange(
  channel: ChannelRenderSettings | undefined,
  fallbackRange: { min: number; max: number; autoMin?: number; autoMax?: number },
): [number, number] {
  const autoMin = fallbackRange.autoMin ?? fallbackRange.min;
  const autoMax = fallbackRange.autoMax ?? fallbackRange.max;
  const min = channel?.min ?? autoMin;
  let max = channel?.max ?? autoMax;

  if (max <= min) {
    max = min + 1;
  }

  return [min, max];
}

function formatUnitInterval(value: number): string {
  return Math.max(0, Math.min(1, value)).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function stripHexPrefix(color: string): string {
  return color.startsWith("#") ? color.slice(1) : color;
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

function hasCurrentPlaneSet(imageState: LoadedImageState): imageState is LoadedImageState & {
  metadata: ZarrImageMetadata;
  planeSet: LoadedPlaneSet;
} {
  if (!imageState.metadata || !imageState.planeSet) return false;

  const expectedT = clampIndex(getImageTimeIndex(imageState), imageState.metadata.shapeTCZYX.t);
  const expectedZ = clampIndex(appState.currentZ, imageState.metadata.shapeTCZYX.z);

  return imageState.planeSet.timeIndex === expectedT
    && imageState.planeSet.zIndex === expectedZ
    && imageState.planeSet.arrayPath === imageState.metadata.arrayPath
    && imageState.planeSet.resolutionTarget === imageState.metadata.resolutionTarget;
}

function getImageLoadEntries(): ImageLoadEntry[] {
  if (!appState.usingFigureLayout) {
    return ZARR_IMAGE_SOURCES.map((source) => ({ source }));
  }

  return orchestraFigureLayout.cells
    .map((cell, index): ImageLoadEntry | undefined => {
      const filePath = cell.file_paths[0];
      if (!filePath) return undefined;

      const url = normalizeFigureUrl(filePath);
      return {
        source: {
          label: makeFigureCellLabel(url, index),
          url,
        },
        timeIndexOverride: cell.t_indices[0] ?? orchestraFigureLayout.default_t,
        placement: {
          row: cell.row,
          col: cell.col,
          rowSpan: cell.row_span,
          colSpan: cell.col_span,
        },
      };
    })
    .filter(isPresent);
}

function getFigureGridLayout(): FigureGridLayout {
  return {
    rows: orchestraFigureLayout.rows,
    cols: orchestraFigureLayout.cols,
    rowLabels: orchestraFigureLayout.row_labels,
    colLabels: orchestraFigureLayout.col_labels,
  };
}

function getFigureLayoutChannelSettings(
  index: number,
): Pick<ChannelRenderSettings, "enabled" | "color" | "min" | "max"> | undefined {
  const lut = orchestraFigureLayout.lut_groups[0]?.channel_luts.find((channel) => channel.channel_idx === index);
  if (!lut) return undefined;

  return {
    enabled: lut.enabled,
    color: rgbToHex(lut.color),
    min: parseOptionalNumber(lut.vmin),
    max: parseOptionalNumber(lut.vmax),
  };
}

function getImageTimeIndex(imageState: LoadedImageState): number {
  return imageState.timeIndexOverride ?? appState.currentT;
}

function normalizeFigureUrl(filePath: string): string {
  return filePath.replace(/\\/g, "/").replace(/^https:\/?\/?/, "https://");
}

function makeFigureCellLabel(url: string, index: number): string {
  const fileName = url.split("/").at(-1);
  return fileName || `Figure cell ${index + 1}`;
}

function rgbToHex(color: readonly [number, number, number]): string {
  return `#${color.map((component) => clampColorByte(component).toString(16).padStart(2, "0")).join("")}`;
}

function clampColorByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
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

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

import "./styles.css";
import {
  getZarrPlaneCacheStats,
  loadChannelPlaneSet,
  loadImageMetadata,
  prefetchChannelPlaneSet,
} from "./zarrLoader";
import { orchestraFigureLayout } from "./orchestraFigureLayout";
import { ZARR_IMAGE_SOURCES } from "./zarrSources";
import type {
  ChannelRenderSettings,
  LoadedChannelPlane,
  LoadedPlaneSet,
  ZarrImageMetadata,
  ZarrImageSource,
} from "./types";
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
const PLAYBACK_FRAME_INTERVAL_MS = 350;
const PREFETCH_START_DELAY_MS = 75;
const PREFETCH_MAX_CONCURRENT_PLANE_SETS = 2;
const PREFETCH_CACHE_HIGH_WATERMARK = 0.95;

interface LoadedImageState {
  source: ZarrImageSource;
  tileId: number;
  channelOverrides: ImageChannelOverride[];
  timeIndexOverride?: number;
  placement?: TilePlacement;
  metadata?: ZarrImageMetadata;
  planeSet?: LoadedPlaneSet;
}

interface ImageChannelOverride {
  enabled: boolean | null;
  color: string | null;
  min: number | null;
  max: number | null;
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
  isPlaying: boolean;
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
const playButton = requireElement<HTMLButtonElement>("#play-button");
const pauseButton = requireElement<HTMLButtonElement>("#pause-button");
const stopButton = requireElement<HTMLButtonElement>("#stop-button");
const channelControls = requireElement<HTMLElement>("#channel-controls");
const datasetModeInputs = Array.from(document.querySelectorAll<HTMLInputElement>("input[name='dataset-mode']"));

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
  isPlaying: false,
};

let renderer: ImageGridRenderer | undefined;
let currentLoadAbortController: AbortController | undefined;
let currentRenderAbortController: AbortController | undefined;
let resolutionReloadTimer = 0;
let playbackTimer = 0;
let playbackFrameInFlight = false;
let prefetchTimer = 0;
let prefetchAbortController: AbortController | undefined;
let closeActiveSettingsPopup: (() => void) | undefined;
let activeSettingsPopupTileId: number | undefined;

init().catch((error) => {
  console.error(error);
  statusText.textContent = getErrorMessage(error);
});

loadButton.addEventListener("click", () => {
  pausePlayback({ announce: false });
  void loadSources();
});
playButton.addEventListener("click", () => startPlayback());
pauseButton.addEventListener("click", () => pausePlayback());
stopButton.addEventListener("click", () => stopPlayback());
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
for (const input of datasetModeInputs) {
  input.addEventListener("change", () => {
    if (!input.checked) return;

    setDatasetMode(input.value === "figure");
  });
}

async function init(): Promise<void> {
  renderer = await createImageGridRenderer(
    grid,
    (stats) => {
      renderStats.textContent = `${stats.rendered} rendered / ${stats.visible} visible`;
    },
    (tileId) => openTileInVole(tileId),
    (tileId, anchor) => openTileSettings(tileId, anchor),
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
  abortPlaybackPrefetch();
  closeActiveSettingsPopup?.();
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
      channelOverrides: [],
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
  abortPlaybackPrefetch();
  closeActiveSettingsPopup?.();
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

      const channels = getImageChannelSettings(imageState);
      activeRenderer.uploadChannelPlanes(imageState.tileId, imageState.planeSet, channels);
      activeRenderer.updateTile(imageState.tileId, {
        subtitle: formatPlaneSetSubtitle(imageState.planeSet, channels),
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

  renderControls();
  const failed = results.filter((result) => result.status === "rejected").length;
  const rendered = imagesWithMetadata.length - failed;
  statusText.textContent = failed
    ? `Rendered ${rendered}; ${failed} failed.`
    : `Rendered ${rendered} image${rendered === 1 ? "" : "s"}.`;
  schedulePlaybackPrefetch();
}

function updateCachedChannelRendering(): void {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  const imagesWithPlanes = appState.images.filter(hasCurrentPlaneSet);
  if (imagesWithPlanes.length === 0) {
    statusText.textContent = "Channel settings will apply after image data finishes loading.";
    return;
  }

  updateLoadedTileChannelSettings();

  statusText.textContent = `Updated channel shader settings for ${imagesWithPlanes.length} image${imagesWithPlanes.length === 1 ? "" : "s"}.`;
}

function updateLoadedTileChannelSettings(): void {
  for (const imageState of appState.images.filter(hasCurrentPlaneSet)) {
    updateImageTileChannelSettings(imageState);
  }
}

function updateImageTileChannelSettings(imageState: LoadedImageState): void {
  const activeRenderer = renderer;
  if (!activeRenderer || !hasCurrentPlaneSet(imageState)) return;

  const channels = getImageChannelSettings(imageState);
  activeRenderer.updateChannelSettings(imageState.tileId, channels);
  activeRenderer.updateTile(imageState.tileId, {
    subtitle: formatPlaneSetSubtitle(imageState.planeSet, channels),
  });
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
  for (const input of datasetModeInputs) {
    input.checked = appState.usingFigureLayout
      ? input.value === "figure"
      : input.value === "sources";
  }

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

  const playbackAvailable = isPlaybackAvailable();
  playButton.disabled = !playbackAvailable || appState.isPlaying;
  pauseButton.disabled = !appState.isPlaying;
  stopButton.disabled = !playbackAvailable && appState.currentT === 0;

  channelControls.textContent = "";
  for (const channel of appState.channels) {
    channelControls.append(createChannelControl(channel));
  }
}

function setDatasetMode(usingFigureLayout: boolean): void {
  if (appState.usingFigureLayout === usingFigureLayout) return;

  pausePlayback({ announce: false });
  appState.usingFigureLayout = usingFigureLayout;
  appState.axesInitialized = false;
  appState.currentT = 0;
  appState.currentZ = 0;
  appState.maxTimeIndex = 0;
  appState.maxZIndex = 0;
  appState.channels = [];
  closeActiveSettingsPopup?.();
  renderControls();
  statusText.textContent = usingFigureLayout
    ? "Switching to orchestra figure layout."
    : "Switching to configured Zarr sources.";
  void loadSources();
}

function startPlayback(): void {
  if (!isPlaybackAvailable()) {
    statusText.textContent = appState.usingFigureLayout
      ? "Switch to Zarr sources to play through time."
      : "Playback needs a loaded dataset with more than one timepoint.";
    renderControls();
    return;
  }

  if (appState.currentT >= appState.maxTimeIndex) {
    appState.currentT = 0;
  }

  appState.isPlaying = true;
  renderControls();
  statusText.textContent = "Playing through time.";
  schedulePlaybackPrefetch();
  queueNextPlaybackFrame(0);
}

function pausePlayback(options: { announce?: boolean } = {}): void {
  if (!appState.isPlaying && playbackTimer === 0 && prefetchTimer === 0 && !prefetchAbortController) return;

  appState.isPlaying = false;
  window.clearTimeout(playbackTimer);
  playbackTimer = 0;
  abortPlaybackPrefetch();
  renderControls();

  if (options.announce ?? true) {
    statusText.textContent = `Paused at T${appState.currentT}.`;
  }
}

function stopPlayback(): void {
  const shouldRender = appState.currentT !== 0 && appState.images.some(hasMetadata);
  pausePlayback({ announce: false });
  appState.currentT = 0;
  renderControls();
  statusText.textContent = "Stopped playback.";

  if (shouldRender) {
    void renderLoadedImages();
  }
}

function queueNextPlaybackFrame(delayMs = PLAYBACK_FRAME_INTERVAL_MS): void {
  window.clearTimeout(playbackTimer);
  if (!appState.isPlaying) return;

  playbackTimer = window.setTimeout(() => {
    void advancePlaybackFrame();
  }, delayMs);
}

async function advancePlaybackFrame(): Promise<void> {
  if (!appState.isPlaying || playbackFrameInFlight) return;

  if (!isPlaybackAvailable()) {
    pausePlayback({ announce: false });
    statusText.textContent = "Playback stopped because time data is no longer available.";
    return;
  }

  playbackFrameInFlight = true;
  try {
    appState.currentT = appState.currentT >= appState.maxTimeIndex
      ? 0
      : appState.currentT + 1;
    renderControls();
    await renderLoadedImages();
  } finally {
    playbackFrameInFlight = false;
    queueNextPlaybackFrame();
  }
}

function isPlaybackAvailable(): boolean {
  return !appState.usingFigureLayout
    && appState.maxTimeIndex > 0
    && appState.images.some(hasMetadata);
}

function schedulePlaybackPrefetch(): void {
  window.clearTimeout(prefetchTimer);
  prefetchTimer = 0;
  if (!appState.isPlaying || !isPlaybackAvailable()) return;

  prefetchTimer = window.setTimeout(() => {
    void prefetchPlaybackFrames();
  }, PREFETCH_START_DELAY_MS);
}

async function prefetchPlaybackFrames(): Promise<void> {
  abortPlaybackPrefetch();
  if (!appState.isPlaying || !isPlaybackAvailable()) return;

  const abortController = new AbortController();
  prefetchAbortController = abortController;
  const zIndex = appState.currentZ;
  const metadata = appState.images.filter(hasMetadata).map((imageState) => imageState.metadata);
  const timepoints = getPlaybackPrefetchTimepoints();
  let prefetchedTimepoints = 0;

  try {
    for (const timeIndex of timepoints) {
      if (abortController.signal.aborted || !isPlaybackAvailable()) return;
      if (!appState.isPlaying) return;

      const cacheStats = getZarrPlaneCacheStats();
      if (
        prefetchedTimepoints > 0
        && cacheStats.bytes >= cacheStats.maxBytes * PREFETCH_CACHE_HIGH_WATERMARK
      ) {
        return;
      }

      const tasks = metadata.map((item) => async (): Promise<void> => {
        await prefetchChannelPlaneSet({
          metadata: item,
          timeIndex,
          zIndex,
          signal: abortController.signal,
        });
      });
      const results = await runLimited(tasks, PREFETCH_MAX_CONCURRENT_PLANE_SETS);
      if (abortController.signal.aborted) return;

      const failed = results.filter((result) => result.status === "rejected").length;
      if (failed > 0) {
        console.debug(`Prefetch skipped ${failed} plane set${failed === 1 ? "" : "s"} for T${timeIndex}.`);
      }
      prefetchedTimepoints++;
    }
  } finally {
    if (prefetchAbortController === abortController) {
      prefetchAbortController = undefined;
    }
  }
}

function getPlaybackPrefetchTimepoints(): number[] {
  const frameCount = appState.maxTimeIndex + 1;
  if (frameCount <= 1) return [];

  return Array.from({ length: frameCount - 1 }, (_, index) => (
    appState.currentT + index + 1
  ) % frameCount);
}

function abortPlaybackPrefetch(): void {
  window.clearTimeout(prefetchTimer);
  prefetchTimer = 0;
  prefetchAbortController?.abort();
  prefetchAbortController = undefined;
}

function scheduleResolutionReload(): void {
  window.clearTimeout(resolutionReloadTimer);
  pausePlayback({ announce: false });
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

function openTileSettings(tileId: number, anchor: HTMLElement): void {
  const imageState = appState.images.find((item) => item.tileId === tileId);
  if (!imageState || !hasCurrentPlaneSet(imageState)) {
    statusText.textContent = "Image data is not ready for per-image settings yet.";
    return;
  }

  if (activeSettingsPopupTileId === tileId && closeActiveSettingsPopup) {
    closeActiveSettingsPopup();
    return;
  }

  closeActiveSettingsPopup?.();

  const popup = document.createElement("div");
  popup.className = "image-settings-popup";
  popup.role = "dialog";
  popup.ariaModal = "false";
  popup.tabIndex = -1;
  popup.style.visibility = "hidden";

  const panel = document.createElement("div");
  panel.className = "image-settings-panel";

  const header = document.createElement("div");
  header.className = "image-settings-header";

  const titleBlock = document.createElement("div");
  titleBlock.className = "image-settings-title";

  const title = document.createElement("h2");
  title.textContent = imageState.source.label || "Image settings";

  const details = document.createElement("p");
  details.textContent = [
    `${imageState.planeSet.width} x ${imageState.planeSet.height}`,
    `T${imageState.planeSet.timeIndex} Z${imageState.planeSet.zIndex}`,
    `level ${imageState.planeSet.multiresolutionLevel}`,
  ].join("  ");

  titleBlock.append(title, details);

  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.className = "popup-close-button";
  closeButton.textContent = "Close";

  header.append(titleBlock, closeButton);

  const rows = document.createElement("div");
  rows.className = "image-threshold-list";
  for (const plane of imageState.planeSet.channelPlanes.slice().sort((a, b) => a.channelIndex - b.channelIndex)) {
    rows.append(createImageThresholdRow(imageState, plane));
  }

  panel.append(header, rows);
  popup.append(panel);
  document.body.append(popup);

  const positionPopup = (): void => positionSettingsPopup(popup, anchor);
  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.key === "Escape") {
      closeActiveSettingsPopup?.();
    }
  };
  const onPointerDown = (event: PointerEvent): void => {
    const target = event.target;
    if (!(target instanceof Node)) return;

    if (!popup.contains(target) && !anchor.contains(target)) {
      closeActiveSettingsPopup?.();
    }
  };
  let pointerListenerAttached = false;
  const pointerListenerTimer = window.setTimeout(() => {
    pointerListenerAttached = true;
    document.addEventListener("pointerdown", onPointerDown);
  }, 0);

  closeActiveSettingsPopup = () => {
    window.clearTimeout(pointerListenerTimer);
    if (pointerListenerAttached) {
      document.removeEventListener("pointerdown", onPointerDown);
    }
    document.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("resize", positionPopup);
    window.removeEventListener("scroll", positionPopup, true);
    popup.remove();
    closeActiveSettingsPopup = undefined;
    activeSettingsPopupTileId = undefined;
  };
  activeSettingsPopupTileId = tileId;
  closeButton.addEventListener("click", () => closeActiveSettingsPopup?.());
  document.addEventListener("keydown", onKeyDown);
  window.addEventListener("resize", positionPopup);
  window.addEventListener("scroll", positionPopup, true);

  positionPopup();
  popup.style.visibility = "";
  popup.focus({ preventScroll: true });
}

function positionSettingsPopup(popup: HTMLElement, anchor: HTMLElement): void {
  const margin = 10;
  const gap = 8;
  const anchorRect = anchor.getBoundingClientRect();
  const popupRect = popup.getBoundingClientRect();
  const left = Math.max(
    margin,
    Math.min(anchorRect.right - popupRect.width, window.innerWidth - popupRect.width - margin),
  );
  const belowTop = anchorRect.bottom + gap;
  const aboveTop = anchorRect.top - popupRect.height - gap;
  const top = belowTop + popupRect.height <= window.innerHeight - margin
    ? belowTop
    : Math.max(margin, aboveTop);

  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
}

function createImageThresholdRow(
  imageState: LoadedImageState & { metadata: ZarrImageMetadata; planeSet: LoadedPlaneSet },
  plane: LoadedChannelPlane,
): HTMLElement {
  const row = document.createElement("div");
  row.className = "image-threshold-row";

  const globalChannel = appState.channels[plane.channelIndex];
  const isGloballyEnabled = globalChannel?.enabled ?? true;
  const readChannelSettings = (): ChannelRenderSettings => getImageChannelSettings(imageState)[plane.channelIndex] ?? {
    index: plane.channelIndex,
    enabled: true,
    color: DEFAULT_CHANNEL_COLORS[plane.channelIndex % DEFAULT_CHANNEL_COLORS.length],
    min: null,
    max: null,
  };

  const channelControl = document.createElement("div");
  channelControl.className = "image-channel-control";

  const enabled = document.createElement("input");
  enabled.type = "checkbox";
  enabled.checked = readChannelSettings().enabled;
  enabled.disabled = !isGloballyEnabled;
  enabled.title = isGloballyEnabled
    ? `Show C${plane.channelIndex} in this image`
    : `C${plane.channelIndex} is disabled by the global channel setting`;
  enabled.setAttribute("aria-label", `Show C${plane.channelIndex} in this image`);

  const color = document.createElement("input");
  color.type = "color";
  color.value = readChannelSettings().color;
  color.disabled = !isGloballyEnabled;
  color.title = `C${plane.channelIndex} color for this image`;
  color.setAttribute("aria-label", `C${plane.channelIndex} color for this image`);

  const labelText = document.createElement("span");
  labelText.textContent = `C${plane.channelIndex}`;
  channelControl.append(enabled, color, labelText);

  const domain = getPlaneThresholdDomain(plane);
  const minControl = createPopupThresholdControl("Min", `C${plane.channelIndex} min threshold`, domain);
  const maxControl = createPopupThresholdControl("Max", `C${plane.channelIndex} max threshold`, domain);

  const autoButton = document.createElement("button");
  autoButton.type = "button";
  autoButton.className = "image-threshold-auto-button";
  autoButton.textContent = "Auto";

  const readRange = (): { min: number; max: number } => {
    return getChannelThresholdDisplayRange(readChannelSettings(), getPlaneAutoThresholdRange(plane));
  };

  const syncControls = (): void => {
    const channelSettings = readChannelSettings();
    const range = readRange();
    enabled.checked = channelSettings.enabled;
    color.value = channelSettings.color;
    minControl.slider.value = String(clampThresholdValue(range.min, domain));
    minControl.number.value = formatThresholdInputValue(range.min);
    maxControl.slider.value = String(clampThresholdValue(range.max, domain));
    maxControl.number.value = formatThresholdInputValue(range.max);
  };

  const writeRange = (
    key: "min" | "max",
    value: unknown,
    options: { syncInvalid?: boolean } = {},
  ): void => {
    const parsed = parseOptionalNumber(value);
    if (parsed === null) {
      if (options.syncInvalid ?? true) {
        syncControls();
      }
      return;
    }

    const current = readRange();
    let min = current.min;
    let max = current.max;
    if (key === "min") {
      min = clampThresholdValue(parsed, domain);
      if (max < min) max = min;
    } else {
      max = clampThresholdValue(parsed, domain);
      if (min > max) min = max;
    }

    setImageChannelThreshold(imageState, plane.channelIndex, min, max);
    syncControls();
    updateImageTileChannelSettings(imageState);
  };

  enabled.addEventListener("change", () => {
    setImageChannelEnabled(imageState, plane.channelIndex, enabled.checked);
    updateImageTileChannelSettings(imageState);
  });
  color.addEventListener("input", () => {
    setImageChannelColor(imageState, plane.channelIndex, color.value);
    updateImageTileChannelSettings(imageState);
  });
  minControl.slider.addEventListener("input", () => writeRange("min", minControl.slider.value));
  minControl.number.addEventListener("input", () => writeRange("min", minControl.number.value, { syncInvalid: false }));
  minControl.number.addEventListener("change", () => writeRange("min", minControl.number.value));
  maxControl.slider.addEventListener("input", () => writeRange("max", maxControl.slider.value));
  maxControl.number.addEventListener("input", () => writeRange("max", maxControl.number.value, { syncInvalid: false }));
  maxControl.number.addEventListener("change", () => writeRange("max", maxControl.number.value));
  autoButton.addEventListener("click", () => {
    if (!applyImageAutoThreshold(imageState, plane.channelIndex)) return;

    syncControls();
    updateImageTileChannelSettings(imageState);
    statusText.textContent = `Set C${plane.channelIndex} auto thresholds for ${imageState.source.label || "image"}.`;
  });

  syncControls();
  row.append(channelControl, minControl.element, maxControl.element, autoButton);
  return row;
}

function buildVoleUrl(imageState: LoadedImageState & { metadata: ZarrImageMetadata }): string {
  const { metadata } = imageState;
  const url = new URL(VOLE_VIEWER_URL);
  const zIndex = clampIndex(appState.currentZ, metadata.shapeTCZYX.z);
  const timeIndex = clampIndex(getImageTimeIndex(imageState), metadata.shapeTCZYX.t);
  const zSlice = metadata.shapeTCZYX.z <= 1 ? 0.5 : zIndex / (metadata.shapeTCZYX.z - 1);
  const channels = getResolvedChannelSettingsForImage(imageState);

  url.searchParams.set("url", metadata.source.url);
  url.searchParams.set("view", "Z");
  url.searchParams.set("t", String(timeIndex));
  url.searchParams.set("slice", `0.5,0.5,${formatUnitInterval(zSlice)}`);

  for (let index = 0; index < metadata.shapeTCZYX.c; index++) {
    const channel = channels[index];
    if (!channel?.enabled) {
      url.searchParams.set(`c${index}`, "ven:0");
      continue;
    }

    url.searchParams.set(`c${index}`, formatVoleChannelSetting(channel));
  }

  return url.toString();
}

function createChannelControl(channel: ChannelRenderSettings): HTMLElement {
  const control = document.createElement("div");
  control.className = "channel-control";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = channel.enabled;
  checkbox.addEventListener("change", () => {
    channel.enabled = checkbox.checked;
    applyGlobalChannelEnabled(channel.index, channel.enabled);
    renderControls();
    updateCachedChannelRendering();
  });

  const color = document.createElement("input");
  color.type = "color";
  color.value = channel.color;
  color.disabled = !channel.enabled;
  color.addEventListener("input", () => {
    channel.color = color.value;
    applyGlobalChannelColor(channel.index, channel.color);
    updateCachedChannelRendering();
  });

  const name = document.createElement("span");
  name.textContent = `C${channel.index}`;

  const auto = document.createElement("button");
  const autoRange = getChannelAutoThresholdRange(channel.index);
  auto.type = "button";
  auto.className = "channel-auto-button";
  auto.textContent = "Auto";
  auto.disabled = !channel.enabled || !autoRange;
  auto.title = `Set C${channel.index} auto thresholds for every loaded image`;
  auto.addEventListener("click", () => {
    applyGlobalAutoThreshold(channel.index);
  });

  control.append(checkbox, color, name, auto);
  return control;
}

interface ThresholdDomain {
  min: number;
  max: number;
  step: string;
}

interface PopupThresholdControl {
  element: HTMLElement;
  slider: HTMLInputElement;
  number: HTMLInputElement;
}

function createPopupThresholdControl(
  labelText: string,
  ariaLabel: string,
  domain: ThresholdDomain,
): PopupThresholdControl {
  const element = document.createElement("label");
  element.className = "image-threshold-control";

  const label = document.createElement("span");
  label.textContent = labelText;

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(domain.min);
  slider.max = String(domain.max);
  slider.step = domain.step;
  slider.setAttribute("aria-label", ariaLabel);

  const number = document.createElement("input");
  number.type = "number";
  number.inputMode = "decimal";
  number.min = String(domain.min);
  number.max = String(domain.max);
  number.step = domain.step;
  number.setAttribute("aria-label", ariaLabel);

  element.append(label, slider, number);
  return { element, slider, number };
}

function applyGlobalAutoThreshold(channelIndex: number): void {
  let updated = 0;
  for (const imageState of appState.images.filter(hasCurrentPlaneSet)) {
    if (applyImageAutoThreshold(imageState, channelIndex)) {
      updated++;
    }
  }

  renderControls();
  if (updated === 0) {
    statusText.textContent = `No loaded C${channelIndex} planes are ready for auto thresholds.`;
    return;
  }

  updateLoadedTileChannelSettings();
  statusText.textContent = `Set C${channelIndex} auto thresholds for ${updated} image${updated === 1 ? "" : "s"}.`;
}

function applyGlobalChannelEnabled(channelIndex: number, enabled: boolean): void {
  for (const imageState of appState.images) {
    setImageChannelEnabled(imageState, channelIndex, enabled);
  }
}

function applyGlobalChannelColor(channelIndex: number, color: string): void {
  for (const imageState of appState.images) {
    setImageChannelColor(imageState, channelIndex, color);
  }
}

function applyImageAutoThreshold(imageState: LoadedImageState, channelIndex: number): boolean {
  const autoRange = getImageChannelAutoThresholdRange(imageState, channelIndex);
  if (!autoRange) return false;

  setImageChannelThreshold(imageState, channelIndex, autoRange.min, autoRange.max);
  return true;
}

function setImageChannelThreshold(
  imageState: LoadedImageState,
  channelIndex: number,
  min: number,
  max: number,
): void {
  const override = getImageChannelOverride(imageState, channelIndex);
  override.min = min;
  override.max = max;
}

function setImageChannelEnabled(
  imageState: LoadedImageState,
  channelIndex: number,
  enabled: boolean,
): void {
  const override = getImageChannelOverride(imageState, channelIndex);
  override.enabled = enabled;
}

function setImageChannelColor(
  imageState: LoadedImageState,
  channelIndex: number,
  color: string,
): void {
  const override = getImageChannelOverride(imageState, channelIndex);
  override.color = color;
}

function getImageChannelOverride(
  imageState: LoadedImageState,
  channelIndex: number,
): ImageChannelOverride {
  const existing = imageState.channelOverrides[channelIndex];
  if (existing) return existing;

  const created = { enabled: null, color: null, min: null, max: null };
  imageState.channelOverrides[channelIndex] = created;
  return created;
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

function getImageChannelSettings(imageState: LoadedImageState): ChannelRenderSettings[] {
  return appState.channels.map((channel) => {
    const channelOverride = imageState.channelOverrides[channel.index];
    return {
      ...channel,
      enabled: channel.enabled && (channelOverride?.enabled ?? true),
      color: channelOverride?.color ?? channel.color,
      min: channelOverride?.min ?? channel.min,
      max: channelOverride?.max ?? channel.max,
    };
  });
}

function getResolvedChannelSettingsForImage(imageState: LoadedImageState): ChannelRenderSettings[] {
  return getImageChannelSettings(imageState).map((channel) => {
    const autoRange = getImageChannelAutoThresholdRange(imageState, channel.index)
      ?? getChannelAutoThresholdRange(channel.index);
    const { min, max } = getChannelThresholdDisplayRange(channel, autoRange);
    return { ...channel, min, max };
  });
}

function getChannelThresholdDisplayRange(
  channel: ChannelRenderSettings,
  autoRange = getChannelAutoThresholdRange(channel.index),
): { min: number; max: number } {
  const min = channel.min ?? autoRange?.min ?? 0;
  let max = channel.max ?? autoRange?.max ?? min + 1;

  if (max <= min) {
    max = min + 1;
  }

  return { min, max };
}

function getImageChannelAutoThresholdRange(
  imageState: LoadedImageState,
  channelIndex: number,
): { min: number; max: number } | undefined {
  if (!hasCurrentPlaneSet(imageState)) return undefined;

  const plane = imageState.planeSet.channelPlanes.find((item) => item.channelIndex === channelIndex);
  if (!plane) return undefined;

  const min = plane.autoMin;
  let max = plane.autoMax;
  if (max <= min) {
    max = min + 1;
  }

  return { min, max };
}

function getPlaneAutoThresholdRange(plane: LoadedChannelPlane): { min: number; max: number } {
  const min = plane.autoMin;
  let max = plane.autoMax;
  if (max <= min) {
    max = min + 1;
  }

  return { min, max };
}

function getPlaneThresholdDomain(plane: LoadedChannelPlane): ThresholdDomain {
  if (plane.nativePixels instanceof Uint8Array || plane.nativePixels instanceof Uint8ClampedArray) {
    return { min: 0, max: 255, step: "1" };
  }

  if (plane.nativePixels instanceof Uint16Array) {
    return { min: 0, max: 65535, step: "1" };
  }

  const min = Math.min(plane.min, plane.autoMin);
  let max = Math.max(plane.max, plane.autoMax);
  if (max <= min) {
    max = min + 1;
  }

  return { min, max, step: "any" };
}

function clampThresholdValue(value: number, domain: ThresholdDomain): number {
  if (!Number.isFinite(value)) return domain.min;
  return Math.max(domain.min, Math.min(value, domain.max));
}

function getChannelAutoThresholdRange(channelIndex: number): { min: number; max: number } | undefined {
  const autoRanges = appState.images
    .filter(hasCurrentPlaneSet)
    .flatMap((imageState) => imageState.planeSet.channelPlanes
      .filter((plane) => plane.channelIndex === channelIndex)
      .map((plane) => ({ min: plane.autoMin, max: plane.autoMax })));

  if (autoRanges.length === 0) return undefined;

  const min = Math.min(...autoRanges.map((range) => range.min));
  let max = Math.max(...autoRanges.map((range) => range.max));

  if (max <= min) {
    max = min + 1;
  }

  return { min, max };
}

function formatThresholdInputValue(value: number): string {
  if (!Number.isFinite(value)) return "0";
  if (Number.isInteger(value)) return String(value);
  return Number(value.toPrecision(6)).toString();
}

function formatVoleChannelSetting(channel: ChannelRenderSettings): string {
  const { min, max } = getChannelThresholdDisplayRange(channel);
  return [
    "ven:1",
    `col:${stripHexPrefix(channel.color)}`,
    `min:${formatVoleNumber(min)}`,
    `max:${formatVoleNumber(max)}`,
  ].join(",");
}

function formatVoleNumber(value: number): string {
  if (!Number.isFinite(value)) return "0";
  if (Number.isInteger(value)) return String(value);
  return Number(value.toPrecision(8)).toString();
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

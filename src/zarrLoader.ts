import * as zarr from "zarrita";
import { loadChannelPlaneInWorker } from "./zarrWorkerPool";
import type {
  DimensionAxis,
  LoadedChannelPlane,
  LoadedPlaneSet,
  TCZYXShape,
  ZarrImageMetadata,
  ZarrImageSource,
  ZarrSelection,
} from "./types";

type ZarrArray = zarr.Array<zarr.DataType, zarr.Readable>;
type ZarrGroup = zarr.Group<zarr.Readable>;

const DEFAULT_ARRAY_PATH_CANDIDATES = ["0", "0/0", "s0"];
const MAX_CHANNEL_PLANE_CACHE_BYTES = 512 * 1024 * 1024;

interface CachedChannelPlane {
  plane: LoadedChannelPlane;
  bytes: number;
}

const channelPlaneCache = new Map<string, CachedChannelPlane>();
let channelPlaneCacheBytes = 0;

interface LoadImageMetadataOptions {
  source: ZarrImageSource;
  resolutionTarget: number;
  signal?: AbortSignal;
}

interface LoadChannelPlaneSetOptions {
  metadata: ZarrImageMetadata;
  timeIndex: number;
  zIndex: number;
  signal?: AbortSignal;
}

interface OpenedArray {
  arr: ZarrArray;
  arrayPath: string;
  multiresolutionLevel: number;
}

interface CandidateArray extends OpenedArray {
  dimensionOrder: DimensionAxis[];
  shapeTCZYX: TCZYXShape;
}

interface PlaneSelection {
  selection: ZarrSelection;
  timeIndex: number;
  zIndex: number;
  channelIndex: number;
}

export async function loadImageMetadata({
  source,
  resolutionTarget,
  signal,
}: LoadImageMetadataOptions): Promise<ZarrImageMetadata> {
  const root = createRootLocation(source);
  const { arr, arrayPath, multiresolutionLevel } = await openSourceArray(
    root,
    source,
    resolutionTarget,
    signal,
  );
  const dimensionOrder = normalizeDimensionOrder(arr, source.dimensionOrder ?? "TCZYX");
  const compatArr = arr as ZarrArray & { chunkShape?: number[]; dataType?: string };

  return {
    source,
    arrayPath,
    multiresolutionLevel,
    resolutionTarget,
    dimensionOrder,
    arrayShape: Array.from(arr.shape ?? []),
    shapeTCZYX: makeTCZYXShape(arr.shape ?? [], dimensionOrder),
    chunks: Array.from(compatArr.chunks ?? compatArr.chunkShape ?? []),
    dtype: String(compatArr.dtype ?? compatArr.dataType ?? "unknown"),
  };
}

export async function loadChannelPlaneSet({
  metadata,
  timeIndex,
  zIndex,
  signal,
}: LoadChannelPlaneSetOptions): Promise<LoadedPlaneSet> {
  throwIfAborted(signal);

  const width = metadata.shapeTCZYX.x;
  const height = metadata.shapeTCZYX.y;
  const sourceUrl = normalizeSourceUrl(metadata.source.url);
  const channelPlaneSlots: Array<LoadedChannelPlane | undefined> = new Array(metadata.shapeTCZYX.c);
  const loadTasks: Array<Promise<void>> = [];

  let actualTimeIndex = clampIndex(timeIndex, metadata.shapeTCZYX.t);
  let actualZIndex = clampIndex(zIndex, metadata.shapeTCZYX.z);

  for (let channelIndex = 0; channelIndex < metadata.shapeTCZYX.c; channelIndex++) {
    const planeSelection = makePlaneSelection(metadata.arrayShape, metadata.dimensionOrder, {
      timeIndex,
      zIndex,
      channelIndex,
    });
    const cacheKey = makeChannelPlaneCacheKey(metadata, planeSelection);
    const cachedPlane = getCachedChannelPlane(cacheKey);

    if (cachedPlane) {
      actualTimeIndex = planeSelection.timeIndex;
      actualZIndex = planeSelection.zIndex;
      channelPlaneSlots[planeSelection.channelIndex] = cachedPlane;
      continue;
    }

    actualTimeIndex = planeSelection.timeIndex;
    actualZIndex = planeSelection.zIndex;
    loadTasks.push(
      loadChannelPlaneInWorker(
        {
          sourceUrl,
          arrayPath: metadata.arrayPath,
          selection: planeSelection.selection,
          width,
          height,
          channelIndex: planeSelection.channelIndex,
        },
        signal,
      ).then((channelPlane) => {
        setCachedChannelPlane(cacheKey, channelPlane);
        channelPlaneSlots[channelPlane.channelIndex] = channelPlane;
      }),
    );
  }

  await Promise.all(loadTasks);
  throwIfAborted(signal);

  return {
    width,
    height,
    channelPlanes: channelPlaneSlots.filter(isPresent),
    timeIndex: actualTimeIndex,
    zIndex: actualZIndex,
    arrayShape: metadata.arrayShape,
    shapeTCZYX: metadata.shapeTCZYX,
    chunks: metadata.chunks,
    dtype: metadata.dtype,
    arrayPath: metadata.arrayPath,
    multiresolutionLevel: metadata.multiresolutionLevel,
    resolutionTarget: metadata.resolutionTarget,
  };
}

function createRootLocation(source: ZarrImageSource): zarr.Location<zarr.FetchStore> {
  const store = new zarr.FetchStore(normalizeSourceUrl(source.url));
  return zarr.root(store);
}

function normalizeSourceUrl(url: string): string {
  return new URL(url, window.location.href).toString();
}

async function openSourceArray(
  root: zarr.Location<zarr.FetchStore>,
  source: ZarrImageSource,
  resolutionTarget: number,
  signal?: AbortSignal,
): Promise<OpenedArray> {
  if (source.arrayPath) {
    return {
      arr: await openArrayAtPath(root, source.arrayPath, signal),
      arrayPath: source.arrayPath,
      multiresolutionLevel: parseMultiresolutionLevel(source.arrayPath),
    };
  }

  const rootArray = await tryOpenArray(root, signal);
  if (rootArray) {
    return { arr: rootArray, arrayPath: "", multiresolutionLevel: 0 };
  }

  const group = await zarr.open(root, { kind: "group", signal });
  const multiscalePaths = findMultiscaleArrayPaths(group);
  const multiscaleCandidate = await chooseBestCandidate(
    root,
    multiscalePaths,
    source.dimensionOrder ?? "TCZYX",
    resolutionTarget,
    signal,
  );
  if (multiscaleCandidate) {
    return multiscaleCandidate;
  }

  const fallbackCandidate = await chooseBestCandidate(
    root,
    DEFAULT_ARRAY_PATH_CANDIDATES,
    source.dimensionOrder ?? "TCZYX",
    resolutionTarget,
    signal,
  );
  if (fallbackCandidate) {
    return fallbackCandidate;
  }

  throw new Error(
    "Could not locate a Zarr array in this store. Set arrayPath explicitly for grouped Zarr stores.",
  );
}

async function openArrayAtPath(
  root: zarr.Location<zarr.FetchStore>,
  path: string,
  signal?: AbortSignal,
): Promise<ZarrArray> {
  return zarr.open(path ? root.resolve(path) : root, { kind: "array", signal });
}

async function tryOpenArray(
  location: zarr.Location<zarr.FetchStore>,
  signal?: AbortSignal,
): Promise<ZarrArray | null> {
  try {
    return await zarr.open(location, { kind: "array", signal });
  } catch (error) {
    if (error instanceof zarr.NotFoundError) {
      return null;
    }
    throw error;
  }
}

async function chooseBestCandidate(
  root: zarr.Location<zarr.FetchStore>,
  paths: readonly string[],
  configuredDimensionOrder: string,
  resolutionTarget: number,
  signal?: AbortSignal,
): Promise<CandidateArray | null> {
  const candidates: CandidateArray[] = [];

  for (const path of paths) {
    const arr = await tryOpenArray(path ? root.resolve(path) : root, signal);
    if (!arr) continue;

    const dimensionOrder = normalizeDimensionOrder(arr, configuredDimensionOrder);
    candidates.push({
      arr,
      arrayPath: path,
      multiresolutionLevel: parseMultiresolutionLevel(path),
      dimensionOrder,
      shapeTCZYX: makeTCZYXShape(arr.shape, dimensionOrder),
    });
  }

  if (candidates.length === 0) return null;
  return candidates.sort((a, b) => {
    const aTarget = getCandidateTargetDimension(a);
    const bTarget = getCandidateTargetDimension(b);
    const aIsLargeEnough = aTarget >= resolutionTarget;
    const bIsLargeEnough = bTarget >= resolutionTarget;

    if (aIsLargeEnough && bIsLargeEnough) return aTarget - bTarget;
    if (aIsLargeEnough) return -1;
    if (bIsLargeEnough) return 1;
    return bTarget - aTarget;
  })[0];
}

function getCandidateTargetDimension(candidate: CandidateArray): number {
  return Math.max(candidate.shapeTCZYX.x, candidate.shapeTCZYX.y);
}

function findMultiscaleArrayPaths(group: ZarrGroup): string[] {
  const omeAttrs = getRecordProperty(group.attrs, "ome");
  const multiscales = group.attrs.multiscales ?? getRecordProperty(omeAttrs, "multiscales");
  if (!Array.isArray(multiscales)) return [];

  const paths: string[] = [];

  for (const multiscale of multiscales) {
    const datasets = getRecordProperty(multiscale, "datasets");
    if (!Array.isArray(datasets)) continue;

    for (const dataset of datasets) {
      const path = getRecordProperty(dataset, "path");
      if (typeof path === "string" && path.length > 0) {
        paths.push(path);
      }
    }
  }

  return Array.from(new Set(paths));
}

function normalizeDimensionOrder(arr: ZarrArray, dimensionOrder: string): DimensionAxis[] {
  const shape = Array.from(arr.shape ?? []);
  const names = Array.from(arr.dimensionNames ?? []);
  if (names.length === shape.length) {
    const normalizedNames = names.map(mapDimensionName);
    if (normalizedNames.every(isPresent)) return normalizedNames;
  }

  const configuredOrder = dimensionOrder.toUpperCase().replace(/[^A-Z]/g, "").split("");
  if (configuredOrder.length === shape.length && configuredOrder.every(isDimensionAxis)) {
    return configuredOrder;
  }

  if (shape.length === 2) return ["Y", "X"];
  if (shape.length === 3) return ["Z", "Y", "X"];
  if (shape.length === 4) return ["C", "Z", "Y", "X"];
  if (shape.length === 5) return ["T", "C", "Z", "Y", "X"];

  throw new Error(`Cannot infer TCZYX dimension order for ${shape.length}D array.`);
}

function makeTCZYXShape(shape: readonly number[], dimensionOrder: readonly DimensionAxis[]): TCZYXShape {
  return {
    t: getAxisSize(shape, dimensionOrder, "T"),
    c: getAxisSize(shape, dimensionOrder, "C"),
    z: getAxisSize(shape, dimensionOrder, "Z"),
    y: getAxisSize(shape, dimensionOrder, "Y"),
    x: getAxisSize(shape, dimensionOrder, "X"),
  };
}

function getAxisSize(
  shape: readonly number[],
  dimensionOrder: readonly DimensionAxis[],
  axis: DimensionAxis,
): number {
  const axisIndex = dimensionOrder.indexOf(axis);
  return axisIndex === -1 ? 1 : shape[axisIndex] ?? 1;
}

function makePlaneSelection(
  shape: readonly number[],
  dimensionOrder: readonly DimensionAxis[],
  requested: { timeIndex: number; zIndex: number; channelIndex: number },
): PlaneSelection {
  if (shape.length !== dimensionOrder.length) {
    throw new Error(
      `Array shape has ${shape.length} dimensions, but dimension order "${dimensionOrder.join("")}" has ${dimensionOrder.length}.`,
    );
  }

  const selection = new Array<number | null>(shape.length).fill(null);
  let timeIndex = 0;
  let zIndex = 0;
  let channelIndex = 0;

  for (let index = 0; index < dimensionOrder.length; index++) {
    const axis = dimensionOrder[index];
    const size = shape[index] ?? 1;

    if (axis === "T") {
      timeIndex = clampIndex(requested.timeIndex, size);
      selection[index] = timeIndex;
    } else if (axis === "C") {
      channelIndex = clampIndex(requested.channelIndex, size);
      selection[index] = channelIndex;
    } else if (axis === "Z") {
      zIndex = clampIndex(requested.zIndex, size);
      selection[index] = zIndex;
    } else if (axis === "Y" || axis === "X") {
      selection[index] = null;
    }
  }

  if (selection.filter((item) => item === null).length !== 2) {
    throw new Error(`Dimension order must leave exactly Y and X unsliced. Got "${dimensionOrder.join("")}".`);
  }

  return { selection, timeIndex, zIndex, channelIndex };
}

function makeChannelPlaneCacheKey(metadata: ZarrImageMetadata, planeSelection: PlaneSelection): string {
  return [
    normalizeSourceUrl(metadata.source.url),
    metadata.multiresolutionLevel,
    metadata.arrayPath || "/",
    `T${planeSelection.timeIndex}`,
    `C${planeSelection.channelIndex}`,
    `Z${planeSelection.zIndex}`,
  ].join("|");
}

function getCachedChannelPlane(cacheKey: string): LoadedChannelPlane | undefined {
  const cached = channelPlaneCache.get(cacheKey);
  if (!cached) return undefined;

  channelPlaneCache.delete(cacheKey);
  channelPlaneCache.set(cacheKey, cached);
  return cached.plane;
}

function setCachedChannelPlane(cacheKey: string, plane: LoadedChannelPlane): void {
  const bytes = getChannelPlaneByteLength(plane);
  if (bytes > MAX_CHANNEL_PLANE_CACHE_BYTES) return;

  const previous = channelPlaneCache.get(cacheKey);
  if (previous) {
    channelPlaneCacheBytes -= previous.bytes;
    channelPlaneCache.delete(cacheKey);
  }

  channelPlaneCache.set(cacheKey, { plane, bytes });
  channelPlaneCacheBytes += bytes;
  trimChannelPlaneCache();
}

function trimChannelPlaneCache(): void {
  while (channelPlaneCacheBytes > MAX_CHANNEL_PLANE_CACHE_BYTES) {
    const oldestKey = channelPlaneCache.keys().next().value;
    if (oldestKey === undefined) return;

    const oldest = channelPlaneCache.get(oldestKey);
    if (oldest) channelPlaneCacheBytes -= oldest.bytes;
    channelPlaneCache.delete(oldestKey);
  }
}

function getChannelPlaneByteLength(plane: LoadedChannelPlane): number {
  return plane.nativePixels.byteLength + plane.pixels.byteLength;
}

function mapDimensionName(name: string): DimensionAxis | null {
  const clean = String(name).toLowerCase().replace(/[^a-z]/g, "");
  if (clean === "t" || clean === "time") return "T";
  if (clean === "c" || clean === "channel" || clean === "channels") return "C";
  if (clean === "z" || clean === "depth" || clean === "plane") return "Z";
  if (clean === "y" || clean === "height") return "Y";
  if (clean === "x" || clean === "width") return "X";
  return null;
}

function parseMultiresolutionLevel(path: string): number {
  const parts = path.split("/").filter(Boolean);
  const lastPart = parts.at(-1);
  const level = lastPart === undefined ? 0 : Number.parseInt(lastPart, 10);
  return Number.isFinite(level) ? level : 0;
}

function clampIndex(value: number, size: number): number {
  if (size <= 1) return 0;
  return Math.max(0, Math.min(Math.round(value), size - 1));
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;

  if (typeof DOMException !== "undefined") {
    throw new DOMException("The operation was aborted.", "AbortError");
  }

  const error = new Error("The operation was aborted.");
  error.name = "AbortError";
  throw error;
}

function getRecordProperty(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object") return undefined;
  return (value as Record<string, unknown>)[key];
}

function isDimensionAxis(value: string): value is DimensionAxis {
  return value === "T" || value === "C" || value === "Z" || value === "Y" || value === "X";
}

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

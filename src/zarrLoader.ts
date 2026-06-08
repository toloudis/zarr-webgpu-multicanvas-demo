import * as zarr from "zarrita";
import type {
  DimensionAxis,
  LoadedChannelPlane,
  LoadedPlaneSet,
  NumericTypedArray,
  TCZYXShape,
  ZarrImageMetadata,
  ZarrImageSource,
  ZarrSelection,
} from "./types";

type ZarrArray = zarr.Array<zarr.DataType, zarr.Readable>;
type ZarrGroup = zarr.Group<zarr.Readable>;
type ChunkView = zarr.Chunk<zarr.DataType> & { offset?: number };
type ArrayData = ArrayLike<unknown> & { get?: (index: number) => unknown };

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

interface LuminancePlane {
  nativePixels: NumericTypedArray;
  pixels: Uint8Array;
  min: number;
  max: number;
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
  const root = createRootLocation(metadata.source);
  const arr = await openArrayAtPath(root, metadata.arrayPath, signal);
  const width = metadata.shapeTCZYX.x;
  const height = metadata.shapeTCZYX.y;
  const channelPlanes: LoadedChannelPlane[] = [];

  let actualTimeIndex = clampIndex(timeIndex, metadata.shapeTCZYX.t);
  let actualZIndex = clampIndex(zIndex, metadata.shapeTCZYX.z);

  for (let channelIndex = 0; channelIndex < metadata.shapeTCZYX.c; channelIndex++) {
    const planeSelection = makePlaneSelection(arr.shape, metadata.dimensionOrder, {
      timeIndex,
      zIndex,
      channelIndex,
    });
    const cacheKey = makeChannelPlaneCacheKey(metadata, planeSelection);
    const cachedPlane = getCachedChannelPlane(cacheKey);

    if (cachedPlane) {
      actualTimeIndex = planeSelection.timeIndex;
      actualZIndex = planeSelection.zIndex;
      channelPlanes.push(cachedPlane);
      continue;
    }

    const view = await zarr.get(arr, planeSelection.selection, { signal }) as ChunkView;

    if (view.shape.length !== 2) {
      throw new Error(`Selection produced ${view.shape.length}D data; expected a YX plane.`);
    }

    const [planeHeight, planeWidth] = view.shape;
    if (planeWidth !== width || planeHeight !== height) {
      throw new Error(
        `Selection produced ${planeWidth} x ${planeHeight}; expected ${width} x ${height}.`,
      );
    }

    const luminance = normalizeToLuminance(view, width, height);

    actualTimeIndex = planeSelection.timeIndex;
    actualZIndex = planeSelection.zIndex;
    const channelPlane: LoadedChannelPlane = {
      channelIndex: planeSelection.channelIndex,
      nativePixels: luminance.nativePixels,
      pixels: luminance.pixels,
      min: luminance.min,
      max: luminance.max,
      selection: planeSelection.selection,
    };

    setCachedChannelPlane(cacheKey, channelPlane);
    channelPlanes.push(channelPlane);
  }

  return {
    width,
    height,
    channelPlanes,
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

function normalizeToLuminance(view: ChunkView, width: number, height: number): LuminancePlane {
  const { data, stride } = view;
  const offset = view.offset ?? 0;
  const strideY = stride?.[0] ?? width;
  const strideX = stride?.[1] ?? 1;
  const nativePixels = createNativePlaneArray(data, width * height);

  let min = Infinity;
  let max = -Infinity;
  let out = 0;

  forEachValue(data, width, height, offset, strideY, strideX, (value) => {
    nativePixels[out++] = value;
    if (!Number.isNaN(value)) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  });

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }

  const scale = max === min ? 0 : 255 / (max - min);
  const pixels = new Uint8Array(width * height);

  for (let index = 0; index < nativePixels.length; index++) {
    const value = nativePixels[index];
    pixels[index] = max === min
      ? 128
      : Math.max(0, Math.min(255, Math.round((value - min) * scale)));
  }

  return { nativePixels, pixels, min, max };
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

function createNativePlaneArray(
  data: zarr.TypedArray<zarr.DataType>,
  length: number,
): NumericTypedArray {
  if (data instanceof Int8Array) return new Int8Array(length);
  if (data instanceof Uint8Array) return new Uint8Array(length);
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(length);
  if (data instanceof Int16Array) return new Int16Array(length);
  if (data instanceof Uint16Array) return new Uint16Array(length);
  if (data instanceof Int32Array) return new Int32Array(length);
  if (data instanceof Uint32Array) return new Uint32Array(length);
  if (data instanceof Float64Array) return new Float64Array(length);
  return new Float32Array(length);
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

function forEachValue(
  data: zarr.TypedArray<zarr.DataType>,
  width: number,
  height: number,
  offset: number,
  strideY: number,
  strideX: number,
  callback: (value: number) => void,
): void {
  const readableData = data as ArrayData;

  for (let y = 0; y < height; y++) {
    const rowOffset = offset + y * strideY;
    for (let x = 0; x < width; x++) {
      const rawValue = readArrayValue(readableData, rowOffset + x * strideX);
      if (typeof rawValue === "bigint") {
        throw new Error("BigInt image arrays are not supported by this grayscale demo.");
      }
      callback(Number(rawValue));
    }
  }
}

function readArrayValue(data: ArrayData, index: number): unknown {
  if (typeof data.get === "function") {
    return data.get(index);
  }
  return data[index];
}

function getRecordProperty(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object") return undefined;
  return (value as Record<string, unknown>)[key];
}

function isDimensionAxis(value: string): value is DimensionAxis {
  return value === "T" || value === "C" || value === "Z" || value === "Y" || value === "X";
}

function isPresent<T>(value: T | null): value is T {
  return value !== null;
}

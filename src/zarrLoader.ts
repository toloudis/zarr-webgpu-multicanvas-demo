import * as zarr from "zarrita";
import type {
  ChannelRange,
  ChannelRenderSettings,
  DimensionAxis,
  LoadedSlice,
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

interface LoadImageMetadataOptions {
  source: ZarrImageSource;
  signal?: AbortSignal;
}

interface LoadBlendedSliceOptions {
  metadata: ZarrImageMetadata;
  timeIndex: number;
  zIndex: number;
  channels: ChannelRenderSettings[];
  signal?: AbortSignal;
}

interface OpenedArray {
  arr: ZarrArray;
  arrayPath: string;
  multiresolutionLevel: number;
}

interface PlaneSelection {
  selection: ZarrSelection;
  timeIndex: number;
  zIndex: number;
  channelIndex: number;
}

interface LuminancePlane {
  pixels: Uint8Array;
  min: number;
  max: number;
}

interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export async function loadImageMetadata({
  source,
  signal,
}: LoadImageMetadataOptions): Promise<ZarrImageMetadata> {
  const root = createRootLocation(source);
  const { arr, arrayPath, multiresolutionLevel } = await openSourceArray(root, source, signal);
  const dimensionOrder = normalizeDimensionOrder(arr, source.dimensionOrder ?? "TCZYX");
  const compatArr = arr as ZarrArray & { chunkShape?: number[]; dataType?: string };

  return {
    source,
    arrayPath,
    multiresolutionLevel,
    dimensionOrder,
    arrayShape: Array.from(arr.shape ?? []),
    shapeTCZYX: makeTCZYXShape(arr.shape ?? [], dimensionOrder),
    chunks: Array.from(compatArr.chunks ?? compatArr.chunkShape ?? []),
    dtype: String(compatArr.dtype ?? compatArr.dataType ?? "unknown"),
  };
}

export async function loadBlendedSlice({
  metadata,
  timeIndex,
  zIndex,
  channels,
  signal,
}: LoadBlendedSliceOptions): Promise<LoadedSlice> {
  const root = createRootLocation(metadata.source);
  const arr = await openArrayAtPath(root, metadata.arrayPath, signal);
  const selectedChannels = channels.filter((channel) => channel.enabled);
  const width = metadata.shapeTCZYX.x;
  const height = metadata.shapeTCZYX.y;
  const rgba = new Uint8Array(width * height * 4);
  const channelRanges: ChannelRange[] = [];
  const channelIndices: number[] = [];
  const selections: ZarrSelection[] = [];

  for (let index = 3; index < rgba.length; index += 4) {
    rgba[index] = 255;
  }

  let actualTimeIndex = clampIndex(timeIndex, metadata.shapeTCZYX.t);
  let actualZIndex = clampIndex(zIndex, metadata.shapeTCZYX.z);

  for (const channel of selectedChannels) {
    if (channel.index >= metadata.shapeTCZYX.c) continue;

    const planeSelection = makePlaneSelection(arr.shape, metadata.dimensionOrder, {
      timeIndex,
      zIndex,
      channelIndex: channel.index,
    });
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
    const color = parseHexColor(channel.color);
    blendLuminance(rgba, luminance.pixels, color);

    actualTimeIndex = planeSelection.timeIndex;
    actualZIndex = planeSelection.zIndex;
    channelRanges.push({
      channelIndex: planeSelection.channelIndex,
      min: luminance.min,
      max: luminance.max,
    });
    channelIndices.push(planeSelection.channelIndex);
    selections.push(planeSelection.selection);
  }

  return {
    rgba,
    width,
    height,
    channelRanges,
    timeIndex: actualTimeIndex,
    zIndex: actualZIndex,
    channelIndices,
    arrayShape: metadata.arrayShape,
    shapeTCZYX: metadata.shapeTCZYX,
    chunks: metadata.chunks,
    dtype: metadata.dtype,
    arrayPath: metadata.arrayPath,
    multiresolutionLevel: metadata.multiresolutionLevel,
    selections,
  };
}

function createRootLocation(source: ZarrImageSource): zarr.Location<zarr.FetchStore> {
  const absoluteUrl = new URL(source.url, window.location.href).toString();
  const store = new zarr.FetchStore(absoluteUrl);
  return zarr.root(store);
}

async function openSourceArray(
  root: zarr.Location<zarr.FetchStore>,
  source: ZarrImageSource,
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
  const multiscalePath = findFirstMultiscaleArrayPath(group);
  if (multiscalePath) {
    return {
      arr: await openArrayAtPath(root, multiscalePath, signal),
      arrayPath: multiscalePath,
      multiresolutionLevel: parseMultiresolutionLevel(multiscalePath),
    };
  }

  for (const path of DEFAULT_ARRAY_PATH_CANDIDATES) {
    const arr = await tryOpenArray(root.resolve(path), signal);
    if (arr) {
      return { arr, arrayPath: path, multiresolutionLevel: parseMultiresolutionLevel(path) };
    }
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

function findFirstMultiscaleArrayPath(group: ZarrGroup): string | null {
  const omeAttrs = getRecordProperty(group.attrs, "ome");
  const multiscales = group.attrs.multiscales ?? getRecordProperty(omeAttrs, "multiscales");
  if (!Array.isArray(multiscales)) return null;

  for (const multiscale of multiscales) {
    const datasets = getRecordProperty(multiscale, "datasets");
    if (!Array.isArray(datasets)) continue;

    for (const dataset of datasets) {
      const path = getRecordProperty(dataset, "path");
      if (typeof path === "string" && path.length > 0) {
        return path;
      }
    }
  }

  return null;
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

  let min = Infinity;
  let max = -Infinity;

  forEachValue(data, width, height, offset, strideY, strideX, (value) => {
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
  let out = 0;

  forEachValue(data, width, height, offset, strideY, strideX, (value) => {
    pixels[out++] = max === min
      ? 128
      : Math.max(0, Math.min(255, Math.round((value - min) * scale)));
  });

  return { pixels, min, max };
}

function blendLuminance(rgba: Uint8Array, luminance: Uint8Array, color: RgbColor): void {
  for (let index = 0; index < luminance.length; index++) {
    const out = index * 4;
    const value = luminance[index];
    rgba[out] = Math.min(255, rgba[out] + Math.round(value * color.r / 255));
    rgba[out + 1] = Math.min(255, rgba[out + 1] + Math.round(value * color.g / 255));
    rgba[out + 2] = Math.min(255, rgba[out + 2] + Math.round(value * color.b / 255));
  }
}

function parseHexColor(color: string): RgbColor {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);
  if (!match) return { r: 255, g: 255, b: 255 };

  return {
    r: Number.parseInt(match[1], 16),
    g: Number.parseInt(match[2], 16),
    b: Number.parseInt(match[3], 16),
  };
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

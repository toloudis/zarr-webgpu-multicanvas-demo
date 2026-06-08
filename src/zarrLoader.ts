import * as zarr from "zarrita";
import type { LoadedSlice, ZarrImageSource, ZarrSelection } from "./types";

type ZarrArray = zarr.Array<zarr.DataType, zarr.Readable>;
type ZarrGroup = zarr.Group<zarr.Readable>;
type DimensionAxis = "T" | "C" | "Z" | "Y" | "X";
type ChunkView = zarr.Chunk<zarr.DataType> & { offset?: number };
type ArrayData = ArrayLike<unknown> & { get?: (index: number) => unknown };

const DEFAULT_ARRAY_PATH_CANDIDATES = ["0", "0/0", "s0"];

interface LoadMiddleZSliceOptions {
  source: ZarrImageSource;
  signal?: AbortSignal;
}

interface MiddleZSelection {
  selection: ZarrSelection;
  zIndex: number;
}

interface NormalizedImage {
  rgba: Uint8Array;
  min: number;
  max: number;
}

export async function loadMiddleZSlice({
  source,
  signal,
}: LoadMiddleZSliceOptions): Promise<LoadedSlice> {
  const absoluteUrl = new URL(source.url, window.location.href).toString();
  const store = new zarr.FetchStore(absoluteUrl);
  const root = zarr.root(store);
  const arr = await openSourceArray(root, source, signal);
  const { selection, zIndex } = makeMiddleZSelection(arr, source.dimensionOrder ?? "TCZYX");
  const view = await zarr.get(arr, selection, { signal }) as ChunkView;

  if (view.shape.length !== 2) {
    throw new Error(`Selection produced ${view.shape.length}D data; expected a YX plane.`);
  }

  const [height, width] = view.shape;
  const normalized = normalizeToRgba(view, width, height);
  const compatArr = arr as ZarrArray & { chunkShape?: number[]; dataType?: string };

  return {
    rgba: normalized.rgba,
    width,
    height,
    min: normalized.min,
    max: normalized.max,
    zIndex,
    arrayShape: Array.from(arr.shape ?? []),
    chunks: Array.from(compatArr.chunks ?? compatArr.chunkShape ?? []),
    dtype: String(compatArr.dtype ?? compatArr.dataType ?? "unknown"),
    selection,
  };
}

async function openSourceArray(
  root: zarr.Location<zarr.FetchStore>,
  source: ZarrImageSource,
  signal?: AbortSignal,
): Promise<ZarrArray> {
  if (source.arrayPath) {
    return openArrayAtPath(root, source.arrayPath, signal);
  }

  const rootArray = await tryOpenArray(root, signal);
  if (rootArray) return rootArray;

  const group = await zarr.open(root, { kind: "group", signal });
  const multiscalePath = findFirstMultiscaleArrayPath(group);
  if (multiscalePath) {
    return openArrayAtPath(root, multiscalePath, signal);
  }

  for (const path of DEFAULT_ARRAY_PATH_CANDIDATES) {
    const arr = await tryOpenArray(root.resolve(path), signal);
    if (arr) return arr;
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
  return zarr.open(root.resolve(path), { kind: "array", signal });
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

function makeMiddleZSelection(arr: ZarrArray, dimensionOrder: string): MiddleZSelection {
  const shape = Array.from(arr.shape ?? []);
  if (shape.length === 2) return { selection: [null, null], zIndex: 0 };

  const order = normalizeDimensionOrder(arr, dimensionOrder);
  if (shape.length !== order.length) {
    throw new Error(
      `Array shape has ${shape.length} dimensions, but dimension order "${order.join("")}" has ${order.length}.`,
    );
  }

  const selection = new Array(shape.length).fill(null);
  const tAxis = order.indexOf("T");
  const cAxis = order.indexOf("C");
  const zAxis = order.indexOf("Z");
  const yAxis = order.indexOf("Y");
  const xAxis = order.indexOf("X");

  if ([tAxis, cAxis, zAxis, yAxis, xAxis].some((axis) => axis === -1)) {
    throw new Error(`Dimension order must include T, C, Z, Y, and X. Got "${order.join("")}".`);
  }

  const zIndex = Math.floor(shape[zAxis] / 2);
  selection[tAxis] = boundedIndex(0, shape[tAxis], "T");
  selection[cAxis] = boundedIndex(0, shape[cAxis], "C");
  selection[zAxis] = boundedIndex(zIndex, shape[zAxis], "Z");
  selection[yAxis] = null;
  selection[xAxis] = null;

  if (selection.filter((item) => item === null).length !== 2) {
    throw new Error(`Dimension order must leave exactly Y and X unsliced. Got "${order.join("")}".`);
  }

  return { selection, zIndex };
}

function normalizeDimensionOrder(arr: ZarrArray, dimensionOrder: string): string[] {
  const names = Array.from(arr.dimensionNames ?? []);
  if (names.length) {
    const normalizedNames = names.map(mapDimensionName);
    if (normalizedNames.every(isPresent)) return normalizedNames;
  }

  return dimensionOrder.toUpperCase().replace(/[^A-Z]/g, "").split("");
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

function boundedIndex(value: number, size: number, axisName: DimensionAxis): number {
  if (value < size) return value;
  throw new Error(`${axisName} index ${value} is outside dimension length ${size}.`);
}

function normalizeToRgba(view: ChunkView, width: number, height: number): NormalizedImage {
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
  const rgba = new Uint8Array(width * height * 4);
  let out = 0;

  forEachValue(data, width, height, offset, strideY, strideX, (value) => {
    const luminance = max === min ? 128 : Math.max(0, Math.min(255, Math.round((value - min) * scale)));
    rgba[out++] = luminance;
    rgba[out++] = luminance;
    rgba[out++] = luminance;
    rgba[out++] = 255;
  });

  return { rgba, min, max };
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

function isPresent<T>(value: T | null): value is T {
  return value !== null;
}

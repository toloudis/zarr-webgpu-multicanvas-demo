import * as zarr from "zarrita";
import type { LoadedChannelPlane, NumericTypedArray } from "./types";
import type {
  ZarrPlaneLoadRequest,
  ZarrPlaneWorkerRequest,
  ZarrPlaneWorkerResponse,
} from "./zarrWorkerTypes";

type ZarrArray = zarr.Array<zarr.DataType, zarr.Readable>;
type ChunkView = zarr.Chunk<zarr.DataType> & { offset?: number };
type ArrayData = ArrayLike<unknown> & { get?: (index: number) => unknown };

interface WorkerContext {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<ZarrPlaneWorkerRequest>) => void,
  ): void;
  postMessage(message: ZarrPlaneWorkerResponse, transfer?: Transferable[]): void;
}

interface PlanePixels {
  nativePixels: NumericTypedArray;
  pixels: Uint8Array;
  min: number;
  max: number;
}

const workerContext = self as unknown as WorkerContext;
const activeRequests = new Map<number, AbortController>();

workerContext.addEventListener("message", (event) => {
  const message = event.data;

  if (message.type === "cancel") {
    activeRequests.get(message.id)?.abort();
    return;
  }

  void loadPlane(message.request);
});

async function loadPlane(request: ZarrPlaneLoadRequest): Promise<void> {
  const abortController = new AbortController();
  activeRequests.set(request.id, abortController);

  try {
    const arr = await openArray(request.sourceUrl, request.arrayPath, abortController.signal);
    const view = await zarr.get(arr, request.selection, {
      signal: abortController.signal,
    }) as ChunkView;

    if (view.shape.length !== 2) {
      throw new Error(`Selection produced ${view.shape.length}D data; expected a YX plane.`);
    }

    const [planeHeight, planeWidth] = view.shape;
    if (planeWidth !== request.width || planeHeight !== request.height) {
      throw new Error(
        `Selection produced ${planeWidth} x ${planeHeight}; expected ${request.width} x ${request.height}.`,
      );
    }

    const planePixels = normalizeToLuminance(view, request.width, request.height);
    const plane: LoadedChannelPlane = {
      channelIndex: request.channelIndex,
      nativePixels: planePixels.nativePixels,
      pixels: planePixels.pixels,
      min: planePixels.min,
      max: planePixels.max,
      selection: request.selection,
    };

    workerContext.postMessage(
      { type: "loaded", id: request.id, plane },
      [
        plane.nativePixels.buffer as ArrayBuffer,
        plane.pixels.buffer as ArrayBuffer,
      ],
    );
  } catch (error) {
    workerContext.postMessage({
      type: "error",
      id: request.id,
      message: getErrorMessage(error),
      name: error instanceof Error ? error.name : undefined,
      stack: error instanceof Error ? error.stack : undefined,
    });
  } finally {
    activeRequests.delete(request.id);
  }
}

async function openArray(
  sourceUrl: string,
  arrayPath: string,
  signal: AbortSignal,
): Promise<ZarrArray> {
  const store = new zarr.FetchStore(sourceUrl);
  const root = zarr.root(store);
  return zarr.open(arrayPath ? root.resolve(arrayPath) : root, { kind: "array", signal });
}

function normalizeToLuminance(view: ChunkView, width: number, height: number): PlanePixels {
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

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

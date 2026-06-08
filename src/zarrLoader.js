import * as zarr from "zarrita";

export async function loadMiddleZSlice({ source, signal }) {
  const absoluteUrl = new URL(source.url, window.location.href).toString();
  const store = new zarr.FetchStore(absoluteUrl);
  const root = zarr.root(store);
  const location = source.arrayPath ? root.resolve(source.arrayPath) : root;
  const arr = await zarr.open(location, { kind: "array", signal });
  const { selection, zIndex } = makeMiddleZSelection(arr, source.dimensionOrder ?? "TCZYX");
  const view = await zarr.get(arr, selection, { signal });

  if (view.shape.length !== 2) {
    throw new Error(`Selection produced ${view.shape.length}D data; expected a YX plane.`);
  }

  const [height, width] = view.shape;
  const normalized = normalizeToRgba(view, width, height);

  return {
    rgba: normalized.rgba,
    width,
    height,
    min: normalized.min,
    max: normalized.max,
    zIndex,
    arrayShape: Array.from(arr.shape ?? []),
    chunks: Array.from(arr.chunks ?? arr.chunkShape ?? []),
    dtype: arr.dtype ?? arr.dataType ?? "unknown",
    selection,
  };
}

function makeMiddleZSelection(arr, dimensionOrder) {
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

function normalizeDimensionOrder(arr, dimensionOrder) {
  const names = Array.from(arr.dimensionNames ?? []);
  if (names.length) {
    const normalizedNames = names.map(mapDimensionName);
    if (normalizedNames.every(Boolean)) return normalizedNames;
  }

  return dimensionOrder.toUpperCase().replace(/[^A-Z]/g, "").split("");
}

function mapDimensionName(name) {
  const clean = String(name).toLowerCase().replace(/[^a-z]/g, "");
  if (clean === "t" || clean === "time") return "T";
  if (clean === "c" || clean === "channel" || clean === "channels") return "C";
  if (clean === "z" || clean === "depth" || clean === "plane") return "Z";
  if (clean === "y" || clean === "height") return "Y";
  if (clean === "x" || clean === "width") return "X";
  return "";
}

function boundedIndex(value, size, axisName) {
  if (value < size) return value;
  throw new Error(`${axisName} index ${value} is outside dimension length ${size}.`);
}

function normalizeToRgba(view, width, height) {
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

function forEachValue(data, width, height, offset, strideY, strideX, callback) {
  for (let y = 0; y < height; y++) {
    const rowOffset = offset + y * strideY;
    for (let x = 0; x < width; x++) {
      const rawValue = data[rowOffset + x * strideX];
      if (typeof rawValue === "bigint") {
        throw new Error("BigInt image arrays are not supported by this grayscale demo.");
      }
      callback(Number(rawValue));
    }
  }
}

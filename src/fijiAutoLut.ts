import type { ChannelHistogram, NumericTypedArray } from "./types";

const IMAGEJ_AUTO_THRESHOLD = 5000;
const IMAGEJ_HISTOGRAM_BINS = 256;

export interface FijiAutoLutRange {
  min: number;
  max: number;
}

export interface FijiAutoLutStats extends FijiAutoLutRange {
  histogram: ChannelHistogram;
}

export interface FijiAutoLutOptions {
  dataMin?: number;
  dataMax?: number;
  autoThreshold?: number;
}

interface HistogramStats {
  histogram: Int32Array;
  histMin: number;
  binSize: number;
  pixelCount: number;
  dataMin: number;
  dataMax: number;
}

export function calculateFijiAutoLutRange(
  pixels: NumericTypedArray,
  options: FijiAutoLutOptions = {},
): FijiAutoLutRange {
  const { min, max } = calculateFijiAutoLutStats(pixels, options);
  return { min, max };
}

export function calculateFijiAutoLutStats(
  pixels: NumericTypedArray,
  options: FijiAutoLutOptions = {},
): FijiAutoLutStats {
  const stats = makeHistogramStats(pixels, options);
  const histogram = makeChannelHistogram(stats);
  if (stats.pixelCount === 0 || stats.dataMax <= stats.dataMin) {
    return { ...makeFallbackRange(stats.dataMin, stats.dataMax), histogram };
  }

  return {
    ...calculateAutoRangeFromHistogram(stats, options),
    histogram,
  };
}

function calculateAutoRangeFromHistogram(
  stats: HistogramStats,
  options: FijiAutoLutOptions,
): FijiAutoLutRange {
  const autoThreshold = Math.max(1, Math.trunc(options.autoThreshold ?? IMAGEJ_AUTO_THRESHOLD));
  const limit = Math.trunc(stats.pixelCount / 10);
  const threshold = Math.trunc(stats.pixelCount / autoThreshold);
  const lastBin = stats.histogram.length - 1;
  let hmin = 0;
  let hmax = lastBin;
  let foundMin = false;
  let foundMax = false;

  for (let index = 0; index <= lastBin; index++) {
    const count = getAutoEligibleBinCount(stats.histogram[index], limit);
    if (count > threshold) {
      hmin = index;
      foundMin = true;
      break;
    }
  }

  for (let index = lastBin; index >= 0; index--) {
    const count = getAutoEligibleBinCount(stats.histogram[index], limit);
    if (count > threshold) {
      hmax = index;
      foundMax = true;
      break;
    }
  }

  if (!foundMin || !foundMax || hmax < hmin) {
    return makeFallbackRange(stats.dataMin, stats.dataMax);
  }

  let min = stats.histMin + hmin * stats.binSize;
  let max = stats.histMin + hmax * stats.binSize;

  if (max <= min) {
    ({ min, max } = makeFallbackRange(stats.dataMin, stats.dataMax));
  }

  return { min, max };
}

function makeChannelHistogram(stats: HistogramStats): ChannelHistogram {
  return {
    bins: stats.histogram,
    min: stats.histMin,
    max: stats.histMin + (stats.histogram.length - 1) * stats.binSize,
    binSize: stats.binSize,
    pixelCount: stats.pixelCount,
  };
}

function makeHistogramStats(
  pixels: NumericTypedArray,
  options: FijiAutoLutOptions,
): HistogramStats {
  const { dataMin, dataMax, pixelCount } = resolveDataRange(pixels, options);
  const { histMin, histMax, binSize } = makeHistogramRange(pixels, dataMin, dataMax);
  const histogram = new Int32Array(IMAGEJ_HISTOGRAM_BINS);

  if (pixelCount === 0 || histMax <= histMin) {
    return { histogram, histMin, binSize: 1, pixelCount, dataMin, dataMax };
  }

  const scale = 1 / binSize;
  for (let index = 0; index < pixels.length; index++) {
    const value = Number(pixels[index]);
    if (!Number.isFinite(value)) continue;

    const histogramIndex = Math.max(
      0,
      Math.min(IMAGEJ_HISTOGRAM_BINS - 1, Math.trunc((value - histMin) * scale)),
    );
    histogram[histogramIndex]++;
  }

  return { histogram, histMin, binSize, pixelCount, dataMin, dataMax };
}

function resolveDataRange(
  pixels: NumericTypedArray,
  options: FijiAutoLutOptions,
): { dataMin: number; dataMax: number; pixelCount: number } {
  let dataMin = getFiniteOption(options.dataMin, Infinity);
  let dataMax = getFiniteOption(options.dataMax, -Infinity);
  let pixelCount = 0;

  for (let index = 0; index < pixels.length; index++) {
    const value = Number(pixels[index]);
    if (!Number.isFinite(value)) continue;

    pixelCount++;
    dataMin = Math.min(dataMin, value);
    dataMax = Math.max(dataMax, value);
  }

  if (!Number.isFinite(dataMin) || !Number.isFinite(dataMax)) {
    dataMin = 0;
    dataMax = 1;
  }

  return { dataMin, dataMax, pixelCount };
}

function makeHistogramRange(
  pixels: NumericTypedArray,
  dataMin: number,
  dataMax: number,
): { histMin: number; histMax: number; binSize: number } {
  if (pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray) {
    return { histMin: 0, histMax: 255, binSize: 1 };
  }

  if (pixels instanceof Uint16Array) {
    const histMin = Math.floor(dataMin);
    const histMax = Math.ceil(dataMax);
    return {
      histMin,
      histMax,
      binSize: Math.max(1 / IMAGEJ_HISTOGRAM_BINS, (histMax - histMin + 1) / IMAGEJ_HISTOGRAM_BINS),
    };
  }

  const histMin = dataMin;
  const histMax = dataMax;
  return {
    histMin,
    histMax,
    binSize: Math.max(Number.EPSILON, (histMax - histMin) / IMAGEJ_HISTOGRAM_BINS),
  };
}

function getAutoEligibleBinCount(count: number, limit: number): number {
  return count > limit ? 0 : count;
}

function makeFallbackRange(min: number, max: number): FijiAutoLutRange {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }

  if (max <= min) {
    return { min, max: min + 1 };
  }

  return { min, max };
}

function getFiniteOption(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

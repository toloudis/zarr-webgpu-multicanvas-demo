export interface ZarrImageSource {
  label?: string;
  url: string;
  arrayPath?: string;
  dimensionOrder?: string;
}

export type DimensionAxis = "T" | "C" | "Z" | "Y" | "X";
export type ZarrSelection = Array<number | null>;
export type NumericTypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export interface TCZYXShape {
  t: number;
  c: number;
  z: number;
  y: number;
  x: number;
}

export interface ChannelRenderSettings {
  index: number;
  enabled: boolean;
  color: string;
  min: number | null;
  max: number | null;
}

export interface ChannelRange {
  channelIndex: number;
  min: number;
  max: number;
  autoMin: number;
  autoMax: number;
}

export interface ChannelHistogram {
  bins: Int32Array;
  min: number;
  max: number;
  binSize: number;
  pixelCount: number;
}

export interface LoadedChannelPlane extends ChannelRange {
  histogram: ChannelHistogram;
  nativePixels: NumericTypedArray;
  selection: ZarrSelection;
}

export interface LoadedPlaneSet {
  width: number;
  height: number;
  channelPlanes: LoadedChannelPlane[];
  timeIndex: number;
  zIndex: number;
  arrayShape: number[];
  shapeTCZYX: TCZYXShape;
  chunks: number[];
  dtype: string;
  arrayPath: string;
  multiresolutionLevel: number;
  resolutionTarget: number;
}

export interface ZarrImageMetadata {
  source: ZarrImageSource;
  arrayPath: string;
  multiresolutionLevel: number;
  resolutionTarget: number;
  dimensionOrder: DimensionAxis[];
  arrayShape: number[];
  shapeTCZYX: TCZYXShape;
  chunks: number[];
  dtype: string;
}

export interface RenderStats {
  rendered: number;
  visible: number;
}

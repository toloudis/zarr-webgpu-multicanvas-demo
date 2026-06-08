export interface ZarrImageSource {
  label?: string;
  url: string;
  arrayPath?: string;
  dimensionOrder?: string;
}

export type DimensionAxis = "T" | "C" | "Z" | "Y" | "X";
export type ZarrSelection = Array<number | null>;

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
}

export interface ChannelRange {
  channelIndex: number;
  min: number;
  max: number;
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

export interface LoadedSlice {
  rgba: Uint8Array;
  width: number;
  height: number;
  channelRanges: ChannelRange[];
  timeIndex: number;
  zIndex: number;
  channelIndices: number[];
  arrayShape: number[];
  shapeTCZYX: TCZYXShape;
  chunks: number[];
  dtype: string;
  arrayPath: string;
  multiresolutionLevel: number;
  resolutionTarget: number;
  selections: ZarrSelection[];
}

export interface RenderStats {
  rendered: number;
  visible: number;
}

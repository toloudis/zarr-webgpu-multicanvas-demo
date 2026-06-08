export interface ZarrImageSource {
  label?: string;
  url: string;
  arrayPath?: string;
  dimensionOrder?: string;
}

export type ZarrSelection = Array<number | null>;

export interface LoadedSlice {
  rgba: Uint8Array;
  width: number;
  height: number;
  min: number;
  max: number;
  zIndex: number;
  arrayShape: number[];
  chunks: number[];
  dtype: string;
  selection: ZarrSelection;
}

export interface RenderStats {
  rendered: number;
  visible: number;
}

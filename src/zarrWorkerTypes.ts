import type { LoadedChannelPlane, ZarrSelection } from "./types";

export interface ZarrPlaneLoadRequest {
  id: number;
  sourceUrl: string;
  arrayPath: string;
  selection: ZarrSelection;
  width: number;
  height: number;
  channelIndex: number;
}

export type ZarrPlaneWorkerRequest =
  | { type: "load"; request: ZarrPlaneLoadRequest }
  | { type: "cancel"; id: number };

export type ZarrPlaneWorkerResponse =
  | { type: "loaded"; id: number; plane: LoadedChannelPlane }
  | { type: "error"; id: number; message: string; name?: string; stack?: string };

# Zarr WebGPU Multi-Canvas Demo

This is a small Vite app that loads 2D `Y,X` planes from a static list of `T,C,Z,Y,X` Zarr URLs with `zarrita`, then renders every plane into its own WebGPU canvas. The app keeps one synchronized global `T` index, one synchronized global `Z` index, one shared resolution target for multiscale level selection, and one shared set of channel visibility/color controls for every loaded image.

For grouped OME-Zarr stores, the loader inspects advertised multiscale datasets and selects the smallest `X,Y` level whose larger image dimension is still at least the resolution target. If every level is smaller than the target, it uses the largest available level.

The renderer follows the multiple-canvas pattern from WebGPU Fundamentals: one device, one configured context per canvas, a shared render pipeline, one command encoder for all visible canvases in a frame, `ResizeObserver` for backing-store sizing, and `IntersectionObserver` so off-screen canvases are skipped.

## Run

```sh
nvs use lts
npm install
npm run dev
```

Open the local Vite URL. The default list loads the included fixtures under `public/demo-zarr`.

## Static URL List

Edit `src/zarrSources.ts`:

```ts
export const ZARR_IMAGE_SOURCES = [
  { label: "Example", url: "https://example.org/image.zarr", arrayPath: "0" },
];
```

The `arrayPath` field is optional and is useful for grouped stores where the array is not at the store root, such as OME-Zarr scale paths.

## References

- WebGPU multiple canvases: https://webgpufundamentals.org/webgpu/lessons/webgpu-multiple-canvases.html
- zarrita: https://zarrita.dev/

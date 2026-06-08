import { mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

const demoRoot = new URL("../public/demo-zarr/", import.meta.url);
const demoRootPath = demoRoot.pathname.replace(/^\/([A-Za-z]:\/)/, "$1");
const legacyRoot = new URL("../public/demo-tczyx.zarr/", import.meta.url);
const legacyRootPath = legacyRoot.pathname.replace(/^\/([A-Za-z]:\/)/, "$1");

const shape = [1, 1, 7, 160, 192];
const chunks = [1, 1, 1, 160, 192];
const [, , zSize, height, width] = shape;
const specimens = ["a", "b", "c", "d", "e", "f"];

await rm(demoRootPath, { recursive: true, force: true });
await rm(legacyRootPath, { recursive: true, force: true });
await mkdir(demoRootPath, { recursive: true });

for (const [specimenIndex, specimen] of specimens.entries()) {
  const rootPath = join(demoRootPath, `specimen-${specimen}.zarr`);
  await mkdir(rootPath, { recursive: true });

  await writeJson(rootPath, ".zarray", {
    zarr_format: 2,
    shape,
    chunks,
    dtype: "|u1",
    compressor: null,
    fill_value: 0,
    order: "C",
    filters: null,
    dimension_separator: ".",
  });

  await writeJson(rootPath, ".zattrs", {
    _ARRAY_DIMENSIONS: ["t", "c", "z", "y", "x"],
    description: "Synthetic TCZYX uint8 fixture for the WebGPU multi-canvas demo.",
  });

  for (let z = 0; z < zSize; z++) {
    const plane = makePlane(specimenIndex, z);
    await writeFile(join(rootPath, `0.0.${z}.0.0`), plane);
  }
}

function makePlane(specimenIndex, z) {
  const plane = new Uint8Array(width * height);
  const angle = specimenIndex * 0.9 + z * 0.22;
  const cx = width * (0.48 + Math.cos(angle) * 0.19);
  const cy = height * (0.48 + Math.sin(angle) * 0.2);
  const sigma = 19 + specimenIndex * 2.5 + z * 3;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const blob = Math.exp(-(distance * distance) / (2 * sigma * sigma)) * 160;
      const ring = Math.max(0, 1 - Math.abs(distance - (30 + specimenIndex * 5 + z * 2)) / 8) * 58;
      const wave =
        (Math.sin((x + z * 17 + specimenIndex * 31) * 0.075) +
          Math.cos((y - z * 9 + specimenIndex * 23) * 0.092)) *
        23;
      const diagonal = ((x + y + z * 21 + specimenIndex * 14) % 56) < 3 ? 33 : 0;
      const value = Math.max(0, Math.min(255, 18 + specimenIndex * 8 + z * 6 + blob + ring + wave + diagonal));
      plane[y * width + x] = Math.round(value);
    }
  }

  return plane;
}

async function writeJson(rootPath, name, value) {
  await writeFile(join(rootPath, name), `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

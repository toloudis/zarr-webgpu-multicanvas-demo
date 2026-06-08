import "./styles.css";
import { ZARR_IMAGE_SOURCES } from "./zarrSources.js";
import { createImageGridRenderer } from "./webgpuImageRenderer.js";
import { loadMiddleZSlice } from "./zarrLoader.js";

const loadButton = document.querySelector("#load-button");
const statusText = document.querySelector("#status-text");
const renderStats = document.querySelector("#render-stats");
const grid = document.querySelector("#canvas-grid");

let renderer;
let currentAbortController;

init().catch((error) => {
  console.error(error);
  statusText.textContent = error.message;
});

loadButton.addEventListener("click", () => loadSources());

async function init() {
  renderer = await createImageGridRenderer(grid, (stats) => {
    renderStats.textContent = `${stats.rendered} rendered / ${stats.visible} visible`;
  });
  statusText.textContent = "WebGPU ready.";
  await loadSources();
}

async function loadSources() {
  if (!renderer) return;

  currentAbortController?.abort();
  currentAbortController = new AbortController();
  const sources = ZARR_IMAGE_SOURCES;

  renderer.clear();
  if (sources.length === 0) {
    statusText.textContent = "No configured Zarr sources.";
    return;
  }

  statusText.textContent = `Loading ${sources.length} slice${sources.length === 1 ? "" : "s"}.`;

  const tasks = sources.map((source, index) => async () => {
    const tile = renderer.addTile({
      title: source.label || `Slice ${index + 1}`,
      subtitle: source.url,
    });

    try {
      const slice = await loadMiddleZSlice({
        source,
        signal: currentAbortController.signal,
      });

      renderer.uploadTile(tile.id, slice);
      renderer.updateTile(tile.id, {
        subtitle: [
          `${slice.width} x ${slice.height}`,
          `T0 C0 Z${slice.zIndex}`,
          `range ${formatNumber(slice.min)} to ${formatNumber(slice.max)}`,
          `shape [${slice.arrayShape.join(", ")}]`,
        ].join("  "),
      });
    } catch (error) {
      if (currentAbortController.signal.aborted) return;
      console.error(error);
      renderer.setTileError(tile.id, error.message);
    }
  });

  const results = await runLimited(tasks, 4);
  const failed = results.filter((result) => result.status === "rejected").length;
  const loaded = sources.length - failed;

  if (!currentAbortController.signal.aborted) {
    statusText.textContent = failed
      ? `Loaded ${loaded}; ${failed} failed.`
      : `Loaded ${loaded} slice${loaded === 1 ? "" : "s"}.`;
  }
}

async function runLimited(tasks, limit) {
  const results = new Array(tasks.length);
  let next = 0;

  async function worker() {
    while (next < tasks.length) {
      const index = next++;
      try {
        await tasks[index]();
        results[index] = { status: "fulfilled" };
      } catch (error) {
        results[index] = { status: "rejected", reason: error };
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toPrecision(4);
}

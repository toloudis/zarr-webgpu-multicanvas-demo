import "./styles.css";
import { ZARR_IMAGE_SOURCES } from "./zarrSources";
import { createImageGridRenderer, type ImageGridRenderer } from "./webgpuImageRenderer";
import { loadMiddleZSlice } from "./zarrLoader";

const loadButton = requireElement<HTMLButtonElement>("#load-button");
const statusText = requireElement<HTMLElement>("#status-text");
const renderStats = requireElement<HTMLElement>("#render-stats");
const grid = requireElement<HTMLElement>("#canvas-grid");

let renderer: ImageGridRenderer | undefined;
let currentAbortController: AbortController | undefined;

init().catch((error) => {
  console.error(error);
  statusText.textContent = getErrorMessage(error);
});

loadButton.addEventListener("click", () => loadSources());

async function init(): Promise<void> {
  renderer = await createImageGridRenderer(grid, (stats) => {
    renderStats.textContent = `${stats.rendered} rendered / ${stats.visible} visible`;
  });
  statusText.textContent = "WebGPU ready.";
  await loadSources();
}

async function loadSources(): Promise<void> {
  const activeRenderer = renderer;
  if (!activeRenderer) return;

  currentAbortController?.abort();
  const abortController = new AbortController();
  currentAbortController = abortController;
  const sources = ZARR_IMAGE_SOURCES;

  activeRenderer.clear();
  if (sources.length === 0) {
    statusText.textContent = "No configured Zarr sources.";
    return;
  }

  statusText.textContent = `Loading ${sources.length} slice${sources.length === 1 ? "" : "s"}.`;

  const tasks = sources.map((source, index) => async (): Promise<void> => {
    const tile = activeRenderer.addTile({
      title: source.label || `Slice ${index + 1}`,
      subtitle: source.url,
    });

    try {
      const slice = await loadMiddleZSlice({
        source,
        signal: abortController.signal,
      });

      activeRenderer.uploadTile(tile.id, slice);
      activeRenderer.updateTile(tile.id, {
        subtitle: [
          `${slice.width} x ${slice.height}`,
          `T0 C0 Z${slice.zIndex}`,
          `range ${formatNumber(slice.min)} to ${formatNumber(slice.max)}`,
          `shape [${slice.arrayShape.join(", ")}]`,
        ].join("  "),
      });
    } catch (error) {
      if (abortController.signal.aborted) return;
      console.error(error);
      activeRenderer.setTileError(tile.id, getErrorMessage(error));
    }
  });

  const results = await runLimited(tasks, 4);
  const failed = results.filter((result) => result.status === "rejected").length;
  const loaded = sources.length - failed;

  if (!abortController.signal.aborted) {
    statusText.textContent = failed
      ? `Loaded ${loaded}; ${failed} failed.`
      : `Loaded ${loaded} slice${loaded === 1 ? "" : "s"}.`;
  }
}

async function runLimited<T>(
  tasks: Array<() => Promise<T>>,
  limit: number,
): Promise<Array<PromiseSettledResult<T>>> {
  const results = new Array<PromiseSettledResult<T>>(tasks.length);
  let next = 0;

  async function worker(): Promise<void> {
    while (next < tasks.length) {
      const index = next++;
      try {
        const value = await tasks[index]();
        results[index] = { status: "fulfilled", value };
      } catch (error) {
        results[index] = { status: "rejected", reason: error };
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toPrecision(4);
}

function requireElement<TElement extends Element>(selector: string): TElement {
  const element = document.querySelector<TElement>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

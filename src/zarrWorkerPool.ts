import type { LoadedChannelPlane } from "./types";
import type { ZarrPlaneLoadRequest, ZarrPlaneWorkerResponse } from "./zarrWorkerTypes";

type ZarrPlaneLoadTask = Omit<ZarrPlaneLoadRequest, "id">;

interface QueueTask {
  request: ZarrPlaneLoadRequest;
  resolve: (plane: LoadedChannelPlane) => void;
  reject: (reason?: unknown) => void;
  signal?: AbortSignal;
  abortListener?: () => void;
  settled: boolean;
  started: boolean;
}

interface WorkerEntry {
  worker: Worker;
  task?: QueueTask;
}

let sharedPool: ZarrPlaneWorkerPool | undefined;

export function loadChannelPlaneInWorker(
  request: ZarrPlaneLoadTask,
  signal?: AbortSignal,
): Promise<LoadedChannelPlane> {
  sharedPool ??= new ZarrPlaneWorkerPool(getDefaultWorkerCount());
  return sharedPool.load(request, signal);
}

class ZarrPlaneWorkerPool {
  private readonly workers: WorkerEntry[] = [];
  private readonly queue: QueueTask[] = [];
  private nextRequestId = 1;

  constructor(workerCount: number) {
    for (let index = 0; index < workerCount; index++) {
      this.workers.push({ worker: this.createWorker() });
    }
  }

  load(request: ZarrPlaneLoadTask, signal?: AbortSignal): Promise<LoadedChannelPlane> {
    if (signal?.aborted) {
      return Promise.reject(createAbortError());
    }

    return new Promise((resolve, reject) => {
      const task: QueueTask = {
        request: {
          ...request,
          id: this.nextRequestId++,
        },
        resolve,
        reject,
        signal,
        settled: false,
        started: false,
      };

      if (signal) {
        task.abortListener = () => this.abortTask(task);
        signal.addEventListener("abort", task.abortListener, { once: true });
      }

      this.queue.push(task);
      this.schedule();
    });
  }

  private createWorker(): Worker {
    const worker = new Worker(new URL("./zarrPlaneWorker.ts", import.meta.url), {
      name: "zarr-plane-worker",
      type: "module",
    });

    worker.addEventListener("message", (event: MessageEvent<ZarrPlaneWorkerResponse>) => {
      const entry = this.workers.find((item) => item.worker === worker);
      if (!entry) return;
      this.handleMessage(entry, event.data);
    });

    worker.addEventListener("error", (event) => {
      const entry = this.workers.find((item) => item.worker === worker);
      if (!entry) return;
      this.replaceFailedWorker(entry, new Error(event.message));
    });

    worker.addEventListener("messageerror", () => {
      const entry = this.workers.find((item) => item.worker === worker);
      if (!entry) return;
      this.replaceFailedWorker(entry, new Error("Could not deserialize a Zarr worker response."));
    });

    return worker;
  }

  private schedule(): void {
    for (const entry of this.workers) {
      if (entry.task) continue;

      let task = this.queue.shift();
      while (task?.settled) {
        task = this.queue.shift();
      }

      if (!task) return;

      this.startTask(entry, task);
    }
  }

  private startTask(entry: WorkerEntry, task: QueueTask): void {
    task.started = true;
    entry.task = task;

    try {
      entry.worker.postMessage({ type: "load", request: task.request });
    } catch (error) {
      entry.task = undefined;
      this.rejectTask(task, error);
      this.schedule();
    }
  }

  private handleMessage(entry: WorkerEntry, response: ZarrPlaneWorkerResponse): void {
    const task = entry.task;
    entry.task = undefined;

    if (!task || task.request.id !== response.id) {
      this.schedule();
      return;
    }

    this.removeAbortListener(task);

    if (!task.settled) {
      task.settled = true;
      if (response.type === "loaded") {
        task.resolve(response.plane);
      } else {
        task.reject(createWorkerError(response));
      }
    }

    this.schedule();
  }

  private abortTask(task: QueueTask): void {
    if (task.settled) return;

    task.settled = true;
    this.removeAbortListener(task);

    const queueIndex = this.queue.indexOf(task);
    if (queueIndex !== -1) {
      this.queue.splice(queueIndex, 1);
    }

    if (task.started) {
      const entry = this.workers.find((item) => item.task === task);
      entry?.worker.postMessage({ type: "cancel", id: task.request.id });
    }

    task.reject(createAbortError());
  }

  private rejectTask(task: QueueTask, reason: unknown): void {
    this.removeAbortListener(task);
    if (task.settled) return;

    task.settled = true;
    task.reject(reason);
  }

  private replaceFailedWorker(entry: WorkerEntry, error: Error): void {
    const failedTask = entry.task;
    entry.task = undefined;
    entry.worker.terminate();
    entry.worker = this.createWorker();

    if (failedTask) {
      this.rejectTask(failedTask, error);
    }

    this.schedule();
  }

  private removeAbortListener(task: QueueTask): void {
    if (task.signal && task.abortListener) {
      task.signal.removeEventListener("abort", task.abortListener);
      task.abortListener = undefined;
    }
  }
}

function getDefaultWorkerCount(): number {
  const hardwareConcurrency = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(4, hardwareConcurrency - 1));
}

function createWorkerError(response: Extract<ZarrPlaneWorkerResponse, { type: "error" }>): Error {
  const error = new Error(response.message);
  error.name = response.name ?? "Error";
  if (response.stack) {
    error.stack = response.stack;
  }
  return error;
}

function createAbortError(): unknown {
  if (typeof DOMException !== "undefined") {
    return new DOMException("The operation was aborted.", "AbortError");
  }

  const error = new Error("The operation was aborted.");
  error.name = "AbortError";
  return error;
}

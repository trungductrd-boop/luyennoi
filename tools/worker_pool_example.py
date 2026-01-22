"""
Safe worker-pool example for CPU-bound audio tasks.

Guidelines demonstrated:
- Use `spawn` start method on POSIX/Windows to avoid copy-on-write surprises when
  the process image contains large in-memory objects.
- Limit `max_workers` to control total RAM used.
- Use `maxtasksperchild` equivalent by recreating executors periodically (not shown)
  or by using short-lived subprocesses for isolated tasks.
- Pass file paths or small objects to workers (avoid sharing large in-memory state).

Run:
    python tools/worker_pool_example.py

This is an example script only; adapt to your environment.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os

# Ensure we use 'spawn' on platforms that support it to avoid fork() memory cloning.
try:
    multiprocessing.set_start_method('spawn')
except Exception:
    # start method may already be set; ignore
    pass


def process_audio_file(path: str) -> dict:
    """Worker function that loads minimal modules and processes a single file path.

    IMPORTANT: Keep imports inside the function so the worker process starts with
    a small memory footprint.
    """
    try:
        # Import locally to avoid pulling large modules into parent's address space
        import librosa
        import numpy as np
        # Lightweight processing: load limited duration and compute simple features
        y, sr = librosa.load(path, sr=16000, mono=True, duration=3.0)
        # compute mean absolute amplitude as a tiny example
        amp = float(np.mean(np.abs(y))) if y is not None else 0.0
        return {"path": path, "amp": amp, "ok": True}
    except Exception as e:
        return {"path": path, "error": str(e), "ok": False}


def run_worker_pool(paths, max_workers=2, chunk_size=1):
    """Run jobs with a bounded ProcessPoolExecutor.

    - `max_workers` controls parallelism (RAM scales roughly with this).
    - `chunk_size` controls how many tasks are sent at once (useful for many small tasks).
    """
    results = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = []
        for p in paths:
            futures.append(exe.submit(process_audio_file, p))
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"error": str(e)})
    duration = time.time() - start
    print(f"Processed {len(paths)} files in {duration:.2f}s using {max_workers} workers")
    return results


if __name__ == '__main__':
    # Example: pick up to 4 small WAV files from data/samples
    sample_dir = os.path.join('data', 'samples')
    if not os.path.isdir(sample_dir):
        print("No samples dir found; create data/samples with a few small audio files to try")
    else:
        files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith('.wav')][:10]
        if not files:
            print("No .wav files found in data/samples")
        else:
            res = run_worker_pool(files, max_workers=2)
            for r in res:
                print(r)

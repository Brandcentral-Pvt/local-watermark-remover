#!/usr/bin/env python3
"""Scale test: run a realistic mixed batch through the GUI worker while watching RSS."""
import os
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, "/home/user/local_app")
import watermark_gui as g  # noqa: E402


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


stop = threading.Event()
samples: list = []


def monitor():
    while not stop.is_set():
        samples.append(rss_mb())
        time.sleep(0.4)


threading.Thread(target=monitor, daemon=True).start()

in_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scale/in"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/scale/out"
model = "/tmp/wm/lama_fp32.onnx"

low_mem = g.default_low_mem()
files = g.collect_inputs(None, in_dir, "image", True)
print(f"queue: {len(files)} files | RSS at start {rss_mb():.0f} MB | low-memory mode: {low_mem}")
t0 = time.time()
job = g.Job()
g.process_images(job, files, out_dir, "gemini_sparkle", True, False, True, model, "auto", 0,
                 low_mem=low_mem)
dt = time.time() - t0
stop.set()
time.sleep(0.6)

counts = Counter(r["status"] for r in job.rows)
print(f"--- {dt:.0f}s total, {dt / max(len(files), 1):.1f}s per file ---")
print("rows:", len(job.rows), "| statuses:", dict(counts))
print(f"RSS: start {samples[0]:.0f} MB, peak {max(samples):.0f} MB, end {samples[-1]:.0f} MB")
outs = [f for f in os.listdir(out_dir) if f.endswith((".jpg", ".png"))]
print("outputs written:", len(outs))
print("previews:", len(os.listdir(os.path.join(out_dir, "_previews"))))
verdicts = Counter(r["verdict"][:26] for r in job.rows if r["status"] == "done")
print("done verdicts:", dict(verdicts))
unexpected = [r for r in job.rows if r["status"] not in ("done", "no mark", "skipped", "error")]
print("unexpected rows:", unexpected[:3] or "none")
print("report rows:", len(open(os.path.join(out_dir, "images_report.csv")).read().splitlines()) - 1)

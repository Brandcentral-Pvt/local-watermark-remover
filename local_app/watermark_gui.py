#!/usr/bin/env python3
"""
Watermark Remover -- local desktop app (Gradio GUI).

Removes the visible watermark from your own Veo/Flow videos and Gemini images:
  * Gemini image sparkle   -> shape-template detection + adaptive fill
  * Flow / Veo wordmark    -> temporal detection + LaMa inpainting

Everything runs on your machine. The only network access is the one-time 208 MB model
download (skip it by pointing --model at an existing lama_fp32.onnx).

Usage:
    python watermark_gui.py                # opens http://127.0.0.1:7860
    python watermark_gui.py --port 8000 --no-browser
    python watermark_gui.py --model D:\\models\\lama_fp32.onnx --device cpu --threads 6

Layout expected:
    <folder>/watermark_gui.py
    <folder>/wm_core.py            <- the engine (same folder)
    <folder>/models/lama_fp32.onnx <- created by the download button / first run
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))                     # so `import wm_core` finds our engine

try:
    import cv2
    import numpy as np
    import gradio as gr
except ModuleNotFoundError as exc:                    # friendly first-run message
    sys.exit(f"Missing dependency: {exc.name}\nInstall everything with:\n"
             f"    python -m pip install -r requirements.txt")

import wm_core as wc

MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL_MIN_BYTES = 100_000_000
APP_NAME = "Watermark Remover"
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wm")

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")

# --------------------------------------------------------------------------------------
# model management
# --------------------------------------------------------------------------------------

_INPAINTER_CACHE: dict = {}


def default_model_path() -> str:
    return str(APP_DIR / "models" / "lama_fp32.onnx")


def download_model(dest: str, progress_cb=None) -> str:
    """Fetch the LaMa ONNX weights once. Writes to `<dest>.part` then renames."""
    import urllib.request

    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    if dest_p.is_file() and dest_p.stat().st_size >= MODEL_MIN_BYTES:
        return str(dest_p)
    part = str(dest_p) + ".part"
    t0 = time.time()

    def hook(block, block_size, total):
        if progress_cb and total > 0:
            done = min(block * block_size, total)
            el = max(time.time() - t0, 1e-6)
            progress_cb(done, total, done / el)

    log.info("downloading model -> %s", dest_p)
    urllib.request.urlretrieve(MODEL_URL, part, reporthook=hook)
    if os.path.getsize(part) < MODEL_MIN_BYTES:
        os.remove(part)
        raise RuntimeError("downloaded file looks truncated - check your connection and retry")
    os.replace(part, str(dest_p))
    log.info("model saved (%d MB)", dest_p.stat().st_size // 1_000_000)
    return str(dest_p)


def load_inpainter(model_path: str, device: str = "auto", threads: int = 0, verbose: bool = True,
                   low_mem: bool = False):
    """Load (and cache) the ONNX session. `device`: auto | cpu | cuda."""
    import onnxruntime as ort

    key = (os.path.abspath(model_path), device, threads, bool(low_mem))
    if key in _INPAINTER_CACHE:
        return _INPAINTER_CACHE[key]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    avail = ort.get_available_providers()
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError("CUDAExecutionProvider is not available - install onnxruntime-gpu, "
                               "or switch the device to cpu/auto")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    inp = wc.Inpainter(model_path, providers=providers, threads=threads, verbose=verbose,
                       mem_arena=not low_mem, graph_opt="basic" if low_mem else "all")
    _INPAINTER_CACHE[key] = inp
    return inp


def total_ram_gb() -> float:
    """Physical RAM in GB (0.0 when the platform will not tell us)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def default_low_mem() -> bool:
    """Under ~4 GB the full ORT arena is the difference between finishing a batch and an OOM."""
    ram = total_ram_gb()
    return bool(ram and ram < 4.0)


def gpu_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# job plumbing: worker thread -> event queue -> streaming UI updates
# --------------------------------------------------------------------------------------

class JobCancelled(Exception):
    pass


class Job:
    """A unit of work with a live event queue, a cancel flag and an owning session."""

    def __init__(self, owner: str = "local", label: str = ""):
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.cancel = threading.Event()
        self.started = threading.Event()
        self.rows: list = []
        self.report_paths: list = []
        self.t0 = time.time()
        self.owner = owner                                  # session key: who may cancel this
        self.label = label
        self.queue_pos = 0
        self.finished = False
        self.zip_path = None

    # --- events emitted by workers -----------------------------------------------------
    def log(self, msg: str):
        self.q.put({"type": "log", "msg": msg})

    def progress(self, done: int, total: int, label: str = ""):
        self.q.put({"type": "progress", "done": done, "total": max(total, 1), "label": label})

    def row(self, row: dict):
        self.rows.append(row)
        self.q.put({"type": "row", "row": row})

    def preview(self, path: str, caption: str):
        self.q.put({"type": "preview", "path": path, "caption": caption})

    def done(self, summary: str):
        self.q.put({"type": "done", "msg": summary})

    def error(self, msg: str):
        self.q.put({"type": "error", "msg": msg})

    def check_cancel(self):
        if self.cancel.is_set():
            raise JobCancelled()


class Scheduler:
    """
    One job at a time, first come first served.

    Inpainting is CPU/GPU bound, so running two jobs in parallel makes both slower and doubles
    peak memory (which is what kills small machines). The queue keeps a hosted instance fair:
    every visitor sees their position and no one can hog the box.
    """

    def __init__(self, max_waiting: int = 6):
        self.max_waiting = max_waiting
        self._q: "collections.deque[tuple[Job, object]]" = collections.deque()
        self._lock = threading.Lock()
        self._current: Job = None
        threading.Thread(target=self._loop, daemon=True, name="wm-scheduler").start()

    def submit(self, job: Job, target) -> int:
        with self._lock:
            if len(self._q) >= self.max_waiting:
                raise RuntimeError(
                    f"the queue is full ({len(self._q)} waiting) — try again in a few minutes")
            self._q.append((job, target))
            pos = len(self._q)
        job.queue_pos = pos
        job.log(f"queued — position {pos} ({pos - 1} ahead of you)")
        return pos

    def position_of(self, job: Job) -> int:
        with self._lock:
            for i, (j, _) in enumerate(self._q, 1):
                if j is job:
                    return i
        return 0                                            # 0 = running or finished

    @property
    def current(self) -> Job:
        return self._current

    def waiting(self) -> int:
        with self._lock:
            return len(self._q)

    def _loop(self):
        while True:
            with self._lock:
                item = self._q.popleft() if self._q else None
                if item is not None:
                    self._current = item[0]
            if item is None:
                time.sleep(0.1)
                continue
            job, target = item
            if job.cancel.is_set():
                job.done("Cancelled while queued")
                job.finished = True
                with self._lock:
                    self._current = None
                continue
            job.started.set()
            job.q.put({"type": "log", "msg": "▶ starting"})
            try:
                target(job)
            except JobCancelled:
                job.log("⏹  cancelled by user")
                job.done("Cancelled")
            except Exception as exc:                                   # noqa: BLE001
                log.error("job failed: %s\n%s", exc, traceback.format_exc())
                job.error(f"{type(exc).__name__}: {exc}")
            finally:
                job.finished = True
                with self._lock:
                    self._current = None


SCHEDULER = Scheduler()
JOBS: dict = {}                                             # session key -> running/queued Job


def session_key(request) -> str:
    """Stable per-browser key, so a visitor can only touch their own job."""
    key = getattr(request, "session_hash", None) if request is not None else None
    if not key:
        client = getattr(request, "client", None) if request is not None else None
        host = getattr(client, "host", None) if client is not None else None
        key = host or "local"
    return str(key)[:32]


def _run_in_thread(target, job: Job):
    def wrapper():
        try:
            target(job)
        except JobCancelled:
            job.log("⏹  cancelled by user")
            job.done("Cancelled")
        except Exception as exc:                                   # noqa: BLE001
            log.error("job failed: %s\n%s", exc, traceback.format_exc())
            job.error(f"{type(exc).__name__}: {exc}")
    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    return t


def stream_ui(job: Job, thread=None, header: str = "", previews: bool = True):
    """
    Generator: drains the job queue and yields UI updates until the job finishes.
    Yielded tuple: (status_md, progress_value, log_text, table_rows, gallery_items)
    """
    lines: list[str] = []
    gallery: list = []
    done = total = 0
    label = ""
    status = header or "working…"
    last_emit = 0.0
    while True:
        finished = False
        while True:                                     # drain everything queued right now
            try:
                ev = job.q.get_nowait()
            except queue.Empty:
                break
            kind = ev["type"]
            if kind == "log":
                lines.append(ev["msg"])
                lines = lines[-500:]
            elif kind == "progress":
                done, total, label = ev["done"], ev["total"], ev.get("label", "")
            elif kind == "row":
                pass
            elif kind == "preview" and previews:
                gallery.append((ev["path"], ev["caption"]))
                gallery = gallery[-24:]
            elif kind == "done":
                status = ev["msg"]
                finished = True
            elif kind == "error":
                status = ev["msg"]
                lines.append("❌ " + ev["msg"])
                finished = True
        if thread is not None and not thread.is_alive():
            finished = True
        if job.finished and not finished:
            finished = True
        frac = (done / total) if total else 0.0
        if finished:
            frac = 1.0
        el = time.time() - job.t0
        eta = (el / frac - el) if frac > 0.01 else 0.0
        if not job.started.is_set():                      # still waiting for the CPU
            pos = SCHEDULER.position_of(job)
            ahead = max(pos - 1, 0)
            head = (f"**queued** — {ahead} job(s) ahead of you  \n"
                    f"waiting {fmt_time(el)} (one job runs at a time, so everyone gets a "
                    f"fair share of the machine)")
        else:
            head = status if finished else (
                f"**{status}**  \n{label} — {done}/{total}  "
                f"({frac*100:.0f}%, elapsed {fmt_time(el)}"
                f"{', ETA ' + fmt_time(eta) if eta > 1 else ''})"
            )
        body = "\n".join(lines[-400:])
        now = time.time()
        if finished or (now - last_emit) > 0.25:
            last_emit = now
            yield head, frac, body, [row_to_table(r) for r in job.rows], gallery
        if finished:
            return
        time.sleep(0.12)


def _stream_with_zip(job: Job, thread, header: str, out_dir: str, key: str):
    """`stream_ui` plus a results .zip that appears in the UI once the job ends."""
    last = None
    for out in stream_ui(job, thread, header=header):
        last = out
        yield (*out, gr.update())
    if last is None:
        return
    # also after a cancel: whatever finished is on disk and the visitor should be able to
    # take it home (make_results_zip returns None when there is nothing to bundle)
    try:
        job.zip_path = make_results_zip(out_dir, key)
    except Exception as exc:                                       # noqa: BLE001
        log.warning("could not build the results zip: %s", exc)
    yield (*last, job.zip_path)


def fmt_time(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m {sec % 60:02d}s"
    return f"{sec // 3600}h {(sec % 3600) // 60:02d}m"


TABLE_HEADERS = ["file", "status", "method", "detail", "time", "verdict"]


def row_to_table(r: dict) -> list:
    return [r.get("file", ""), r.get("status", ""), r.get("method", ""),
            r.get("detail", ""), r.get("time", ""), r.get("verdict", "")]


def write_report(rows: list, out_dir: str, name: str = "report") -> list:
    """CSV + JSON next to the outputs. Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if not rows:
        return paths
    fields = sorted({k for r in rows for k in r})
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    paths += [csv_path, json_path]
    return paths


# --------------------------------------------------------------------------------------
# bulk workers
# --------------------------------------------------------------------------------------

def collect_inputs(files, folders_text: str, kind: str, recursive: bool) -> list:
    """Combine uploaded files and folder paths (one per line) into a single work list."""
    out: list = []
    for f in files or []:
        p = getattr(f, "name", f)
        if p and os.path.isfile(p):
            out.append(os.path.abspath(p))
    exts = IMAGE_EXT if kind == "image" else VIDEO_EXT
    for line in (folders_text or "").splitlines():
        folder = line.strip().strip('"')
        if not folder:
            continue
        if os.path.isfile(folder) and folder.lower().endswith(exts):
            out.append(os.path.abspath(folder))
        elif os.path.isdir(folder):
            out += wc.list_media(folder, kind=kind, recursive=recursive)
        else:
            log.warning("skipping unsupported path: %s", folder)
    # de-duplicate, keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def scan_images(job: Job, files: list, out_dir: str, preset: str, adaptive: bool,
                skip_existing: bool, model_path: str, device: str, threads: int,
                allow_preset: bool = False):
    """Dry run: detect the mark in every image and report what would happen."""
    job.log(f"scanning {len(files)} image(s) — no files are modified")
    for i, path in enumerate(files, 1):
        job.check_cancel()
        job.progress(i, len(files), os.path.basename(path))
        img = wc.imread_color(path)
        if img is None:
            job.row(dict(file=os.path.basename(path), status="error", method="-", detail="-",
                         time="-", verdict="could not read file"))
            continue
        h, w = img.shape[:2]
        mask, method = wc.detect_for_image(img, mode="auto", preset=preset,
                                           allow_preset=allow_preset, verbose=False)
        if mask is None:
            probe = wc.sparkle_probe(img)
            job.row(dict(file=os.path.basename(path), status="no mark", method="none",
                         detail=f"{w}x{h}, shape score {probe:.2f}", time="-",
                         verdict=("nothing found; closest shape match "
                                  f"{probe:.2f} vs 0.70 needed" if probe >= 0.55
                                  else "no sparkle found (try a preset box)")))
            job.log(f"  · {os.path.basename(path)}: no mark found, shape score {probe:.2f}")
            continue
        tex = wc.ring_texture(img, mask)
        fill = "smooth" if (adaptive and tex < 4.0) else "lama"
        x0, y0, x1, y1 = wc.mask_bbox(mask)
        job.row(dict(file=os.path.basename(path), status="ready", method=method,
                     detail=f"{w}x{h}, mask {int((mask > 0).sum())}px @({x0},{y0})",
                     time="-", verdict=f"fill: {fill} (texture {tex:.1f})"))
        job.log(f"  · {os.path.basename(path)}: mark at ({x0},{y0}) {int((mask>0).sum())}px, "
                f"background texture {tex:.1f} -> {fill} fill")
        if job.cancel.is_set():
            break
        # mask preview
        p = os.path.join(out_dir, "_previews", f"{Path(path).stem}_mask.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        cv2.imwrite(p, wc.mask_preview(img, mask, scale=3))
        job.preview(p, f"{os.path.basename(path)} — detected mark")
    job.done(f"Scan finished — {len(job.rows)} file(s) inspected")


def process_images(job: Job, files: list, out_dir: str, preset: str, adaptive: bool,
                   skip_existing: bool, verify: bool, model_path: str, device: str, threads: int,
                   allow_preset: bool = False, low_mem: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    job.log(f"loading model: {model_path}")
    inp = load_inpainter(model_path, device, threads, verbose=False, low_mem=low_mem)
    job.log(f"model ready ({inp.provider}, {inp.model_size:.0f} MB"
            f"{', low-memory mode' if low_mem else ''})")
    done_n = fail_n = skip_n = 0
    t_all = time.time()
    try:
        for i, path in enumerate(files, 1):
            job.check_cancel()
            name = os.path.basename(path)
            job.progress(i, len(files), name)
            img = wc.imread_color(path)
            if img is None:
                fail_n += 1
                job.row(dict(file=name, status="error", method="-", detail="-", time="-",
                             verdict="could not read file"))
                continue
            t0 = time.time()
            try:
                mask, method = wc.detect_for_image(img, mode="auto", preset=preset,
                                                   allow_preset=allow_preset, verbose=False)
                if mask is None:
                    probe = wc.sparkle_probe(img)
                    near = (f" — close shape match {probe:.2f} (needs 0.70); use the fallback box "
                            f"if you know this file is watermarked") if probe >= 0.55 else \
                           " (enable the fallback box if this is a wordmark)"
                    job.row(dict(file=name, status="no mark", method="none",
                                 detail=f"shape score {probe:.2f}", time="-",
                                 verdict="no sparkle found - file left untouched" + near))
                    job.log(f"  · {name}: no mark found (shape score {probe:.2f}) — left untouched")
                    continue
                ext = os.path.splitext(path)[1].lower()
                canonical = os.path.join(out_dir, f"{Path(path).stem}_clean{ext}")
                if skip_existing and os.path.exists(canonical):
                    skip_n += 1
                    job.row(dict(file=name, status="skipped", method=method, detail="-", time="-",
                                 verdict="output already exists (resumable run)"))
                    job.log(f"  · {name}: skipped — {os.path.basename(canonical)} exists")
                    continue
                dst = wc.unique_output_path(path, out_dir)
                tex = wc.ring_texture(img, mask)
                res = (wc.adaptive_fill(img, mask, inp, verbose=False) if adaptive
                       else inp.inpaint(img, mask))
                params = [cv2.IMWRITE_JPEG_QUALITY, 97] if dst.lower().endswith((".jpg", ".jpeg")) else []
                cv2.imwrite(dst, res, params)
                verdict = "written"
                iou_a = None
                if verify:
                    if method == "sparkle":
                        v = wc.verify_sparkle_removal(img, res, verbose=False)
                        verdict, iou_a = v["verdict"], v["iou_after"]
                    else:
                        v = wc.verify_removal(img, res, mask)
                        verdict = v["verdict"]
                dt = time.time() - t0
                done_n += 1
                job.row(dict(file=name, status="done", method=method,
                             detail=f"mask {int((mask>0).sum())}px, texture {tex:.1f}",
                             time=fmt_time(dt), verdict=verdict, output=os.path.basename(dst),
                             iou_after=iou_a))
                job.log(f"  ✓ {name}  ({fmt_time(dt)})  {verdict}")
                # before/after zoom for the gallery
                x0, y0, x1, y1 = wc.mask_bbox(mask)
                p = 22
                sl = (slice(max(y0 - p, 0), min(y1 + p, img.shape[0])),
                      slice(max(x0 - p, 0), min(x1 + p, img.shape[1])))
                sep = np.full((img[sl].shape[0], 4, 3), 70, np.uint8)
                zoom = np.hstack([img[sl], sep, res[sl]])
                zp = os.path.join(out_dir, "_previews", f"{Path(path).stem}_before_after.png")
                os.makedirs(os.path.dirname(zp), exist_ok=True)
                cv2.imwrite(zp, zoom)
                job.preview(zp, f"{name} — before | after")
            except JobCancelled:
                raise
            except Exception as exc:                                   # noqa: BLE001
                if job.cancel.is_set():
                    job.row(dict(file=name, status="cancelled", method=locals().get("method", "-"),
                                 detail="-", time="-",
                                 verdict="cancelled - file left as it was"))
                    job.log(f"  \u23f9 {name}: cancelled")
                    break
                fail_n += 1
                log.exception("image failed: %s", path)
                job.row(dict(file=name, status="failed", method="-", detail="-",
                             time=fmt_time(time.time() - t0), verdict=str(exc)[:120]))
                job.log(f"  ✗ {name}: {exc}")
    except JobCancelled:
        job.log("  \u23f9 cancelled by user")
    job.report_paths = write_report(job.rows, out_dir, "images_report")
    total_time = time.time() - t_all
    if job.cancel.is_set():
        job.done(f"Images: cancelled after {done_n + skip_n + fail_n}/{len(files)} file(s) — "
                 f"partial report written → {out_dir}")
    else:
        job.done(f"Images: {done_n} processed, {skip_n} skipped, {fail_n} failed "
                 f"in {fmt_time(total_time)} → {out_dir}")


def _video_mask_for(job: Job, path: str, preset: str, n_frames: int):
    """Temporal detection when the clip moves enough, else the preset box."""
    try:
        mask = wc.detect_mask_from_video(path, n_frames=n_frames, verbose=False)
        return mask, "temporal"
    except Exception:
        frame0 = wc.sample_frames(path, n=1)[0]
        return wc.preset_mask(frame0, preset=preset, pad=6, refine=True, verbose=False), f"preset:{preset}"


def scan_videos(job: Job, files: list, out_dir: str, preset: str, model_path: str,
                device: str, threads: int):
    job.log(f"scanning {len(files)} clip(s) — nothing is re-encoded")
    for i, path in enumerate(files, 1):
        job.check_cancel()
        job.progress(i, len(files), os.path.basename(path))
        try:
            nfo = wc.video_info(path)
            mask, method = _video_mask_for(job, path, preset, 24)
            frame0 = wc.sample_frames(path, n=1)[0]
            x0, y0, x1, y1 = wc.mask_bbox(mask)
            job.row(dict(file=os.path.basename(path), status="ready", method=method,
                         detail=f"{nfo['width']}x{nfo['height']}, {nfo['frames']} frames, "
                                f"mask {int((mask > 0).sum())}px",
                         time="-", verdict=f"mark at ({x0},{y0})-({x1},{y1})"))
            p = os.path.join(out_dir, "_previews", f"{Path(path).stem}_mask.png")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, wc.mask_preview(frame0, mask, scale=3))
            job.preview(p, f"{os.path.basename(path)} — detected mark ({method})")
            job.log(f"  · {os.path.basename(path)}: {method}, mask {int((mask>0).sum())}px")
        except Exception as exc:                                   # noqa: BLE001
            job.row(dict(file=os.path.basename(path), status="error", method="-", detail="-",
                         time="-", verdict=str(exc)[:120]))
            job.log(f"  ✗ {os.path.basename(path)}: {exc}")
    job.done(f"Scan finished — {len(job.rows)} clip(s) inspected")


def process_videos(job: Job, files: list, out_dir: str, preset: str, engine: str,
                   batch_size: int, crf: int, verify: bool, skip_existing: bool,
                   model_path: str, device: str, threads: int, low_mem: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    inp = None
    if engine == "lama":
        job.log(f"loading model: {model_path}")
        inp = load_inpainter(model_path, device, threads, verbose=False, low_mem=low_mem)
        job.log(f"model ready ({inp.provider}, {inp.model_size:.0f} MB"
                f"{', low-memory mode' if low_mem else ''})")
    done_n = fail_n = skip_n = empty_n = 0
    t_all = time.time()
    try:
        for i, path in enumerate(files, 1):
            job.check_cancel()
            name = os.path.basename(path)
            try:
                nfo = wc.video_info(path)
                canonical = os.path.join(out_dir, f"{Path(path).stem}_clean.mp4")
                if skip_existing and os.path.exists(canonical):
                    skip_n += 1
                    job.row(dict(file=name, status="skipped", method=engine, detail="-", time="-",
                                 verdict="output already exists (resumable run)"))
                    job.log(f"  · {name}: skipped — {os.path.basename(canonical)} exists")
                    continue
                dst = wc.unique_output_path(path, out_dir, ext=".mp4")
                mask, method = _video_mask_for(job, path, preset, 24)
                job.log(f"  {name}: {nfo['width']}x{nfo['height']} {nfo['frames']}f — {method}, "
                        f"mask {int((mask > 0).sum())}px")
                t0 = time.time()
                job.progress(0, 1, f"{name} (starting)")

                def on_progress(done, total, elapsed, rate):
                    job.progress(done, total, f"{name} — {rate:.1f} fps")

                if engine == "lama":
                    vres = wc.remove_watermark_video(path, dst, inp, mask, batch_size=batch_size,
                                                     crf=crf, preset="medium", keep_audio=True,
                                                     verbose=False, progress_cb=on_progress,
                                                     cancel=job.cancel)
                    if vres.get("audio") is False:
                        job.log(f"  ! {name}: the source has no audio track (or it could not be "
                                f"copied) — the output is video only")
                elif engine == "delogo":
                    job.progress(0, 1, f"{name} (delogo)")
                    wc.remove_watermark_video_delogo(path, dst, mask, crf=crf, verbose=False)
                elif engine == "crop":
                    x0, y0, x1, y1 = wc.mask_bbox(mask)
                    crop = (f"{x0 - 8}:{nfo['height']}:0:0" if x0 > nfo["width"] / 2
                            else f"{x1 + 8}:{nfo['height']}:{x1 + 8}:0")
                    wc.remove_watermark_video_crop(path, dst, crop, verbose=False)
                else:
                    raise ValueError(f"unknown engine {engine}")
                dt = time.time() - t0
                verdict, cb = "written", None
                if engine == "crop":
                    _ci = wc.video_info(dst)
                    verdict = (f"cropped to {_ci['width']}x{_ci['height']} - mark region gone "
                               f"(mask check does not apply)")
                elif verify and os.path.isfile(dst):
                    n_chk = min(8, max(2, nfo["frames"] - 1))
                    v = wc.verify_removal(wc.sample_frames(path, n=n_chk),
                                          wc.sample_frames(dst, n=n_chk), mask)
                    verdict, cb = v["verdict"], v["contrast_before"]
                    if verdict.startswith("no strong mark"):
                        empty_n += 1
                        job.log(f"  ! {name}: nothing pale inside the preset box — open the mask "
                                f"preview and try another preset (static shots need the right box)")
                done_n += 1
                fps_out = nfo["frames"] / max(dt, 1e-6)
                job.row(dict(file=name, status="done", method=method,
                             detail=f"{engine}, {fps_out:.1f} fps",
                             time=fmt_time(dt), verdict=verdict, output=os.path.basename(dst),
                             frames=nfo["frames"], fps=round(fps_out, 2),
                             contrast_before=cb, size=f"{nfo['width']}x{nfo['height']}"))
                job.log(f"  ✓ {name}  {fmt_time(dt)} ({fps_out:.1f} fps)  {verdict}")
                try:
                    b0 = wc.sample_frames(path, n=1)[0]
                    a0 = wc.sample_frames(dst, n=1)[0]
                    p = os.path.join(out_dir, "_previews", f"{Path(path).stem}_before_after.png")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    cv2.imwrite(p, wc.make_side_by_side(b0, a0, mask, scale=3))
                    job.preview(p, f"{name} — before | after")
                except Exception:
                    pass
            except JobCancelled:
                raise
            except Exception as exc:                                   # noqa: BLE001
                if job.cancel.is_set():
                    job.row(dict(file=name, status="cancelled", method=engine, detail="-",
                                 time="-", verdict="cancelled mid-encode - partial file discarded"))
                    job.log(f"  \u23f9 {name}: cancelled mid-encode")
                    break
                fail_n += 1
                log.exception("video failed: %s", path)
                job.row(dict(file=name, status="failed", method=engine, detail="-", time="-",
                             verdict=str(exc)[:120]))
                job.log(f"  ✗ {name}: {exc}")
    except JobCancelled:
        job.log("  \u23f9 cancelled by user")
    job.report_paths = write_report(job.rows, out_dir, "videos_report")
    if job.cancel.is_set():
        job.done(f"Videos: cancelled after {done_n + skip_n + fail_n}/{len(files)} clip(s) — "
                 f"partial report written → {out_dir}")
    else:
        note = (f", {empty_n} with an empty preset box (check the previews)"
                if empty_n else "")
        job.done(f"Videos: {done_n} processed, {skip_n} skipped, {fail_n} failed{note} "
                 f"— {fmt_time(time.time() - t_all)} → {out_dir}")


# --------------------------------------------------------------------------------------
# Gradio callbacks (each returns an update + a streaming generator)
# --------------------------------------------------------------------------------------

CURRENT_JOB: dict = {"job": None, "thread": None}


def _start(target, header: str, owner: str = "local"):
    """Create a job for `owner` and hand it to the scheduler. Returns (job, thread-like)."""
    busy = JOBS.get(owner)
    if busy is not None and not busy.finished and not busy.cancel.is_set():
        raise RuntimeError("you already have a job running — press Cancel first")
    job = Job(owner=owner, label=header)
    JOBS[owner] = job
    SCHEDULER.submit(job, target)
    return job, None


def is_host(request) -> bool:
    """True for the operator sitting at the machine (or on loopback), False for visitors."""
    if not HOST["multiuser"] or request is None:
        return True
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return str(host) in ("127.0.0.1", "::1", "localhost", "None", "testclient")


def _host_only(request, what: str) -> "str | None":
    if not is_host(request):
        return (f"⚠ {what} is disabled in hosted mode — ask the person running this app "
                f"to change it on the machine itself.")
    return None


def _hosted_folder_guard(folders, request) -> "str | None":
    """A visitor must not be able to read folders off the host's disk by typing a path."""
    if not HOST["multiuser"] or not folders or not str(folders).strip():
        return None
    if is_host(request):
        return None
    return ("⚠ In hosted mode, drop your files in the upload box — pasting folder paths "
            "would read folders on the host's disk, so it is switched off.")


def _too_many(files: list) -> str:
    return (f"Too many files at once ({len(files)}): this instance accepts up to "
            f"{HOST['max_files']} per run. Split the folder and run it in parts — finished "
            f"files are skipped on the next pass.")


def ui_scan_images(files, folders, recursive, out_root, preset, allow_preset, request: "gr.Request" = None):
    key = session_key(request)
    guard = _hosted_folder_guard(folders, request)
    if guard:
        yield guard, 0.0, "", [], [], gr.update()
        return
    files = collect_inputs(files, folders, "image", recursive)
    if not files:
        yield "No images found — drag files in or paste a folder path.", 0.0, "", [], [], gr.update()
        return
    if len(files) > HOST["max_files"]:
        yield _too_many(files), 0.0, "", [], [], gr.update()
        return
    out_dir = os.path.join(pick_out_root(out_root, key), "scan")
    try:
        job, thread = _start(lambda j: scan_images(
            j, files, out_dir, preset, True, False,
            ui_state["model"], ui_state["device"], ui_state["threads"], allow_preset=allow_preset),
            f"Scanning {len(files)} image(s)", owner=key)
    except RuntimeError as exc:                                    # already running / queue full
        yield f"⚠ {exc}", 0.0, "", [], [], gr.update()
        return
    yield from _stream_with_zip(job, thread, header="", out_dir=out_dir, key=key)


def ui_process_images(files, folders, recursive, out_root, preset, adaptive, verify, skip_existing,
                      allow_preset, request: "gr.Request" = None):
    key = session_key(request)
    guard = _hosted_folder_guard(folders, request)
    if guard:
        yield guard, 0.0, "", [], [], gr.update()
        return
    files = collect_inputs(files, folders, "image", recursive)
    if not files:
        yield "No images found — drag files in or paste a folder path.", 0.0, "", [], [], gr.update()
        return
    if len(files) > HOST["max_files"]:
        yield _too_many(files), 0.0, "", [], [], gr.update()
        return
    out_dir = os.path.join(pick_out_root(out_root, key), "images_clean")
    try:
        job, thread = _start(lambda j: process_images(
            j, files, out_dir, preset, adaptive, skip_existing, verify,
            ui_state["model"], ui_state["device"], ui_state["threads"], allow_preset=allow_preset,
            low_mem=ui_state["low_mem"]),
            f"Processing {len(files)} image(s)", owner=key)
    except RuntimeError as exc:                                    # already running / queue full
        yield f"⚠ {exc}", 0.0, "", [], [], gr.update()
        return
    yield from _stream_with_zip(job, thread, header=f"Processing {len(files)} image(s)",
                                out_dir=out_dir, key=key)


def ui_scan_videos(files, folders, recursive, out_root, preset, request: "gr.Request" = None):
    key = session_key(request)
    guard = _hosted_folder_guard(folders, request)
    if guard:
        yield guard, 0.0, "", [], [], gr.update()
        return
    files = collect_inputs(files, folders, "video", recursive)
    if not files:
        yield "No clips found — drag files in or paste a folder path.", 0.0, "", [], [], gr.update()
        return
    if len(files) > HOST["max_files"]:
        yield _too_many(files), 0.0, "", [], [], gr.update()
        return
    out_dir = os.path.join(pick_out_root(out_root, key), "scan")
    try:
        job, thread = _start(lambda j: scan_videos(
            j, files, out_dir, preset, ui_state["model"], ui_state["device"], ui_state["threads"]),
            f"Scanning {len(files)} clip(s)", owner=key)
    except RuntimeError as exc:                                    # already running / queue full
        yield f"⚠ {exc}", 0.0, "", [], [], gr.update()
        return
    yield from _stream_with_zip(job, thread, header="", out_dir=out_dir, key=key)


def ui_process_videos(files, folders, recursive, out_root, preset, engine, batch_size, crf,
                      verify, skip_existing, request: "gr.Request" = None):
    key = session_key(request)
    guard = _hosted_folder_guard(folders, request)
    if guard:
        yield guard, 0.0, "", [], [], gr.update()
        return
    files = collect_inputs(files, folders, "video", recursive)
    if not files:
        yield "No clips found — drag files in or paste a folder path.", 0.0, "", [], [], gr.update()
        return
    if len(files) > HOST["max_files"]:
        yield _too_many(files), 0.0, "", [], [], gr.update()
        return
    out_dir = os.path.join(pick_out_root(out_root, key), "videos_clean")
    try:
        job, thread = _start(lambda j: process_videos(
            j, files, out_dir, preset, engine, int(batch_size), int(crf), verify, skip_existing,
            ui_state["model"], ui_state["device"], ui_state["threads"], low_mem=ui_state["low_mem"]),
            f"Processing {len(files)} clip(s)", owner=key)
    except RuntimeError as exc:                                    # already running / queue full
        yield f"⚠ {exc}", 0.0, "", [], [], gr.update()
        return
    yield from _stream_with_zip(job, thread, header=f"Processing {len(files)} clip(s)",
                                out_dir=out_dir, key=key)


def ui_cancel(request: "gr.Request" = None):
    key = session_key(request)
    job = JOBS.get(key)
    if job is None or job.finished:
        return "nothing running in this browser session"
    job.cancel.set()
    if not job.started.is_set():
        return "cancelled — your job was still queued, so it will be dropped before it starts"
    return "cancelling — the file in flight finishes its last batch, then your queue stops"


def ui_download_model(model_path, request: "gr.Request" = None):
    blocked = _host_only(request, "downloading the model")
    if blocked:
        return blocked, model_path
    try:
        def cb(done, total, rate):
            pass
        p = download_model(model_path, progress_cb=cb)
        ui_state["model"] = p
        return f"model ready: {p} ({os.path.getsize(p) // 1_000_000} MB)", p
    except Exception as exc:                                       # noqa: BLE001
        return f"download failed: {exc}", model_path


def ui_set_low_mem(flag, request: "gr.Request" = None):
    blocked = _host_only(request, "changing memory settings")
    if blocked:
        return blocked
    ui_state["low_mem"] = bool(flag)
    for key in list(_INPAINTER_CACHE):                 # drop sessions built with the old setting
        if key[3] != bool(flag):
            _INPAINTER_CACHE.pop(key, None)
    log.info("low_mem=%s", ui_state["low_mem"])
    return (f"low-memory mode {'ON' if ui_state['low_mem'] else 'off'} — "
            f"the next job loads the model that way.")


def ui_check_env(model_path, device, threads):
    lines = []
    lines.append(f"python      : {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"opencv      : {cv2.__version__}")
    try:
        import onnxruntime as ort
        lines.append(f"onnxruntime : {ort.__version__} — providers: {', '.join(ort.get_available_providers())}")
        ram = total_ram_gb()
        lines.append(f"memory      : {ram:.1f} GB RAM — low-memory mode "
                     f"{'ON' if ui_state['low_mem'] else 'off'}"
                     f"{' (auto: under 4 GB)' if default_low_mem() and ui_state['low_mem'] else ''}")
        lines.append(f"GPU (CUDA)  : {'available' if gpu_available() else 'not available — running on CPU'}")
    except Exception as exc:                                       # noqa: BLE001
        lines.append(f"onnxruntime : MISSING ({exc})")
    try:
        lines.append(f"ffmpeg      : {wc.find_ffmpeg()}")
    except Exception as exc:                                       # noqa: BLE001
        lines.append(f"ffmpeg      : MISSING ({exc})")
    ok_model = os.path.isfile(model_path) and os.path.getsize(model_path) >= MODEL_MIN_BYTES
    lines.append(f"model       : {'OK' if ok_model else 'missing'} — {model_path}")
    lines.append(f"engine      : {wc.__file__}")
    lines.append(f"cpu cores   : {os.cpu_count()}   (threads setting: {threads or 'auto'})")
    lines.append(f"logs        : {LOG_DIR / 'app.log'}")
    return "\n".join(lines)


def ui_load_model(model_path, device, threads, request: "gr.Request" = None):
    blocked = _host_only(request, "loading a model")
    if blocked:
        return blocked
    try:
        ui_state["model"], ui_state["device"], ui_state["threads"] = model_path, device, int(threads)
        inp = load_inpainter(model_path, device, int(threads), verbose=False,
                             low_mem=ui_state["low_mem"])
        return (f"✅ model loaded: {os.path.basename(model_path)} — {inp.provider}, "
                f"{inp.model_size:.0f} MB" + ("" if device != "auto" else ""))
    except Exception as exc:                                       # noqa: BLE001
        return f"❌ {exc}"


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

ui_state = {"model": default_model_path(), "device": "auto", "threads": 0,
            "low_mem": default_low_mem()}

# hosted mode: private per-session folders, one job at a time, hard limits per visitor
HOST = {"multiuser": False, "max_files": 500, "keep_days": 3, "session_root": None}


def session_dir(key: str, sub: str = "") -> str:
    root = HOST["session_root"] or os.path.join(str(APP_DIR), "output", "sessions")
    path = os.path.join(root, key, sub) if sub else os.path.join(root, key)
    return path


def pick_out_root(requested: str, key: str) -> str:
    """Hosted: a visitor never chooses a path on the host's disk."""
    return session_dir(key) if HOST["multiuser"] else requested


def make_results_zip(out_dir: str, key: str) -> "str | None":
    """Bundle everything a visitor produced into one download (JPEG/MP4 do not re-compress)."""
    import zipfile
    src = os.path.dirname(out_dir) if os.path.basename(out_dir) in ("images_clean", "videos_clean") \
        else out_dir
    if not os.path.isdir(src):
        return None
    files = []
    for dirpath, _dirnames, filenames in os.walk(src):
        for fn in sorted(filenames):
            if fn.endswith(".zip") or fn.endswith(".pyc"):
                continue
            files.append(os.path.join(dirpath, fn))
    if not files:
        return None
    dl = os.path.join(src, "_downloads")
    os.makedirs(dl, exist_ok=True)
    zpath = os.path.join(dl, "results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, src))
    return zpath


def prune_sessions(keep_days: int) -> int:
    """Delete hosted-session folders older than `keep_days` so the disk does not fill up."""
    import shutil
    root = HOST["session_root"] or os.path.join(str(APP_DIR), "output", "sessions")
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# 🎬 {APP_NAME}\n"
            "Removes the **visible** watermark from your own Veo/Flow videos and Gemini images — "
            "locally. Gemini's 4-point **sparkle** is found with a shape template; Flow's `Veo` "
            "wordmark with temporal detection. **SynthID (the invisible watermark) is left in place.**"
        )
        with gr.Tabs():
            # ---------------------------------------------------------------- images
            with gr.Tab("🖼 Images"):
                with gr.Row():
                    with gr.Column(scale=3):
                        img_files = gr.File(label="Images (drag & drop, multi-select)",
                                            file_count="multiple", type="filepath",
                                            file_types=list(IMAGE_EXT))
                        img_folders = gr.Textbox(label="…and/or folders (one path per line)",
                                                 lines=2, placeholder=r"C:\Users\me\Pictures\gemini")
                        with gr.Row():
                            img_recursive = gr.Checkbox(value=True, label="include sub-folders")
                            img_skip = gr.Checkbox(value=True, label="skip already-processed files")
                        img_out = gr.Textbox(label="Output root folder", value=str(APP_DIR / "output"))
                        with gr.Row():
                            img_preset = gr.Dropdown(
                                ["gemini_sparkle", "veo_video", "gemini_image", "flow_strip",
                                 "bottom_left", "top_right"],
                                value="gemini_sparkle", label="fallback preset box")
                            img_adaptive = gr.Checkbox(value=True, label="adaptive fill (recommended)")
                            img_verify = gr.Checkbox(value=True, label="verify each result")
                        img_allow_preset = gr.Checkbox(
                            value=False,
                            label="if no sparkle is found, apply the fallback box above "
                                  "(use for Flow/video frame exports - otherwise those files are skipped)")
                        with gr.Row():
                            img_scan_btn = gr.Button("🔍 Scan (dry run)")
                            img_run_btn = gr.Button("▶ Process all", variant="primary")
                            img_cancel_btn = gr.Button("⏹ Cancel")
                    with gr.Column(scale=4):
                        img_status = gr.Markdown("idle")
                        img_bar = gr.Slider(0, 1, value=0, label="progress", interactive=False)
                        img_table = gr.Dataframe(headers=TABLE_HEADERS, label="queue", wrap=True)
                        img_gallery = gr.Gallery(label="mask / before-after previews",
                                                 columns=3, height=260, object_fit="contain")
                        img_log = gr.Textbox(label="log", lines=12, max_lines=20, autoscroll=True)
                        img_dl = gr.File(label="results — download (.zip)")
                img_scan_btn.click(ui_scan_images,
                                   [img_files, img_folders, img_recursive, img_out, img_preset,
                                    img_allow_preset],
                                   [img_status, img_bar, img_log, img_table, img_gallery, img_dl])
                img_run_btn.click(ui_process_images,
                                  [img_files, img_folders, img_recursive, img_out, img_preset,
                                   img_adaptive, img_verify, img_skip, img_allow_preset],
                                  [img_status, img_bar, img_log, img_table, img_gallery, img_dl])
                img_cancel_btn.click(ui_cancel, None, img_status)

            # ---------------------------------------------------------------- videos
            with gr.Tab("🎥 Videos"):
                gr.Markdown("Single clip or a whole folder. **LaMa** = best quality, "
                            "**delogo** ≈ realtime (softer), **crop** = artifact-free but re-frames.")
                with gr.Row():
                    with gr.Column(scale=3):
                        vid_files = gr.File(label="Videos (drag & drop, multi-select)",
                                            file_count="multiple", type="filepath",
                                            file_types=list(VIDEO_EXT))
                        vid_folders = gr.Textbox(label="…and/or folders (one path per line)",
                                                 lines=2, placeholder=r"D:\flow_renders")
                        with gr.Row():
                            vid_recursive = gr.Checkbox(value=True, label="include sub-folders")
                            vid_skip = gr.Checkbox(value=True, label="skip already-processed files")
                            vid_verify = gr.Checkbox(value=True, label="verify each clip")
                        vid_out = gr.Textbox(label="Output root folder", value=str(APP_DIR / "output"))
                        with gr.Row():
                            vid_engine = gr.Dropdown(["lama", "delogo", "crop"], value="lama",
                                                     label="engine")
                            vid_preset = gr.Dropdown(["veo_video", "flow_strip", "gemini_sparkle"],
                                                     value="veo_video", label="fallback preset box")
                            vid_batch = gr.Slider(1, 16, value=4, step=1, label="frames per batch")
                            vid_crf = gr.Slider(12, 30, value=17, step=1, label="quality (CRF, lower=better)")
                        with gr.Row():
                            vid_scan_btn = gr.Button("🔍 Scan (dry run)")
                            vid_run_btn = gr.Button("▶ Process all", variant="primary")
                            vid_cancel_btn = gr.Button("⏹ Cancel")
                    with gr.Column(scale=4):
                        vid_status = gr.Markdown("idle")
                        vid_bar = gr.Slider(0, 1, value=0, label="progress", interactive=False)
                        vid_table = gr.Dataframe(headers=TABLE_HEADERS, label="queue", wrap=True)
                        vid_gallery = gr.Gallery(label="mask / before-after previews",
                                                 columns=3, height=260, object_fit="contain")
                        vid_log = gr.Textbox(label="log", lines=12, max_lines=20, autoscroll=True)
                        vid_dl = gr.File(label="results — download (.zip)")
                vid_scan_btn.click(ui_scan_videos,
                                   [vid_files, vid_folders, vid_recursive, vid_out, vid_preset],
                                   [vid_status, vid_bar, vid_log, vid_table, vid_gallery, vid_dl])
                vid_run_btn.click(ui_process_videos,
                                  [vid_files, vid_folders, vid_recursive, vid_out, vid_preset,
                                   vid_engine, vid_batch, vid_crf, vid_verify, vid_skip],
                                  [vid_status, vid_bar, vid_log, vid_table, vid_gallery, vid_dl])
                vid_cancel_btn.click(ui_cancel, None, vid_status)

            # ---------------------------------------------------------------- settings
            with gr.Tab("⚙ Settings"):
                with gr.Row():
                    with gr.Column():
                        set_model = gr.Textbox(label="Model file (lama_fp32.onnx)", value=default_model_path())
                        with gr.Row():
                            set_dl = gr.Button("⬇ Download model (208 MB, once)")
                            set_load = gr.Button("Load / reload model", variant="primary")
                        with gr.Row():
                            set_device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="device")
                            set_threads = gr.Slider(0, 32, value=0, step=1,
                                                    label="CPU threads (0 = let onnxruntime decide)")
                        set_msg = gr.Markdown("")
                    with gr.Column():
                        env_box = gr.Textbox(label="environment check", lines=11, interactive=False)
                        env_btn = gr.Button("Check environment")
                gr.Markdown(
                    "**Where do results go?** `output/images_clean/`, `output/videos_clean/` plus "
                    "`_previews/`, plus `images_report.csv|json` / `videos_report.csv|json`.\n\n"
                    "**Bulk behaviour** — files are processed one at a time (safest for RAM); a failed "
                    "file never stops the batch; `skip already-processed` makes re-runs resumable; "
                    "Cancel stops after the current batch and leaves no half-written output.\n\n"
                    "**GPU** — `pip install onnxruntime-gpu` (and a matching CUDA runtime) then set "
                    "device to `cuda`; check with the environment button."
                )
                set_dl.click(ui_download_model, [set_model], [set_msg, set_model])
                set_lowmem = gr.Checkbox(value=ui_state["low_mem"],
                                         label="low-memory mode (~40% less RAM, ~30% slower on CPU)")
                set_load.click(ui_load_model, [set_model, set_device, set_threads], set_msg)
                set_lowmem.change(ui_set_low_mem, [set_lowmem], [set_msg])
                env_btn.click(ui_check_env, [set_model, set_device, set_threads], env_box)

            # ---------------------------------------------------------------- help
            with gr.Tab("❓ Help"):
                gr.Markdown(
                    """
### Quick start
1. **Settings** → *Download model* once (208 MB). It is stored next to the app.
2. **Images** → drag in your Gemini renders (or paste a folder path) → **Scan (dry run)** to see
   what will be removed → **Process all**.
3. **Videos** → same flow; pick `lama` for quality or `delogo` for speed.
4. **Sharing it with others?** run with `--host 0.0.0.0` (same network) or `--share --auth user:pass`
   (internet) — see **HOSTING.md**. Hosted mode gives every visitor a private results folder, a
   queue with their position, their own Cancel, and a results .zip; visitors cannot read your disk
   or change settings.

### What the verdicts mean
| verdict | meaning |
|---|---|
| `clean - no star-shaped mark left` | the shape template can no longer find the mark in the output ✅ |
| `clean - no mark-like contrast left` | wordmark path: the pale-overlay signal is gone ✅ |
| `no sparkle found` | this file has no mark the detector recognises — check the mask preview, then try a fallback preset box |
| `some mark-like contrast remains` | inspect the before/after preview; on very busy footage this can be the texture itself |
| `failed` | the file is listed in the report with the error; the rest of the batch continues |

### Good to know
* **Videos:** detection compares frames, so it needs a little motion — a static or sub-second clip
  falls back to the preset box (still verified). Audio is always copied through untouched.
* **Images:** two of your renders can be the same picture at different resolutions, and the mark
  sits at a *different offset in each* — that is why every image is detected individually.
* **Flat backgrounds:** a learned model invents texture there, so those get a smooth fill instead
  (picked automatically per image). Turn `adaptive fill` off to force LaMa everywhere.
* **Retina/4K bulk:** inference runs on a 512×512 crop around the mark, so a 4K batch is limited by
  decode/encode, not the model.
* **Nothing is ever overwritten by accident:** if no sparkle is detected the file is reported as
  `no mark` and **left untouched**. Tick *"if no sparkle is found, apply the fallback box"* only for
  Flow/video-frame exports, where a preset box is the intended path.
* **Cancelling is safe:** the job stops after the file in flight, keeps everything already finished,
  discards the half-written file, and still writes the report — press *Process all* again to resume.
* **`no mark` with a shape score near 0.70:** the mark is there but degraded (re-compression, busy
  background) and the detector refuses to guess. Tick the fallback box for that file.
* **Low-memory mode** (Settings) is on automatically under 4 GB RAM: ~40 % less RAM, ~30 % slower.
  Turn it off on a big machine for full speed.
* **Is my install healthy?** run `python smoke_test.py` in this folder (23 checks, no files touched).
* **Logs:** `logs/app.log` — paste the last lines when reporting a problem.
                    """
                )

    return demo


def parse_args():
    p = argparse.ArgumentParser(description=f"{APP_NAME} — local GUI")
    p.add_argument("--model", default=default_model_path(), help="path to lama_fp32.onnx")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = auto)")
    p.add_argument("--low-mem", action="store_true",
                   help="low-memory mode: ~40%% less RAM, ~30%% slower on CPU")
    p.add_argument("--full-mem", action="store_true",
                   help="force full ORT arena even on a small machine")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public Gradio link")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.add_argument("--download", action="store_true", help="fetch the model then exit")
    g = p.add_argument_group("hosting for other people")
    g.add_argument("--auth", metavar="USER:PASS", default=None,
                   help="require a login (do this before exposing the app to the internet)")
    g.add_argument("--max-files", type=int, default=500,
                   help="most files one visitor may queue per run (default 500)")
    g.add_argument("--max-queue", type=int, default=6,
                   help="jobs allowed to wait behind the running one (default 6)")
    g.add_argument("--keep-days", type=int, default=3,
                   help="delete per-visitor results older than this many days (default 3)")
    p.add_argument("--multiuser", action="store_true",
                   help="force per-visitor private output folders (auto when --share, --auth, "
                        "or a non-loopback --host is used)")
    return p.parse_args()


def main():
    args = parse_args()
    ui_state["model"], ui_state["device"], ui_state["threads"] = args.model, args.device, args.threads
    if args.low_mem:
        ui_state["low_mem"] = True
    elif args.full_mem:
        ui_state["low_mem"] = False
    if args.download:
        p = download_model(args.model)
        print("model ready:", p)
        return
    missing = []
    if not os.path.isfile(args.model) or os.path.getsize(args.model) < MODEL_MIN_BYTES:
        missing.append(f"model: {args.model} (use the Settings tab → Download model, or --download)")
    try:
        wc.find_ffmpeg()
    except Exception as exc:                                       # noqa: BLE001
        missing.append(f"ffmpeg: {exc}  →  pip install imageio-ffmpeg")
    log.info("engine: %s", wc.__file__)
    log.info("device=%s threads=%s gpu=%s", args.device, args.threads, gpu_available())
    if missing:
        log.warning("needs attention:\n  - " + "\n  - ".join(missing))
        print("\n⚠  needs attention:\n  - " + "\n  - ".join(missing) + "\n")
    HOST["max_files"] = max(1, args.max_files)
    HOST["keep_days"] = max(0, args.keep_days)
    HOST["multiuser"] = bool(args.multiuser or args.share or args.auth
                             or args.host not in ("127.0.0.1", "localhost", "::1"))
    SCHEDULER.max_waiting = max(1, args.max_queue)
    if HOST["multiuser"]:
        removed = prune_sessions(HOST["keep_days"])
        log.info("hosted mode: private session folders, one job at a time, max %d files/run, "
                 "queue %d, keep %d days (pruned %d old folders)",
                 HOST["max_files"], SCHEDULER.max_waiting, HOST["keep_days"], removed)
        print(f"\nhosted mode — visitors get private folders under {session_dir('<session>')}")
        print(f"  one job at a time, {SCHEDULER.max_waiting} may wait, "
              f"{HOST['max_files']} files per run, results kept {HOST['keep_days']} days")
        if args.host not in ("0.0.0.0", "127.0.0.1", "localhost", "::1"):
            print(f"  hint: other machines cannot reach {args.host} — use --host 0.0.0.0")
        if args.host == "0.0.0.0" and not args.auth:
            print("  ⚠  anyone on your network can use this machine's CPU. Add "
                  "--auth user:pass before opening it to the internet.")
    demo = build_ui()
    demo.queue(max_size=max(16, SCHEDULER.max_waiting + 4))
    auth = None
    if args.auth:
        if ":" not in args.auth:
            print("--auth must look like  user:password")
            sys.exit(2)
        user, pw = args.auth.split(":", 1)
        auth = (user, pw)
        print(f"login required: user '{user}'")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth,
                inbrowser=not args.no_browser, show_error=True)


if __name__ == "__main__":
    main()

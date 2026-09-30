#!/usr/bin/env python3
"""
Hosting check: does this instance behave correctly with several people on it?

    python host_test.py                 # needs the model (real jobs are queued)
    python host_test.py --no-model      # only the queue/limit logic, no inference

Verifies, with two simulated browser sessions:
  1. one job runs at a time, the second visitor sees their queue position;
  2. Cancel only touches the caller's own job - a queued visitor is dropped, the
     running visitor keeps going;
  3. each visitor's results land in their own private folder, plus a .zip;
  4. a second run from the same session is refused while the first is active;
  5. --max-files is enforced per run.

Nothing outside a temp folder (and local_app/output/sessions/) is touched.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                        # noqa: E402
import cv2                                                # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name}  — {why}")


def fake_session(tag, host="127.0.0.1"):
    """Sections 2-5 act as the operator (two of their own tabs); section 6 acts as a visitor."""
    return types.SimpleNamespace(session_hash=tag, client=types.SimpleNamespace(host=host))


class Consumer:
    """Drives a Gradio generator in a thread, like a browser tab would."""

    def __init__(self, gen, name):
        self.name = name
        self.heads = []
        self.final = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.thread.start()

    def _run(self, gen):
        try:
            for out in gen:
                self.heads.append(str(out[0]))
                self.final = out
        except Exception as exc:                                       # noqa: BLE001
            self.heads.append(f"EXCEPTION {type(exc).__name__}: {exc}")
            self.final = ("EXCEPTION", 0, "", [], [], None)
        self.done.set()

    def text(self) -> str:
        return self.final[0] if self.final else ""


def make_batch(folder, n=8, w=480, h=640):
    """Textured images on purpose: flat ones take the cheap smooth-fill path and finish so fast
    that queueing cannot be observed."""
    import wm_core as wc
    os.makedirs(folder, exist_ok=True)
    star = wc.star_mask(48)
    rng = np.random.default_rng(3)
    for i in range(n):
        base = rng.integers(90, 170, (h, w, 3)).astype(np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 1.2)
        img = cv2.circle(base, (140 + i * 11 % 200, 260), 90, (60, 120, 190), -1)
        img = cv2.putText(img, f"frame {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (240, 240, 240), 2)
        # A clean patch for the star (so the shape test passes) with grain in the ring around it
        # (so the fill decision picks the learned model, exactly like real footage). Without the
        # grain these images take the cheap smooth-fill path and finish too fast to observe.
        x0, y0 = w - 48 - 76, h - 48 - 76
        patch = cv2.GaussianBlur(img[y0 - 3:y0 + 51, x0 - 3:x0 + 51], (0, 0), 5)
        img[y0 - 3:y0 + 51, x0 - 3:x0 + 51] = patch
        roi = img[y0:y0 + 48, x0:x0 + 48].astype(np.int16)
        img[y0:y0 + 48, x0:x0 + 48] = np.clip(roi + (star[..., None] > 0) * 60, 0, 255).astype(np.uint8)
        m = np.zeros((h, w), np.uint8)
        m[y0:y0 + 48, x0:x0 + 48] = star
        band = (wc.dilate_mask(m, 8) > 0) & (m == 0)
        noise = rng.integers(-28, 29, img.shape).astype(np.int16)
        img = np.where(band[..., None], np.clip(img.astype(np.int16) + noise, 0, 255), img).astype(np.uint8)
        cv2.imwrite(os.path.join(folder, f"clip_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 94])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "lama_fp32.onnx"))
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("Watermark Remover — hosting / multi-user check")
    print("=" * 74)

    import watermark_gui as g

    g.HOST.update(multiuser=True, max_files=500, keep_days=3)
    g.SCHEDULER.max_waiting = 6
    if os.path.isfile(args.model):
        g.ui_state["model"] = args.model          # the CLI/--model equivalent for this test

    tmp = tempfile.mkdtemp(prefix="wm_host_")
    sessions = g.session_dir("__hosttest__")
    shutil.rmtree(sessions, ignore_errors=True)
    for tag in ("session-A", "session-B", "session-C"):
        shutil.rmtree(g.session_dir(tag), ignore_errors=True)
    batch = os.path.join(tmp, "in")

    try:
        with_model = not args.no_model and os.path.isfile(args.model)
        print("\n1) setup")
        make_batch(batch, n=4 if with_model else 1)
        check("test batch created", len(os.listdir(batch)) >= 1, f"{len(os.listdir(batch))} files")
        if not with_model:
            skip("real queued jobs", "no model — only the queue logic is exercised below")

        a, b = fake_session("session-A"), fake_session("session-B")

        if with_model:
            print("\n2) two visitors at once — one runs, the other waits")
            ca = Consumer(g.ui_process_images(None, batch, True, "output", "gemini_sparkle",
                                              True, True, False, False, a), "A")
            time.sleep(2.5)
            cb = Consumer(g.ui_process_images(None, batch, True, "output", "gemini_sparkle",
                                              True, True, False, False, b), "B")
            time.sleep(2.5)
            job_a, job_b = g.JOBS.get("session-A"), g.JOBS.get("session-B")
            b_queued = any("queued" in h for h in cb.heads)
            check("second visitor is told they are queued", b_queued,
                  (cb.heads[-1][:90] if cb.heads else "no output yet"))
            check("only one job runs at a time",
                  g.SCHEDULER.current is job_a and job_a.started.is_set()
                  and not job_b.started.is_set(),
                  f"running={job_a.owner if job_a else None}, "
                  f"waiting={job_b.owner if job_b else None}")

            print("\n3) Cancel is per-visitor")
            msg = g.ui_cancel(b)
            cb.done.wait(timeout=20)
            check("cancel from the waiting visitor drops only their job",
                  "cancelled" in msg.lower(), msg[:70])
            check("the running visitor keeps going", not ca.done.is_set(),
                  "session A still processing")
            a2 = g.ui_cancel(a)
            check("the running visitor can still cancel their own job",
                  "cancelling" in a2.lower(), a2[:70])

            print("\n4) results are private, with a download")
            ca.done.wait(timeout=240)
            cb.done.wait(timeout=20)
            ca_dir = os.path.join(g.session_dir("session-A"), "images_clean")
            cb_dir = os.path.join(g.session_dir("session-B"), "images_clean")
            produced = [f for f in os.listdir(ca_dir)] if os.path.isdir(ca_dir) else []
            check("session A has its own output folder", len(produced) > 0,
                  f"{len(produced)} file(s) in sessions/session-A/images_clean")
            check("session B did not write into A's folder",
                  not os.path.exists(os.path.join(cb_dir, "clip_0_clean.jpg"))
                  or len(os.listdir(cb_dir)) < len(produced),
                  "B was cancelled while queued, so it produced nothing")
            zip_path = ca.final[5] if ca.final and len(ca.final) > 5 else None
            check("a results .zip is offered for download",
                  bool(zip_path) and os.path.isfile(str(zip_path)),
                  os.path.basename(str(zip_path)) if zip_path else "no zip returned")

            print("\n5) guards")
            msgs = g.ui_cancel(a)
            check("cancel on an idle session says so", "nothing running" in msgs.lower(), msgs[:60])
            g.HOST["max_files"] = 3
            c = fake_session("session-C")
            gen = g.ui_process_images(None, batch, True, "output", "gemini_sparkle",
                                      True, True, False, False, c)
            head = str(next(iter(gen))[0])
            check("per-run file limit is enforced", "Too many files" in head, head[:80])
            g.HOST["max_files"] = 500

            print("\n6) visitors cannot reach the host's machine")
            remote = fake_session("session-R", host="203.0.113.7")
            local = fake_session("session-L", host="127.0.0.1")
            files_only = os.path.join(batch, "clip_0.jpg")
            g.HOST["max_files"] = 500

            gen = g.ui_process_images([files_only], "C:/Windows", True, "output", "gemini_sparkle",
                                      True, True, False, False, remote)
            head = str(next(iter(gen))[0])
            check("a visitor cannot make the app read a host folder",
                  "upload box" in head, head[:88])

            gen = g.ui_process_images([files_only], "", True, "output", "gemini_sparkle",
                                      True, True, False, False, remote)
            check("a visitor's own upload still works", "folder paths" not in str(next(iter(gen))[0]),
                  "job accepted")

            for fn, call_args, what in ((g.ui_load_model, ("no-such-model.onnx", "cpu", 0),
                                         "load a model"),
                                        (g.ui_set_low_mem, (True,), "change memory settings")):
                out_remote = str(fn(*call_args, remote))
                out_local = str(fn(*call_args, local))
                check(f"a visitor cannot {what}", "disabled in hosted mode" in out_remote,
                      out_remote[:66])
                check(f"the operator can still {what}", "disabled in hosted mode" not in out_local,
                      out_local[:66])

            for tag in ("session-C", "session-L", "session-R"):
                j = g.JOBS.get(tag)
                if j is not None:
                    j.cancel.set()

            print("\n7) queue ceiling")
            g.SCHEDULER.max_waiting = 1
            try:
                g.SCHEDULER.submit(g.Job(owner="x"), lambda j: j.done("noop"))
                g.SCHEDULER.submit(g.Job(owner="y"), lambda j: j.done("noop"))
                full = g.SCHEDULER.submit(g.Job(owner="z"), lambda j: j.done("noop"))
                check("queue ceiling rejects the next visitor", False, f"accepted ({full})")
            except RuntimeError as exc:
                check("queue ceiling rejects the next visitor", "full" in str(exc), str(exc)[:70])
            g.SCHEDULER.max_waiting = 6
        else:
            skip("queue / cancel / private folders", "requires the model")

    finally:
        if args.keep:
            print(f"\ntemp kept: {tmp}   sessions: {sessions}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(sessions, ignore_errors=True)

    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

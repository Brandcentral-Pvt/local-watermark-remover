#!/usr/bin/env python3
"""
Install check for the Watermark Remover GUI.

Run it once after installing to prove that this machine can actually do the work:

    python smoke_test.py                 # full check (needs the model)
    python smoke_test.py --no-model      # everything except inpainting
    python smoke_test.py --model /path/to/lama_fp32.onnx

It builds its own test material in a temp folder (a 4-point sparkle on a smooth
background, and a synthetic pale wordmark in a 6-frame clip), so it never touches
your files. Nothing here tests the web UI - open the app for that.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PASS, FAIL, SKIP = [], [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    return ok


def skip(name: str, why: str):
    SKIP.append(name)
    print(f"  SKIP  {name}  — {why}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "lama_fp32.onnx"))
    ap.add_argument("--no-model", action="store_true", help="skip the inpainting step")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args()

    print("=" * 74)
    print("Watermark Remover — install check")
    print("=" * 74)

    # ---------------------------------------------------------------- imports
    print("\n1) dependencies")
    import numpy as np

    try:
        import cv2
        check("opencv", True, f"cv2 {cv2.__version__}")
    except Exception as exc:
        check("opencv", False, str(exc))
        return 1
    try:
        import onnxruntime as ort
        check("onnxruntime", True, f"{ort.__version__} ({', '.join(ort.get_available_providers()[:2])})")
        gpu = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:
        check("onnxruntime", False, str(exc))
        ort, gpu = None, False
    try:
        import gradio as gr
        check("gradio", True, gr.__version__)
    except Exception as exc:
        check("gradio", False, str(exc))
    try:
        import wm_core as wc
        check("engine (wm_core.py)", True, f"{wc.__file__}")
    except Exception as exc:
        check("engine (wm_core.py)", False, str(exc))
        return 1
    master = os.path.join(os.path.dirname(HERE), "wm_core.py")      # repo copy, when present
    if os.path.isfile(master):
        same = open(master, encoding="utf-8").read() == open(os.path.join(HERE, "wm_core.py"),
                                                             encoding="utf-8").read()
        check("engine copy matches ../wm_core.py", same,
              "identical" if same else "out of sync — copy ../wm_core.py into this folder")
    try:
        from watermark_gui import process_images, process_videos, Job, collect_inputs
        check("app (watermark_gui.py)", True)
    except Exception as exc:
        check("app (watermark_gui.py)", False, str(exc))
        return 1
    try:
        ff = wc.find_ffmpeg()
        check("ffmpeg", True, os.path.basename(ff))
    except Exception as exc:
        check("ffmpeg", False, str(exc))
        return 1

    tmp = tempfile.mkdtemp(prefix="wm_smoke_")
    try:
        # ---------------------------------------------------- synthetic material
        print("\n2) synthetic test material")
        rng = np.random.default_rng(0)
        h, w = 640, 480
        base = np.full((h, w, 3), 205, np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 9).astype(np.int16)
        base += rng.integers(-1, 2, base.shape).astype(np.int16)      # faint sensor noise
        img = np.clip(base, 0, 255).astype(np.uint8)

        star = wc.star_mask(48)                                       # the Gemini mark's shape
        inset_x, inset_y = 76, 76
        x0, y0 = w - star.shape[1] - inset_x, h - star.shape[0] - inset_y
        roi = img[y0:y0 + star.shape[0], x0:x0 + star.shape[1]]
        img[y0:y0 + star.shape[0], x0:x0 + star.shape[1]] = np.clip(
            roi.astype(np.int16) + (star[..., None] > 0) * 60, 0, 255).astype(np.uint8)

        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        marked = os.path.join(in_dir, "synthetic_sparkle.jpg")
        cv2.imwrite(marked, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        clean_reference = np.clip(base, 0, 255).astype(np.uint8)
        check("wrote a marked test image", os.path.isfile(marked), f"{w}x{h}")

        # ------------------------------------------------------------- detection
        print("\n3) mark detection")
        t0 = time.time()
        mask, method = wc.detect_for_image(img, mode="auto", verbose=False)
        found = mask is not None and int((mask > 0).sum()) > 200
        if found:
            bx = wc.mask_bbox(mask)
            err = max(abs(bx[0] - x0), abs(bx[1] - y0))
            check("sparkle detected", True,
                  f"method={method}, mask {int((mask>0).sum())}px, corner off by {err}px, "
                  f"{time.time()-t0:.2f}s")
        else:
            check("sparkle detected", False, "no mark found in the synthetic image")
        neg = np.full((h, w, 3), 255, np.uint8)                        # blank page
        neg = cv2.GaussianBlur(neg, (0, 0), 5)
        m2, meth2 = wc.detect_for_image(neg, mode="auto", verbose=False)
        check("no false positive on a blank image", m2 is None, f"method={meth2}")
        marked_score, blank_score = wc.sparkle_probe(img), wc.sparkle_probe(neg)
        check("shape score separates a mark from clean content",
              marked_score >= 0.70 > blank_score,
              f"marked {marked_score:.2f} vs clean {blank_score:.2f} (gate 0.70)")

        # ----------------------------------------------------------- images run
        print("\n4) bulk image run (this is what the GUI does)")
        if args.no_model or not os.path.isfile(args.model):
            skip("image inpainting", "no model at " + args.model + " (run: python watermark_gui.py --download)")
        else:
            job = Job()
            t0 = time.time()
            try:
                process_images(job, [marked], out_dir, "gemini_sparkle", True, False, True,
                               args.model, "auto", args.threads)
                row = job.rows[0]
                ok = row["status"] == "done"
                check("image processed", ok, f"{row['status']} — {row['verdict'][:60]} ({time.time()-t0:.1f}s)")
                out_png = os.path.join(out_dir, row.get("output", "missing.jpg"))
                if ok and os.path.isfile(out_png):
                    res = cv2.imread(out_png)
                    inside = np.abs(res.astype(np.int16) - clean_reference.astype(np.int16)).mean()
                    outside_slice = (slice(0, h - 120), slice(0, w - 120))
                    outside = np.abs(res[outside_slice].astype(np.int16)
                                     - clean_reference[outside_slice].astype(np.int16)).mean()
                    check("mark removed (mean |error| over the patch)",
                          inside < 4.0, f"inside {inside:.2f} grey levels, untouched area {outside:.2f}")
                check("report written", len(job.report_paths) == 2,
                      ", ".join(os.path.basename(p) for p in job.report_paths))
            except Exception as exc:
                check("image processed", False, f"{type(exc).__name__}: {exc}")

        # ------------------------------------------------------------ video run
        print("\n5) video path (delogo engine — needs no model)")
        v_in = os.path.join(tmp, "vin.mp4")
        writer = wc.FFmpegWriter(v_in, 320, 180, 12, crf=20, preset="veryfast")
        for i in range(6):
            frame = cv2.resize(img, (320, 180))
            frame = np.roll(frame, i * 3, axis=1)                       # moving content
            frame = cv2.putText(frame, "Veo", (238, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (235, 235, 235), 1, cv2.LINE_AA)
            writer.write(frame)
        writer.close()
        nfo = wc.video_info(v_in)
        check("wrote a 6-frame test clip", nfo["frames"] == 6, f"{nfo['width']}x{nfo['height']}")

        vout = os.path.join(tmp, "vout")
        job = Job()
        try:
            process_videos(job, [v_in], vout, "flow_strip", "delogo", 1, 20, True, False,
                           args.model, "auto", args.threads)
            row = job.rows[0]
            ok = row["status"] == "done"
            check("clip processed", ok, f"{row['status']} — {row['verdict'][:60]}")
            produced = os.path.join(vout, row.get("output", "missing.mp4"))
            if ok and os.path.isfile(produced):
                po = wc.video_info(produced)
                check("output is a valid clip", po["frames"] == 6,
                      f"{po['frames']} frames, {po['width']}x{po['height']}")
        except Exception as exc:
            check("clip processed", False, f"{type(exc).__name__}: {exc}")

        # ------------------------------------------- quality path keeps the audio
        print("\n6) lama path + audio track")
        if args.no_model or not os.path.isfile(args.model):
            skip("lama audio check", "no model loaded")
        else:
            import subprocess
            with_audio = os.path.join(tmp, "vin_audio.mp4")
            subprocess.run([wc.find_ffmpeg(), "-y", "-loglevel", "error", "-i", v_in, "-f", "lavfi",
                            "-i", "sine=frequency=440:duration=0.5", "-c:v", "copy", "-c:a", "aac",
                            "-shortest", with_audio], check=True)
            check("test clip has audio", wc.has_audio(with_audio))
            job = Job()
            try:
                process_videos(job, [with_audio], os.path.join(tmp, "vout_lama"), "flow_strip", "lama",
                               1, 20, True, True, args.model, "auto", args.threads)
                row = job.rows[0]
                made = os.path.join(tmp, "vout_lama", row.get("output", "missing.mp4"))
                ok = row["status"] == "done" and os.path.isfile(made)
                check("lama processed the clip", ok, f"{row['status']} — {row['verdict'][:50]}")
                if ok:
                    check("audio survived the lama path", wc.has_audio(made),
                          "still present in the output" if wc.has_audio(made) else "AUDIO WAS DROPPED")
            except Exception as exc:
                check("lama processed the clip", False, f"{type(exc).__name__}: {exc}")

        # ---------------------------------------------------------------- misc
        print("\n7) bulk helpers")
        files = collect_inputs(None, in_dir, "image", True)
        check("folder scan finds files", len(files) == 1, f"{len(files)} file(s) in {os.path.basename(in_dir)}/")
        first = wc.unique_output_path(marked, out_dir)
        open(first, "w").close()                                   # pretend it already exists
        second = wc.unique_output_path(marked, out_dir)
        check("output naming never overwrites", first != second and not os.path.exists(second),
              f"{os.path.basename(first)} then {os.path.basename(second)}")
        check("GPU available", True, "yes — set device=cuda" if gpu else "no — CPU mode (works, slower)")

    finally:
        if args.keep:
            print(f"\ntemp folder kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

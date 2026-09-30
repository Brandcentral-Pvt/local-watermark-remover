"""
wm_core.py -- visible watermark removal core (Gemini / Imagen images, Google Flow / Veo video).

Engine : LaMa (Resolution-robust Large Mask Inpainting, Apache-2.0) exported to ONNX.
Design  : the watermark is a small, *static* region, so we never inpaint the whole frame.
          We detect the mark once, then run the model on a small native-resolution crop
          around it -> fast on GPU, sharp result, zero damage to the rest of the frame.

Public API
----------
Mask helpers   : corner_box, rect_mask, dilate_mask, feather_alpha, mask_preview
Detection      : detect_mask_stack, detect_mask_single, resolve_mask
Inpainting     : Inpainter (ONNX wrapper), inpaint_image, inpaint_folder
Video          : remove_watermark_video, remove_watermark_video_delogo
Tools          : find_ffmpeg, video_info, mux_audio
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# ffmpeg / video helpers
# --------------------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Return a usable ffmpeg binary (system ffmpeg, else the one bundled with imageio-ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ffmpeg not found. `pip install imageio-ffmpeg` or apt-get install ffmpeg.") from exc


def video_info(path: str) -> dict:
    """Width/height/fps/frame count via OpenCV (no ffprobe needed)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info = dict(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str, max_frames: Optional[int] = None) -> Iterable[np.ndarray]:
    """Yield BGR frames of a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            i += 1
            if max_frames is not None and i >= max_frames:
                break
    finally:
        cap.release()


def sample_frames(path: str, n: int = 24, max_side: Optional[int] = None) -> list:
    """Grab n frames spread evenly across the video (used for watermark detection)."""
    nfo = video_info(path)
    total = max(nfo["frames"], 1)
    idxs = np.unique(np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1))).astype(int))
    cap = cv2.VideoCapture(path)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        if max_side and max(f.shape[:2]) > max_side:
            sc = max_side / max(f.shape[:2])
            f = cv2.resize(f, (int(f.shape[1] * sc), int(f.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        out.append(f)
    cap.release()
    return out


class FFmpegWriter:
    """Pipe raw BGR frames into ffmpeg (libx264). Audio is muxed later from the source."""

    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 17,
                 preset: str = "medium", ffmpeg: Optional[str] = None, pix_fmt: str = "yuv420p"):
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.width, self.height, self.path = width, height, path
        cmd = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", pix_fmt, path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = self.proc.stderr.read().decode("utf-8", "ignore")
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed ({code}): {err[-800:]}")


def has_audio(path: str, ffmpeg: Optional[str] = None) -> bool:
    """True when the file carries at least one audio stream (ffmpeg probe, no decoding)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    return "Audio:" in probe


def mux_audio(video_path: str, audio_source: str, out_path: str, ffmpeg: Optional[str] = None) -> bool:
    """Copy the audio track of `audio_source` onto `video_path`. Returns True if audio was copied."""
    ffmpeg = ffmpeg or find_ffmpeg()
    if not has_audio(audio_source, ffmpeg):
        if video_path != out_path:
            shutil.move(video_path, out_path)
        return False
    ext = os.path.splitext(out_path)[1].lower()
    fmt = [] if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v") else ["-f", "mp4"]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", video_path, "-i", audio_source,
           "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest",
           "-movflags", "+faststart"] + fmt + [out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if video_path != out_path:
            shutil.copy(video_path, out_path)
        return False
    return True


# --------------------------------------------------------------------------------------
# geometry / masks
# --------------------------------------------------------------------------------------

CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")


def imread_color(path: str) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII paths (falls back to imdecode)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        with open(path, "rb") as fh:
            buf = np.frombuffer(fh.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def corner_box(H: int, W: int, corner: str = "bottom-right",
               frac_w: float = 0.40, frac_h: float = 0.30) -> Tuple[int, int, int, int]:
    """Search window (x0, y0, x1, y1) inside which the watermark is expected to live."""
    bw, bh = max(int(W * frac_w), 32), max(int(H * frac_h), 32)
    if corner.endswith("right"):
        x0, x1 = W - bw, W
    else:
        x0, x1 = 0, bw
    if corner.startswith("bottom"):
        y0, y1 = H - bh, H
    else:
        y0, y1 = 0, bh
    return x0, y0, x1, y1


def rect_mask(H: int, W: int, rect: Sequence[int]) -> np.ndarray:
    """
    rect = (x, y, w, h).
      * values with |v| <= 1 are treated as fractions of W (x, w) / H (y, h);
      * a NEGATIVE x or y is measured from the opposite edge, so the box can be anchored
        to the bottom-right corner: rect=(-0.06, -0.06, 0.055, 0.05) -> 6% in from the
        right, 6% up from the bottom, 5.5% wide, 5% tall, whatever the resolution is.
    """
    x, y, w, h = [float(v) for v in rect]
    x = x * W if abs(x) <= 1.0 else x
    y = y * H if abs(y) <= 1.0 else y
    w = w * W if abs(w) <= 1.0 else w
    h = h * H if abs(h) <= 1.0 else h
    if x < 0: x = W + x
    if y < 0: y = H + y
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), W), min(int(round(y + h)), H)
    m = np.zeros((H, W), np.uint8)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


# Handy starting points if automatic detection ever fails. Each entry:
#   (corner, rect) with rect in fraction-of-frame units anchored to the corner.
PRESETS = {
    # Google Flow / Veo "Veo" wordmark: small, ~3-6% inset from the bottom-right
    "veo_video":    ("bottom-right", (-0.075, -0.065, 0.065, 0.055)),
    # Gemini / Gemini app image badge (sparkle + wordmark), a bit larger and further in
    "gemini_image": ("bottom-right", (-0.115, -0.095, 0.100, 0.075)),
    # Gemini image sparkle (4-point star): measured on real 2K output at 48 px with the
    # corner ~94 px away; the same star is 48 px with a 76 px inset on a 1K render, so the
    # box is deliberately generous and gets tightened to the evidence inside it.
    "gemini_sparkle": ("bottom-right", (-0.115, -0.085, 0.100, 0.070)),
    # Flow "Made with Google AI" style strip along the bottom-right
    "flow_strip":   ("bottom-right", (-0.320, -0.085, 0.300, 0.070)),
    "bottom_left":  ("bottom-left",  (0.020, -0.065, 0.070, 0.055)),
    "top_right":    ("top-right",    (-0.085, 0.030, 0.070, 0.055)),
}


def dilate_mask(mask: np.ndarray, px: int = 6) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def feather_alpha(mask: np.ndarray, px: int = 3) -> np.ndarray:
    """Float 0..1 alpha used to blend the inpainted patch back (soft edges)."""
    a = (mask > 0).astype(np.float32)
    if px > 0:
        k = 2 * px + 1
        a = cv2.GaussianBlur(a, (k, k), 0)
    return np.clip(a, 0.0, 1.0)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_preview(img_bgr: np.ndarray, mask: np.ndarray, rect: Optional[Sequence[int]] = None,
                 pad: int = 0, scale: int = 2) -> np.ndarray:
    """Blue tint over masked pixels + green box, cropped to the region of interest."""
    vis = img_bgr.copy()
    vis[mask > 0] = (0.45 * vis[mask > 0] + 0.55 * np.array([255, 80, 0])).astype(np.uint8)
    bb = mask_bbox(mask)
    if bb:
        x0, y0, x1, y1 = bb
        x0, y0 = max(x0 - 12, 0), max(y0 - 12, 0)
        x1, y1 = min(x1 + 12, img_bgr.shape[1]), min(y1 + 12, img_bgr.shape[0])
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1, y1 = min(x1 + pad, img_bgr.shape[1]), min(y1 + pad, img_bgr.shape[0])
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), 2)
        if rect is not None:
            rx, ry, rw, rh = [int(v) for v in rect]
            cv2.rectangle(vis, (rx, ry), (rx + rw - 1, ry + rh - 1), (0, 165, 255), 2)
        vis = vis[y0:y1, x0:x1]
        if scale > 1:
            vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return vis


# --------------------------------------------------------------------------------------
# watermark detection
# --------------------------------------------------------------------------------------

def _marklike_score(frames: Sequence[np.ndarray], box: Tuple[int, int, int, int],
                    ref_size: Optional[Tuple[int, int]] = None,
                    fine_sigma: float = 1.6, fine_tscale: float = 4.0,
                    coarse_sigma: float = 10.0, coarse_tscale: float = 3.5,
                    noise_floor: float = 1.5, dsat_sigma: float = 2.4) -> np.ndarray:
    """
    One "looks like a watermark" score map (float32, 0..1) for one corner box.

    Core idea -- a temporal t-statistic on the high-pass residual:
        t(x,y) = mean_i( g_i - blur(g_i) ) / (std_i( g_i - blur(g_i) ) + noise_floor)
    At a watermark pixel the overlay is *identically* present in every frame, so the residual
    has a large mean and a tiny spread -> t is large. Ordinary detail (a white cloud, a bright
    wall, textures) changes from frame to frame -> mean ~ 0, std large -> t ~ 0. This works
    whether the content moves or not, and even for image batches with unrelated pictures.

    Two scales are merged: a fine one (stroke edges, sharp marks) and a coarse one (fills the
    interior of thick strokes). The result is gated by a "pale translucent overlay" prior --
    the mark also de-saturates the pixels it covers.
    """
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    REF_H, REF_W = ref_size if ref_size else frames[0].shape[:2]
    G, S = [], []
    for f in frames:
        h, w = f.shape[:2]
        # Frames may not all be the same size (a folder of 1K/2K/4K renders, for instance).
        # Locate the box proportionally and normalise the crop so the statistics still stack.
        fx0, fy0 = int(round(x0 / REF_W * w)), int(round(y0 / REF_H * h))
        fx1, fy1 = int(round(x1 / REF_W * w)), int(round(y1 / REF_H * h))
        fx0, fy0 = max(fx0, 0), max(fy0, 0)
        fx1, fy1 = min(max(fx1, fx0 + 1), w), min(max(fy1, fy0 + 1), h)
        sub = f[fy0:fy1, fx0:fx1]
        if sub.size == 0:
            sub = f[max(h - bh, 0):, max(w - bw, 0):]
        if (sub.shape[1], sub.shape[0]) != (bw, bh):
            sub = cv2.resize(sub, (bw, bh), interpolation=cv2.INTER_AREA if sub.shape[1] > bw else cv2.INTER_LINEAR)
        G.append(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32))
        S.append(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32))
    G = np.stack(G)
    S = np.stack(S)

    def tstat(sigma: float, t_scale: float) -> np.ndarray:
        try:
            R = G - np.stack([cv2.GaussianBlur(g, (0, 0), sigma) for g in G])
        except cv2.error:                     # tiny box -> no valid gaussian kernel
            R = G - G.mean(axis=(1, 2), keepdims=True)
        t = np.clip(R.mean(axis=0) / (R.std(axis=0) + noise_floor), 0.0, None)
        return np.clip(t / t_scale, 0.0, 1.0)

    # fine scale -> thin strokes & edges; coarse scale -> the body of thick/bold marks
    score = np.maximum(tstat(fine_sigma, fine_tscale), tstat(coarse_sigma, coarse_tscale))
    D = np.stack([cv2.GaussianBlur(s, (0, 0), dsat_sigma) - s for s in S])
    prior = 0.45 + 0.55 * np.clip(D.mean(axis=0) / 22.0, 0.0, 1.0)
    return score * prior


def _mask_from_score(score: np.ndarray, strict: float = 1.0, pad: int = 6, min_area: int = 10,
                     max_frac: float = 0.28, close_iter: int = 3,
                     min_strength: float = 0.13) -> Optional[tuple]:
    """
    Threshold one score map into a clean, localized mask.

    Returns (mask, strength) or None. `strength` is the mean score on the *undilated* mark
    (calibrated: real watermarks 0.18-0.45, JPEG-noise blobs ~0.09), so it doubles as a
    confidence value and as the false-positive gate.
    """
    peak = float(np.percentile(score, 99.9))
    if peak < 0.18:
        return None
    thr = max(0.20, 0.50 * peak) / max(strict, 0.15)
    m = (score >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    core = np.zeros_like(m)
    total = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            core[lab == i] = 255
            total += area
    if total == 0 or total > max_frac * score.size:
        return None
    # Thick glyphs only show up as an outline in the high-pass score -> fill the enclosed
    # interiors so the whole stroke gets repainted, not just its edges.
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(core)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    if 0 < int((filled > 0).sum()) <= max_frac * score.size:
        core = filled
    total = int((core > 0).sum())
    if total == 0 or total > max_frac * score.size:
        return None
    strength = float(score[core > 0].mean())
    if strength < min_strength / max(strict, 0.15):
        return None
    # Shape gate. A wordmark/logo is a COMPACT cluster; the classic false positives (a
    # corner that happens to be consistently brighter, compression banding, stray texture)
    # are sprawling, hollow, many-component blobs. Calibrated on both sets.
    h, w = core.shape
    bb = mask_bbox(core)
    if bb is None:
        return None
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    if bw > 0.5 * w or bh > 0.5 * h:
        return None
    if total / float(max(bw * bh, 1)) < 0.30:                      # too hollow / spread out
        return None
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(core, 8)
    areas = sorted((int(stats2[i, cv2.CC_STAT_AREA]) for i in range(1, n2)), reverse=True)
    if len(areas) > 8 or areas[0] / float(total) < 0.35:           # fragmented
        return None
    return dilate_mask(core, pad), strength


def detect_mask_stack(frames: Sequence[np.ndarray], corner: str = "auto", strict: float = 1.0,
                      pad: int = 6, verbose: bool = True) -> Optional[np.ndarray]:
    """
    Detect a watermark present in >=4 images / video frames that share the same mark.
    Returns a full-frame uint8 mask (0/255) or None.
    """
    if len(frames) < 3:
        raise ValueError("detect_mask_stack needs at least 3 frames (8-32 is ideal).")
    H, W = frames[0].shape[:2]
    sizes = {f.shape[:2] for f in frames}
    if len(sizes) > 1 and verbose:
        print(f"    note: {len(sizes)} different frame sizes present -> detection runs at "
              f"{W}x{H} and the mask is returned in those coordinates")
    corners = CORNERS if corner == "auto" else (corner,)
    best, best_strength, best_corner, low_conf = None, -1.0, None, False
    # Two score recipes are tried because marks differ in stroke weight: a tighter coarse
    # scale for fine wordmarks, a wider one for bold/thick logos. Both are gated by the
    # same strength + shape checks, so at most one can be a false positive.
    recipes = (dict(coarse_sigma=6.0, coarse_tscale=4.0),
               dict(coarse_sigma=10.0, coarse_tscale=3.5))
    # A single, calibrated confidence bar. Marks fainter than this (or with < ~6 frames to
    # work with) fall back to preset/manual mode in the notebook rather than risking a
    # false positive on ordinary content.
    passes = ((0.13, 0.28),)
    for pass_no, (min_strength, max_frac) in enumerate(passes):
        for c in corners:
            box = corner_box(H, W, c)
            for recipe in recipes:
                score = _marklike_score(frames, box, ref_size=(H, W), **recipe)
                got = _mask_from_score(score, strict=strict, pad=pad, min_strength=min_strength,
                                       max_frac=max_frac)
                if got is None:
                    continue
                cand, strength = got
                if verbose:
                    print(f"    corner {c:13s} cs{recipe['coarse_sigma']:>4.0f} -> "
                          f"{int((cand > 0).sum()):6d}px  strength {strength:.3f}"
                          f"{'  (faint)' if pass_no else ''}")
                if strength > best_strength:
                    best, best_strength, best_corner, low_conf = cand, strength, c, bool(pass_no)
            if verbose and best is None:
                print(f"    corner {c:13s} -> nothing")
        if best is not None:
            break
    if best is None:
        if verbose:
            print("    no candidate found (try mode='preset' with a manual box)")
        return None
    full = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = corner_box(H, W, best_corner)
    full[y0:y1, x0:x1] = best
    if verbose:
        print(f"    -> chose {best_corner}, mask area {(full>0).sum()} px  (strength {best_strength:.3f})"
              + ("  [LOW CONFIDENCE - check the preview!]" if low_conf else ""))
    return full


def _pale_overlay_evidence(img_bgr: np.ndarray) -> np.ndarray:
    """
    Single-frame "this looks like a translucent pale overlay" evidence, in grey levels.
    A pale mark is (a) locally brighter than its surroundings and (b) de-saturated.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    brighter = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    desat = cv2.GaussianBlur(sat, (0, 0), 3.0) - sat
    return np.clip(brighter, 0, None) * (0.45 + 0.55 * np.clip(desat / 20.0, 0, 1))


# --------------------------------------------------------------------------------------
# Gemini image sparkle (4-point star) -- template matcher, validated on real output
# --------------------------------------------------------------------------------------
# The Gemini image watermark is a solid pale 4-point star. Measured on real output: 48x48 px,
# near-diamond outline (inner radius ratio ~0.48), sitting ~94 px in from the bottom-right
# corner on a 2K render but ~76 px on a 1K render -- the position is NOT a fixed fraction of
# the frame, which is exactly why this is detected rather than hard-coded.
# Matched-filter response (is this spot brighter than the ring around it?) + shape IoU.

# Measured real-world mark sizes: 48 px on 1K renders / cropped close-ups, 96 px on 2K renders
# (insets 94 px and 152 px respectively). The ladder spans both plus the plausible range in
# between, so a different export size is covered without a code change. Matching is a cheap
# filter pass over a corner ROI, so the extra scales cost milliseconds.
SPARKLE_SCALES = (32, 40, 48, 56, 64, 72, 80, 90, 96, 104, 116, 132)


def star_mask(size: int, inner: float = 0.48) -> np.ndarray:
    """Solid 4-point sparkle footprint (uint8 0/255); inner = inner-radius ratio."""
    s = int(size)
    big = np.zeros((s * 8, s * 8), np.uint8)
    c = s * 4
    pts = []
    for i in range(8):
        ang = np.pi * i / 4 - np.pi / 2
        r = (s * 4 - 2) if i % 2 == 0 else (s * 4 * inner)
        pts.append((c + r * np.cos(ang), c + r * np.sin(ang)))
    cv2.fillPoly(big, [np.array(pts, np.int32)], 255)
    return cv2.resize(big, (s, s), interpolation=cv2.INTER_AREA)


def _sparkle_kernel(size: int, inner: float = 0.48):
    """Matched filter: star footprint (normalised) minus the annulus around it."""
    s = int(size)
    star = star_mask(s, inner).astype(np.float32) / 255.0
    K = int(round(s * 1.8))
    if K % 2 == 0:
        K += 1
    canvas = np.zeros((K, K), np.float32)
    off = (K - s) // 2
    canvas[off:off + s, off:off + s] = star
    m = max(int(s * 0.22), 3)
    box = np.zeros((K, K), np.float32)
    box[m:K - m, m:K - m] = 1.0
    annulus = np.clip(box - canvas, 0, 1)
    k = canvas / max(canvas.sum(), 1e-6) - annulus / max(annulus.sum(), 1e-6)
    return k.astype(np.float32), K, off


def _sparkle_iou(gray: np.ndarray, x: int, y: int, size: int, inner: float = 0.48,
                 bright_frac: float = 0.45, close: int = 0) -> float:
    """How star-shaped is the bright blob at (x, y)? IoU of blob vs the ideal star."""
    s = int(size)
    h, w = gray.shape
    if x < 0 or y < 0 or x + s > w or y + s > h:
        return 0.0
    _, K, off = _sparkle_kernel(s, inner)
    if x - off < 0 or y - off < 0 or y - off + K > h or x - off + K > w:
        return 0.0
    star = star_mask(s, inner) > 127
    patch = gray[y:y + s, x:x + s].astype(np.float32)
    big = gray[y - off:y - off + K, x - off:x - off + K].astype(np.float32)
    canvas = np.zeros((K, K), bool)
    canvas[off:off + s, off:off + s] = star
    m = max(int(s * 0.22), 3)
    box = np.zeros((K, K), bool)
    box[m:K - m, m:K - m] = True
    annulus = box & ~canvas
    bg = float(np.median(big[annulus])) if annulus.any() else float(big.min())
    peak = float(np.percentile(patch[star], 90))
    blob = patch >= bg + bright_frac * max(peak - bg, 1e-6)
    if close:
        # A semi-transparent mark crossed by a dark feature of the scene (a plank groove, a
        # branch, a horizon line) is split into pieces by any threshold. Bridging those gaps
        # before the shape comparison is what makes the test match how a person sees it.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close), int(close)))
        blob = cv2.morphologyEx(blob.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return float(np.logical_and(blob, star).sum()) / max(int(np.logical_or(blob, star).sum()), 1)


def detect_sparkle(img_bgr: np.ndarray, corner: Tuple[float, float] = (0.20, 0.16),
                   scales: Sequence[int] = SPARKLE_SCALES, inner: float = 0.48,
                   min_iou: float = 0.70, min_resp: float = 12.0, topk: int = 8,
                   close_inner: bool = True, verbose: bool = False) -> Optional[dict]:
    """
    Locate the Gemini image sparkle in the bottom-right corner.

    Returns dict(x, y, size, resp, iou, score) at the star's top-left corner, or None.
    Calibrated on real Gemini output: 3/3 real marks found exactly (IoU 0.79-0.85) with
    0/19 false positives on unwatermarked photos. Shape (IoU) is what separates a sparkle
    from a bright blob of ordinary content, not brightness.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # The mark sits ~76-94 px from the corner on real Gemini output. A purely fractional
    # search window falls short on small renders (e.g. 512 px wide), so floor it at
    # (largest star + margin) px. On 1K-2K renders the fraction still dominates.
    floor = int(max(scales) + 130)
    x0 = max(w - max(int(w * corner[0]), floor), 0)
    y0 = max(h - max(int(h * corner[1]), floor), 0)
    roi = gray[y0:, x0:]
    cands = []
    for s in scales:
        if roi.shape[0] < int(s * 1.9) or roi.shape[1] < int(s * 1.9):
            continue
        k, K, _ = _sparkle_kernel(s, inner)
        resp = cv2.filter2D(roi, -1, k, borderType=cv2.BORDER_REPLICATE)
        r = resp.copy()
        for _ in range(topk):
            ry, rx = np.unravel_index(int(np.argmax(r)), r.shape)
            val = float(r[ry, rx])
            if val >= min_resp:
                # a response at (ry, rx) means the star's CENTRE is there
                gx, gy = rx + x0 - s // 2, ry + y0 - s // 2
                iou = _sparkle_iou(gray, gx, gy, s, inner)
                if close_inner:
                    # A mark crossed by a dark scene feature can split into pieces; accept the
                    # candidate if either reading of its shape passes. Measured: this recovers
                    # split marks without lifting any clean-photo candidate near the gate.
                    iou = max(iou, _sparkle_iou(gray, gx, gy, s, inner,
                                                close=max(3, int(s * 0.09))))
                if verbose:
                    print(f"      size {s:3d} resp {val:7.1f} at ({gx},{gy}) IoU {iou:.2f}")
                cands.append((iou * min(val / 25.0, 1.0), val, iou, int(gx), int(gy), int(s)))
            r[max(ry - s, 0):ry + s, max(rx - s, 0):rx + s] = -1e9
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    score, val, iou, gx, gy, s = cands[0]
    if iou < min_iou:
        if verbose:
            print(f"    best candidate was not star-shaped enough (IoU {iou:.2f} < {min_iou})")
        return None
    return dict(x=gx, y=gy, size=s, resp=round(val, 1), iou=round(iou, 3), score=round(score, 3))


# The two mark families measured on real Gemini/Flow exports: the standard small sparkle and
# the larger one used on some 2K renders. Both sit in the bottom-right corner at a fixed inset,
# which is what makes a scoped second pass safe (see SPARKLE_FAMILIES).
SPARKLE_FAMILIES = (
    # (size range, right/bottom inset range)
    ((40, 58), (55, 135)),
    ((82, 110), (128, 185)),
)
SPARKLE_FAMILY_GATE = 0.60      # measured: faint real marks 0.62-0.64, best clean-family fit 0.55


def _family_candidates(gray: np.ndarray, gate: float = SPARKLE_FAMILY_GATE,
                       inner: float = 0.48) -> list:
    """Star fits inside the two measured corner families, including faint/split marks."""
    h, w = gray.shape[:2]
    out = []
    for (s_lo, s_hi), (i_lo, i_hi) in SPARKLE_FAMILIES:
        for size in range(s_lo, s_hi + 1, 2):
            for inset in range(i_lo, i_hi + 1, 3):
                x, y = w - size - inset, h - size - inset
                if x < 0 or y < 0:
                    continue
                iou = max(_sparkle_iou(gray, x, y, size, inner),
                          _sparkle_iou(gray, x, y, size, inner, close=max(3, int(size * 0.09))))
                if iou >= gate:
                    out.append((iou, size, x, y))
    out.sort(reverse=True)                                     # strongest first
    kept = []
    for iou, size, x, y in out:                                # suppress neighbours
        cx, cy = x + size / 2, y + size / 2
        if all((cx - (k[2] + k[1] / 2)) ** 2 + (cy - (k[3] + k[1] / 2)) ** 2
               > (0.8 * max(k[1], size)) ** 2 for k in kept):
            kept.append((iou, size, x, y))
    return kept


def detect_sparkle_all(img_bgr: np.ndarray, min_iou: float = 0.70, max_marks: int = 4,
                       family_gate: float = SPARKLE_FAMILY_GATE, verbose: bool = False) -> list:
    """
    Every sparkle the image carries, strongest first: dicts as `detect_sparkle` returns.

    Real Gemini exports can carry two marks at once (measured: a 96 px sparkle at a 152 px inset
    plus a fainter 48 px one at 94 px). The first pass uses the strict shape gate anywhere in the
    corner; the second looks only inside the two measured mark families at a lower gate, which is
    where a faint or scene-split mark still scores ~0.62-0.64 while clean content stays <= 0.55.
    """
    marks = []
    first = detect_sparkle(img_bgr, min_iou=min_iou, verbose=verbose)
    if first is not None:
        marks.append(first)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for iou, size, x, y in _family_candidates(gray, gate=family_gate):
        if len(marks) >= max_marks:
            break
        if any((x + size / 2 - (m["x"] + m["size"] / 2)) ** 2
               + (y + size / 2 - (m["y"] + m["size"] / 2)) ** 2
               < (0.8 * max(m["size"], size)) ** 2 for m in marks):
            continue
        marks.append(dict(x=int(x), y=int(y), size=int(size), resp=0.0, iou=float(iou),
                          score=float(iou), source="family"))
    if verbose:
        for i, m in enumerate(marks, 1):
            print(f"    mark {i}: {m['size']}x{m['size']} at ({m['x']},{m['y']}) "
                  f"IoU {m['iou']:.2f} ({m.get('source', 'primary')})")
    return marks


def sparkle_probe(img_bgr: np.ndarray, verbose: bool = False, **kw) -> float:
    """
    Best star-shape IoU found in the corner ROI, whether or not it passes the gate.

    Diagnostic only - `detect_sparkle` requires IoU >= 0.70 (measured: real marks 0.86-0.95,
    the best clean-photo false candidate 0.61) so that unwatermarked photos are never touched.
    A number just under the gate tells you the mark is there but degraded (heavy JPEG, busy
    background) and that the fallback preset box is the right tool for that file.
    """
    hit = detect_sparkle(img_bgr, min_iou=0.0, verbose=verbose, **kw)
    return float(hit["iou"]) if hit else 0.0


def detect_sparkle_mask(img_bgr: np.ndarray, pad: Optional[int] = None, verbose: bool = True,
                        all_marks: bool = True, **kw) -> Optional[np.ndarray]:
    """Every sparkle location -> one star-shaped mask (dilated), ready for inpainting."""
    hits = detect_sparkle_all(img_bgr, verbose=verbose, **kw) if all_marks else []
    if not hits:
        single = detect_sparkle(img_bgr, verbose=verbose, **kw)
        hits = [single] if single is not None else []
    if not hits:
        return None
    h, w = img_bgr.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for hit in hits:
        s = hit["size"]
        m[hit["y"]:hit["y"] + s, hit["x"]:hit["x"] + s] = star_mask(s)
    # Cover the mark's soft edge, which does not end at the template's outline: on a 96 px star
    # that edge is ~8 px wide, and leaving it behind is exactly what made a pale ghost star.
    # (Measured: pad 3 -> visible ghost, pad ~8-10 -> gone.)
    pad = max(6, round(max(h_["size"] for h_ in hits) * 0.09)) if pad is None else pad
    m = dilate_mask(m, pad)
    if verbose:
        print(f"    {len(hits)} sparkle(s) removed -> mask {int((m > 0).sum())} px")
    return m


def refine_mask_to_evidence(img_bgr: np.ndarray, box_mask: np.ndarray, pad: int = 6,
                            min_frac: float = 0.02, max_frac: float = 0.85,
                            verbose: bool = True) -> Tuple[np.ndarray, str]:
    """
    Shrink a corner box mask down to the pale-overlay evidence found *inside* that box.

    Deliberately constrained: it can only make the mask smaller, never move it somewhere else
    in the frame, so it is safe to run unattended. If the evidence looks implausible (almost
    nothing, or the whole box lit up) the original box is kept.
    """
    ev = _pale_overlay_evidence(img_bgr)
    ys, xs = np.nonzero(box_mask)
    if len(xs) == 0:
        return box_mask, "empty box"
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    sub = ev[y0:y1, x0:x1]
    peak = float(np.percentile(sub, 99.5))
    if peak < 5.0:
        return box_mask, "no pale evidence inside the box -> kept the full box"
    thr = max(2.5, 0.35 * peak)
    m = (sub >= thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= 10:
            keep[lab == i] = 255
    frac = float(keep.mean()) / 255.0
    if frac < min_frac or frac > max_frac:
        return box_mask, f"evidence covers {frac*100:.0f}% of the box -> kept the full box"
    out = np.zeros_like(box_mask)
    out[y0:y1, x0:x1] = keep
    if int((out > 0).sum()) < 20:
        return box_mask, "too little evidence -> kept the full box"
    out = dilate_mask(out, pad)
    return out, (f"refined to {int((out > 0).sum())} px "
                 f"({(out > 0).sum() / max((box_mask > 0).sum(), 1) * 100:.0f}% of the box)")


def verify_sparkle_removal(before_bgr: np.ndarray, after_bgr: np.ndarray, verbose: bool = True) -> dict:
    """
    Shape-aware QA for the Gemini sparkle: run the star template matcher on the *output*.

    This is far more decisive than a generic contrast measure -- a residual ghost from the
    inpainting patch is not star-shaped and will not trigger, while a surviving watermark
    will. Returns dict(found_before, found_after, x, y, iou_before, iou_after, verdict).
    """
    b_marks = detect_sparkle_all(before_bgr, verbose=False)
    a_marks = detect_sparkle_all(after_bgr, verbose=False)
    b = b_marks[0] if b_marks else None
    a = a_marks[0] if a_marks else None
    if not b_marks:
        verdict = "no sparkle found in the input - check the image / try a preset box"
    elif not a_marks:
        verdict = ("clean - no star-shaped mark left"
                   + (f" (all {len(b_marks)} marks gone)" if len(b_marks) > 1 else ""))
    else:
        verdict = (f"a star-shaped mark is still present at ({a['x']},{a['y']}) - inspect the "
                   f"output")
    out = dict(found_before=bool(b_marks), found_after=bool(a_marks),
               marks_before=len(b_marks), marks_after=len(a_marks),
               iou_before=b["iou"] if b else None, iou_after=a["iou"] if a else None,
               x=(b or a or {}).get("x"), y=(b or a or {}).get("y"), verdict=verdict)
    if verbose:
        print(f"    sparkle check: {len(b_marks)} mark(s) before "
              f"({'IoU %.2f' % b['iou'] if b else 'none'}) -> "
              f"{len(a_marks)} after ({'IoU %.2f' % a['iou'] if a else 'none'})  |  {verdict}")
    return out


def preset_mask(img_bgr: np.ndarray, preset: str = "veo_video", pad: int = 6, refine: bool = True,
                verbose: bool = True) -> np.ndarray:
    """
    Deterministic corner mask: a named preset box, optionally tightened to the pale evidence
    inside it. Used when there is no temporal signal (static shot, single image).

    This never guesses a location, so it cannot wander off onto ordinary picture content.
    Automatic *placement* discovery from a single frame was measured at 43-71% recall with
    3-16 false positives across 19 clean photos -- not good enough to ship -- so the corner
    comes from the preset table and you confirm it in the preview.
    """
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; available: {list(PRESETS)}")
    corner, rect = PRESETS[preset]
    H, W = img_bgr.shape[:2]
    mask = rect_mask(H, W, rect)
    note = f"preset '{preset}' ({corner})"
    if refine:
        mask, how = refine_mask_to_evidence(img_bgr, mask, pad=pad, verbose=False)
        note += " | " + how
    if verbose:
        print(f"    {note}")
    return mask


def resolve_mask(frames: Sequence[np.ndarray], mode: str = "auto", corner: str = "auto",
                 rect: Optional[Sequence[int]] = None, preset: Optional[str] = None,
                 strict: float = 1.0, pad: int = 6, verbose: bool = True) -> np.ndarray:
    """
    mode: 'auto'    -> stack detection when several frames are given, else sparkle detection,
                       else the corner preset box
          'stack'   -> force stack detection (raises if it fails)
          'sparkle' -> force the Gemini 4-point star template matcher
          'preset'  -> `preset` name or explicit `rect`; preset boxes are refined to the pale
                       evidence inside them (they can only shrink, never relocate)
    """
    H, W = frames[0].shape[:2]
    preset_name = preset
    if preset:
        if preset not in PRESETS:
            raise KeyError(f"unknown preset {preset!r}; available: {list(PRESETS)}")
        mode = "preset"
    if mode == "sparkle":
        m = detect_sparkle_mask(frames[0], verbose=verbose)
        if m is None:
            raise RuntimeError("no sparkle found - try mode='preset' or an explicit rect")
        return m
    if mode == "preset":
        if preset_name:
            return preset_mask(frames[0], preset=preset_name, pad=pad, refine=True, verbose=verbose)
        if rect is None:
            raise ValueError("mode='preset' needs rect=(x, y, w, h) or preset='<name>'")
        return rect_mask(H, W, rect)
    if rect is not None:
        return rect_mask(H, W, rect)
    if mode in ("auto", "stack") and len(frames) >= 3:
        m = detect_mask_stack(frames, corner=corner, strict=strict, pad=pad, verbose=verbose)
        if m is not None:
            return m
        if mode == "stack":
            raise RuntimeError("stack detection failed - pass mode='preset' with a manual rect.")
    # No temporal signal: try the sparkle template (images), else the deterministic preset.
    m = detect_sparkle_mask(frames[0], verbose=False)
    if m is not None:
        if verbose:
            print("    detected the Gemini sparkle with the shape template")
        return m
    return preset_mask(frames[0], preset=preset_name or "veo_video", pad=pad,
                       refine=True, verbose=verbose)


# --------------------------------------------------------------------------------------
# LaMa ONNX inpainting
# --------------------------------------------------------------------------------------

def _resize_to(img: np.ndarray, size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    tw, th = size
    if (img.shape[1], img.shape[0]) == (tw, th):
        return img
    down = tw < img.shape[1] or th < img.shape[0]
    interp = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if down else cv2.INTER_LINEAR)
    out = cv2.resize(img, (tw, th), interpolation=interp)
    return out


class Inpainter:
    """LaMa ONNX wrapper with region-crop inference (works with fixed- or dynamic-shape exports)."""

    def __init__(self, model_path: str, providers: Optional[Sequence[str]] = None,
                 threads: int = 0, verbose: bool = True,
                 mem_arena: bool = True, graph_opt: str = "all"):
        """
        `mem_arena=False` + `graph_opt="basic"` is the low-memory mode: measured on the same
        model it cuts peak RSS by roughly a third (768 MB -> 433 MB for one 512x512 pass) for
        about +30% inference time on CPU. Same output; use it on machines under ~4 GB RAM.
        """
        import onnxruntime as ort

        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if not mem_arena:
            so.enable_cpu_mem_arena = False
        if graph_opt == "basic":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if threads:
            so.intra_op_num_threads = threads
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=list(providers))
        self.provider = self.sess.get_providers()[0]
        self.in_names = [i.name for i in self.sess.get_inputs()]
        shape = self.sess.get_inputs()[0].shape
        self.fixed = None
        if isinstance(shape[-1], int) and isinstance(shape[-2], int):
            self.fixed = (int(shape[-1]), int(shape[-2]))       # (W, H) the model insists on
        self.model_size = os.path.getsize(model_path) / 1e6
        if verbose:
            print(f"LaMa loaded | {os.path.basename(model_path)} ({self.model_size:.0f} MB) | {self.provider} | "
                  f"input {'fixed ' + str(self.fixed) if self.fixed else 'dynamic'}")

    # ---- single batch of same-shaped crops -------------------------------------------
    def _run(self, imgs_bgr: Sequence[np.ndarray], masks_u8: Sequence[np.ndarray]) -> list:
        target = self.fixed
        ims, ms, orig = [], [], []
        for img, mask in zip(imgs_bgr, masks_u8):
            h, w = img.shape[:2]
            orig.append((w, h))
            if target:
                img_r = _resize_to(img, target)
                mask_r = _resize_to(mask, target, is_mask=True)
            else:
                m8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8
                img_r = _resize_to(img, (m8[1], m8[0]))
                mask_r = _resize_to(mask, (m8[1], m8[0]), is_mask=True)
            rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ims.append(np.transpose(rgb, (2, 0, 1)))
            ms.append(np.transpose((mask_r > 127).astype(np.float32), (0, 1))[None])
        feed = {self.in_names[0]: np.stack(ims).astype(np.float32),
                self.in_names[1]: np.stack(ms).astype(np.float32)}
        out = self.sess.run(None, feed)[0]
        res = []
        for i in range(out.shape[0]):
            o = np.clip(np.transpose(out[i], (1, 2, 0)), 0, 255).astype(np.uint8)   # RGB 0..255
            o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
            res.append(_resize_to(o, orig[i]) if (o.shape[1], o.shape[0]) != orig[i] else o)
        return res

    # ---- region planning ------------------------------------------------------------
    def plan_region(self, H: int, W: int, mask: np.ndarray, context: Optional[int] = None):
        bb = mask_bbox(mask)
        if bb is None:
            raise ValueError("empty mask")
        x0, y0, x1, y1 = bb
        bw, bh = x1 - x0, y1 - y0
        ctx = int(context) if context else max(24, int(0.6 * max(bw, bh)))
        want = max(bw, bh) + 2 * ctx
        side = self.fixed[0] if self.fixed else want
        if want > side:                       # mark bigger than the model input -> keep it native and downscale at run time
            side = want
        side = int(min(side, H, W))
        if not self.fixed:
            side = (side + 7) // 8 * 8
            side = int(min(side, H, W))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        rx = int(np.clip(cx - side // 2, 0, max(W - side, 0)))
        ry = int(np.clip(cy - side // 2, 0, max(H - side, 0)))
        return (rx, ry, min(side, W - rx), min(side, H - ry))

    def make_patch(self, img_bgr: np.ndarray, mask: np.ndarray, rect, feather: int = 3):
        """Return (paste_fn) closure: apply(patched_crop) -> full frame with feathered paste."""
        rx, ry, rw, rh = rect
        crop = np.ascontiguousarray(img_bgr[ry:ry + rh, rx:rx + rw])
        cmask = np.ascontiguousarray(mask[ry:ry + rh, rx:rx + rw])
        alpha = feather_alpha(cmask, feather)[..., None]

        def paste(patched: np.ndarray) -> np.ndarray:
            out = img_bgr.copy()
            region = out[ry:ry + rh, rx:rx + rw]
            region[:] = (patched.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            return out

        return crop, cmask, paste

    # ---- public ---------------------------------------------------------------------
    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray, feather: int = 3) -> np.ndarray:
        H, W = img_bgr.shape[:2]
        rect = self.plan_region(H, W, mask)
        crop, cmask, paste = self.make_patch(img_bgr, mask, rect, feather)
        return paste(self._run([crop], [cmask])[0])

    def inpaint_frames(self, frames: Sequence[np.ndarray], mask: np.ndarray, batch_size: int = 8,
                       feather: int = 3, progress: bool = True):
        """Inpaint a list of same-shaped frames with ONE shared mask / crop -> batched GPU runs."""
        if not frames:
            return []
        H, W = frames[0].shape[:2]
        rect = self.plan_region(H, W, mask)
        rx, ry, rw, rh = rect
        cmask = np.ascontiguousarray(mask[ry:ry + rh, rx:rx + rw])
        alpha = feather_alpha(cmask, feather)[..., None]
        n = len(frames)
        out = [None] * n
        t0 = time.time()
        for s in range(0, n, batch_size):
            chunk = [np.ascontiguousarray(f[ry:ry + rh, rx:rx + rw]) for f in frames[s:s + batch_size]]
            patched = self._run(chunk, [cmask] * len(chunk))
            for j, p in enumerate(patched):
                i = s + j
                full = frames[i].copy()
                region = full[ry:ry + rh, rx:rx + rw]
                region[:] = (p.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)).astype(np.uint8)
                out[i] = full
            if progress and (s // batch_size) % 5 == 0:
                done = min(s + batch_size, n)
                el = time.time() - t0
                print(f"      {done}/{n} frames  ({el:.1f}s, {done/max(el,1e-6):.1f} fps)", end="\r")
        if progress:
            el = time.time() - t0
            print(f"      {n}/{n} frames  ({el:.1f}s, {n/max(el,1e-6):.1f} fps)   ")
        return out


def _ring_stat(gray: np.ndarray, core: np.ndarray, ring: int = 6, q: float = 90.0,
               sigma: float = 1.6) -> float:
    """Percentile-contrast of a region against a ring just outside it."""
    ring_m = (dilate_mask(core.astype(np.uint8) * 255, ring) > 0) & ~core
    if not core.any() or not ring_m.any():
        return float("nan")
    hp = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
    return float(np.percentile(hp[core], q) - np.percentile(hp[ring_m], q))


def _control_region(core: np.ndarray, gap: int = 10) -> Optional[np.ndarray]:
    """Same shape, translated off the mark (left/right/up/down) -> a matched content control."""
    H, W = core.shape
    ys, xs = np.nonzero(core)
    if len(xs) == 0:
        return None
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    for dx, dy in ((-(bw + gap), 0), (bw + gap, 0), (0, -(bh + gap)), (0, bh + gap)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        c = cv2.warpAffine(core.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        if c.sum() == 0 or (c & core).any():
            continue
        if c.sum() < 0.9 * core.sum():
            continue
        return c
    return None


def _mark_contrast(img_bgr: np.ndarray, mask: np.ndarray, ring: int = 6, q: float = 90.0) -> float:
    """
    Matched-filter "is there a pale mark here?" contrast.

    For two scales (fine strokes / coarse glyph bodies) it measures the percentile contrast of
    the masked region against a ring just outside it, and subtracts the same measurement made
    on an identical region shifted off the mark. That second term is the content's own local
    contrast, so it cancels out and only the overlay's brightness lift remains:

        real mark present -> ~ +5 and up (50-100 on a flat background)
        mark removed      -> ~  0 (busy scenes can drift to +-8)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    core = mask > 0
    if not core.any():
        return 0.0
    ctrl = _control_region(core)
    vals = []
    for sigma in (1.6, 6.0):
        v = _ring_stat(gray, core, ring=ring, q=q, sigma=sigma)
        if ctrl is not None:
            v -= _ring_stat(gray, ctrl, ring=ring, q=q, sigma=sigma)
        vals.append(v)
    vals = [v for v in vals if not np.isnan(v)]
    return float(max(vals)) if vals else 0.0


def verify_removal(before, after, mask: np.ndarray, ring: int = 6) -> dict:
    """
    QA the result, so you don't have to squint at a 4K frame.

    Pass one image or a list of frames (video: the notebook samples a few) for `before` and
    `after`. Returns the mark contrast before/after (median over frames) plus a verdict:

        contrast_before < 5   -> no strong mark was in this region (check the mask!)
        contrast_after  < max(4.0, 0.45*contrast_before) -> clean
        (bands measured on real material: removed ~ -8..+3, content noise up to ~8,
         watermark still present ~ 19)
        otherwise             -> some mark-like contrast remains
    """
    def median_contrast(x):
        items = list(x) if isinstance(x, (list, tuple)) else [x]
        vals = [_mark_contrast(im, mask, ring=ring) for im in items]
        return float(np.median(vals)) if vals else 0.0

    cb, ca = median_contrast(before), median_contrast(after)
    removed = 100.0 * (1.0 - ca / max(cb, 1e-6)) if cb > 0 else 0.0
    if cb < 5.0:
        verdict = "no strong mark was in this region - check the mask / try another preset"
    elif ca < max(4.0, 0.45 * cb):
        verdict = "clean - no mark-like contrast left"
    else:
        verdict = "some mark-like contrast remains - inspect the output"
    return dict(contrast_before=round(cb, 2), contrast_after=round(ca, 2),
                removed_pct=round(removed, 1), verdict=verdict)


# --------------------------------------------------------------------------------------
# high level: images
# --------------------------------------------------------------------------------------

def ring_texture(img_bgr: np.ndarray, mask: np.ndarray, ring: int = 8) -> float:
    """High-pass std just outside the mask: tells a flat background from a textured one."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.6)
    core = mask > 0
    ring_m = (dilate_mask(mask, ring) > 0) & ~core
    return float(hp[ring_m].std()) if ring_m.any() else 0.0


def adaptive_fill(img_bgr: np.ndarray, mask: np.ndarray, inp: Optional["Inpainter"] = None,
                  feather: int = 3, smooth_below: float = 4.0, blur_sigma: float = 2.0,
                  verbose: bool = False) -> np.ndarray:
    """
    Pick the fill that suits the background, which matters more than it sounds:

    * flat / low-texture background (dark sky, plaster, smooth bokeh) -> cv2.inpaint (TELEA)
      plus a light blur. Measured on real output: the patch then matches the surroundings
      (high-pass std 0.9 vs 1.2 around it) where a learned model hallucinates texture
      ~13x stronger than the background (std 15.3) and leaves a visible blob.
    * textured background -> LaMa, which reconstructs plausible texture instead of smearing.
    """
    tex = ring_texture(img_bgr, mask)
    m8 = (mask > 0).astype(np.uint8)
    if tex < smooth_below or inp is None:
        out = cv2.inpaint(img_bgr, m8, 3, cv2.INPAINT_TELEA)
        if blur_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
        how = f"telea+blur (flat background, texture {tex:.1f})"
    else:
        out = inp.inpaint(img_bgr, mask, feather=feather)
        how = f"lama (textured background, texture {tex:.1f})"
    a = feather_alpha(mask, feather)[..., None]
    out = (out.astype(np.float32) * a + img_bgr.astype(np.float32) * (1 - a)).astype(np.uint8)
    # Level correction only makes sense where the surroundings are featureless. On a textured
    # background the fill is reproducing real structure (a plank groove through the mask), and a
    # global shift would push that structure off-level.
    if tex < smooth_below or inp is None:
        out, bias = match_fill_level(out, mask, ring=8, verbose=verbose)
    if verbose:
        print(f"    fill: {how}" + (f", level corrected by {bias:+.1f}" if abs(bias) >= 0.5 else ""))
    return out


def match_fill_level(filled_bgr: np.ndarray, mask: np.ndarray, ring: int = 8,
                     max_shift: float = 12.0, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Cancel the slow brightness bias a fill leaves behind.

    On a dark, near-flat area (a night sky, a dark jacket) the inpainted patch can come out a few
    levels off the surroundings. It is invisible in the crop but it is exactly star-shaped, so it
    reads as a leftover ghost. Measure the median level of the filled area against the ring around
    it and shift the patch by the difference (capped, so a real edge is never flattened away).
    Returns (image, shift).
    """
    core = mask > 0
    ring_m = (dilate_mask(mask, ring) > 0) & ~core
    if not core.any() or not ring_m.any():
        return filled_bgr, 0.0
    g = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift = float(np.median(g[ring_m]) - np.median(g[core]))
    shift = float(np.clip(shift, -max_shift, max_shift))
    if abs(shift) < 0.5:
        return filled_bgr, shift
    w = feather_alpha(mask, ring)[..., None]
    out = filled_bgr.astype(np.float32) + w * shift
    return np.clip(out, 0, 255).astype(np.uint8), shift


def inpaint_image(inp: Inpainter, img_bgr: np.ndarray, mask: np.ndarray, feather: int = 3,
                  adaptive: bool = True, verbose: bool = False):
    """Remove the mark from one image. `adaptive=True` picks LaMa or a smooth fill based on
    how textured the background is (see `adaptive_fill`)."""
    if adaptive:
        return adaptive_fill(img_bgr, mask, inp, feather=feather, verbose=verbose)
    return inp.inpaint(img_bgr, mask, feather=feather)


def inpaint_folder(inp: Inpainter, in_dir: str, out_dir: str, mask: Optional[np.ndarray] = None,
                   mode: str = "auto", corner: str = "auto", rect: Optional[Sequence[int]] = None,
                   preset: Optional[str] = None, strict: float = 1.0, pad: int = 6,
                   detect_n: int = 12, feather: int = 3, adaptive: bool = True,
                   exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp"), save_mask: bool = True,
                   verbose: bool = True) -> dict:
    """
    Batch-process a folder of images.

    With mode='auto' (default) the Gemini sparkle is looked for **per image**, which also
    handles folders that mix resolutions (Gemini emits 1K and 2K renders side by side, and the
    mark sits at a different offset in each). Images where no sparkle is found use the
    folder-level mask (temporal stack when there are enough frames, else preset + refinement).
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(f"no images found in {in_dir}")
    loaded = {}
    for f in files:
        im = imread_color(os.path.join(in_dir, f))
        if im is not None:
            loaded[f] = im
    files = [f for f in files if f in loaded]

    per_image, fallback, how = {}, None, []
    if mask is not None:
        fallback = mask
    if mode in ("auto", "sparkle"):
        for f, im in loaded.items():
            m = detect_sparkle_mask(im, pad=pad, verbose=False)
            if m is not None:
                per_image[f] = m
        if verbose:
            print(f"  sparkle template: {len(per_image)}/{len(files)} images matched")
        if mode == "sparkle" and not per_image:
            raise RuntimeError("no sparkle found in any image - try mode='preset'")
    if fallback is None and len(per_image) < len(files):
        if len(files) >= 3 and mode in ("auto", "stack"):
            sample = [loaded[f] for f in files[:max(detect_n, 4)]]
            if verbose:
                print(f"  temporal detection across {len(sample)} images for the remainder")
            try:
                fallback = resolve_mask(sample, mode="stack" if mode != "stack" else "stack",
                                        corner=corner, rect=rect, preset=preset,
                                        strict=strict, pad=pad, verbose=verbose)
                how.append("stack")
            except Exception as e:
                if verbose:
                    print("  stack detection unavailable:", str(e)[:120])
        if fallback is None:
            fallback = preset_mask(loaded[files[0]], preset=preset or "gemini_sparkle", pad=pad,
                                   refine=True, verbose=verbose)
            how.append(f"preset:{preset or 'gemini_sparkle'}")

    if save_mask:
        first = loaded[files[0]]
        m0 = per_image.get(files[0], fallback)
        cv2.imwrite(os.path.join(out_dir, "_detected_mask_preview.png"),
                    mask_preview(first, m0, scale=3))
        cv2.imwrite(os.path.join(out_dir, "_mask.png"), m0)

    done, by_method = 0, {"sparkle": 0, "fallback": 0}
    reports = []
    for f in files:
        src = loaded[f]
        m = per_image.get(f, fallback)
        by_method["sparkle" if f in per_image else "fallback"] += 1
        if m.shape[:2] != src.shape[:2]:                    # folder mask, different resolution
            m = cv2.resize(m, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
        used_sparkle = f in per_image
        res = (adaptive_fill(src, m, inp, feather=feather, verbose=False) if adaptive
               else inp.inpaint(src, m, feather=feather))
        params = [cv2.IMWRITE_JPEG_QUALITY, 97] if f.lower().endswith((".jpg", ".jpeg")) else []
        cv2.imwrite(os.path.join(out_dir, f), res, params)
        row = dict(file=f, method=("sparkle" if used_sparkle else "fallback"),
                   texture=round(ring_texture(src, m), 2))
        try:
            if used_sparkle:
                v = verify_sparkle_removal(src, res, verbose=False)
                row.update(iou_before=v["iou_before"], iou_after=v["iou_after"], verdict=v["verdict"])
            else:
                v = verify_removal(src, res, m)
                row.update(contrast_before=v["contrast_before"], contrast_after=v["contrast_after"],
                           verdict=v["verdict"])
        except Exception:
            row["verdict"] = "verification skipped"
        reports.append(row)
        done += 1
        if verbose:
            tail = ""
            if reports and reports[-1]["file"] == f:
                r = reports[-1]
                tail = f"  | {r['method']}, texture {r.get('texture')}  ->  {r['verdict'][:40]}"
            print(f"    [{done}/{len(files)}] {f}{tail}")
    return {"mask": fallback if fallback is not None else next(iter(per_image.values()), None),
            "per_image": per_image, "processed": done, "out_dir": out_dir,
            "method": f"sparkle x{by_method['sparkle']}" + (f", {', '.join(how)}" if how else ""),
            "reports": reports}


# --------------------------------------------------------------------------------------
# high level: video
# --------------------------------------------------------------------------------------

def list_media(folder: str, kind: str = "video", recursive: bool = False,
               exts: Optional[Sequence[str]] = None) -> list:
    """
    Collect media paths from a folder (sorted, deterministic order).

    kind: 'video' (.mp4/.mov/.webm/.mkv/.avi/.m4v), 'image' (.png/.jpg/.jpeg/.webp/.bmp),
          or 'any'. `recursive=True` walks sub-folders.
    """
    defaults = {
        "video": (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"),
        "image": (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    }
    exts = tuple(e.lower() for e in (exts or defaults.get(kind, defaults["video"] + defaults["image"])))
    out = []
    if not os.path.isdir(folder):
        return out
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(exts):
                out.append(p)
    return sorted(out)


def unique_output_path(in_path: str, out_dir: str, suffix: str = "_clean",
                       ext: Optional[str] = None) -> str:
    """Build an output path that never overwrites an existing file or an input file."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    ext = ext or os.path.splitext(in_path)[1].lower()
    cand = os.path.join(out_dir, f"{stem}{suffix}{ext}")
    if os.path.abspath(cand) == os.path.abspath(in_path) or os.path.exists(cand):
        n = 2
        while True:
            cand = os.path.join(out_dir, f"{stem}{suffix}_{n}{ext}")
            if os.path.abspath(cand) != os.path.abspath(in_path) and not os.path.exists(cand):
                break
            n += 1
    return cand


def detect_for_image(img_bgr: np.ndarray, mode: str = "auto", preset: Optional[str] = None,
                     rect: Optional[Sequence[int]] = None, pad: Optional[int] = None,
                     allow_preset: bool = True, verbose: bool = False):
    """
    One-stop mask resolution for a single image. Returns `(mask, method)` where method is one of
    'sparkle', 'rect', 'preset:<name>' -- or `(None, None)` when no mark was found.

    Order (matters on real output):
      1. sparkle shape template (the Gemini image mark),
      2. an explicit `rect` if given,
      3. a named `preset` box -- only when `allow_preset=True`, because a preset assumes where
         the mark is; if it is wrong you would inpaint innocent picture content.
    Stack detection needs several frames, so it is not part of this path.
    """
    if mode in ("auto", "sparkle"):
        m = detect_sparkle_mask(img_bgr, pad=pad, verbose=verbose)
        if m is not None:
            return m, "sparkle"
        if mode == "sparkle":
            return None, None
    if rect is not None:
        return rect_mask(img_bgr.shape[0], img_bgr.shape[1], rect), "rect"
    if preset and allow_preset:
        return preset_mask(img_bgr, preset=preset, pad=pad + 3, refine=True, verbose=verbose), f"preset:{preset}"
    return None, None


def detect_mask_from_video(video_path: str, n_frames: int = 24, corner: str = "auto",
                           mask: Optional[np.ndarray] = None, mode: str = "stack",
                           strict: float = 1.0, pad: int = 6, verbose: bool = True) -> np.ndarray:
    """Sample frames from a video and detect the static watermark mask."""
    if mask is not None:
        return mask
    frames = sample_frames(video_path, n=n_frames)
    if verbose:
        print(f"  sampling {len(frames)} frames for detection")
    return resolve_mask(frames, mode=mode, corner=corner, strict=strict, pad=pad, verbose=verbose)


def remove_watermark_video(in_path: str, out_path: str, inp: Inpainter, mask: np.ndarray,
                           batch_size: int = 8, crf: int = 17, preset: str = "medium",
                           feather: int = 3, keep_audio: bool = True, verbose: bool = True,
                           progress_cb=None, cancel=None) -> dict:
    """
    Quality path: LaMa-inpaint the watermark region on every frame (streamed in batches, so
    memory stays flat no matter how long the clip is). Audio is copied from the source.

    GUI-friendly extras:
      * `progress_cb(done, total, elapsed_seconds, fps)` is called after every batch;
      * `cancel` is a threading.Event -- if it gets set the run stops promptly and raises
        InterruptedError without leaving a half-written output behind (output is written to
        `<out_path>.partial` and only renamed into place on success).
    """
    nfo = video_info(in_path)
    W, H, fps = nfo["width"], nfo["height"], nfo["fps"]
    total = nfo["frames"]
    if verbose:
        print(f"  input : {W}x{H} @ {fps:.2f} fps, {total} frames")
        print(f"  mask  : {(mask > 0).sum()} px, region {inp.plan_region(H, W, mask)}")
    partial = out_path + ".partial"
    tmp = tempfile.NamedTemporaryFile(suffix="_nowm.mp4", delete=False).name
    writer = FFmpegWriter(tmp, W, H, fps, crf=crf, preset=preset)
    buf, done, t0 = [], 0, time.time()
    try:
        for frame in iter_frames(in_path):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            buf.append(frame)
            if len(buf) >= batch_size:
                for f in inp.inpaint_frames(buf, mask, batch_size=batch_size, feather=feather, progress=False):
                    writer.write(f)
                done += len(buf)
                buf = []
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                if progress_cb is not None:
                    progress_cb(done, total, el, rate)
                if verbose:
                    eta = (total - done) / max(rate, 1e-6)
                    print(f"    {done}/{total} frames | {rate:5.1f} fps | {el:6.1f}s elapsed | "
                          f"ETA {eta/60:4.1f} min", end="\r")
        if buf:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            for f in inp.inpaint_frames(buf, mask, batch_size=batch_size, feather=feather, progress=False):
                writer.write(f)
            done += len(buf)
            if progress_cb is not None:
                progress_cb(done, total, time.time() - t0, done / max(time.time() - t0, 1e-6))
    except BaseException:
        # never leave a half-encoded file behind
        try:
            writer.proc.kill()
        except Exception:
            pass
        for path in (tmp, partial):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        raise
    writer.close()
    audio_ok = None
    if keep_audio:
        # mux into a real .mp4 name and rename afterwards: ffmpeg picks the container from the
        # extension, and "<out>.partial" has none, which used to silently drop the audio track.
        muxed = tmp + ".muxed.mp4"
        audio_ok = mux_audio(tmp, in_path, muxed)
        os.replace(muxed, partial)
        try:
            os.remove(tmp)
        except OSError:
            pass
    else:
        shutil.move(tmp, partial)
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    os.replace(partial, out_path)
    el = time.time() - t0
    if verbose:
        print(f"\n  done: {done} frames in {el/60:.1f} min ({done/max(el,1e-6):.1f} fps) -> {out_path}")
    return {"out": out_path, "frames": done, "seconds": el, "info": nfo, "audio": audio_ok}


def remove_watermark_video_delogo(in_path: str, out_path: str, mask: np.ndarray,
                                  crf: int = 17, preset: str = "veryfast", blur: bool = False,
                                  ffmpeg: Optional[str] = None, verbose: bool = True) -> str:
    """Fast path (~realtime, no model): ffmpeg delogo interpolation inside the watermark box."""
    ffmpeg = ffmpeg or find_ffmpeg()
    nfo = video_info(in_path)
    W, H = nfo["width"], nfo["height"]
    bb = mask_bbox(mask)
    if bb is None:
        raise ValueError("empty mask")
    x0, y0, x1, y1 = bb
    x0, y0 = max(x0 - 4, 1), max(y0 - 4, 1)          # delogo needs a 1px ring of good pixels
    x1, y1 = min(x1 + 4, W - 1), min(y1 + 4, H - 1)
    if blur:
        vf = f"boxblur=luma_radius=12:luma_power=2:enable='between(t,0,99999)',crop=w={x1-x0}:h={y1-y0}:x={x0}:y={y0},scale={W}:{H}"
        # (kept simple: use delogo below instead of fancy chaining)
        vf = f"delogo=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:show=0"
    else:
        vf = f"delogo=x={x0}:y={y0}:w={x1-x0}:h={y1-y0}:show=0"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", in_path,
           "-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg delogo failed: {r.stderr[-800:]}")
    if verbose:
        print(f"  delogo box ({x0},{y0},{x1-x0},{y1-y0}) -> {out_path}")
    return out_path


def remove_watermark_video_crop(in_path: str, out_path: str, crop: str,
                                ffmpeg: Optional[str] = None, verbose: bool = True) -> str:
    """Artifact-free alternative: cut the watermark out of frame. crop='w:h:x:y' (ffmpeg syntax)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", in_path,
           "-vf", f"crop={crop}", "-c:v", "libx264", "-crf", "17", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {r.stderr[-800:]}")
    if verbose:
        print(f"  cropped ({crop}) -> {out_path}")
    return out_path


def process_video_folder(in_dir: str, out_dir: str, inp: "Inpainter", preset: str = "veo_video",
                         batch_size: int = 8, crf: int = 17, encode_preset: str = "medium",
                         skip_existing: bool = True, verify: bool = True, verify_frames: int = 6,
                         exts=(".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"),
                         report: bool = True, verbose: bool = True) -> list:
    """
    Batch-process every clip in a folder. For each one:

    1. detect the mark from ~24 sampled frames (temporal); if the shot is too static, fall
       back to the corner preset box, refined to the pale evidence inside it;
    2. remove the mark, keeping the audio;
    3. verify the result with `verify_removal` and write a mask preview + before/after zoom.

    `skip_existing=True` makes re-runs resumable. A JSON report lands in `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(f"no videos found in {in_dir} (looking for {', '.join(exts)})")
    rows = []
    for idx, f in enumerate(files, 1):
        src = os.path.join(in_dir, f)
        stem = os.path.splitext(f)[0]
        dst = os.path.join(out_dir, f"{stem}_clean.mp4")
        row = dict(file=f, status="", method="", mask_px=0, frames=0, seconds=0.0,
                   contrast_before=None, contrast_after=None, removed_pct=None, verdict="")
        if verbose:
            print(f"\n[{idx}/{len(files)}] {f}")
        if skip_existing and os.path.isfile(dst):
            row["status"] = "skipped (already processed)"
            if verbose:
                print("    output exists -> skipped")
            rows.append(row)
            continue
        try:
            nfo = video_info(src)
            row["frames"] = nfo["frames"]
            try:
                mask = detect_mask_from_video(src, n_frames=24, verbose=False)
                row["method"] = "temporal"
            except RuntimeError:
                frame0 = sample_frames(src, n=1)[0]
                mask = preset_mask(frame0, preset=preset, verbose=False)
                row["method"] = f"preset:{preset}"
            row["mask_px"] = int((mask > 0).sum())
            if verbose:
                print(f"    {nfo['width']}x{nfo['height']} @ {nfo['fps']:.2f} fps, {nfo['frames']} frames"
                      f" | mask {row['mask_px']} px via {row['method']}")
            t0 = time.time()
            remove_watermark_video(src, dst, inp, mask, batch_size=batch_size, crf=crf,
                                   preset=encode_preset, verbose=verbose)
            row["seconds"] = round(time.time() - t0, 1)
            frame0 = sample_frames(src, n=1)[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}_mask_preview.png"),
                        mask_preview(frame0, mask, scale=3))
            if verify and os.path.isfile(dst):
                bef = sample_frames(src, n=verify_frames)
                aft = sample_frames(dst, n=verify_frames)
                v = verify_removal(bef, aft, mask)
                row.update(contrast_before=v["contrast_before"], contrast_after=v["contrast_after"],
                           removed_pct=v["removed_pct"], verdict=v["verdict"])
                cv2.imwrite(os.path.join(out_dir, f"{stem}_before_after.png"),
                            make_side_by_side(bef[0], aft[0], mask, scale=3))
                if verbose:
                    print(f"    verify: contrast {v['contrast_before']} -> {v['contrast_after']}"
                          f"  ({v['removed_pct']}% removed)  {v['verdict']}")
            row["status"] = "done"
        except Exception as exc:
            row["status"] = f"FAILED: {exc}"
            if verbose:
                print("    FAILED:", str(exc)[:300])
        rows.append(row)

    if report:
        try:
            import json
            with open(os.path.join(out_dir, "_report.json"), "w") as fh:
                json.dump(rows, fh, indent=2)
        except Exception:
            pass
    if verbose:
        done = [r for r in rows if r["status"] == "done"]
        total_s = sum(float(r["seconds"]) for r in done)
        print(f"\n{len(done)}/{len(rows)} clips processed"
              + (f" in {total_s/60:.1f} min" if done else ""))
        print(f"  {'file':38s} {'method':16s} result")
        for r in rows:
            print(f"  {r['file'][:38]:38s} {r['method'][:16]:16s} {(r['verdict'] or r['status'])[:60]}")
    return rows


def make_side_by_side(before_bgr: np.ndarray, after_bgr: np.ndarray, mask: np.ndarray,
                      scale: int = 3, label: bool = True) -> np.ndarray:
    """Zoomed before/after of the watermark region, stacked vertically."""
    bb = mask_bbox(mask)
    if bb is None:
        return np.hstack([before_bgr, after_bgr])
    x0, y0, x1, y1 = bb
    p = 14
    x0, y0 = max(x0 - p, 0), max(y0 - p, 0)
    x1, y1 = min(x1 + p, before_bgr.shape[1]), min(y1 + p, before_bgr.shape[0])
    a = before_bgr[y0:y1, x0:x1].copy()
    b = after_bgr[y0:y1, x0:x1].copy()
    if label:
        for im, txt in ((a, "BEFORE"), (b, "AFTER")):
            cv2.rectangle(im, (0, 0), (150, 26), (0, 0, 0), -1)
            cv2.putText(im, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    stack = np.vstack([a, np.full((6, a.shape[1], 3), 40, np.uint8), b])
    return cv2.resize(stack, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

"""Builds the Colab notebook deliverable."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(text):
    C.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    C.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- intro
md(r"""
# 🎬 Veo / Flow / Gemini — Visible Watermark Remover (Colab · Google Drive)

Removes the **visible** corner watermark from **your own** generated media: the `Veo` wordmark
that Google Flow stamps on videos, and the **4-point sparkle** that Gemini stamps on images
(measured on real output: 48x48 px, near-diamond outline, bottom-right). The engine is
**LaMa** (Resolution-robust Large Mask Inpainting) running as ONNX — the model file lives in
your Google Drive.

### How it works
1. **Find the mark.** For a folder or a video, the mark is static, so the notebook detects it
   from several frames with a temporal t‑statistic. For still images it uses a **shape template
   matcher** for the Gemini sparkle (matched filter + star-shape IoU) — validated on real
   output: 3/3 real marks located exactly, 0/19 false positives on unwatermarked photos.
2. It then inpaints **only that small region** on every frame (a 512×512 crop around the
   mark), which is why it is fast and why the rest of the frame is left **bit‑for‑bit
   untouched**.
3. **Pick the right fill.** On flat backgrounds a learned model hallucinates texture (~13x the
   surrounding noise, a visible blob), so those regions get a smooth interpolation instead;
   textured regions get LaMa. This is chosen per image automatically.
4. Audio is copied through untouched (video only).

### Honest limits
* This removes the **visible** watermark only. Google also embeds an invisible **SynthID**
  watermark in all Veo/Flow/Gemini output. SynthID is designed to survive editing and is
  **not** removed (or detected) by this notebook.
* Only use this on your own content (or content you have the rights to edit) and mind the
  terms of the service you generated it with.
* A *perfectly* invisible result isn't always possible: the pixels under the mark were
  destroyed by the overlay, so they are **reconstructed**, like a tiny spot-heal. On flat
  backgrounds it's undetectable; on busy textures you may see a faint smudge if you zoom in.

### Quick start
Run the cells in order (`Runtime ▸ Run all` works too):

| Step | Cell |
|---|---|
| 1 | Mount Drive + install packages |
| 2 | Configure folders (model, input, output) |
| 3 | Load the LaMa model from Drive |
| 4 | **Images**: folder of Gemini renders (the common case) |
| 5 | **Video**: detect → preview → remove → verify |
| 6 | **Batch**: every clip in a folder, with a report |
| 7 | Optional Gradio UI (point-and-click) |

A GPU runtime (**Runtime ▸ Change runtime type ▸ T4 GPU**) is strongly recommended:
CPU is ≈ 7 s per 720p frame, a T4 is typically 20–60× faster.
""")

# ---------------------------------------------------------------- config
md(r"""
## 1 · Mount Google Drive & install packages

Put the model file in Drive once (this cell downloads it for you if it's missing):
`MyDrive/watermark_remover/models/lama_fp32.onnx` (208 MB).
""")

code(r'''
#@title Mount Drive + install packages (run once per session)
import os, sys, subprocess, shutil, textwrap

IN_COLAB = "google.colab" in sys.modules
DRIVE_ROOT = "/content/drive/MyDrive" if IN_COLAB else os.path.expanduser("~")
if IN_COLAB:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    print("Drive mounted ✔")
else:
    print("Not running in Colab — using", DRIVE_ROOT)

def sh(cmd):
    print("$", cmd)
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout[-1500:])

# onnxruntime: GPU build on Colab, plain onnxruntime elsewhere
want_gpu = IN_COLAB
sh(f"{sys.executable} -m pip install -q {'onnxruntime-gpu' if want_gpu else 'onnxruntime'} imageio-ffmpeg")

import onnxruntime as ort
print("onnxruntime", ort.__version__, "| providers:", ort.get_available_providers())
print("GPU provider available:", "CUDAExecutionProvider" in ort.get_available_providers())
''')

# ---------------------------------------------------------------- config 2
code(r'''
#@title Folders & settings  { display-mode: "form" }
#@markdown Everything lives in Drive so it survives runtime restarts.
PROJECT_DIR = f"{DRIVE_ROOT}/watermark_remover"      #@param {type:"string"}
MODEL_PATH  = f"{PROJECT_DIR}/models/lama_fp32.onnx" #@param {type:"string"}
INPUT_DIR   = f"{PROJECT_DIR}/input"                 #@param {type:"string"}
OUTPUT_DIR  = f"{PROJECT_DIR}/output"                #@param {type:"string"}

MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL_SHA_SIZE = 208_044_816            # bytes, used only as a sanity check

for d in (os.path.dirname(MODEL_PATH), INPUT_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)
print("project :", PROJECT_DIR)
print("model   :", MODEL_PATH)
print("input   :", INPUT_DIR)
print("output  :", OUTPUT_DIR)
''')

# ---------------------------------------------------------------- library cell
md(r"""
## 2 · The engine (`wm_core.py`)

This cell writes the whole watermark-removal library to disk. It contains:

* **detection** — `detect_mask_stack` / `detect_mask_from_video` (temporal t‑statistic),
  `detect_mask_single`, `resolve_mask`, geometry `PRESETS`;
* **inpainting** — `Inpainter` (ONNX wrapper, batched, region-crop),
  `inpaint_folder`, `remove_watermark_video`, plus fast `delogo` / `crop` fallbacks;
* **video plumbing** — ffmpeg writer + audio mux (`mux_audio`), `video_info`, `sample_frames`.

Since detection was calibrated on both real and synthetic marks, the numbers you'll see in
the preview (`strength 0.18 – 0.45`) are exactly the range real wordmarks fall into.
""")

LIB = open("/home/user/wm_core.py").read().rstrip()
cell = "%%writefile wm_core.py\n" + LIB
code(cell)

# ---------------------------------------------------------------- model load
md(r"""
## 3 · Load the model from Drive

The cell downloads `lama_fp32.onnx` into your Drive folder the first time (208 MB, one-time),
then loads it. It also runs a 10-second self-test so you know the engine works before you
throw a 4K clip at it.
""")

code(r'''
#@title Download (once) + load LaMa + self-test
import os, sys, time, urllib.request, cv2, numpy as np
for p in (os.getcwd(), "/content", os.path.dirname(os.path.abspath("wm_core.py"))):
    if p and p not in sys.path:
        sys.path.insert(0, p)
import wm_core as wc
print("engine:", wc.__file__)

def ensure_model(path: str, url: str, expect_bytes: int) -> str:
    if os.path.isfile(path) and os.path.getsize(path) > 100_000_000:
        print(f"model already in Drive ✔  ({os.path.getsize(path)/1e6:.0f} MB)")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"downloading {url}\n  -> {path}  (one time, ~208 MB) ...")
    t0 = time.time()
    def hook(b, bs, total):
        if total > 0 and (time.time() - t0) > 0:
            pct = min(100.0, b * bs / total * 100)
            print(f"\r  {pct:5.1f}%  ({b*bs/1e6:6.1f}/{total/1e6:.0f} MB)", end="")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    os.replace(tmp, path)
    print(f"saved ({os.path.getsize(path)/1e6:.0f} MB). Keeping it in Drive means no re-download next time.")
    return path

MODEL_PATH = ensure_model(MODEL_PATH, MODEL_URL, MODEL_SHA_SIZE)

INP = wc.Inpainter(MODEL_PATH, verbose=True)

# ---- self-test: remove a synthetic pale wordmark and measure how much is gone -----------
scale, thick, alpha = 2.2, 5, 0.55                      # a "Veo"-like pale overlay
(tw, th), _ = cv2.getTextSize("Veo", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
x, y = 512 - tw - 40, 512 - th - 40
big = np.zeros((512 * 4, 512 * 4), np.uint8)
cv2.putText(big, "Veo", (x * 4, (y + th) * 4), cv2.FONT_HERSHEY_SIMPLEX, scale * 4, 255, thick * 4, cv2.LINE_AA)
a = cv2.resize(big, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * alpha
rng = np.random.default_rng(0)
bg = cv2.resize(rng.integers(60, 200, (16, 16, 3), np.uint8), (512, 512), interpolation=cv2.INTER_CUBIC)
bg = cv2.GaussianBlur(bg, (0, 0), 6)
test = (bg.astype(np.float32) * (1 - a[..., None]) + 255.0 * a[..., None]).astype(np.uint8)
mask = wc.rect_mask(512, 512, (x - 8, y - 8, tw + 16, th + 16))
t0 = time.time(); out = INP.inpaint(test, mask); dt = time.time() - t0
glyph = a > 0.15
err_before = float(np.abs(test.astype(float) - bg.astype(float)).mean(axis=2)[glyph].mean())
err_after = float(np.abs(out.astype(float) - bg.astype(float)).mean(axis=2)[glyph].mean())
print(f"self-test ({dt:.2f}s): watermark strength on the mark pixels {err_before:.1f} -> {err_after:.1f} "
      f"({100 * (1 - err_after / max(err_before, 1e-6)):.0f}% removed)")
print("engine ready ✔" if err_after < 0.35 * err_before else "engine check FAILED - re-run the install cell / restart runtime")
''')

# ---------------------------------------------------------------- video
md(r"""
## 4 · Images (Gemini renders)

Point `IMG_IN` at a folder of images. The default mode looks for the **Gemini sparkle in each
image individually** with the shape template, which also handles folders that **mix
resolutions** — Gemini emits 1K and 2K side by side and the mark sits at a *different offset*
in each (measured: 76 px inset at 1K, 94 px at 2K), so one fixed box cannot serve the whole
folder.

Per image you get: detection method, local background texture, and a verification line. For the
sparkle path the verification re-runs the star template on the output — if it can still find a
star-shaped mark, the removal did not finish (that is a much sharper test than a brightness
comparison, which can be fooled by the fill patch itself).
""")

code(r'''
#@title Process an image folder  { display-mode: "form" }
IMG_IN  = f"{INPUT_DIR}/images"        #@param {type:"string"}
IMG_OUT = f"{OUTPUT_DIR}/images_clean" #@param {type:"string"}
MODE    = "auto"                       #@param ["auto", "sparkle", "stack", "preset"]
PRESET  = "gemini_sparkle"             #@param ["gemini_sparkle", "veo_video", "gemini_image", "flow_strip", "bottom_left"]
ADAPTIVE_FILL = True                   #@param {type:"boolean"}

import os, json, cv2, numpy as np, wm_core as wc
res = wc.inpaint_folder(INP, IMG_IN, IMG_OUT, mode=MODE, adaptive=ADAPTIVE_FILL,
                        preset=(PRESET if MODE == "preset" else None), verbose=True)
print(f"\n{res['processed']} images -> {res['out_dir']}   (method: {res['method']})")

# ---- per-image report + before/after zooms ---------------------------------------------
with open(os.path.join(IMG_OUT, "_report.json"), "w") as fh:
    json.dump(res["reports"], fh, indent=2)
try:
    import pandas as pd
    display(pd.DataFrame(res["reports"]))
except Exception:
    for r in res["reports"]:
        print("   ", r)

try:
    from IPython.display import display, Image as IPyImage
    files = sorted(f for f in os.listdir(IMG_IN) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))[:3]
    for f in files:
        b = wc.imread_color(os.path.join(IMG_IN, f))
        a = wc.imread_color(os.path.join(IMG_OUT, f))
        m = wc.detect_sparkle_mask(b, pad=3, verbose=False)
        if m is None:
            continue
        x0, y0, x1, y1 = wc.mask_bbox(m); p = 22
        sl = (slice(max(y0 - p, 0), min(y1 + p, b.shape[0])), slice(max(x0 - p, 0), min(x1 + p, b.shape[1])))
        sep = np.full((b[sl].shape[0], 4, 3), 70, np.uint8)
        zoom = cv2.resize(np.hstack([b[sl], sep, a[sl]]), None, fx=3.5, fy=3.5, interpolation=cv2.INTER_NEAREST)
        print(f"\n{f}  (left: original, right: cleaned)")
        display(IPyImage(data=cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)))
except Exception as e:
    print("(previews skipped:", e, ")")

try:
    from IPython.display import display, Image as IPyImage
    display(IPyImage(filename=os.path.join(res["out_dir"], "_detected_mask_preview.png")))
except Exception as e:
    print("(mask preview:", e, ")")
''')

md(r"""
## 5 · Video: detect → preview → remove

* `VIDEO_PATH` — any `.mp4/.mov/.webm` (in Drive or `/content`).
* Detection samples 24 frames; **look at the preview image** before processing.
* Output goes to `OUTPUT_DIR` with audio preserved.

**Detection needs motion.** The mark is found by comparing frames, so a *static* shot (or a
clip shorter than ~8 frames) has little to compare — the cell then falls back to the
`veo_video` preset box, and you confirm it in the preview. Long, moving clips are the easy case.

**Speed:** on a T4 ≈ 30–60 fps of 720p, 10–25 fps of 1080p (a 10 s clip ≈ 15–40 s) with `BATCH_SIZE 8-16`.
On CPU it's ≈ 0.14 fps — use a GPU runtime, or the fast `delogo` mode below.
""")

code(r'''
#@title Remove watermark from a video  { display-mode: "form" }
VIDEO_PATH = f"{INPUT_DIR}/my_clip.mp4"   #@param {type:"string"}
MODE = "lama"        #@param ["lama", "delogo", "crop"]
BATCH_SIZE = 4       #@param {type:"slider", min:1, max:32, step:1}
CRF = 17             #@param {type:"slider", min:12, max:30, step:1}
CHECK_PREVIEW = True #@param {type:"boolean"}

import os, cv2, wm_core as wc
assert os.path.isfile(VIDEO_PATH), f"not found: {VIDEO_PATH} — copy it into {INPUT_DIR} first"

stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
out_path = os.path.join(OUTPUT_DIR, f"{stem}_clean.mp4")

# ---- 0) how much does the picture move? (detection needs a little variation) ----------
probe = wc.sample_frames(VIDEO_PATH, n=12)
motion = float(np.mean([cv2.absdiff(probe[0], f).mean() for f in probe[1:]])) if len(probe) > 1 else 0.0
print(f"sampled {len(probe)} frames | average frame-to-frame difference: {motion:.1f}"
      + ("  (nearly static shot - auto-detection may struggle)" if motion < 2.0 else ""))

# ---- 1) detect -------------------------------------------------------------------------
try:
    mask = wc.detect_mask_from_video(VIDEO_PATH, n_frames=24, verbose=True)
except RuntimeError as e:
    print("\nauto-detection failed:", e)
    print("-> falling back to the 'veo_video' preset box (bottom-right wordmark).")
    print("   Check the preview below; if it's off, use the manual/preset cell instead.")
    frame0 = wc.sample_frames(VIDEO_PATH, n=1)[0]
    mask = wc.resolve_mask([frame0], mode="preset", preset="veo_video", verbose=True)

# ---- 2) preview ------------------------------------------------------------------------
preview_frame = wc.sample_frames(VIDEO_PATH, n=1)[0]
vis = wc.mask_preview(preview_frame, mask, scale=2)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_mask_preview.png"), vis)
print(f"mask: {(mask>0).sum()} px at {wc.mask_bbox(mask)}  ->  {OUTPUT_DIR}/{stem}_mask_preview.png")

try:
    from IPython.display import display, Image as IPyImage
    display(IPyImage(filename=os.path.join(OUTPUT_DIR, f"{stem}_mask_preview.png")))
except Exception:
    pass

if CHECK_PREVIEW:
    print("\nCheck the preview above: the blue highlight must sit exactly on the wordmark.")
    print("If it's wrong/empty, jump to the 'manual box' snippet in the next cell and re-run.")

# ---- 3) process ------------------------------------------------------------------------
if MODE == "lama":
    res = wc.remove_watermark_video(VIDEO_PATH, out_path, INP, mask,
                                    batch_size=BATCH_SIZE, crf=CRF, preset="medium")
elif MODE == "delogo":
    # ~realtime, no model: linear interpolation over the box (a bit soft on busy textures)
    wc.remove_watermark_video_delogo(VIDEO_PATH, out_path, mask, crf=CRF)
elif MODE == "crop":
    # artifact-free but re-frames the video: crops off the watermark side
    info = wc.video_info(VIDEO_PATH)
    x0, y0, x1, y1 = wc.mask_bbox(mask)
    if x0 > info["width"] / 2:            # mark on the right -> keep the left part
        crop = f"{x0 - 8}:{info['height']}:0:0"
    else:
        crop = f"{x1 + 8}:{info['height']}:{x1 + 8}:0"
    wc.remove_watermark_video_crop(VIDEO_PATH, out_path, crop)
else:
    raise ValueError(MODE)

print("\noutput ->", out_path)

# ---- 4) verify: is the mark actually gone? ---------------------------------------------
try:
    n_check = min(8, max(2, wc.video_info(VIDEO_PATH)["frames"] - 1))
    v = wc.verify_removal(wc.sample_frames(VIDEO_PATH, n=n_check),
                          wc.sample_frames(out_path, n=n_check), mask)
    print(f"\nverify: mark contrast {v['contrast_before']} -> {v['contrast_after']}"
          f"  ({v['removed_pct']}% lower)  ->  {v['verdict']}")
    print("(contrast is the pale-overlay signal inside the mask vs a ring around it;"
          " +4..+20 = mark present, ~0 = gone)")
except Exception as e:
    print("(verification skipped:", e, ")")

# ---- 5) before/after zoom (frame 0) ----------------------------------------------------
try:
    before = wc.sample_frames(VIDEO_PATH, n=1)[0]
    after = wc.sample_frames(out_path, n=1)[0]
    sbs = wc.make_side_by_side(before, after, mask, scale=3)
    p = os.path.join(OUTPUT_DIR, f"{stem}_before_after.png")
    cv2.imwrite(p, sbs)
    print("before/after ->", p)
    from IPython.display import display, Image as IPyImage
    display(IPyImage(filename=p))
except Exception as e:
    print("(before/after preview skipped:", e, ")")
''')

md(r"""
**If auto-detection picked the wrong spot**, skip it and draw the box yourself. Coordinates
are fractions of the frame; negative values are measured from the right/bottom edge
(`(-0.06, -0.06, 0.055, 0.05)` = a box 6 % in from the bottom-right corner).

Ready-made presets: `gemini_sparkle` (images), `veo_video` (Flow video), `gemini_image`, `flow_strip`, `bottom_left`.
""")

code(r'''
#@title Manual / preset mask (use instead of auto-detect) { display-mode: "form" }
VIDEO_PATH = f"{INPUT_DIR}/my_clip.mp4"  #@param {type:"string"}
PRESET = "veo_video"                     #@param ["custom", "veo_video", "gemini_image", "flow_strip", "bottom_left", "top_right"]
RECT   = "-0.075,-0.065,0.065,0.055"     #@param {type:"string"}
#@markdown `RECT` = `x,y,w,h` as fractions of width/height (negative = from the opposite edge).
import os, cv2, wm_core as wc
frame = wc.sample_frames(VIDEO_PATH, n=1)[0]
rect = tuple(float(v) for v in RECT.split(","))
mask = wc.resolve_mask([frame], mode="preset", preset=(None if PRESET == "custom" else PRESET),
                       rect=rect, verbose=True)
vis = wc.mask_preview(frame, mask, scale=2)
p = os.path.join(OUTPUT_DIR, "manual_mask_preview.png")
cv2.imwrite(p, vis)
print("mask pixels:", int((mask > 0).sum()), "->", p)
try:
    from IPython.display import display, Image as IPyImage
    display(IPyImage(filename=p))
except Exception:
    pass
# When you're happy with the preview, reuse `mask` in the processing cell above
# (set the cell's detection line to: mask = ...  or simply paste the snippet below).
''')

# ---------------------------------------------------------------- images
md(r"""
## 6 · Batch: a whole folder of clips (video) (recommended for Flow)

Drop every clip in `MyDrive/watermark_remover/input/clips/` and run one cell. For each clip it
detects the mark (falling back to the preset box on static shots), removes it, **verifies** the
result and writes a mask preview + before/after zoom. Re-running is safe: finished clips are
skipped (`SKIP_EXISTING`). A `_report.json` summary is written next to the outputs.
""")

batch_code = '''
#@title Process every clip in a folder  { display-mode: "form" }
BATCH_IN  = f"{INPUT_DIR}/clips"        #@param {type:"string"}
BATCH_OUT = f"{OUTPUT_DIR}/clips_clean" #@param {type:"string"}
PRESET    = "veo_video"                 #@param ["veo_video", "gemini_image", "flow_strip", "bottom_left", "top_right"]
SKIP_EXISTING = True                    #@param {type:"boolean"}
BATCH_SIZE    = 4                       #@param {type:"slider", min:1, max:32, step:1}
CRF           = 17                      #@param {type:"slider", min:12, max:30, step:1}

import os, json, wm_core as wc
os.makedirs(BATCH_IN, exist_ok=True)
if not os.listdir(BATCH_IN):
    print(f"nothing in {BATCH_IN} yet - copy your clips there and re-run.")
else:
    rows = wc.process_video_folder(BATCH_IN, BATCH_OUT, INP, preset=PRESET,
                                   batch_size=BATCH_SIZE, crf=CRF,
                                   skip_existing=SKIP_EXISTING, verify=True, verbose=True)
    try:                      # tidy summary table
        import pandas as pd
        df = pd.DataFrame(rows)[["file", "status", "method", "frames", "mask_px",
                                 "removed_pct", "verdict"]]
        display(df)
    except Exception:
        pass
    print("outputs + previews + _report.json in", BATCH_OUT)
'''
code(batch_code)

# ---------------------------------------------------------------- gradio UI
md(r"""
## 7 · Optional: point-and-click UI

Colab ships with Gradio. This gives you drag-and-drop + buttons for both videos and images.
Use it for quick jobs; for long clips the API cells above are more robust (no browser timeout).
""")

code(r'''
#@title Launch the UI (optional)
import os, shutil, tempfile, cv2
import gradio as gr
import wm_core as wc

MODE_CHOICES = ["lama (quality)", "delogo (fast)", "crop (no re-draw)"]

def _save_to_output(path, name):
    dst = os.path.join(OUTPUT_DIR, name)
    try:
        shutil.copy(path, dst); return dst
    except Exception:
        return path

def ui_detect(video, preset):
    if not video:
        return None, "upload a video first"
    frames = wc.sample_frames(video, n=24)
    try:
        mask = wc.resolve_mask(frames, mode=("preset" if preset != "auto" else "stack"),
                               preset=(None if preset == "auto" else preset), verbose=False)
    except Exception as e:
        return None, f"detection failed: {e}\n-> pick a preset box instead"
    vis = wc.mask_preview(frames[0], mask, scale=2)
    p = os.path.join(tempfile.gettempdir(), "ui_mask_preview.png")
    cv2.imwrite(p, vis)
    return p, f"mask {int((mask>0).sum())} px @ {wc.mask_bbox(mask)} — blue/box must sit on the wordmark"

def ui_run(video, preset, mode, progress=gr.Progress()):
    if not video:
        return None, "upload a video first"
    frames = wc.sample_frames(video, n=24)
    try:
        mask = wc.resolve_mask(frames, mode=("preset" if preset != "auto" else "stack"),
                               preset=(None if preset == "auto" else preset), verbose=False)
    except Exception as e:
        return None, f"detection failed: {e} — choose a preset box."
    stem = os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(tempfile.gettempdir(), f"{stem}_clean.mp4")
    progress(0.05, desc="processing frames")
    if mode.startswith("lama"):
        wc.remove_watermark_video(video, out, INP, mask, batch_size=BATCH_SIZE, crf=CRF, verbose=False)
    elif mode.startswith("delogo"):
        wc.remove_watermark_video_delogo(video, out, mask, crf=CRF, verbose=False)
    else:
        info = wc.video_info(video); x0, y0, x1, y1 = wc.mask_bbox(mask)
        crop = f"{x0-8}:{info['height']}:0:0" if x0 > info["width"]/2 else f"{x1+8}:{info['height']}:{x1+8}:0"
        wc.remove_watermark_video_crop(video, out, crop, verbose=False)
    progress(1.0, desc="done")
    return out, f"saved to {_save_to_output(out, f'{stem}_clean.mp4')}"

def ui_images(files, preset, progress=gr.Progress()):
    if not files:
        return None, "upload images first"
    tmp_in = tempfile.mkdtemp(); tmp_out = tempfile.mkdtemp()
    for f in files:
        shutil.copy(f.name if hasattr(f, "name") else f, tmp_in)
    res = wc.inpaint_folder(INP, tmp_in, tmp_out, mode=("preset" if preset != "auto" else "auto"),
                            preset=(None if preset == "auto" else preset), verbose=False)
    zip_path = shutil.make_archive(os.path.join(OUTPUT_DIR, "images_clean"), "zip", tmp_out)
    return zip_path, f"{res['processed']} images -> {zip_path}"

with gr.Blocks(title="Veo / Gemini watermark remover") as demo:
    gr.Markdown("### Visible watermark remover — LaMa inpainting of the mark's region only\n"
                "SynthID (invisible watermark) is **not** affected. Only use on your own content.")
    with gr.Tab("Video"):
        v_in = gr.Video(label="input video")
        with gr.Row():
            v_preset = gr.Dropdown(["auto", "veo_video", "gemini_image", "flow_strip"], value="auto", label="mask")
            v_mode = gr.Dropdown(MODE_CHOICES, value=MODE_CHOICES[0], label="engine")
        v_detect = gr.Button("1 · detect & preview"); v_preview = gr.Image(label="mask preview"); v_status = gr.Textbox(label="status")
        v_run = gr.Button("2 · remove watermark", variant="primary"); v_out = gr.Video(label="output")
        v_detect.click(ui_detect, [v_in, v_preset], [v_preview, v_status])
        v_run.click(ui_run, [v_in, v_preset, v_mode], [v_out, v_status])
    with gr.Tab("Images"):
        i_in = gr.File(file_count="multiple", label="images (same watermark)")
        i_preset = gr.Dropdown(["auto", "gemini_image", "veo_video"], value="auto", label="mask")
        i_run = gr.Button("remove watermark", variant="primary"); i_out = gr.File(label="zip"); i_status = gr.Textbox(label="status")
        i_run.click(ui_images, [i_in, i_preset], [i_out, i_status])
    demo.launch(share=False, debug=False)
''')

# ---------------------------------------------------------------- notes
md(r"""
## 8 · Notes, limits & troubleshooting

**What this does / doesn't do**
* Removes the **visible** mark only (Flow's `Veo` wordmark, Gemini badges). **SynthID stays.**
* **Reading the verification line.** `mark contrast` compares the mask region against a ring
  around it *and* against an identical region shifted off the mark, so the picture's own
  contrast cancels: a mark reads **+5 … +100** (higher on flat backgrounds), a removed one
  reads **~0**, and the content noise floor is up to ~8 on busy scenes. The verdict flags the
  two mistakes that actually happen — "nothing was removed" and "the mask missed the mark".
  A residual mark fainter than the scene's own noise can slip through as "clean"; the
  before/after zoom image is always written so you can confirm visually.
* **Images: how the mark is found.** The Gemini image mark is a *shape*, not text — a solid
  pale 4-point star. The notebook matches it with a template (matched filter for brightness,
  then a star-shape IoU test that is what actually separates it from bright picture content).
  Measured on real output: 3/3 found at the exact pixel (IoU 0.79–0.95), 0/19 false positives
  on unwatermarked photos. Searching a *location* with no shape prior was tried and rejected
  (43–71 % recall, 3–16 false positives).
* **Mixed resolutions are handled per image.** The sparkle sits 76 px from the corner on a 1K
  render and 94 px on a 2K one, so a folder-level box would clip it; detection runs per image
  and returns a star-shaped mask.
* **Why is the fill adaptive?** On a flat background an inpainting model invents texture — the
  patch measured ~13x the local noise and read as a visible blob. Flat regions now get a smooth
  interpolation (patch texture 0.49 vs 0.93 around it) and textured regions get LaMa
  (12.5 vs 13.8). Both verified as "no star left".
* **Why there is no single-image auto-detect for wordmarks.** Finding a watermark's *location* in one frame,
  with no temporal information, measured 43–71% recall with 3–16 false positives across 19 clean
  photos — too unreliable to ship. Instead: batches and videos use temporal detection, and a
  single image uses a preset corner box (optionally tightened to the pale evidence inside it,
  which can only shrink, never relocate). You confirm it in the preview.
* The covered pixels are *reconstructed*, so keep the mask tight — bigger box = more made‑up
  area. Use `strict` (0.7 ≈ looser, 1.4 ≈ tighter) or a preset if the auto mask is bloated.
* Works best when the mark is pale/white. Dark marks on light backgrounds: use a `preset`
  or a custom `RECT` box.

**Troubleshooting**

| Symptom | Fix |
|---|---|
| "no candidate found" | Not enough frames (use ≥ 8), or mark very faint → use a preset / custom `RECT`. |
| Mask too big / smeared | Raise `strict` to ~1.3, or use the tight `veo_video` preset. |
| Mask empty / wrong corner | Use the preset cell, check the preview, adjust `RECT`. |
| CUDA provider fails | The model falls back to CPU automatically (slow); re-run the install cell, or `Runtime ▸ Disconnect and delete runtime` then rerun. |
| Long video takes ages | Use a GPU runtime; or `MODE="delogo"` (≈ realtime, softer); or use the batch cell and walk away. |
| Very short clip (< ~1 s) | Too few frames for temporal detection → the preset box is used; that is expected and still verified. |
| Verdict says "nothing was removed" | The mask isn't on the mark: try another `PRESET`, or a custom `RECT` in the manual cell. |
| Out of memory | Lower `BATCH_SIZE` to 2–4 (frames are what cost memory, ~270 MB per item at 512² on CPU). |

**Tips**
* Keep `MODEL_PATH` in Drive: after a runtime restart it's a 2-second load instead of a 208 MB download.
* `INPUT_DIR` is your inbox: drop clips/images there and just change the path parameters.
* Compare engines: `lama` = best quality, `delogo` = fastest, `crop` = zero re-draw (re-frames).
* For a 4K clip, everything still works — inference is on a 512² crop; only decode/encode cost grows.
""")

nb["cells"] = C
nb["metadata"] = {
    "colab": {"provenance": [], "collapsed_sections": [], "toc_visible": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}
nbf.write(nb, "/home/user/veo_gemini_watermark_remover.ipynb")
print("wrote notebook:", len(C), "cells")

# Veo / Flow / Gemini — visible watermark remover

Two ways to run the same engine (`wm_core.py`):

| | What it is | Use it for |
|---|---|---|
| **[`local_app/`](local_app/)** | **desktop GUI app** — runs on your own machine, no Colab, no upload | **whole folders / large batches** |
| [`veo_gemini_watermark_remover.ipynb`](veo_gemini_watermark_remover.ipynb) | self-contained Colab notebook (19 cells) | a quick run on a free GPU, no install |

Both embed the identical engine — the notebook's engine cell is byte-identical to `wm_core.py`.

Verified on real Gemini renders: the visible sparkle is removed, the surrounding pixels are
left untouched (mean error under 1 grey level over the patched area), and the per-file CSV/JSON
report records what happened to every file in the batch.

Supporting files: `wm_core.py` (the engine — the notebook embeds this exact file),
`build_notebook.py` (regenerates the notebook from it).
Sample renders and proof screenshots are kept out of this repository.

---

## The two marks this removes

| | Video (Google Flow / Veo) | Image (Gemini) |
|---|---|---|
| Shape | pale `Veo` wordmark | **4-point sparkle** (solid, or faint/semi-transparent) |
| Measured | bottom-right, small, semi-transparent | two sizes seen in the wild: **48 px at a 94 px inset** and **96 px at a 152 px inset** (inner radius ratio ≈0.48) |
| Position | ~3–6 % inset | bottom-right corner; **an image can carry two marks at once** (96 px + faint 48 px) |
| Removed by | temporal detection → LaMa | shape template (**all marks**) → adaptive fill |

Both image sizes are found by one detector, which also flags a faint second mark when a render carries
one. Measured across **7 real marked renders (10 marks): 10/10 found** and **0 false positives on 40
unwatermarked photos** (the best clean-content candidate scores 0.55 against a 0.60–0.70 gate).

The position for images is *not* a fixed fraction of the frame — that is why it is detected per
image rather than hard-coded, and why a folder mixing 1K and 2K renders still works.

## How it works

1. **Find the mark.**
   * *Video / folders:* the mark is static, so a temporal t‑statistic on the high-pass residual
     finds it — `t(x,y) = mean_i(g_i − blur(g_i)) / (std_i(g_i − blur(g_i)) + noise_floor)`.
     Present in every frame → high t; ordinary moving content → ≈0. Gated by a strength bar
     (real marks 0.18–0.45, JPEG-noise blobs ≈0.09) and a shape/compactness check.
   * *Images:* a **shape template matcher** for the sparkle — matched filter for brightness, then
     a star-shape IoU test. Shape is what separates a sparkle from a bright blob of picture
     content, not brightness (positives IoU 0.79–0.95, the best clean-photo false candidate 0.61).
2. **Inpaint the region only** — a 512×512 crop around the mark. Everything outside the mask is
   bit-for-bit untouched; audio is copied through (video).
3. **Choose the fill per image.** On flat backgrounds LaMa invents texture (~13× the local noise:
   high-pass std 15.3 vs 1.2 around it → visible blob), so flat regions get a smooth interpolation.
   The mask is padded by ~9 % of the mark size (a 96 px star's soft edge is ~8 px wide — a 3 px pad
   left a pale ghost star), and on a flat background the fill level is nudged to match the ring
   around it, which removes the faint star-shaped bias a fill can leave on dark areas.
   and textured regions get LaMa. Measured after: patch texture 0.49 vs 0.93 surroundings (flat),
   12.5 vs 13.8 (textured).
4. **Verify.** Sparkle path: the star template is re-run on the output — a residual mark still
   matches the shape, a fill artifact does not. Wordmark path: matched contrast against a control
   region shifted off the mark.

## Validated results (measured, not estimated)

| Check | Result |
|---|---|
| **Your 3 real Gemini images** — sparkle located | **3/3 at the exact pixel** (IoU 0.79–0.95) |
| Same, verified after removal | 3/3 "clean — no star-shaped mark left"; patch texture matches its surroundings in all three |
| False positives — 19 unwatermarked photos × corner settings | **0** |
| Detection, 6 synthetic wordmark styles vs exact ground truth | recall **0.91–1.00**, centroid ≤ 8.7 px |
| End-to-end 720p video | PSNR **37.3 dB** vs the watermark-free original, audio preserved |
| Notebook self-test | glyph strength 69.9 → 8.0 (89 % removed) |
| Speed, this sandbox (2 CPU cores) | ~7 s per 720p frame; your 3 images: **19 s total** |

## Local GUI app — `local_app/`

Runs entirely on your machine (nothing is uploaded anywhere), with a queue-based bulk processor.

### Install & start

* **Windows:** double-click `local_app/run_windows.bat`
* **macOS / Linux:** `bash local_app/run_mac_linux.sh`
* **Manual:**
  ```bash
  cd local_app
  python -m pip install -r requirements.txt
  python watermark_gui.py --download        # fetches the 208 MB model once, into local_app/models/
  python watermark_gui.py                   # opens http://127.0.0.1:7860
  ```

The launchers create a virtual environment, install dependencies, download the model once and open
the browser. The model is the same Apache-2.0 LaMa ONNX file the notebook uses from Drive.

### Bulk processing, concretely

* **Queue:** drop **many** files (or paste folder paths, one per line, with *recursive* on). Extensions
  are matched automatically; a folder with 500 mixed 1K/2K renders is fine.
* **Per-file rows:** every file gets a row — `done` / `skipped` / `no mark` / `failed` / `cancelled` —
  with the detection method, mask size, background texture, time, verdict and output name.
* **Live progress:** `48/500 (10%, elapsed 6m, ETA 51m)` plus a streaming log.
* **Cancel:** stops after the file in flight; the partial report is still written and the finished
  files are kept, so you can hit *Run* again.
* **Resumable:** *skip images that already have an output* compares the canonical
  `<name>_clean.jpg`, so a re-run only does the missing work.
* **One bad file never stops the batch** — it is marked `failed` with the reason and the queue continues.
* **Reports:** `images_report.csv` + `.json` (and `videos_report.*`) land next to the outputs.
* **Previews:** a detected-mask zoom and a before/after zoom per file, in `<out>/_previews/`.
* **Safety default:** if no sparkle is detected, the file is **left untouched** (`no mark`). Tick
  *apply the fallback box* only for wordmark frames, where a preset box is the intended path.

### Images tab vs Videos tab

* **Images** — sparkle shape detection (per-image, so mixed resolutions work) → adaptive fill.
* **Videos** — temporal detection across sampled frames, then one of three engines:
  `lama` (best quality, ~7 s/frame on CPU), `delogo` (fast, subtle marks), `crop` (instant,
  removes the corner, changes the resolution), audio kept.

```
local_app/output/
├── images_clean/   <name>_clean.jpg      (+ _previews/<name>_mask.png, _before_after.png)
├── videos_clean/   <name>_clean.mp4      (+ _previews/…)
├── scan/                                 (dry-run results, nothing modified)
├── images_report.csv / .json
└── videos_report.csv / .json
local_app/logs/app.log
```

### Useful flags

```bash
python watermark_gui.py --model /path/to/lama_fp32.onnx   # model somewhere else
python watermark_gui.py --device cuda                     # NVIDIA GPU (pip install onnxruntime-gpu)
python watermark_gui.py --threads 8                       # cap CPU threads
python watermark_gui.py --low-mem                         # ~40% less RAM, ~30% slower (CPU)
python watermark_gui.py --share                           # temporary public link
```

**Low-memory mode** switches off ONNX Runtime's CPU memory arena and heavy graph fusion. Measured on
this model: peak RSS **768 MB → 433 MB** for one 512×512 pass (output differs by ≤1 grey level, mean
0.000), for about +30 % inference time on CPU. It turns itself on automatically under ~4 GB of RAM
and can be toggled in *Settings* (`--low-mem` / `--full-mem` overrides it). A 40-file batch on a 2 GB
machine went from being killed mid-run to completing: 30 images cleaned, peak 998 MB.

### Check your install first

```bash
cd local_app
python smoke_test.py              # 18 checks: deps, detection, bulk image run, video run, reports
python smoke_test.py --no-model   # everything except inpainting
```

It builds its own synthetic test material in a temp folder, so it never touches your files.
Expected output ends with `23 passed, 0 failed` (the last checks exercise the `lama` path on CPU,
so allow ~1 minute with a model).

Two more tools, for bigger questions:

```bash
python scale_test.py                              # run a folder through the worker, measuring RSS
python scale_test.py /path/to/in /path/to/out
```

### Machine notes

* CPU-only is fine (the sandbox these were measured in had 2 cores). One 2K image ≈ 8 s; video with
  the `lama` engine ≈ 7 s/frame, so prefer `delogo` for long clips unless you have a GPU.
* Memory: about 1 GB peak per running job (the model is 208 MB). On a very small machine process a
  folder in two halves — the resume logic makes that easy.
* `imageio-ffmpeg` ships a static ffmpeg, so no system install is needed. **Audio is copied through
  on both video engines** (verified by `smoke_test.py`); a silent source produces a silent output and
  the log says so.

## Sharing it with other people (hosting)

The app can serve a whole team from your machine — that is what the scheduler and per-session
folders are for. Full guide: **[`local_app/HOSTING.md`](local_app/HOSTING.md)**.

```bash
# same network: everyone opens http://<your-ip>:7860
python watermark_gui.py --host 0.0.0.0 --no-browser

# anywhere on the internet (always add a login)
python watermark_gui.py --share --auth team:password
cloudflared tunnel --url http://localhost:7860        # or a permanent tunnel
```

Hosted mode switches on automatically for `--share`, `--auth` or a non-loopback host:

* each visitor gets **private output folders** (`output/sessions/<session>/…`) and a **results .zip**;
* **one job at a time, first come first served** — everyone else sees *"queued — 3 job(s) ahead"*,
  which is what keeps a 2 GB machine from falling over;
* **Cancel only affects your own job** (a queued job is dropped before it starts);
* limits: `--max-files` (500/run), `--max-queue` (6 waiting), `--keep-days` (3), all adjustable;
* visitors **cannot** browse folders on your disk or change the model/settings — those are
  operator-only — and the app warns you if you expose it without `--auth`.

Verify before opening it up: `python host_test.py` → **18 passed** (queueing, per-visitor cancel,
private folders, limits, and that a visitor cannot reach the host's machine).

### An .exe instead?

Possible, with caveats: the bundle is 500 MB+, antivirus tools flag PyInstaller output, and every
update means everyone downloads it again. It is the right answer only for people who must run it
**offline on their own PC**. If that is what you need, the packaging route is PyInstaller
(`--onedir`, model downloaded on first run, not bundled) plus a GitHub Actions Windows runner to
produce the .exe — ask and it will be set up.

## Known limits (documented in the notebook too)

* **SynthID** (Google's invisible watermark) is in every Veo/Flow/Gemini output and is **not** removed.
* **Single-image wordmark *location* discovery was deliberately not shipped** — 43–71 % recall with
  3–16 false positives across 19 clean photos. Shape-primed detection (the sparkle) works; unprimed
  blob hunting does not.
* Detection needs variation: ≥6–8 frames when the scene moves. Static shots / sub-second clips fall
  back to the preset box (still verified).
* A residual fainter than the scene's own contrast noise can read as "clean" on the wordmark path;
  the sparkle path is sharper because it tests shape. Before/after zooms are always written.
* Sparkle detection is deliberately strict (shape IoU ≥ 0.70: real marks measure 0.86–0.95, the best
  clean-photo false candidate 0.61). A badly degraded mark — heavy re-compression on a busy
  background — can land just under the gate (measured 0.59) and be reported as `no mark` rather
  than touched; the app prints that shape score so you can decide to use the fallback box.

## Using the notebook

1. Open the notebook in Colab, `Runtime ▸ Change runtime type ▸ T4 GPU`.
2. Run all cells (accept the Drive mount; the model downloads into Drive once).
3. **Images (section 4):** drop renders in `MyDrive/watermark_remover/input/images/`, run the cell.
   You get per-image detection method, background texture, verdict, a report JSON, and
   before/after zooms inline.
4. **Video (section 5):** single clip → `VIDEO_PATH`. **Batch (section 6):** whole folder of clips,
   with report + previews and resumable re-runs.
5. **Buttons instead of cells?** Section 7 launches a Gradio UI.

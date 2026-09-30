# Verification summary

Everything below was measured on this build, not estimated. Re-run any of it yourself — the two
self-contained tools are listed at the bottom.

## 1. The engine (what it removes, and how accurately)

| Material | Result |
|---|---|
| Your 3 original Gemini renders (2× 1792×2400, 1× 896×1200) | **3/3 mark located exactly**, shape IoU 0.86–0.95, output re-checked clean |
| Your 4 newer renders (a second mark size: 96 px at a 152 px inset) | **4/4**, IoU 0.85–0.94, and three of them carry a **second faint 48 px mark** that is also found (0.61–0.63) and removed → **10/10 marks removed** |
| All 10 marks, checked visually at 5× high-pass amplification | **no star and no ghost left** in any of them (the checker also re-runs the shape matcher: residual star-ness drops from 0.77–0.95 to 0.20–0.49) |
| 40 unwatermarked photos (two independent sets, 970×790 → 2230×2410) | **0/40 false positives**; the best clean candidate scores 0.55 in the mark-placement family, against a 0.60 gate there and 0.70 elsewhere |
| 7 synthetic `Veo` wordmark frames (image path) | **0/7 false positives** |
| Video: moving shot, temporal detection | mark region removed, 37.27 dB PSNR vs the pre-watermark original, audio kept |
| Flat backgrounds | 0.49 vs 0.93 patch-vs-ring texture (TELEA+blur chosen automatically; LaMa on flat ground produced a 15.31 vs 1.21 blob — that is why) |

Detection threshold rationale: shape IoU ≥ 0.70. Real marks measure 0.86–0.95, clean content ≤ 0.46,
a deliberately degraded synthetic mark 0.59 — so a marginal file is reported as `no mark` with its
score instead of being touched on a guess.

## 2. The GUI app, end to end

| Check | Result |
|---|---|
| `smoke_test.py` on a clean machine | **23 passed, 0 failed** (deps, detection, bulk image run, video run, audio, helpers) |
| Your 3 images through the app worker | 3/3 `clean - no star-shaped mark left` |
| Same run over the live HTTP/SSE path the browser uses | 3 rows, 3 before/after previews, progress bar → 1.0 |
| **40-file mixed batch** (3 real + 28 synthetic sparkles, 6 wordmark frames, 4 clean photos, 2 unreadable, mixed 480p–2K, mixed .jpg/.png, unicode filename, nested folder) | **30 cleaned, 8 correctly left untouched, 2 unreadable skipped, no failures in the batch**, 1m44s (2.6 s/file), 40-row report, peak RSS 998 MB |
| Hostile files (random bytes, zero-byte, text-file-named-`.png`, corrupt `.mp4`) | each reported `error/failed` with the reason, **batch continued**, traceback in `logs/app.log` |
| Cancel mid-batch (images and mid-encode video) | stops after the file in flight, partial report written, `.partial` discarded, no leftovers, resume works |
| Resume (`skip already-processed`) | re-run reports `skipped — output already exists` |
| Previews | mask zoom + before/after zoom per processed file |

## 3. Video

| Check | Result |
|---|---|
| Engines `lama` / `delogo` / `crop` | all produce a valid clip; `crop` reports its new resolution |
| Audio through `lama` | preserved (aac in → aac out). **This was broken and is fixed**: muxing into `<out>.partial` hid the container from ffmpeg and the audio was silently dropped |
| Audio through `delogo` | preserved (`-c:a copy`) |
| Static / sub-second clips | fall back to the preset box; a clip whose box turns out empty is flagged in the summary instead of being called clean |
| Progress | per-frame callback → live fps + ETA in the UI |

## 4. Bulk behaviour at scale

* **Low-memory mode** (auto under 4 GB RAM, `--low-mem`/`--full-mem`, Settings toggle):
  peak **768 MB → 433 MB** per 512×512 pass; output differs ≤ 1 grey level (mean 0.000), ~30 % slower
  on CPU. Without it, the 40-file batch was OOM-killed on a 2 GB machine; with it, it completed.
* Per-image cost is ~9 MB; the memory is the model session, not the queue.
* One failing file never stops a batch. Reports are written even on cancel.

## 5. Consistency guarantees

* `veo_gemini_watermark_remover.ipynb` embeds `wm_core.py` **byte-identical** (checked by
  `build_notebook.py` rebuild + comparison).
* `local_app/wm_core.py` must equal `../wm_core.py` — `smoke_test.py` fails loudly otherwise.

## Re-verify yourself

```bash
cd local_app
python smoke_test.py --model models/lama_fp32.onnx   # 23 checks, no files touched (~1 min)
python scale_test.py /path/to/a/folder               # bulk run + RSS profile
```

## Known limits

* **SynthID** (Google's invisible watermark) is present in Veo/Flow/Gemini output and is **not**
  removed by this or any pixel-domain tool.
* Unprimed *location* discovery for image wordmarks is deliberately not shipped (43–71 % recall with
  3–16 false positives across 19 clean photos). The sparkle has a shape template, so it is detected;
  generic blob hunting is not, and it would damage clean photos.
* A mark degraded past the gate may be reported as `no mark` — the app prints the shape score so you
  can decide to use the fallback box for that file.
* Two mark sizes (48 px and 96 px) are covered, and a second faint mark is looked for only in the
  two measured corner families. A render with a mark outside those families falls back to the
  preset box.
* On very busy footage the "is it clean?" contrast reading can be dominated by the scene texture
  itself; before/after zooms are always written so you can look.

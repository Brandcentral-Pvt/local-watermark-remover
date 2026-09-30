# Hosting it for other people

Your machine does the work; everyone else just opens a browser link. Nothing to install for them,
one model download for you, one place to update.

The app already handles the multi-user parts, so this is mostly a networking decision:

| for you | what it is | what it needs |
|---|---|---|
| **you + your own tabs** | default | nothing |
| **people on the same Wi‑Fi / office network** | LAN hosting | one command + a firewall rule |
| **anyone with a link, anywhere** | tunnel or Gradio share | the same command + a tunnel + **a login** |

---

## 1. Same network (recommended start)

```bash
python watermark_gui.py --host 0.0.0.0 --port 7860 --no-browser
```

`--host 0.0.0.0` means "accept connections from other machines". The app detects this and switches
itself into hosted mode (see §4). Then:

* **Find your address:** Windows `ipconfig` (look for *IPv4 Address*), macOS/Linux `ip addr` or
  `ifconfig`. It looks like `192.168.1.42`.
* **Send people:** `http://192.168.1.42:7860`

**Open the firewall** (once):

| OS | command |
|---|---|
| Windows (admin PowerShell) | `New-NetFirewallRule -DisplayName "Watermark Remover" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Private` |
| Linux (ufw) | `sudo ufw allow 7860/tcp` |
| macOS | System Settings → Network → Firewall → Options → allow incoming for Python |

If Windows shows the "Allow access?" dialog on first run, tick **Private networks** and Allow.

## 2. Anyone on the internet

Two options. Both need `--auth` — this app will happily burn 100 % of your CPU for a stranger.

```bash
# quick demo link (temporary, ~72 h, relayed through Gradio's servers)
python watermark_gui.py --share --auth team:pick-a-real-password

# permanent link on your own tunnel (free, keeps running as long as the command does)
python watermark_gui.py --port 7860 --no-browser --auth team:pick-a-real-password
cloudflared tunnel --url http://localhost:7860        # prints an https://....trycloudflare.com URL
```

* `cloudflared` is a single binary from Cloudflare — no account needed for the quick tunnel; a named
  tunnel with your own domain is a few more commands if you want a stable URL.
* `ngrok http 7860` does the same job.
* `--share` is fine for a demo, but the URL is temporary and the traffic goes through a third party.
* Behind nginx/Apache on a VPS, proxy to `127.0.0.1:7860` and keep `--auth` on anyway.

Tell people: **keep the tab open while a batch runs** (the queue is per browser session).

## 3. What you get from hosted mode (automatic)

Enabled when you use `--share`, `--auth`, or a non-loopback `--host` (force it with `--multiuser`):

* **Private results per visitor** — outputs go to `output/sessions/<session>/images_clean/…`, never
  mixed with anyone else's, and every job ends with a **results .zip** download button.
* **One job at a time, first come first served** — everyone else sees *"queued — 3 job(s) ahead of
  you"*. This keeps the machine responsive and memory flat instead of running 5 jobs in parallel.
* **Cancel is per visitor** — your Cancel stops your job, not someone else's. A job that is still
  waiting in the queue is dropped before it starts.
* **Limits** — `--max-files` (default 500 per run), `--max-queue` (default 6 waiting),
  `--keep-days` (default 3, older visitor folders are deleted at startup).
* **Visitors cannot touch your machine** — pasting a host folder path is refused (they upload
  files instead), the output path is ignored, and the Settings tab (model reload, device, memory
  mode) is operator-only. Only someone on the machine itself can change those.
* **A warning if you expose it without a login.**

## 4. Capacity — the honest numbers

Measured on a 2-core CPU (no GPU), 2 GB RAM, low-memory mode:

| work | time |
|---|---|
| one 2K image (mark + adaptive fill) | ~8 s |
| one 1080p video frame with `lama` | ~7 s |
| one 1080p video frame with `delogo` | ~30 fps (near realtime) |
| 40-image mixed batch | 1 m 44 s, peak 998 MB |

So a CPU-only box gives roughly **400 images/hour** and *one* user at a time in the queue. A CUDA
GPU (`pip install onnxruntime-gpu`, then `--device cuda`) is typically an order of magnitude faster
and is the only way "everyone uses it" feels good at scale.

Rules of thumb: <5 people doing occasional images → CPU is fine. A team of 10+ doing folders →
GPU, and set `--max-files 100` so one person cannot occupy the queue all afternoon.

Want more throughput without a GPU? Run a second instance on another port
(`--port 7861 --model ...`) — two processes, each with its own model session (mind the RAM).

## 5. Privacy and rules

* **Everything people upload is written to your disk** (`local_app/output/sessions/…` plus Gradio's
  temp upload dir). Say so, and keep `--keep-days` short.
* The app never uploads anything to a third party — except the `--share` mode, which relays through
  Gradio, and the model download from Hugging Face on first run.
* **Only remove watermarks from material you have the right to modify.** The visible mark going away
  does not remove **SynthID**, so Google can still identify the content as AI-generated, and some
  platforms require AI labels. Losing the visible mark does not make content yours.

## 6. Keep it running

* **Windows:** Task Scheduler → *At log on* → `python local_app\watermark_gui.py --host 0.0.0.0 --auth user:pass --no-browser`
  (or use `nssm` to install it as a service).
* **Linux (systemd):**
  ```ini
  [Unit]
  Description=Watermark Remover
  After=network.target
  [Service]
  User=you
  WorkingDirectory=/home/you/veo-watermark-remover/local_app
  ExecStart=/home/you/veo-watermark-remover/local_app/.venv/bin/python watermark_gui.py \
            --host 0.0.0.0 --port 7860 --auth user:pass --no-browser
  Restart=always
  [Install]
  WantedBy=multi-user.target
  ```

## 7. Before you open it up

```bash
cd local_app
python host_test.py          # 18 checks: queueing, per-visitor cancel, private folders,
                             # limits, and that visitors cannot reach your machine
python smoke_test.py         # 23 checks: engine, bulk run, video, audio
```

Run `host_test.py` again after changing limits or auth, and keep an eye on `logs/app.log`.

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| other machines cannot connect | firewall rule missing, or you used `--host 127.0.0.1` (the app prints a hint) |
| "queued" forever | someone else's job is running; check the queue line, or raise throughput with a GPU |
| both users' batches feel slow | that is the queue doing its job — one job at a time keeps memory flat |
| out of memory on a small host | low-memory mode is automatic under 4 GB; also lower `--max-files` |
| "no sparkle found … shape score 0.59" | degraded mark; the file is left untouched — see the score hint in the UI |
| someone's files are gone after a few days | `--keep-days` cleanup; raise it or download the .zip per job |

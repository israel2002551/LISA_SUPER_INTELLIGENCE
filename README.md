# LISA_SUPER_INTELLIGENCE

LISA is a self-evolving edge AI. An ESP32-S3 runs a compressed Gemma-style model from an external W25Q128 flash chip and controls hardware over UART. The cloud side learns from everything the device does and ships improved models back over the air.

```
ESP32-S3 + W25Q128  ──telemetry──►  Supabase Edge Function (lisa-api)  ──►  Supabase (Postgres + Storage)
        ▲                                                                      ▲      │
        │                     GitHub Actions: training worker every 6 h ──────┘      │
        └──────────── signed OTA model / delta (verified, trial-booted) ◄─────────────┘
```

| Folder | What it is |
|---|---|
| `supabase/functions/lisa-api/` | the device API as a Supabase Edge Function (production) |
| `server/` | Python: training worker (`worker.py`), the same API in FastAPI (`app/`, local dev / Render fallback), scripts, tests |
| `render.yaml` | optional: the FastAPI API on Render's free plan |
| `.github/workflows/lisa-evolution.yml` | the training worker, on GitHub Actions (free for public repos) |
| `supabase/migrations/` | Database schema, storage buckets, role grants (run in order) |
| `edge/` | ESP32-S3 firmware: UART discovery, A/B model slots, ring log, OTA, telemetry |
| `gemma4/` | Inference engine, model format, W25Q128 driver, model image tools |
| `tools/fake_peripheral.py` | Simulated UART device for testing the board without real hardware |
| `docs/` | `ARCHITECTURE.md` (how it works) and `BRINGUP.md` (step-by-step testing) |

## Deploy for free

Everything runs on free tiers:

| Job | Where |
|---|---|
| API | Supabase Edge Function, always on |
| Training | GitHub Actions, free for public repos |
| Database + storage | Supabase |

1. **Supabase database:** in the SQL Editor, run the three files in `supabase/migrations/` in order.
2. **Supabase Edge Function:**
   - **Edge Functions → Deploy a new function → Via Editor**, name it **`lisa-api`**, paste `supabase/functions/lisa-api/index.ts`, and deploy.
   - In its settings, turn **off** JWT verification. Boards authenticate with their own node key, not a Supabase JWT.
   - Optional: add an `ADMIN_TOKEN` secret to enable the admin routes. `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are provided automatically.
   - CLI alternative: `supabase functions deploy lisa-api --no-verify-jwt`.
3. **Check:** `https://<project-ref>.supabase.co/functions/v1/lisa-api/healthz` returns `{"ok": true, "ai": "LISA", ...}`. Put `https://<project-ref>.supabase.co/functions/v1/lisa-api` into `EDGE_SERVER_BASE_URL` in the device's `edge/secrets.h`.
4. **GitHub Actions:**
   - **Settings → Secrets and variables → Actions:** add the repository secrets `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and `OTA_SIGNING_KEY_PEM`. The last is the full contents of your OTA private key, the one whose public key is in the device's `edge/secrets.h`.
   - Then **Actions → LISA evolution → Run workflow** to train now. It also runs every 6 hours on its own.

**Notes:**
- **Render (optional):** `render.yaml` runs the same API (FastAPI) as a free Render web service. Free Render hours are shared across a workspace and can run out mid-month, which suspends the service.
- **Training on your PC instead:** `cd server && python worker.py --once --trigger`. It reads `server/.env` and takes ~3 minutes on a CPU. Don't run it while the GitHub workflow is running.
- **Optional:** `TEACHER=claude` + `ANTHROPIC_API_KEY` lets Claude label data and classify unknown UART devices.

## Firmware

```bash
cp edge/secrets.example.h edge/secrets.h     # then fill it in (never commit it)
python -m esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x210000 tokenizer.bin
pio run -e edge_node -t upload --upload-port COM3
python gemma4/w25q128_upload.py COM3 <model image>.w25img
```

## Tests

```bash
cd server
pip install -r requirements-dev.txt
python -m pytest -m "not slow"     # contracts with the firmware, API, pipeline, signing, deltas
python -m pytest -m slow -s        # trains LISA on CPU and checks it learned (~3 min)
```

## Secrets

Nothing secret lives in this repository, and `.gitignore` blocks the usual places:
- `server/.env`
- `server/secrets/`
- `edge/secrets.h`
- `*.pem`

The Supabase service-role key and the OTA private key belong only in Render's environment and your local `server/.env`. Anyone holding the OTA private key can push models to every device.

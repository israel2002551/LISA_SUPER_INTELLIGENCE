# LISA Bring-Up & Testing Guide

Bring the system up one layer at a time. Each step has a check that must pass before the next one. Commands assume Windows with the repo at `c:\Users\USER\VIM\vim`, and `COM3` for the ESP32-S3 (its **UART** USB port).

| Step | Needs | Proves |
|---|---|---|
| 1 | Python | server logic, contracts with the firmware, real training |
| 2 | Python | the HTTP API end to end, no hardware, no cloud |
| 3 | Supabase project | schema, storage, idempotent writes |
| 4 | ESP32-S3 + W25Q128 | flash driver, A/B slots, provisioning, rollback |
| 5 | + USB-UART adapter | auto-baud, IDENTIFY, bus upgrade, actions, ring log |
| 6 | + Wi-Fi | telemetry into Supabase |
| 7 | all of the above | evolution → OTA delta → trial → commit/rollback |
| 8 | Render | production deployment |

---

## 1. Server tests (no hardware, no cloud)

```bash
cd server
pip install -r requirements-dev.txt
python -m pytest -m "not slow"     # 39 tests, ~10 s
python -m pytest -m slow -s        # real CPU training, ~3 min
```
`test_contract.py` checks that the server and the firmware agree on model shape, image header, slot size and prompt budgets.

## 2. Local dev server + simulated node

```bash
cd server
uvicorn app.main:create_app --factory --port 8000          # dev mode: in-memory DB, files in server/.data
# second terminal:
python scripts/register_node.py --server http://localhost:8000 --node-id edge-01
python gemma4/gemma4_slice_int16.py                           # (from the repo root) builds the v0 image if needed
python scripts/bootstrap_model.py --server http://localhost:8000 --image ../cluster_binaries/gemma4_w25q128_int16.bin
python scripts/simulate_node.py --server http://localhost:8000 --node-id edge-01 --api-key <key> --events 300
curl -X POST localhost:8000/api/v1/evolution/trigger -H "Authorization: Bearer dev-admin-token" \
     -H "Content-Type: application/json" -d "{\"trainer\":\"torch\",\"steps\":200}"
curl localhost:8000/api/v1/evolution/runs -H "Authorization: Bearer dev-admin-token"
```
**Check:** the run ends `published`, with `candidate.exact_match` above `baseline.exact_match`. If the gate rejects it, the run says why. Use `{"force_publish":true}` with the mock trainer to exercise the publish path. In dev mode the server generates `server/.data/ota_signing_key.pem` and logs its public key.

## 3. Supabase

1. Create a project. Apply the migrations, either with `supabase link` + `supabase db push`, or by pasting both files from `supabase/migrations/` into the SQL editor in order.
2. **Check (SQL editor):** `select * from storage.buckets;` shows `models` and `deltas`, both `public = false`.
3. Create `server/.env` from `.env.example`:
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
   - a real `ADMIN_TOKEN`
   - `OTA_SIGNING_KEY_PEM` (`python scripts/gen_signing_key.py`)
4. Repeat step 2 against this server.
5. **Check:**
   - rows appear in `edge_nodes`, `uart_devices`, `interaction_logs` and `model_versions`, and `v1.w25img` appears in the `models` bucket;
   - running `simulate_node.py` again with the same `--start-seq` inserts **0** new rows (idempotent).

## 4. Edge firmware: flash, slots, provisioning

Wiring (yours): W25Q128 `DI`→GPIO11, `CLK`→GPIO12, `DO`→GPIO13, `CS`→GPIO10. `WP#` and `HOLD#` go to 3.3 V, and `VCC` is 3.3 V.

```bash
python -m esptool --chip esp32s3 -p COM3 -b 921600 write_flash 0x210000 tokenizer.bin
pio run -e edge_node -t upload --upload-port COM3        # no secrets.h yet: runs offline
pio device monitor -e edge_node                           # 921600 baud
```
**Check (boot log):**
- `[W25Q128] JEDEC ID 0xEF4018, 16777216 bytes`
- `[Slots] No boot control record; scanning slots.`
- `[Ring] 1008 sectors ...`
- `No usable model`

Provision the versioned image from step 2 or 3 (close the monitor first):
```bash
python gemma4/w25q128_upload.py COM3 cluster_binaries/gemma4_w25q128_int16.v1.w25img
```
**Check:** the upload ends `LOADED into slot B; rebooting`. After the reboot, `/status` shows the active slot `good`, `model_version: 1` and `selftest_ok: true`.

**Rollback test:**
1. Provision a second image; it goes to the other slot.
2. Run `/rollback`: the node reboots into the first one and marks the second `bad`.
3. Power-cycle mid-upload. The node must still boot the previous good slot.

## 5. UART: auto-baud, handshake, bus upgrade

Wire a 3.3 V USB-UART adapter: adapter TX→GPIO18, adapter RX→GPIO17, GND→GND.

```bash
python tools/fake_peripheral.py COM5 --baud 115200          # smart thermostat
```
**Check (edge monitor):**
- `[UART1] Identified 'thermo-01' (thermostat) at 2000000 baud, link uart_fast` (if your adapter can't do 2 Mbaud you'll see `upgrade_failed_reverted` instead, which also exercises the fallback).
- The fake peripheral prints the actions it receives and its own good/bad verdicts.
- `/ring 10` on the edge console shows `handshake`, `device` and `interaction` records.

More cases:
- `--baud 9600` or `--baud 921600`: auto-baud must find each one.
- `--mode raw-nmea --baud 9600`: `Unidentified device ... format guess 'nmea'`, observations logged, **no commands sent**.
- `--mode raw-bin`: format guess `modbus_rtu`.
- Stop the fake device: after ~30 s, `stopped answering; rediscovering`.

With the random v1 weights most outputs are `invalid_output` or `nop`. That's expected: those logs are what the server learns from.

## 6. Network + telemetry

1. Copy `edge/secrets.example.h` to `edge/secrets.h` and fill in Wi-Fi, `EDGE_SERVER_BASE_URL`, and the `EDGE_NODE_ID`/`EDGE_NODE_API_KEY` that `register_node.py` printed.
2. Add the OTA public key (`python server/scripts/gen_signing_key.py --show-public key.pem`).
3. **TLS roots:**
   - Run `openssl s_client -connect <host>:443 -showcerts </dev/null`. Read the issuer of the last certificate shown (e.g. *ISRG Root X1*, *GTS Root R1/R4*), and download that root's PEM from the CA (letsencrypt.org / pki.goog).
   - Do this for both your Render host and `<project>.supabase.co`, and paste all PEMs into `EDGE_TLS_ROOT_CA`.
   - **LAN testing against a laptop:** use `#define EDGE_ALLOW_PLAIN_HTTP` instead, with `PUBLIC_BASE_URL=http://<laptop-ip>:8000` and the server started with `--host 0.0.0.0`.
4. Rebuild and upload: `pio run -e edge_node -t upload`.

**Check:**
- `[Net] Uploaded events up to seq N` on the device;
- new `interaction_logs` rows, and `edge_nodes.last_heartbeat_at` updating;
- unplug Wi-Fi, generate events, reconnect: the backlog uploads from the ring with no gaps (`status.ring.lost` stays 0).

## 7. Evolution → OTA → trial → commit

```bash
curl -X POST <server>/api/v1/evolution/trigger -H "Authorization: Bearer <ADMIN_TOKEN>" \
     -H "Content-Type: application/json" -d "{\"trainer\":\"torch\"}"
```
Then run `/ota` on the device console, or wait 5 minutes.

**Check (device):**
1. `[OTA] download vN delta`
2. `Slot X now active (model vN, trial)`
3. a reboot, then `[Infer] Self-test passed`
4. about 60 s later, `Committed slot X`

**Check (server):** `edge_nodes.active_model_version = N`.

**Rollback paths:**
- **Server-side:** `POST /api/v1/admin/models/1/activate`, and devices converge back to v1.
- **Device-side:** a model that fails its self-test, or reboots 3 times before committing, is rolled back automatically.

## 8. Render

1. Push the repo, including `tokenizer.bin`, or let `bootstrap_model.py` upload it to storage.
2. In Render: **New → Blueprint** and select the repo. `render.yaml` creates `lisa-api` (starter plan) and `lisa-worker` (standard plan, CPU torch).
3. Fill the `lisa` env group: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` and `OTA_SIGNING_KEY_PEM`. `ADMIN_TOKEN` is generated.
4. **Check:**
   - `https://<api>.onrender.com/healthz` returns `dev_mode: false`;
   - a triggered run is picked up by the worker (its logs show `running queued evolution run`).
5. **Optional:**
   - `TEACHER=claude` + `ANTHROPIC_API_KEY` for LLM labels and stream classification;
   - `TRAINER=remote` + `REMOTE_TRAINER_URL` for an external GPU worker (contract in `server/app/evolution/trainer.py`).

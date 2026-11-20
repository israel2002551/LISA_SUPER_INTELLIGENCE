# LISA: Self-Evolving Edge AI Architecture

Three parts form one learning loop:

```
 ┌──────────── ESP32-S3 edge node (edge/) ────────────┐        ┌──────── Render (server/) ────────┐
 │ Core 1  inference_task   Gemma INT16 in PSRAM       │ HTTPS  │ lisa-api (FastAPI)      │
 │ Core 0  uart_task        auto-baud, IDENTIFY,       │───────►│  POST /telemetry   GET /ota/check │
 │                          bus upgrade, actions       │◄───────│  POST /uart/classify  admin/*     │
 │         ring_task        logs -> W25Q Partition C   │        │                                   │
 │         net_task         telemetry, OTA, classify   │        │ lisa-worker             │
 │                                                     │        │  curate -> synthesize -> train    │
 │ W25Q128 on SPI2: control | slot A | slot B | ring   │        │  -> gate -> quantize -> sign      │
 └───────────────▲─────────────────────────────────────┘        └──────────────┬───────────────────┘
                 │ signed URL download (model or delta)                         │ service_role
                 │                                    ┌────────── Supabase ─────▼──────────────────┐
                 └────────────────────────────────────│ Postgres: edge_nodes, uart_devices,         │
                                                      │ interaction_logs, training_datasets,        │
                                                      │ model_versions, evolution_runs              │
                                                      │ Storage:  models/  deltas/  (private)       │
                                                      └─────────────────────────────────────────────┘
```

**The loop.** The device acts on UART events with its local model. Every outcome is logged: the peripheral's ack, a reject, a timeout or invalid output, and optional feedback. The server turns those outcomes into training data and fine-tunes. It publishes the new model only if it beats its parent on held-out data. Devices download the new model (a block delta when possible), check its signature and hash, and boot it on trial. They keep it only if it stays healthy, and otherwise roll back on their own.

---

## 1. File tree

```
edge/                          ESP32-S3 firmware   (PlatformIO env: edge_node)
  edge_main.cpp                setup, task creation, boot-slot selection, USB console
  edge_config.h                pins, timings, task priorities, prompt contract
  secrets.example.h            copy to secrets.h: Wi-Fi, server URL, node key, CA, OTA public key
  w25q_layout.h                W25Q128 partition map
  slot_manager.h               A/B slots, ping-pong boot control, trial boot, rollback, SHA-256
  ring_log.h                   Partition C circular log (power-safe, queued writes)
  uart_link.h                  auto-baud, IDENTIFY handshake, raw analyzer, UART upgrade, action loop
  action_schema.h              manifest parsing, prompt builder, action validator (safety gate)
  inference_task.h             Core 1 worker around the engine
  net_sync.h                   Wi-Fi, telemetry, OTA (full + delta), cloud classify
  ota_verify.h                 ECDSA P-256 manifest signature check (mbedtls)
  edge_messages.h              queue messages between UART and inference tasks
gemma4/
  gemma4_engine.h              shared inference engine (loader, forward pass, sampler, tokenizer)
  gemma4_w25q128_config.h      model shape + image header (the contract with the server)
  w25q128_flash.h              SPI2 driver for the external W25Q128 + serial upload
  sha256_util.h                streaming SHA-256
  gemma4_w25q128_cluster_main.cpp   the earlier serial-chat build (same engine)
supabase/migrations/
  20261120000100_core_schema.sql    tables, indexes, FKs, RLS, activate_model_version()
  20261120000200_storage_buckets.sql models + deltas buckets
server/                        Render service
  app/main.py                  app factory (Supabase or in-memory dev mode)
  app/config.py                settings (env vars)
  app/security.py              node API keys (hashed), admin token
  app/ingest.py                telemetry -> rows (idempotent)
  app/uart_analysis.py         heuristic stream classifier
  app/routers/device.py        /telemetry, /uart/classify, /ota/check
  app/routers/admin.py         /evolution/*, /admin/* (nodes, bootstrap, activate, rules, tokenizer)
  app/routers/blobs.py         dev-mode signed downloads from local disk
  app/db/                      Repository/BlobStore interfaces; Supabase (httpx) + in-memory
  app/evolution/
    model_spec.py              shape/contract constants (tested against the C headers)
    tokenizer.py               exact port of bpe_tokenizer.h
    prompt_format.py           prompt/target contract
    action_schema.py           twin of edge/action_schema.h
    image_codec.py             INT16 image <-> float weights, budget checks
    model.py                   PyTorch twin of the firmware forward pass (+ LoRA)
    trainer.py                 MockTrainer | TorchTrainer | RemoteTrainer (GPU hook)
    teacher.py                 RuleTeacher | ClaudeTeacher (labels + stream classification)
    curator.py                 outcomes -> rewards -> examples
    synth.py                   rule-boundary + safety (NOP) synthetic data
    delta.py                   W25D block deltas
    signing.py                 ECDSA manifest signing
    pipeline.py                one evolution run, gate, publish, bootstrap
  worker.py                    background runner (queue + threshold/schedule policy)
  scripts/                     gen_signing_key, register_node, bootstrap_model, simulate_node
  tests/                       39 fast tests + 1 slow real-training test
tools/fake_peripheral.py       smart/raw UART device simulator for bring-up
render.yaml                    Render blueprint (api + worker)
```

## 2. Edge node

### Tasks and cores
| Task | Core | Priority | Job |
|---|---|---|---|
| `uart_task` | 0 | 10 | discovery, handshake, upgrade, events, actions, acks |
| `ring_task` | 0 | 4 | the only writer to Partition C |
| `net_task` | 0 | 3 | Wi-Fi, telemetry, OTA, classify |
| GEMM helper | 0 | 1 | Core 0 half of every matmul |
| `inference_task` | 1 | 5 | the only user of the engine |
| `loop()` | 1 | 1 | USB console, trial commit |

The GEMM helper has the lowest priority on Core 0. Busy I/O therefore slows inference down, but inference never delays I/O. The UART task never touches flash: log records go through a queue, so a long OTA erase can't stall it.

### W25Q128 layout (`edge/w25q_layout.h`)
| Range | Size | Use |
|---|---|---|
| `0x000000` / `0x001000` | 2 × 4 KiB | boot control, ping-pong (higher valid `seq` wins) |
| `0x010000` | 6 MiB | Partition A: model slot 0 |
| `0x610000` | 6 MiB | Partition B: model slot 1 |
| `0xC10000` | ~3.9 MiB | Partition C: ring log (1,008 sectors) |

The model itself runs from PSRAM (the W25Q is on SPI2 and can't be memory-mapped). The slots are persistent storage, and are only read at boot.

### Slot lifecycle
`EMPTY → (download, SHA-256 verified) → TRIAL → GOOD`.
- **Commit:** a trial slot is committed after `EDGE_TRIAL_COMMIT_MS` (60 s) of uptime with a passing self-test.
- **Rollback:**
  - after `EDGE_MAX_TRIAL_BOOTS` (3) boots without a commit;
  - on a load or SHA failure;
  - on a failed self-test;
  - on operator `/rollback`;
  - when the server activates an older version.
- **Atomic switching:** a new control record goes to the *other* sector with `seq+1` and a CRC. A power cut mid-write leaves the old record valid.

### UART protocol (device side: `tools/fake_peripheral.py`)
```
host -> {"cmd":"IDENTIFY","host":"edge-01","proto":1}                  (sent at 9600, 115200, 921600, 2000000)
dev  -> {"id":"thermo-01","type":"thermostat","emits":["temp"],
         "accepts":[{"cmd":"set_fan","args":{"level":"int:0..3"}}],
         "needs":["alerts"],"acks":true,"bus":{"max_baud":2000000,"links":["uart"]},
         "rules":[{"when":{"ev":"temp","gt":30},"do":{"cmd":"set_fan","args":{"level":2}}}]}
host -> {"cmd":"UPGRADE","link":"uart","baud":2000000}     dev -> {"ack":"UPGRADE","ok":true}
both switch; host -> {"cmd":"PING","n":123}                dev -> {"pong":123}
         (the device reverts to the old baud if no PING arrives within 1 s)
dev  -> {"ev":"temp","v":31.2}                             an event: goes to the model
host -> {"cmd":"set_fan","args":{"level":2},"id":42}       validated action
dev  -> {"ack":42,"ok":true}       dev -> {"fb":"good","id":42}   (feedback is optional)
```
- **No IDENTIFY reply:** the node listens passively at each baud rate. It scores 0x55 sync bytes, printable ratio and framing errors, then guesses the format (`json_lines`, `nmea`, `csv`, `ascii_lines`, `modbus_rtu`, `binary`). It logs samples and asks `/uart/classify`, but never sends commands to that device.
- **Idle identified device:** if it goes quiet, the node pings it and rediscovers after two missed PONGs.
- **ESP-NOW / TCP / SPI upgrades:** these are a marked hook (`try_fast_link_upgrade`). The node stays on UART until that's implemented.

### Prompt contract (`edge_config.h` ↔ `server/app/evolution/prompt_format.py`)
```
prompt = "<d>" + device_key + ":" + device_type + "<e>" + event[:120 bytes, CR/LF -> space] + "<a>"
tokens = [BOS] + bpe(prompt)[:79]              (80-token prompt budget)
output = greedy decode, up to 48 tokens, first line
target = "NOP"  |  {"cmd":"...","args":{...}}  (compact JSON, args sorted)
```
`tests/test_contract.py` parses the C headers and fails if either side drifts.

### Safety
- **Validation:** model output is never sent to hardware unless it is `NOP` or a command from the device's own manifest. Every argument must be present and in range. Anything else is logged as `invalid_output`, which becomes training signal.
- **Unidentified devices:** these never receive commands.
- **OTA manifests:** signed (ECDSA P-256 over `W25OTA1|version|size|sha256`), and the downloaded image must match the signed SHA-256. A compromised bucket or URL can't install weights.

## 3. Supabase

| Table | Holds |
|---|---|
| `edge_nodes` | node id, hashed API key, W25Q info, active slot/model, partition state, last heartbeat |
| `uart_devices` | per node + device key: baud, bus, format, manifest, inferred intent, protocol schema, learned rules |
| `interaction_logs` | every uploaded ring record; unique `(node_id, device_seq)` makes re-uploads idempotent; `reward`, `used_for_training` |
| `training_datasets` | prompt → target, `source_type` (`real_log` / `synthetic_distill` / `failure_recovery`), stable 90/10 `split`, `dedupe_key` |
| `model_versions` | version, sha256, signature, storage path, optional delta (from version, path, sha), metrics, `is_active` (at most one) |
| `evolution_runs` | status, trigger, stats (curation counts, baseline vs candidate metrics, gate decision), produced version |

- **Access:** RLS is enabled with no policies, so only the server's `service_role` key can read or write.
- **Activation:** `activate_model_version()` switches the active model in one transaction.
- **Storage:** both buckets are private, and devices download through 1-hour signed URLs.

## 4. Evolution loop (`server/app/evolution/pipeline.py`)

1. **Curate.**
   - **Reward from outcomes:** `ok` 1.0, `timeout` 0.3, `rejected` 0.1, `invalid_output` 0.0. Peripheral feedback overrides this.
   - **Rule match:** a matching rule (from the device manifest or an operator's `learned_rules`) beats the hardware outcome.
   - **No rule:** where no rule applies, a well-rewarded real outcome wins. That's how the model learns behavior beyond its rules.
2. **Synthesize.**
   - **Threshold sweeps:** events swept around each rule's threshold, where a small model is most likely wrong.
   - **Safety examples:** NOP for events no rule covers, and for raw devices.
3. **Train.**
   - `TorchTrainer` fine-tunes the exact firmware architecture on CPU (full or LoRA, merged).
   - The token table is frozen by default, which saves ~2 MiB per OTA delta.
   - Examples whose target can't fit the 48-token budget are skipped, not truncated.
4. **Gate.** The candidate must beat its parent on the same validation split (exact match, or equal exact match with ≥2 % lower loss). A failed gate means `rejected`: nothing ships.
5. **Quantize.** The candidate is quantized to the INT16 image and checked against the 6 MiB slot and the PSRAM budget.
6. **Publish.** The server uploads `v{n}.w25img`. It also builds a W25D block delta against the parent if that's ≤70 % of the full size. It signs the manifest and activates the version.

**Teachers.** `RuleTeacher` is free and deterministic. `ClaudeTeacher` (`TEACHER=claude`) asks `claude-opus-5-5` only when no rule decides:
- answers come back as structured JSON;
- an answer is kept only if it passes manifest validation and has confidence ≥0.6;
- calls are capped at `TEACHER_MAX_CALLS_PER_RUN`;
- refusal fallback goes to `claude-opus-4-8`.

It also helps `/uart/classify` when the heuristics aren't confident.

## 5. Measured so far, and known limits

**Measured (in this repo's tests):**
- From random weights, 400 CPU training steps on 400 thermostat examples raised held-out exact match from 0.00 to 0.85, with 100 % valid actions.
- The behavior survives INT16 quantization.
- An end-to-end HTTP run (register → bootstrap → 226 simulated records → evolution) published a model the OTA path delivered as a verified delta.

**Not measured:** anything on real hardware. Inference speed, Wi-Fi/TLS behavior, actual OTA timing and UART timing on the ESP32-S3 all need the bring-up in `BRINGUP.md`.

**Limits:**
- **Speed:** the model is tiny (2.75M parameters), and a compact JSON action is ~35 tokens, each a full forward pass on the device. A shorter action syntax would cut that by ~3×.
- **Fast links:** ESP-NOW/TCP/SPI bus upgrades are a hook, not an implementation.
- **Worker:** run one worker only; runs aren't locked.
- **TLS:** the device needs the root CA PEMs for your Render and Supabase hosts in `secrets.h` (see `BRINGUP.md`).

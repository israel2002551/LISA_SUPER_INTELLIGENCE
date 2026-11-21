# LISA: Self-Evolving Edge AI Architecture

Three parts form one learning loop:

```
 ┌──────────── ESP32-S3 edge node (edge/) ────────────┐        ┌── Supabase Edge Function ─────────┐
 │ Core 1  inference_task   Gemma INT16 in PSRAM       │ HTTPS  │ lisa-api (or the FastAPI server)  │
 │ Core 0  uart_task        auto-baud, IDENTIFY,       │───────►│  POST /telemetry   GET /ota/check │
 │                          bus upgrade, actions       │◄───────│  POST /uart/classify  admin/*     │
 │         ring_task        logs -> W25Q Partition C   │        ├── GitHub Actions (or your PC) ────┤
 │         net_task         telemetry, OTA, classify   │        │ worker.py --once, every 6 h       │
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
  20261120000300_service_role_grants.sql  table privileges for service_role only
supabase/functions/lisa-api/index.ts    the device API as an Edge Function (TypeScript port of
                                        server/app/routers/device.py + ingest.py + uart_analysis.py)
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
render.yaml                    Render blueprint (free web service: the API)
.github/workflows/             (LISA repo) lisa-evolution.yml: the training worker on GitHub Actions
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
device:  prompt = "<d>" + device_key + ":" + device_type + "<e>" + event[:120 bytes, CR/LF -> space] + "<a>"
         target = "NOP"  |  {"cmd":"...","args":{...}}  (compact JSON, args sorted)
chat:    prompt = "<u>" + text[CR/LF -> space, trimmed, :150 bytes] + "<l>"
         target = one-line reply
tokens = [BOS] + bpe(prompt)[:79]              (80-token prompt budget)
output = greedy decode, up to 48 tokens, first line
```
`tests/test_contract.py` parses the C headers and fails if either side drifts.

**Tokenizer.** `lisa_tokenizer.bin` is a 4096-entry BPE vocabulary built by `tools/build_lisa_tokenizer.py`:
- **Entries:** byte fallbacks, every printable ASCII character, and the 3742 highest-priority ASCII merges of the Llama-2 tokenizer.
- **On the device:** it is built into the firmware (`edge/lisa_tokenizer_data.h`, checked against the file by the tests).
- **Each image records its tokenizer** with `W25_MODEL_FLAG_TOK_LISA`. The firmware and the server use the matching one, so models v1–v3 keep their byte tokens.
- **Why it was needed:** the first 4096 entries of the Llama tokenizer (`tokenizer.bin`) contain no single characters. Every character fell back to a byte token, and no merge ever happened. The new vocabulary needs 2–2.4× fewer tokens for the same text.

**Memory.** For follow-ups only (`FOLLOWUPS` in `prompt_format.py`, the same list in `edge/action_schema.h`: "say that again", "tell me more", "why", ...), her last reply is prepended: `"<l>" + reply + "<u>" + text + "<l>"`, if that fits 160 bytes. It expires after 10 minutes or on `/new`. The corpus trains "say that again" (repeat it) and "tell me more" (per-topic follow-ups) after each answer.

Why only follow-ups: we first gave context to every message. On the board, 4 of 12 ordinary questions then got her previous reply copied back (`who are you?` → her "how are you" answer), and the gate missed it because it sampled the oldest validation rows. Now:
- every other message is single-turn;
- the gate samples each task's validation set with a stable shuffle.

**Chat.** Anything typed on the USB console that isn't a `/command` is a chat message (`/ask <text>` also works). `/good` and `/bad` rate her last reply.
- **Corpus:** `server/app/evolution/chat_corpus.py` holds about 80 intents. Each has many phrasings and one reply; out-of-scope questions get "I don't know that yet".
- **Variants:** each phrasing is expanded with case, punctuation and "lisa, ..." forms.
- **Validation:** whole phrasings are held out for validation, so the score measures understanding of new wordings.
- **Mixed training:** the corpus trains alongside the device examples. The gate rejects a candidate if either task loses more than 0.05 exact match.

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
   - **Chats** (`curate_chats`) are logged with no device and status `nop`. Who decides what she should have said, in order:
     1. **You:** `/good` keeps her reply. `/bad` asks for a better one from the corpus or the teacher.
     2. **The corpus,** when it covers the message. It's free and consistent with her training.
     3. **The teacher model** (`TEACHER=claude`). `ClaudeTeacher.review_chats` judges each remaining reply and writes the one she should have given.

     With a teacher, the corpus's "I don't know that yet" doesn't count as an answer, so the teacher can teach her the fact. When a real conversation and the corpus give the same prompt different targets, the real conversation wins and the newest correction wins (`resolve_chat_conflicts`).
2. **Synthesize.**
   - **Threshold sweeps:** events swept around each rule's threshold, where a small model is most likely wrong.
   - **Safety examples:** NOP for events no rule covers, and for raw devices.
   - **Chat:** the corpus, plus `TEACHER_LESSONS_PER_RUN` new topics written by the teacher model (`teach_lessons`; one API call per run).
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

**Measured on hardware (ESP32-S3 N16R8 + W25Q128, node `lisa-01`, simulated thermostat via `/sim`):**
- **First full learning cycle:** v1 (random weights) scored 0/24 on random thermostat events between 0 and 45 °C. v3 scored 24/24, including 29.7, 30.8 and 14.1 °C, just inside each threshold. The steps were: board events → Supabase → CPU training → signed delta OTA → trial boot → commit.
- **Training recipe:** 1200 steps at lr 1e-3 with warmup plus cosine decay, on 400 synthesized examples per device spread across the rules' whole range. Measured on held-out examples:

  | Setup | Held-out exact match |
  |---|---|
  | This recipe | 0.97 (35 examples) |
  | Constant lr 3e-3 | 0.00–0.34; garbled JSON |
  | Cosine at 3e-3 | 0.57–0.66; memorizes |
  | 68 examples clustered at the thresholds (v2) | "NOP" for everything |

  The gate now needs at least 20 validation examples, because v2 passed on 4.
- **Inference (byte tokens, v1–v3):** `NOP` took about 4 s; a JSON action (~36 tokens) took 8–11 s.
- **Chat and the new tokenizer (v4):** trained from scratch on 2254 examples (chat corpus plus thermostat) for 3000 steps at lr 1e-3 with cosine decay and trainable embeddings, in 12 minutes on a desktop CPU.
  - **Held-out score:** thermostat 1.00; chat 0.42 exact on phrasings never seen in training.
  - **On the board:** a 14-turn scripted conversation got 14/14 replies right, at 2.3–5.6 s per reply.
  - **Thermostat after chat training:** still 24/24; a fan command now takes ~5 s instead of 9–10 s.
  - **Calibration, held-out chat scores:** frozen embeddings gave 0.45 but sometimes blended replies into nonsense; trainable embeddings gave 0.43 with always-whole replies; lr 2e-3 gave 0.35; 6000 steps gave 0.41 (memorizes).
  - **Known limit:** unusual wordings get a coherent reply that is right about 40 % of the time.
- **Learning from conversation (v5):** the first chat-curation run took 7 logged turns.
  - **Kept** 2 (her lessons agreed). **Corrected** 4: 3 by her lessons and 1 by a `/bad`.
  - **Skipped** 1 ("do you like pizza": needs the teacher model).
  - **Training:** fine-tuned from v4 for 3000 steps with frozen embeddings, published as a 3.4 MB delta.
  - **On the board:** 12/14 in a scripted conversation with follow-ups. It fixed "what is your favorite food" (wrong on v4). "say that again" and "tell me more" work after jokes and about herself. Thermostat 12/12.
- **OTA:** the v2→v3 delta was 3.4 MB of a 5.5 MB image. Erasing the 5.4 MB staging area took 14.8 s, longer than the 5 s task watchdog, so erase, patch and verify feed the watchdog from the net task (`edge/wdt_scope.h`). Model load from the W25Q into PSRAM takes 2.2 s.

Not yet measured on hardware: a physical UART peripheral (only the simulator so far) and ESP-NOW/TCP links.

**Limits:**
- **Speed:** the model is tiny (2.75M parameters), and a compact JSON action is ~35 tokens, each a full forward pass on the device. A shorter action syntax would cut that by ~3×.
- **Fast links:** ESP-NOW/TCP/SPI bus upgrades are a hook, not an implementation.
- **Worker:** run one worker only; runs aren't locked. The GitHub Actions workflow enforces this with a `concurrency` group; don't also run `worker.py` on your PC at the same time.
- **Two API implementations:** the Edge Function (production, free, always on) and the Python server (local dev, tests, and the Render fallback). They must stay in step. When an endpoint changes, change both and run the edge-function test against both.
- **Free Render plan (fallback only):** it sleeps after 15 minutes without traffic, and its free hours are shared across the workspace and can run out mid-month.
- **TLS:** the device needs the root CA PEMs for your Render and Supabase hosts in `secrets.h` (see `BRINGUP.md`).

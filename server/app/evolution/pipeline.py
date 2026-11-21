"""The self-improvement loop, one evolution run at a time.

  1. curate      interaction_logs -> scored training examples
  2. synthesize  rule-boundary, full-range and safety (NOP) examples per device
  3. train       Trainer (mock | torch | remote) from the active model's weights
  4. gate        the candidate must beat its parent on the same val split
  5. quantize    INT16 image, checked against the W25Q slot + PSRAM budget
  6. package     upload models/v{n}.w25img, block delta vs the parent if smaller,
                 sign the OTA manifest, insert model_versions, activate
Devices pick the new version up on their next /ota/check.
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..db.base import BlobStore, Repository
from . import image_codec
from .curator import curate
from .delta import build_delta, changed_blocks
from .model_spec import BASE_MODEL, QUANTIZATION_FORMAT, SPEC
from .signing import ManifestSigner
from .chat_corpus import chat_rows, known_topics, lesson_rows
from .synth import synthesize
from .teacher import ClaudeTeacher, DeviceContext, RuleTeacher, Teacher
from .tokenizer import BpeTokenizer
from .trainer import Example, TrainConfig, make_trainer

log = logging.getLogger(__name__)
TOKENIZER_BLOB = ("models", "assets/tokenizer.bin")
LISA_TOKENIZER_BLOB = ("models", "assets/lisa_tokenizer.bin")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionContext:
    settings: Settings
    repo: Repository
    blobs: BlobStore
    signer: ManifestSigner
    teacher: Teacher
    _tokenizer: BpeTokenizer | None = None
    _legacy_tokenizer: BpeTokenizer | None = None

    def tokenizer(self) -> BpeTokenizer:
        """The tokenizer new models train with: lisa_tokenizer.bin from disk, else from storage."""
        if self._tokenizer is None:
            path = self.settings.tokenizer_path
            data = path.read_bytes() if path.is_file() else self.blobs.get(*LISA_TOKENIZER_BLOB)
            self._tokenizer = BpeTokenizer(data, SPEC.vocab)
        return self._tokenizer

    def legacy_tokenizer(self) -> BpeTokenizer:
        """tokenizer.bin, whose first 4096 entries hold no single characters: byte tokens only.
        Models without IMAGE_FLAG_TOK_LISA were trained (and run on the device) with it."""
        if self._legacy_tokenizer is None:
            path = self.settings.legacy_tokenizer_path
            data = path.read_bytes() if path.is_file() else self.blobs.get(*TOKENIZER_BLOB)
            self._legacy_tokenizer = BpeTokenizer(data, SPEC.vocab)
        return self._legacy_tokenizer

    def tokenizer_for(self, lisa_tokenizer: bool) -> BpeTokenizer:
        return self.tokenizer() if lisa_tokenizer else self.legacy_tokenizer()


def make_teacher(settings: Settings) -> Teacher:
    if settings.teacher == "claude":
        return ClaudeTeacher.from_env(settings.teacher_model, settings.teacher_effort, settings.teacher_max_calls_per_run)
    return RuleTeacher()


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def publish_image(ctx: EvolutionContext, image: bytes, *, parent: dict[str, Any] | None, label: str,
                  metrics: dict[str, Any] | None = None, run_id: str | None = None, activate: bool = True) -> dict[str, Any]:
    info = image_codec.read_header(image)
    if info.model_version <= 0:
        raise ValueError("image header must carry its model_version")
    path = f"v{info.model_version}.w25img"
    ctx.blobs.put("models", path, image)
    row: dict[str, Any] = {
        "version": info.model_version, "label": label, "base_model": BASE_MODEL,
        "parent_version": parent["version"] if parent else None, "quantization_format": QUANTIZATION_FORMAT,
        "size_bytes": info.size, "sha256": info.sha256, "storage_path": path,
        "signature": ctx.signer.sign(info.model_version, info.size, info.sha256),
        "metrics_json": metrics or {}, "training_run_id": run_id,
        "delta_from_version": None, "delta_storage_path": None, "delta_size_bytes": None, "delta_sha256": None,
    }
    if parent:
        base = ctx.blobs.get("models", parent["storage_path"])
        delta = build_delta(base, image)
        if len(delta) <= ctx.settings.delta_max_ratio * len(image):
            dpath = f"v{parent['version']}_to_v{info.model_version}.w25d"
            ctx.blobs.put("deltas", dpath, delta)
            row.update(delta_from_version=parent["version"], delta_storage_path=dpath, delta_size_bytes=len(delta),
                       delta_sha256=hashlib.sha256(delta).hexdigest())
            row["metrics_json"] = {**row["metrics_json"], "delta_changed_blocks": changed_blocks(delta)}
    model = ctx.repo.insert_model(row)
    if activate:
        ctx.repo.activate_model(info.model_version)
    return model


def bootstrap_model(ctx: EvolutionContext, image: bytes, version: int = 1, label: str = "bootstrap") -> dict[str, Any]:
    """Registers an existing image (e.g. from gemma4_slice_int16.py) as the first version."""
    if ctx.repo.get_model(version):
        raise ValueError(f"model version {version} already exists")
    image = image_codec.set_model_version(image, version)
    for blob, path in ((LISA_TOKENIZER_BLOB, ctx.settings.tokenizer_path),
                       (TOKENIZER_BLOB, ctx.settings.legacy_tokenizer_path)):
        if path.is_file():   # so workers without the files in their checkout can still train
            ctx.blobs.put(*blob, path.read_bytes())
    return publish_image(ctx, image, parent=None, label=label)


def _base_weights(ctx: EvolutionContext, parent: dict[str, Any] | None) -> tuple[image_codec.Weights, str, bool]:
    """(weights, name, trained with lisa_tokenizer.bin?)"""
    if parent:
        image = ctx.blobs.get("models", parent["storage_path"])
        return (image_codec.decode_image(image), f"v{parent['version']}",
                image_codec.read_header(image).lisa_tokenizer)
    p = ctx.settings.bootstrap_image_path
    if p and p.is_file():
        image = p.read_bytes()
        return image_codec.decode_image(image), str(p), image_codec.read_header(image).lisa_tokenizer
    return image_codec.random_weights(), "random-init", True


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def gate(baseline: dict[str, Any], metrics: dict[str, Any], min_improvement: float, force: bool,
         min_val: int = 0, max_task_regression: float = 1.0) -> tuple[bool, str]:
    if force:
        return True, "forced"
    if metrics.get("mock"):
        return False, "mock trainer produces no real improvement; pass force_publish to publish anyway"
    # A handful of examples can't tell a model that learned the rules from one that always
    # gives the majority answer (on hardware, 4 examples let an "always NOP" model through).
    n_val = metrics.get("n_val", baseline.get("n_val"))
    if n_val is not None and n_val < min_val:
        return False, f"only {n_val} validation examples (< {min_val}); too few to judge"
    # Getting better at chat must not cost the device skills (and vice versa).
    for task, b in (baseline.get("tasks") or {}).items():
        c = (metrics.get("tasks") or {}).get(task)
        if c is not None and c["exact_match"] < b["exact_match"] - max_task_regression:
            return False, f"{task} regressed (exact match {b['exact_match']:.3f} -> {c['exact_match']:.3f})"
    b_em, c_em = baseline.get("exact_match"), metrics.get("exact_match")
    if c_em is None or b_em is None:
        return False, "no validation examples to compare on"
    if c_em > b_em + min_improvement:
        return True, f"exact match {b_em:.3f} -> {c_em:.3f}"
    b_loss, c_loss = baseline.get("val_loss"), metrics.get("val_loss")
    if c_em >= b_em and b_loss is not None and c_loss is not None and c_loss < b_loss * 0.98:
        return True, f"exact match held at {c_em:.3f}, val loss {b_loss:.3f} -> {c_loss:.3f}"
    return False, f"no improvement (exact match {b_em:.3f} -> {c_em:.3f})"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def create_run(ctx: EvolutionContext, trigger: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return ctx.repo.create_run({"trigger": trigger, "status": "queued", "trainer": (params or {}).get("trainer")
                                or ctx.settings.trainer, "params": params or {}})


def run_evolution(ctx: EvolutionContext, run_id: str) -> dict[str, Any]:
    repo, s = ctx.repo, ctx.settings
    run = repo.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run {run_id}")
    params = run.get("params") or {}
    force = bool(params.get("force_publish"))
    repo.update_run(run_id, {"status": "running", "started_at": _now()})
    stats: dict[str, Any] = {}
    try:
        if isinstance(ctx.teacher, ClaudeTeacher):
            ctx.teacher.reset_budget()

        # 1-2. Curate + synthesize
        stats["curation"] = curate(repo, ctx.teacher, s.curate_batch)
        devices = [DeviceContext.from_row(d) for d in repo.list_uart_devices()]
        synth_rows = synthesize(devices, s.synth_per_device)   # stable per device: re-runs add nothing new
        stats["synthetic_inserted"] = repo.insert_examples(synth_rows)
        if s.chat_enabled:
            stats["chat_inserted"] = repo.insert_examples(chat_rows())   # stable corpus: re-runs add nothing
            if hasattr(ctx.teacher, "lessons") and s.teacher_lessons_per_run > 0:
                stats["lessons"] = teach_lessons(ctx, s.teacher_lessons_per_run)

        manifests = {d.key: d.manifest for d in devices}

        def to_example(r: dict[str, Any]) -> Example:
            key = (r.get("context_json") or {}).get("device_key", "")
            task = (r.get("context_json") or {}).get("task", "action")
            return Example(r["prompt"].encode("utf-8"), r["target_response"], manifests.get(key, {}), task)

        train = [to_example(r) for r in resolve_chat_conflicts(repo.list_examples("train"))]
        val = [to_example(r) for r in resolve_chat_conflicts(repo.list_examples("val"))]
        stats.update(n_train=len(train), n_val=len(val))
        if len(train) < s.min_train_examples and not force:
            return _finish(ctx, run_id, "rejected", stats, f"only {len(train)} training examples (< {s.min_train_examples})")

        # 3. Train. Normally from the active model; params.base_version restarts from an earlier
        # one (e.g. when the active model has collapsed to always giving the same answer).
        parent = repo.get_active_model()
        start = parent
        if params.get("base_version") is not None:
            start = repo.get_model(int(params["base_version"]))
            if start is None:
                raise ValueError(f"unknown base_version {params['base_version']}")
        base, base_name, _ = _base_weights(ctx, start)   # retrained with the current tokenizer either way
        cfg = TrainConfig(steps=int(params.get("steps", s.train_steps)), batch_size=s.train_batch_size,
                          lr=float(params.get("lr", s.train_lr)), method=params.get("method", s.train_method),
                          lora_rank=s.lora_rank, seed=len(train),
                          freeze_embeddings=bool(params.get("freeze_embeddings", s.freeze_embeddings)),
                          schedule=s.train_schedule,
                          warmup_steps=max(s.train_warmup_steps, int(params.get("steps", s.train_steps)) // 20))
        trainer = make_trainer(params.get("trainer") or s.trainer, s.remote_trainer_url,
                               s.remote_trainer_token.get_secret_value() if s.remote_trainer_token else None)
        result = trainer.train(base, train, val, ctx.tokenizer(), cfg)
        baseline = result.baseline
        if parent is not None:
            parent_weights, _, parent_lisa_tok = _base_weights(ctx, parent)
            # The gate compares with what devices run now, scored with the tokenizer that model
            # runs with: not with a restart point, and not through a tokenizer it never saw.
            if start is not parent or not parent_lisa_tok:
                if not hasattr(trainer, "evaluate_weights"):
                    raise ValueError(f"can't score the active model with {trainer.name}; it has no evaluate_weights")
                baseline = trainer.evaluate_weights(parent_weights, val, ctx.tokenizer_for(parent_lisa_tok), cfg)
        stats.update(base=base_name, baseline_model=f"v{parent['version']}" if parent else base_name,
                     trainer=trainer.name, baseline=baseline, candidate=result.metrics)

        # 4. Gate
        ok, reason = gate(baseline, result.metrics, s.gate_min_improvement, force, s.gate_min_val_examples,
                          s.gate_max_task_regression)
        stats["gate"] = reason
        if not ok:
            return _finish(ctx, run_id, "rejected", stats, reason)

        # 5-6. Quantize + package
        version = repo.latest_model_version() + 1
        image = image_codec.encode_image(result.weights, version)
        model = publish_image(ctx, image, parent=parent, label=f"run {run_id[:8]}", metrics=result.metrics,
                              run_id=run_id, activate=s.auto_activate)
        stats["published"] = {"version": version, "size": model["size_bytes"], "sha256": model["sha256"],
                              "delta_from": model.get("delta_from_version"), "delta_bytes": model.get("delta_size_bytes")}
        repo.update_run(run_id, {"produced_version": version})
        return _finish(ctx, run_id, "published", stats, None)
    except Exception as e:   # noqa: BLE001 - a run must always end in a terminal state
        log.exception("evolution run %s failed", run_id)
        stats["traceback"] = traceback.format_exc()[-2000:]
        return _finish(ctx, run_id, "failed", stats, f"{type(e).__name__}: {e}")


def resolve_chat_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One target per chat prompt. Lessons from real conversations (you, the teacher model)
    beat the built-in corpus, and the newest one wins, so a correction replaces what came
    before instead of fighting it. Device examples are left alone."""
    best: dict[str, dict[str, Any]] = {}
    out = []
    for r in rows:
        if (r.get("context_json") or {}).get("task") != "chat":
            out.append(r)
            continue
        rank = (r.get("source_log_id") is not None, r.get("id") or 0)
        cur = best.get(r["prompt"])
        if cur is None or rank > (cur.get("source_log_id") is not None, cur.get("id") or 0):
            best[r["prompt"]] = r
    return out + list(best.values())


def teach_lessons(ctx: EvolutionContext, n: int) -> dict[str, Any]:
    """The teacher model picks n topics LISA doesn't know yet and writes them as training examples."""
    taught = {(r.get("context_json") or {}).get("intent", "") for r in ctx.repo.list_examples()}
    known = known_topics() + sorted(i.removeprefix("lesson:") for i in taught if i.startswith("lesson:"))
    lessons = ctx.teacher.lessons(n, known)
    rows = [row for l in lessons for row in lesson_rows(l.topic, l.phrasings, l.reply, ctx.teacher.name)]
    for l in lessons:
        log.info("lesson %r: %r -> %r", l.topic, l.phrasings[0], l.reply)
    return {"topics": [{"topic": l.topic, "ask": l.phrasings[0], "reply": l.reply} for l in lessons],
            "inserted": ctx.repo.insert_examples(rows)}


def _finish(ctx: EvolutionContext, run_id: str, status: str, stats: dict[str, Any], error: str | None) -> dict[str, Any]:
    ctx.repo.update_run(run_id, {"status": status, "stats": stats, "error": error, "finished_at": _now()})
    log.info("evolution run %s -> %s %s", run_id, status, error or "")
    return ctx.repo.get_run(run_id) or {}


def should_trigger(ctx: EvolutionContext, now_ts: float, last_run_ts: float | None) -> str | None:
    """Worker policy: a threshold of new interactions, or the schedule with anything new.
    Never two runs within EVOLUTION_MIN_GAP_S (a run that keeps failing can't spin)."""
    if last_run_ts is not None and now_ts - last_run_ts < ctx.settings.evolution_min_gap_s:
        return None
    pending = ctx.repo.count_untrained_interactions()
    if pending >= ctx.settings.evolution_min_new_logs:
        return "threshold"
    if pending > 0 and (last_run_ts is None or now_ts - last_run_ts >= ctx.settings.evolution_interval_s):
        return "schedule"
    return None


__all__ = ["EvolutionContext", "bootstrap_model", "create_run", "gate", "make_teacher",
           "publish_image", "run_evolution", "should_trigger"]

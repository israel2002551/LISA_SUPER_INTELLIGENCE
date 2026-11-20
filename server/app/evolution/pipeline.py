"""The self-improvement loop, one evolution run at a time.

  1. curate      interaction_logs -> scored training examples
  2. synthesize  rule-boundary + safety (NOP) examples per device
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
from .synth import synthesize
from .teacher import ClaudeTeacher, DeviceContext, RuleTeacher, Teacher
from .tokenizer import BpeTokenizer
from .trainer import Example, TrainConfig, make_trainer

log = logging.getLogger(__name__)
TOKENIZER_BLOB = ("models", "assets/tokenizer.bin")


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

    def tokenizer(self) -> BpeTokenizer:
        """tokenizer.bin from disk, else from storage (models/assets/tokenizer.bin)."""
        if self._tokenizer is None:
            path = self.settings.tokenizer_path
            data = path.read_bytes() if path.is_file() else self.blobs.get(*TOKENIZER_BLOB)
            self._tokenizer = BpeTokenizer(data, SPEC.vocab)
        return self._tokenizer


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
    if ctx.settings.tokenizer_path.is_file():
        ctx.blobs.put(*TOKENIZER_BLOB, ctx.settings.tokenizer_path.read_bytes())
    return publish_image(ctx, image, parent=None, label=label)


def _base_weights(ctx: EvolutionContext, parent: dict[str, Any] | None) -> tuple[image_codec.Weights, str]:
    if parent:
        return image_codec.decode_image(ctx.blobs.get("models", parent["storage_path"])), f"v{parent['version']}"
    p = ctx.settings.bootstrap_image_path
    if p and p.is_file():
        return image_codec.decode_image(p.read_bytes()), str(p)
    return image_codec.random_weights(), "random-init"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def gate(baseline: dict[str, Any], metrics: dict[str, Any], min_improvement: float, force: bool) -> tuple[bool, str]:
    if force:
        return True, "forced"
    if metrics.get("mock"):
        return False, "mock trainer produces no real improvement; pass force_publish to publish anyway"
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
        synth_rows = synthesize(devices, s.synth_per_device, seed=len(repo.list_runs(1000)))
        stats["synthetic_inserted"] = repo.insert_examples(synth_rows)

        manifests = {d.key: d.manifest for d in devices}

        def to_example(r: dict[str, Any]) -> Example:
            key = (r.get("context_json") or {}).get("device_key", "")
            return Example(r["prompt"].encode("utf-8"), r["target_response"], manifests.get(key, {}))

        train = [to_example(r) for r in repo.list_examples("train")]
        val = [to_example(r) for r in repo.list_examples("val")]
        stats.update(n_train=len(train), n_val=len(val))
        if len(train) < s.min_train_examples and not force:
            return _finish(ctx, run_id, "rejected", stats, f"only {len(train)} training examples (< {s.min_train_examples})")

        # 3. Train
        parent = repo.get_active_model()
        base, base_name = _base_weights(ctx, parent)
        cfg = TrainConfig(steps=int(params.get("steps", s.train_steps)), batch_size=s.train_batch_size, lr=s.train_lr,
                          method=params.get("method", s.train_method), lora_rank=s.lora_rank,
                          freeze_embeddings=s.freeze_embeddings, seed=len(train))
        trainer = make_trainer(params.get("trainer") or s.trainer, s.remote_trainer_url,
                               s.remote_trainer_token.get_secret_value() if s.remote_trainer_token else None)
        result = trainer.train(base, train, val, ctx.tokenizer(), cfg)
        stats.update(base=base_name, trainer=trainer.name, baseline=result.baseline, candidate=result.metrics)

        # 4. Gate
        ok, reason = gate(result.baseline, result.metrics, s.gate_min_improvement, force)
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


def _finish(ctx: EvolutionContext, run_id: str, status: str, stats: dict[str, Any], error: str | None) -> dict[str, Any]:
    ctx.repo.update_run(run_id, {"status": status, "stats": stats, "error": error, "finished_at": _now()})
    log.info("evolution run %s -> %s %s", run_id, status, error or "")
    return ctx.repo.get_run(run_id) or {}


def should_trigger(ctx: EvolutionContext, now_ts: float, last_run_ts: float | None) -> str | None:
    """Worker policy: a threshold of new interactions, or the schedule with anything new."""
    pending = ctx.repo.count_untrained_interactions()
    if pending >= ctx.settings.evolution_min_new_logs:
        return "threshold"
    if pending > 0 and (last_run_ts is None or now_ts - last_run_ts >= ctx.settings.evolution_interval_s):
        return "schedule"
    return None


__all__ = ["EvolutionContext", "bootstrap_model", "create_run", "gate", "make_teacher",
           "publish_image", "run_evolution", "should_trigger"]

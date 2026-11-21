"""Trainers: take the current weights + curated examples, return new weights.

MockTrainer    no ML: nudges final_norm so a new, valid image is produced.
               For exercising the pipeline end to end (dry-run on Render).
TorchTrainer   real fine-tuning on CPU with the PyTorch twin of the firmware
               (full fine-tune or LoRA, merged before quantization). The edge
               model has ~2.75M parameters, so a few hundred steps run on a
               normal CPU instance.
RemoteTrainer  HOOK for an external GPU worker (see the contract below).

Quantization to the W25Q image happens afterwards in image_codec.encode_image.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .action_schema import validate_action
from .image_codec import Weights
from .model_spec import ACTION_MAX_TOKENS
from .prompt_format import canonicalize_output, prompt_tokens, target_fits, training_sequence
from .tokenizer import BpeTokenizer

log = logging.getLogger(__name__)


@dataclass
class Example:
    prompt: bytes
    target: str
    manifest: dict[str, Any] = field(default_factory=dict)
    task: str = "action"          # action | chat


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 16
    lr: float = 3e-4
    method: str = "full"          # full | lora
    lora_rank: int = 8
    freeze_embeddings: bool = True
    seed: int = 0
    eval_limit: int = 64          # val examples decoded for exact-match
    # "cosine": linear warmup, then cosine decay to lr * min_lr_ratio. With a constant LR the
    # final weights sit wherever the last noisy step left them; on character-level JSON that
    # garbles the output (seen as '{"cmd":"set_fan",","an":...' after 1200 steps at 3e-3).
    schedule: str = "constant"    # constant | cosine
    warmup_steps: int = 0
    min_lr_ratio: float = 0.05


@dataclass
class TrainResult:
    weights: Weights
    metrics: dict[str, Any]       # candidate, on the val split
    baseline: dict[str, Any]      # starting weights, on the same val split


class Trainer(Protocol):
    name: str

    def train(self, base: Weights, train: list[Example], val: list[Example], tok: BpeTokenizer,
              cfg: TrainConfig) -> TrainResult: ...


def lr_factor(step: int, cfg: TrainConfig) -> float:
    """Multiplier on cfg.lr for optimizer step `step` (0-based)."""
    if cfg.schedule == "constant":
        return 1.0
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    span = max(1, cfg.steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class MockTrainer:
    name = "mock"

    def train(self, base, train, val, tok, cfg) -> TrainResult:
        rng = np.random.default_rng(cfg.seed + len(train))
        w = {k: v.copy() for k, v in base.items()}
        w["final_norm"] = (w["final_norm"] + rng.standard_normal(w["final_norm"].shape).astype(np.float32) * 1e-3)
        info = {"mock": True, "n_train": len(train), "n_val": len(val)}
        return TrainResult(w, info, dict(info))

    def evaluate_weights(self, weights, val, tok, cfg) -> dict[str, Any]:
        return {"mock": True, "n_val": len(val)}


class TorchTrainer:
    name = "torch"

    def evaluate(self, model, val: list[Example], tok: BpeTokenizer, cfg: TrainConfig) -> dict[str, Any]:
        import torch

        if not val:
            return {"n_val": 0, "val_loss": None, "exact_match": None, "valid_action_rate": None}
        model.eval()
        losses = []
        with torch.no_grad():
            for ex in val:
                seq, n_prompt = training_sequence(tok, ex.prompt, ex.target)
                logits = model(torch.tensor([seq[:-1]]))[0]
                tgt = torch.tensor(seq[1:])
                losses.append(float(torch.nn.functional.cross_entropy(logits[n_prompt - 1:], tgt[n_prompt - 1:])))
        # Decode up to eval_limit examples per task. exact_match is the mean over tasks, so a
        # few dozen device examples weigh as much as hundreds of chat lines.
        tasks: dict[str, dict[str, Any]] = {}
        valid = n_action = 0
        for task in sorted({ex.task for ex in val}):
            subset = [ex for ex in val if ex.task == task][: cfg.eval_limit]
            exact = 0
            for ex in subset:
                out = model.greedy(prompt_tokens(tok, ex.prompt), ACTION_MAX_TOKENS)
                text = tok.decode(out).decode("utf-8", "replace").split("\n", 1)[0]
                if task == "chat":
                    exact += text.strip() == ex.target.strip()
                else:
                    exact += canonicalize_output(text) == ex.target
                    valid += validate_action(ex.manifest, text)[0] != "invalid"
                    n_action += 1
            tasks[task] = {"n": len(subset), "exact_match": exact / len(subset)}
        return {"n_val": len(val), "val_loss": float(np.mean(losses)),
                "exact_match": float(np.mean([t["exact_match"] for t in tasks.values()])),
                "valid_action_rate": valid / n_action if n_action else None, "tasks": tasks}

    def evaluate_weights(self, weights: Weights, val: list[Example], tok: BpeTokenizer, cfg: TrainConfig) -> dict[str, Any]:
        from .model import EdgeGemma

        return self.evaluate(EdgeGemma(weights), val, tok, cfg)

    def train(self, base, train, val, tok, cfg) -> TrainResult:
        import torch

        from .model import EdgeGemma

        torch.manual_seed(cfg.seed)
        model = EdgeGemma(base)
        baseline = self.evaluate(model, val, tok, cfg)

        if cfg.method == "lora":
            model.enable_lora(cfg.lora_rank, seed=cfg.seed)
        elif cfg.freeze_embeddings:
            model.params["embed"].requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, cfg))

        usable = [ex for ex in train if target_fits(tok, ex.target)]
        if len(usable) < len(train):
            log.warning("skipping %d examples whose target exceeds the device's action token budget",
                        len(train) - len(usable))
        if not usable:
            raise ValueError("no training example fits the device's action token budget")
        seqs = [training_sequence(tok, ex.prompt, ex.target) for ex in usable]
        rng = np.random.default_rng(cfg.seed)
        model.train()
        t0 = time.time()
        loss_val = float("nan")
        for step in range(cfg.steps):
            batch = [seqs[i] for i in rng.integers(0, len(seqs), size=min(cfg.batch_size, len(seqs)))]
            T = max(len(s) for s, _ in batch) - 1
            inp = torch.zeros(len(batch), T, dtype=torch.long)
            tgt = torch.zeros(len(batch), T, dtype=torch.long)
            mask = torch.zeros(len(batch), T)
            for b, (s, n_prompt) in enumerate(batch):
                n = len(s) - 1
                inp[b, :n] = torch.tensor(s[:-1])
                tgt[b, :n] = torch.tensor(s[1:])
                mask[b, n_prompt - 1:n] = 1.0      # loss only on target tokens + EOS
            logits = model(inp)
            per_tok = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                        reduction="none")
            loss = (per_tok * mask.reshape(-1)).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            loss_val = loss.detach().item()
            if step % 50 == 0:
                log.info("train step %d/%d loss %.4f", step, cfg.steps, loss_val)

        metrics = self.evaluate(model, val, tok, cfg)
        metrics.update(train_loss=loss_val, steps=cfg.steps, n_train=len(usable), method=cfg.method,
                       train_seconds=round(time.time() - t0, 1))
        return TrainResult(model.export_weights(), metrics, baseline)


class RemoteTrainer:
    """HOOK: hand training to an external GPU worker.

    Contract (implement it on the worker):
      POST {url}/train   json {"weights_npz_b64": str, "train": [{"prompt_b64","target"}], "val": [...],
                               "config": {...TrainConfig}}            -> {"job_id": str}
      GET  {url}/jobs/{job_id}                                        -> {"status": "queued|running|done|failed",
                               "weights_npz_b64"?: str, "metrics"?: {}, "baseline"?: {}, "error"?: str}
    The worker must use the same architecture as app/evolution/model.py (or import it).
    """

    name = "remote"

    def __init__(self, url: str, token: str | None = None, poll_s: float = 10.0, timeout_s: float = 6 * 3600) -> None:
        self.url, self.token, self.poll_s, self.timeout_s = url.rstrip("/"), token, poll_s, timeout_s

    def train(self, base, train, val, tok, cfg) -> TrainResult:
        import httpx

        def npz(w: Weights) -> str:
            buf = io.BytesIO()
            np.savez_compressed(buf, **w)
            return base64.b64encode(buf.getvalue()).decode()

        def ex(e: Example) -> dict[str, str]:
            return {"prompt_b64": base64.b64encode(e.prompt).decode(), "target": e.target}

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        with httpx.Client(timeout=120, headers=headers) as http:
            job = http.post(f"{self.url}/train", json={"weights_npz_b64": npz(base), "train": [ex(e) for e in train],
                                                       "val": [ex(e) for e in val], "config": cfg.__dict__})
            job.raise_for_status()
            job_id = job.json()["job_id"]
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                st = http.get(f"{self.url}/jobs/{job_id}").json()
                if st["status"] == "done":
                    data = np.load(io.BytesIO(base64.b64decode(st["weights_npz_b64"])))
                    return TrainResult({k: data[k] for k in data.files}, st.get("metrics", {}), st.get("baseline", {}))
                if st["status"] == "failed":
                    raise RuntimeError(f"remote training failed: {st.get('error')}")
                time.sleep(self.poll_s)
        raise TimeoutError("remote training timed out")


def make_trainer(kind: str, remote_url: str | None = None, remote_token: str | None = None) -> Trainer:
    if kind == "torch":
        return TorchTrainer()
    if kind == "remote":
        if not remote_url:
            raise ValueError("TRAINER=remote needs REMOTE_TRAINER_URL")
        return RemoteTrainer(remote_url, remote_token)
    return MockTrainer()

"""Experience evaluation & curation: interaction_logs -> training_datasets.

Rewards come from what actually happened on the hardware:
  ok (peripheral acked)       1.0      timeout           0.3
  rejected (peripheral nack)  0.1      invalid_output    0.0
  nop                         decided by the teacher
  feedback {"fb":"good"|"bad"} from the peripheral overrides to 1.0 / 0.0
Each log becomes at most one example:
  real_log          the device did the right thing (high reward, or agrees with the teacher)
  failure_recovery  it didn't, and the teacher knows the right answer
Logs nobody can label are skipped (still marked used, so they aren't re-read).

Precedence: a matching rule (confidence 1.0) beats the hardware outcome, so
operators can correct behavior. Where no rule speaks (the rules' default NOP
is 0.9), a well-rewarded real outcome wins: that's how the model learns
behavior beyond its rules. To forbid something, write a rule for it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..db.base import Repository
from .prompt_format import build_prompt, canonicalize_output
from .teacher import DeviceContext, Teacher

REWARD_BY_STATUS = {"ok": 1.0, "timeout": 0.3, "rejected": 0.1, "invalid_output": 0.0}


def initial_reward(status: str | None) -> float | None:
    return REWARD_BY_STATUS.get(status or "")


def example_row(prompt: str, target: str, source_type: str, teacher: str, quality: float,
                context: dict[str, Any], source_log_id: int | None = None, uart_device_id: str | None = None) -> dict:
    key = hashlib.sha256(prompt.encode("utf-8") + b"\x00" + target.encode("utf-8")).hexdigest()
    return {
        "prompt": prompt, "target_response": target, "source_type": source_type, "teacher": teacher,
        "quality": round(float(quality), 3), "context_json": context, "source_log_id": source_log_id,
        "uart_device_id": uart_device_id, "dedupe_key": key,
        "split": "val" if int(key[:8], 16) % 10 == 0 else "train",   # stable 90/10 split
    }


def curate(repo: Repository, teacher: Teacher, limit: int) -> dict[str, int]:
    logs = repo.fetch_untrained_interactions(limit)
    devices = {d["id"]: DeviceContext.from_row(d) for d in repo.list_uart_devices()}
    stats = {"logs": len(logs), "real_log": 0, "failure_recovery": 0, "skipped": 0, "inserted": 0}
    rows = []
    for log_row in logs:
        if log_row.get("status") == "error":            # no model loaded / link reset: no signal
            stats["skipped"] += 1
            continue
        payload = log_row.get("payload") or {}
        dev = devices.get(log_row.get("uart_device_id")) or DeviceContext(key=payload.get("dev", "unknown"))
        event = log_row.get("raw_input") or ""
        prompt = log_row.get("prompt") or build_prompt(dev.key, dev.type, event).decode("utf-8", "replace")
        output = canonicalize_output(log_row.get("model_output"))
        reward = log_row.get("reward")
        label = teacher.label(dev, event)
        ctx = {"node_id": log_row["node_id"], "device_key": dev.key, "status": log_row.get("status"),
               "model_output": log_row.get("model_output"), "reward": reward, "model_version": log_row.get("model_version")}
        common = dict(context=ctx, source_log_id=log_row["id"], uart_device_id=log_row.get("uart_device_id"))

        if label is not None and label.confidence >= 0.95 and label.target != output:
            # An authoritative label (operator/device rule) disagrees: learn the rule.
            rows.append(example_row(prompt, label.target, "failure_recovery", label.teacher, label.confidence, **common))
            stats["failure_recovery"] += 1
        elif reward is not None and reward >= 0.8 and output is not None:
            rows.append(example_row(prompt, output, "real_log", "log", reward, **common))
            stats["real_log"] += 1
        elif label is not None and label.target == output:
            rows.append(example_row(prompt, output, "real_log", label.teacher, 0.7, **common))
            stats["real_log"] += 1
        elif label is not None:
            rows.append(example_row(prompt, label.target, "failure_recovery", label.teacher, label.confidence, **common))
            stats["failure_recovery"] += 1
        else:
            stats["skipped"] += 1

    stats["inserted"] = repo.insert_examples(rows)
    repo.mark_logs_used([l["id"] for l in logs])
    return stats

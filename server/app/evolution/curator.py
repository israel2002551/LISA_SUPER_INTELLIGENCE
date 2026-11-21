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
import logging
from typing import Any

from ..db.base import Repository
from .prompt_format import build_prompt, canonicalize_output, is_chat_prompt, split_chat_prompt
from .teacher import ChatTurn, DeviceContext, Teacher

log = logging.getLogger(__name__)

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


def same_answer(a: str, b: str) -> bool:
    """True if one answer contains the other, ignoring case and punctuation (\"Paris is the capital.\"
    vs \"I know Paris is the capital.\"). Deliberately no fuzzy similarity: \"Jupiter is the largest
    planet\" and \"Saturn is the largest planet\" are 90 % alike and must stay a correction."""
    norm = lambda t: " ".join("".join(c for c in t.lower() if c.isalnum() or c.isspace()).split())
    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and (x in y or y in x)


def _is_chat(log_row: dict[str, Any]) -> bool:
    return (log_row.get("payload") or {}).get("task") == "chat" or is_chat_prompt((log_row.get("prompt") or "").encode())


def curate_chats(logs: list[dict[str, Any]], teacher: Teacher, stats: dict[str, Any]) -> list[dict]:
    """Chat logs -> examples. Who decides what LISA should have said, in order:
      1. you: /good keeps her reply; /bad asks for a better one (below)
      2. the corpus, when it covers the message (free, and consistent with her training)
      3. the teacher model (LLMTeacher.review_chats: Claude or Groq), for everything else
    A reply nobody can judge is skipped. With a teacher, the corpus's "I don't know that
    yet" doesn't count as an answer: the teacher may know it, and then she learns it."""
    from .chat_corpus import DONT_KNOW, corpus_reply   # chat_corpus imports example_row from here

    turns = []
    for log_row in logs:
        prev, msg = split_chat_prompt(log_row.get("prompt") or "")
        turns.append(ChatTurn(prev, msg, (log_row.get("model_output") or "").strip()))
    corpus = [corpus_reply(t.message, t.prev_reply) for t in turns]
    can_teach = hasattr(teacher, "review_chats")
    if can_teach:   # "I don't know that yet" is a placeholder a teacher can replace with the answer
        corpus = [None if c == DONT_KNOW else c for c in corpus]
    # Ask the teacher only where the corpus has no answer, or you said the corpus answer was bad.
    ask = [i for i, (t, c, l) in enumerate(zip(turns, corpus, logs))
           if c is None or (l.get("reward") == 0.0 and c == t.reply)]
    reviews: dict[int, Any] = {}
    if ask and can_teach:
        reviews = dict(zip(ask, teacher.review_chats([turns[i] for i in ask])))
    teacher_name = getattr(teacher, "name", "teacher")

    rows = []
    for i, (log_row, t) in enumerate(zip(logs, turns)):
        reward, review, known = log_row.get("reward"), reviews.get(i), corpus[i]
        ctx = {"task": "chat", "intent": "log", "node_id": log_row["node_id"], "model_output": t.reply,
               "reward": reward, "model_version": log_row.get("model_version")}
        prompt = log_row.get("prompt") or ""
        if reward == 1.0 and t.reply:
            target, source, who, why = t.reply, "real_log", "user", "you said /good"
        elif reward == 0.0:
            better = review.better_reply if review is not None and review.better_reply != t.reply else (
                known if known not in (None, t.reply) else None)
            target, source, who, why = better, "failure_recovery", teacher_name if review else "chat_corpus", "you said /bad"
        elif known is not None:
            ok = known == t.reply
            target, source, who, why = known, "real_log" if ok else "failure_recovery", "chat_corpus", \
                "matches her lessons" if ok else "her lessons say otherwise"
        elif review is not None:
            # A "better" reply that only rewords hers is agreement: retraining on it churns her.
            ok = review.verdict == "good" or same_answer(t.reply, review.better_reply)
            target = t.reply if ok else review.better_reply
            source, who, why = ("real_log" if ok else "failure_recovery"), teacher_name, f"teacher: {review.verdict}"
        else:
            target = None
        if not target:
            stats["chat_skipped"] += 1
            continue
        rows.append(example_row(prompt, target, source, who, 1.0 if who == "user" else 0.9, context=ctx,
                                source_log_id=log_row["id"]))
        stats["chat_" + source] += 1
        if len(stats["chat_review"]) < 40:
            stats["chat_review"].append({"you": t.message[:80], "lisa": t.reply[:120], "learns": target[:120], "why": why})
            log.info("chat %-40r LISA %r -> learns %r (%s)", t.message[:40], t.reply[:60], target[:60], why)
    return rows


def curate(repo: Repository, teacher: Teacher, limit: int) -> dict[str, Any]:
    all_logs = repo.fetch_untrained_interactions(limit)
    chat_logs = [l for l in all_logs if _is_chat(l)]
    logs = [l for l in all_logs if not _is_chat(l)]
    devices = {d["id"]: DeviceContext.from_row(d) for d in repo.list_uart_devices()}
    stats: dict[str, Any] = {"logs": len(logs), "real_log": 0, "failure_recovery": 0, "skipped": 0, "inserted": 0,
                             "chat_logs": len(chat_logs), "chat_real_log": 0, "chat_failure_recovery": 0,
                             "chat_skipped": 0, "chat_review": []}
    rows = curate_chats(chat_logs, teacher, stats) if chat_logs else []
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
    repo.mark_logs_used([l["id"] for l in all_logs])
    if not stats["chat_review"]:
        del stats["chat_review"]
    return stats

"""Background evolution worker (Render "worker" service).

Runs queued evolution runs (created by POST /api/v1/evolution/trigger) and
starts new ones on its own when enough new experience arrives:
  threshold  >= EVOLUTION_MIN_NEW_LOGS untrained interactions
  schedule   EVOLUTION_INTERVAL_S since the last run, if anything is new

Run one worker only (runs are claimed without locking).
    python worker.py            loop forever
    python worker.py --once     process the queue / policy once and exit (cron-style)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

from app.config import get_settings
from app.evolution.pipeline import create_run, run_evolution, should_trigger
from app.main import build_context

log = logging.getLogger("worker")
failed_runs: list[str] = []


def _last_run_ts(ctx) -> float | None:
    runs = ctx.repo.list_runs(1)
    if not runs:
        return None
    return datetime.fromisoformat(str(runs[0]["created_at"]).replace("Z", "+00:00")).timestamp()


def tick(ctx) -> bool:
    """One pass. Returns True if it did any work."""
    run = ctx.repo.next_queued_run()
    if run:
        log.info("running queued evolution run %s", run["id"])
        result = run_evolution(ctx, run["id"])
        log.info("run %s finished: %s %s", run["id"], result.get("status"), result.get("error") or "")
        if result.get("status") == "failed":
            failed_runs.append(run["id"])
        return True
    trigger = should_trigger(ctx, time.time(), _last_run_ts(ctx))
    if trigger:
        run = create_run(ctx, trigger)
        log.info("created %s-triggered run %s", trigger, run["id"])
        return True
    return False


MAX_STEPS_PER_ONCE = 6   # --once: create + finish at most 3 runs, so a broken run can't loop forever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="process the queue / policy, then exit (cron, CI)")
    parser.add_argument("--trigger", action="store_true", help="start a manual evolution run first")
    parser.add_argument("--force", action="store_true", help="with --trigger: publish even if the gate rejects")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    ctx = build_context(settings)
    if settings.dev_mode:
        log.warning("Worker in dev mode has its own in-memory database; it can't see the web service's data.")
    if args.trigger:
        run = create_run(ctx, "manual", {"force_publish": True} if args.force else {})
        log.info("created manual run %s%s", run["id"], " (force publish)" if args.force else "")
    if args.once:
        for _ in range(MAX_STEPS_PER_ONCE):
            if not tick(ctx):
                break
        if failed_runs:   # "rejected" by the gate is normal; "failed" is a bug or outage
            raise SystemExit(f"evolution run(s) failed: {', '.join(failed_runs)}")
        return
    while True:
        if not tick(ctx):
            time.sleep(settings.worker_poll_s)


if __name__ == "__main__":
    main()

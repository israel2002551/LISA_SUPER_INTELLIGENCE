"""Simulated edge node: sends telemetry exactly like edge/net_sync.h, so the
server and Supabase can be tested without hardware.

  python scripts/simulate_node.py --server http://localhost:8000 --node-id edge-01 --api-key <key> --events 300

A fake thermostat (same manifest as tools/fake_peripheral.py) produces events;
the simulated "model" picks the right action with probability --accuracy and
otherwise answers NOP, a wrong level, or broken JSON, which is the data the
evolution loop learns from. Ends with an OTA check, like the device.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evolution.prompt_format import build_prompt  # noqa: E402

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from fake_peripheral import THERMO_MANIFEST, SmartThermostat, compact  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--node-id", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--events", type=int, default=200)
    p.add_argument("--accuracy", type=float, default=0.5)
    p.add_argument("--start-seq", type=int, default=1, help="ring-log seq of the first record (re-use = no duplicates)")
    p.add_argument("--current-sha256", default="0" * 64)
    args = p.parse_args()

    base = args.server.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}", "X-Node-Id": args.node_id}
    rng = random.Random(1)
    dev = SmartThermostat(115200, rng)
    seq = args.start_seq
    events = [{"seq": seq, "type": 2, "data": {"kind": "device", "dev": "thermo-01", "port": 1, "identified": True,
                                                "baud": 2000000, "link": "uart_fast", "format": "json_lines",
                                                "type": "thermostat", "manifest": THERMO_MANIFEST}}]
    for i in range(args.events):
        seq += 1
        dev.temp = rng.uniform(5, 40)
        temp = round(dev.temp, 1)
        dev.last_temp = temp
        event = compact({"ev": "temp", "v": temp})
        want = dev.expected_action(temp)
        right = compact(want) if want else "NOP"
        if rng.random() < args.accuracy:
            output = right
        else:
            output = rng.choice(["NOP", compact({"cmd": "set_fan", "args": {"level": rng.randint(0, 3)}}), '{"cmd":"set_f'])
        if output == "NOP":
            status = "nop"
        elif output.endswith("}"):
            status = "ok"     # valid action, the peripheral acked it
        else:
            status = "invalid_output"
        events.append({"seq": seq, "type": 1, "data": {
            "kind": "interaction", "dev": "thermo-01", "act_id": 1_000_000 + i, "input": event,
            "prompt": build_prompt("thermo-01", "thermostat", event).decode(), "output": output, "status": status,
            "lat": rng.randint(60, 400), "mv": 1}})
        if status == "ok":
            seq += 1
            events.append({"seq": seq, "type": 6, "data": {"kind": "feedback", "dev": "thermo-01",
                                                            "ref_act_id": 1_000_000 + i,
                                                            "value": "good" if output == right else "bad"}})

    status_block = {"model_version": 1, "model_loaded": True, "selftest_ok": True, "uptime_s": 123,
                    "partitions": {"active_slot": 0, "slots": [
                        {"slot": "A", "state": "good", "version": 1, "sha256": args.current_sha256},
                        {"slot": "B", "state": "empty", "version": 0, "sha256": ""}]},
                    "w25q": {"jedec_id": "ef4018", "size": 16777216}, "uart": {"state": "identified"}}
    inserted = 0
    for i in range(0, len(events), 100):
        r = httpx.post(f"{base}/api/v1/telemetry", headers=headers, timeout=60, json={
            "node_id": args.node_id, "fw_version": "edge-sim", "status": status_block, "devices": [],
            "events": events[i:i + 100]})
        if r.is_error:
            raise SystemExit(f"telemetry failed: {r.status_code} {r.text}")
        inserted += r.json()["inserted"]
    print(f"Sent {len(events)} records (seq {args.start_seq}..{seq}), server inserted {inserted} new.")

    r = httpx.get(f"{base}/api/v1/ota/check", headers=headers, timeout=60, params={
        "node_id": args.node_id, "current_version": 1, "current_sha256": args.current_sha256})
    print("OTA check:", json.dumps(r.json(), indent=1)[:600])


if __name__ == "__main__":
    main()

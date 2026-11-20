"""tools/fake_peripheral.py implements the device side of the UART protocol
documented in edge/uart_link.h; the edge node is tested against it on hardware."""

from __future__ import annotations

import json
import random
import sys

from app.evolution.action_schema import validate_action

from .conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
from fake_peripheral import THERMO_MANIFEST, SmartThermostat  # noqa: E402


def test_identify_upgrade_ping_and_revert():
    d = SmartThermostat(115200, random.Random(0))
    out, baud = d.on_line('{"cmd":"IDENTIFY","host":"edge-01","proto":1}', 0.0)
    assert json.loads(out[0])["id"] == "thermo-01" and baud is None
    assert d.on_line("\x00\xffgarbage at the wrong baud", 0.0) == ([], None)

    out, baud = d.on_line('{"cmd":"UPGRADE","link":"uart","baud":2000000}', 1.0)
    assert json.loads(out[0]) == {"ack": "UPGRADE", "ok": True} and baud == 2000000
    d.baud = baud
    assert d.on_line('{"cmd":"PING","n":77}', 1.2)[0] == ['{"pong":77}']
    assert d.tick(5.0)[1] is None                          # PING arrived: no revert

    d.on_line('{"cmd":"UPGRADE","link":"uart","baud":921600}', 10.0)
    assert d.tick(11.5)[1] == 2000000                      # no PING within 1 s: back to the old baud


def test_actions_are_acked_and_judged():
    d = SmartThermostat(115200, random.Random(0))
    d.last_temp = 33.0
    out, _ = d.on_line('{"cmd":"set_fan","args":{"level":2},"id":5}', 0.0)
    assert [json.loads(o) for o in out] == [{"ack": 5, "ok": True}, {"fb": "good", "id": 5}]
    out, _ = d.on_line('{"cmd":"reboot","id":6}', 0.0)
    assert json.loads(out[0]) == {"ack": 6, "ok": False, "err": "unknown cmd"}
    # The manifest it declares is one the edge validator accepts actions against.
    assert validate_action(THERMO_MANIFEST, '{"cmd":"set_fan","args":{"level":2}}')[0] == "ok"

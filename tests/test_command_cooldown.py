from pathlib import Path

from scripts.command_cooldown import cooldown_payload, cooldown_remaining_from_state, cooldown_status


def test_cooldown_remaining_from_state_uses_until_timestamp():
    remaining = cooldown_remaining_from_state({"cooldown_until": 105.0}, now=101.5)

    assert remaining == 3.5


def test_cooldown_payload_clears_expired_values():
    payload = cooldown_payload(1.0, 2.0, "left")

    assert payload == {
        "cooldown_until": 0.0,
        "cooldown_seconds": 0.0,
        "cooldown_source": "",
    }


def test_cooldown_status_reads_shared_state_file(tmp_path: Path):
    state_file = tmp_path / "gamepad_state.json"
    state_file.write_text('{"cooldown_until": 50.0, "cooldown_source": "right"}', encoding="utf-8")

    blocked, remaining, source = cooldown_status(state_file, now=48.0)

    assert blocked is True
    assert remaining == 2.0
    assert source == "right"

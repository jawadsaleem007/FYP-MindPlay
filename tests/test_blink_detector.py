import sys

import pytest

from scripts import blink_detector


class FakeInfo:
    def nominal_srate(self):
        return 2.0

    def channel_count(self):
        return 1

    def name(self):
        return "Fake EEG"

    def desc(self):
        raise RuntimeError("no channel metadata")


class FakeInlet:
    def __init__(self):
        self.samples = [[0.0], [10.0], [20.0]]

    def pull_sample(self):
        if not self.samples:
            raise StopIteration
        return self.samples.pop(0), None


def test_cooldown_blocked_blink_still_slides_buffer(monkeypatch):
    fake_inlet = FakeInlet()
    monkeypatch.setattr(sys, "argv", [
        "blink_detector.py",
        "--sfreq", "2",
        "--picks", "0",
        "--window", "1.0",
        "--threshold-uv", "1",
        "--refractory", "0",
    ])
    monkeypatch.setattr(blink_detector, "ensure_admin_privileges", lambda: None)
    monkeypatch.setattr(blink_detector, "find_eeg_stream", lambda: FakeInfo())
    monkeypatch.setattr(blink_detector, "StreamInlet", lambda info: fake_inlet)
    monkeypatch.setattr(blink_detector, "bandpass", lambda data, **kwargs: data)
    monkeypatch.setattr(blink_detector, "cooldown_status", lambda state_file, now=None: (True, 1.1, "backward"))

    with pytest.raises(StopIteration):
        blink_detector.main()
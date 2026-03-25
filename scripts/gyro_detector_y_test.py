"""Testing launcher: run gyro detector with Y-axis behavior matching left/right.

This script starts `gyro_detector.py` with `--y-like-lr` plus gamepad defaults.
You can override any parameter by passing extra args after this script.

Example:
  python scripts/gyro_detector_y_test.py
  python scripts/gyro_detector_y_test.py --vel-forward 450 --vel-backward 450 --gamepad-repeat-interval 0.2
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def main() -> None:
    base_args = [
        str(SCRIPTS / "gyro_detector.py"),
        "--gyro-channels", "5,6,7",
        "--sfreq", "500",
        "--scale-factor", "1.0",
        "--vel-left", "99999",
        "--vel-right", "99999",
        "--vel-forward", "450",
        "--vel-backward", "450",
        "--vel-return", "180",
        "--deadzone-y", "150",
        "--smoothing-window", "14",
        "--gamepad-mode",
        "--gamepad-ud-only",
        "--y-like-lr",
        "--gamepad-repeat-interval", "0.20",
        "--output-keys",
        "--verbose",
    ]

    # Allow user to override/append any args from command line.
    cmd = [sys.executable] + base_args + sys.argv[1:]
    print("Launching:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

"""Real-time intentional blink detection from an LSL EEG stream.

This script listens to an LSL stream (type='EEG'), focuses on frontal channels
if available (e.g., Fp1/Fp2/Fz), and detects blink events based on the
band-limited peak-to-peak amplitude within short windows.

Parameters allow tuning the detection threshold (in microvolts), window length,
and a refractory interval to avoid multiple triggers for a single blink.

Optionally, it can press a key (e.g., 'b') on detection.

Example:
  python scripts/blink_detector.py --sfreq 500 --picks Fp1,Fp2 --window 0.5 \
      --threshold-uv 80 --refractory 0.8 --key b
"""
import time
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, filtfilt

try:
    from pynput.keyboard import Controller as KBController
    _kb = KBController()
except Exception:
    _kb = None


def butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass(data: np.ndarray, fs: float, low: float = 0.5, high: float = 10.0):
    b, a = butter_bandpass(low, high, fs, order=4)
    return filtfilt(b, a, data, axis=-1)


def find_eeg_stream(timeout=5.0):
    streams = resolve_byprop('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError('No EEG LSL stream found')
    return streams[0]


def get_channel_labels(info) -> List[str]:
    labels = []
    try:
        desc = info.desc()
        ch = desc.child('channels').child('channel')
        while ch and ch.name():
            label = ch.child_value('label') or ''
            labels.append(label)
            ch = ch.next_sibling()
    except Exception:
        pass
    if len(labels) != info.channel_count():
        labels = [''] * info.channel_count()
    return labels


def parse_picks(picks_arg: str) -> Tuple[bool, List[str]]:
    parts = [p.strip() for p in picks_arg.split(',') if p.strip()]
    is_index = all(p.isdigit() for p in parts)
    return is_index, parts


def resolve_picks(labels: List[str], picks_arg: str, default_names=("Fp1", "Fp2")) -> List[int]:
    is_index, vals = parse_picks(picks_arg) if picks_arg else (False, list(default_names))
    if is_index:
        return [int(v) for v in vals]
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    idxs = []
    for name in vals:
        i = lower_map.get(name.lower())
        if i is not None:
            idxs.append(i)
    if idxs:
        return idxs
    # fallback: first channel
    return [0]


def press_key_once(k='b'):
    if _kb is None:
        return False
    try:
        _kb.press(k)
        _kb.release(k)
        return True
    except Exception:
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sfreq', type=float, default=0.0, help='Sampling rate; if 0, use stream nominal rate')
    ap.add_argument('--picks', type=str, default='Fp1,Fp2', help='Channel names or indices CSV')
    ap.add_argument('--window', type=float, default=0.5, help='Detection window length (seconds)')
    ap.add_argument('--threshold-uv', type=float, default=80.0, help='Peak-to-peak threshold (microvolts) to declare blink')
    ap.add_argument('--refractory', type=float, default=0.8, help='Minimum seconds between blink detections')
    ap.add_argument('--scale-to-uv', action='store_true', help='Multiply incoming volts by 1e6 to convert to microvolts')
    ap.add_argument('--key', type=str, default=None, help='Optional key to press on blink (e.g., b)')
    args = ap.parse_args()

    info = find_eeg_stream()
    inlet = StreamInlet(info)
    stream_sfreq = info.nominal_srate()
    sfreq = args.sfreq if args.sfreq > 0 else stream_sfreq
    if sfreq <= 0:
        raise RuntimeError('Sampling rate unknown; provide --sfreq')

    labels = get_channel_labels(info)
    picks = resolve_picks(labels, args.picks)

    n_ch = len(picks)
    win_samples = int(round(args.window * sfreq))
    buf = np.zeros((n_ch, win_samples), dtype=float)
    pos = 0
    last_event = 0.0

    print(f'Stream "{info.name()}" ch={info.channel_count()} sfreq={sfreq} Hz | picks={picks} window={args.window}s thr={args.threshold_uv}uV')
    print('Blink detection started. Press Ctrl+C to exit.')

    while True:
        sample, _ = inlet.pull_sample()
        if sample is None:
            time.sleep(0.001)
            continue
        arr = np.asarray(sample, dtype=float)
        if args.scale_to_uv:
            arr = arr * 1e6
        buf[:, pos] = arr[picks]
        pos += 1

        if pos >= win_samples:
            # band-limit and compute peak-to-peak per channel
            filtered = bandpass(buf, fs=sfreq, low=0.5, high=10.0)
            ptp = np.ptp(filtered, axis=1)  # peak-to-peak
            max_ptp = float(np.max(ptp))
            now = time.time()
            print(f'win_ptp_max={max_ptp:.1f}uV thr={args.threshold_uv:.1f}uV')

            if max_ptp >= args.threshold_uv and (now - last_event) >= args.refractory:
                last_event = now
                print(f'Blink detected! (ptp={max_ptp:.1f}uV)')
                if args.key:
                    ok = press_key_once(args.key)
                    print(f'Action: key "{args.key}" pressed status={ok}')

            # slide window by half for responsiveness
            shift = win_samples // 2
            if shift <= 0:
                shift = 1
            buf = np.roll(buf, -shift, axis=1)
            pos = win_samples - shift


if __name__ == '__main__':
    # Ensure project root on sys.path (not strictly needed here)
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()

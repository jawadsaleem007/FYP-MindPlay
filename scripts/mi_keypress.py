"""Real-time MI detector that presses a key when probability threshold is exceeded.

It connects to the first LSL EEG stream (type='EEG'), buffers windows, and
uses a trained FBCSP+LDA model to estimate class probabilities. When the
smoothed probability of the MI class exceeds a threshold, it presses the
configured key (default 'a').

Labels convention used during recording: 0 = MI (hand imagery), 1 = Rest.
If your labels differ, set --mi-label accordingly.

Example:
  python scripts/mi_keypress.py --model fbcsp_lda_S02.joblib --sfreq 500 \
      --window 4.0 --step 0.5 --picks C3,Cz,C4 --vote-k 5 \
      --threshold 0.65 --key a --min-interval 1.0
"""
import time
import sys
from collections import deque
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, resolve_byprop

# keyboard control
try:
    from pynput.keyboard import Controller, Key
    _kb = Controller()
except Exception as e:
    _kb = None

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fbcsp import FBCSP


def find_eeg_stream(timeout=5.0):
    streams = resolve_byprop('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError('No EEG LSL stream found')
    return streams[0]


def get_channel_labels(info):
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


def parse_picks(picks_arg):
    if not picks_arg:
        return None, None
    parts = [p.strip() for p in picks_arg.split(',') if p.strip()]
    is_index = all(p.isdigit() for p in parts)
    if is_index:
        return [int(p) for p in parts], None
    return None, parts


def resolve_picks(info, picks_arg, default_names=('Cz', 'C3', 'C4')):
    labels = get_channel_labels(info)
    idxs, names = parse_picks(picks_arg) if picks_arg else (None, None)
    if idxs is not None:
        return idxs
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    chosen = []
    names_to_use = names if names else list(default_names)
    for nm in names_to_use:
        i = lower_map.get(nm.lower())
        if i is not None:
            chosen.append(i)
    if len(chosen) == len(names_to_use) and chosen:
        return chosen
    return list(range(min(3, info.channel_count())))


def press_key_once(k='a'):
    if _kb is None:
        print('Warning: keyboard controller unavailable (pynput not installed).')
        return False
    try:
        _kb.press(k)
        _kb.release(k)
        return True
    except Exception as e:
        print('Keyboard press failed:', e)
        return False


def run(model_path, sfreq, window_s, step_s, picks, scale_to_uV, vote_k,
        mi_label, threshold, key, min_interval, block):
    model = FBCSP.load(model_path)
    model.sfreq = sfreq

    info = find_eeg_stream()
    inlet = StreamInlet(info)
    pick_idxs = resolve_picks(info, picks)

    n_channels = len(pick_idxs)
    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)
    buffer = np.zeros((n_channels, window_samples))
    buf_pos = 0

    print(f'Listening to stream "{info.name()}" using channels {pick_idxs}, window {window_s}s, step {step_s}s')

    proba_buf = deque(maxlen=max(1, int(vote_k)))
    classes_order = getattr(model.lda, 'classes_', np.array([0, 1]))
    # index of MI label in classifier classes
    mi_idx = int(np.where(classes_order == mi_label)[0][0]) if mi_label in classes_order else 0
    last_press = 0.0

    while True:
        sample, ts = inlet.pull_sample()
        if sample is None:
            time.sleep(0.001)
            continue
        arr = np.asarray(sample, dtype=float)
        if scale_to_uV:
            arr = arr * 1e6
        buffer[:, buf_pos] = arr[pick_idxs]
        buf_pos += 1

        if buf_pos >= window_samples:
            epoch = np.copy(buffer)
            proba = model.predict_proba(epoch)
            proba_buf.append(proba)
            voted = proba if len(proba_buf) == 1 else np.mean(np.vstack(proba_buf), axis=0)
            p_mi = float(voted[mi_idx])
            pred_lbl = int(classes_order[int(np.argmax(voted))])
            print(f'p_mi={p_mi:.3f} pred={pred_lbl} proba={proba} voted={voted}')

            now = time.time()
            if p_mi >= threshold and (now - last_press) >= min_interval:
                ok = press_key_once(key)
                last_press = now
                print(f'Action: key "{key}" pressed (p_mi={p_mi:.2f} >= {threshold}) status={ok}')

            if block:
                buffer[:] = 0.0
                buf_pos = 0
            else:
                buffer = np.roll(buffer, -step_samples, axis=1)
                buf_pos = window_samples - step_samples


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--sfreq', type=float, required=True)
    ap.add_argument('--window', type=float, default=4.0)
    ap.add_argument('--step', type=float, default=0.5)
    ap.add_argument('--picks', type=str, default='C3,Cz,C4')
    ap.add_argument('--scale-to-uv', action='store_true')
    ap.add_argument('--vote-k', type=int, default=5)
    ap.add_argument('--mi-label', type=int, default=0, help='Which label corresponds to MI (default 0)')
    ap.add_argument('--threshold', type=float, default=0.65, help='Probability threshold for MI trigger')
    ap.add_argument('--key', type=str, default='a', help='Key to press on trigger')
    ap.add_argument('--min-interval', type=float, default=1.0, help='Minimum seconds between key presses')
    ap.add_argument('--block', action='store_true', help='Use non-overlapping windows')
    args = ap.parse_args()

    run(model_path=args.model, sfreq=args.sfreq, window_s=args.window, step_s=args.step,
        picks=args.picks, scale_to_uV=args.scale_to_uv, vote_k=args.vote_k,
        mi_label=args.mi_label, threshold=args.threshold, key=args.key,
        min_interval=args.min_interval, block=args.block)


if __name__ == '__main__':
    main()

"""Real-time classifier connecting to an EEG LSL stream.

It expects the incoming samples to be in microvolts (uV). If your stream provides volts, set `scale_to_uV=True`.

This script buffers a sliding window and classifies each window with the saved FBCSP+LDA model.
"""
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from pathlib import Path
import sys
from collections import deque

# Ensure project root is on sys.path to import src
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
    # fallback first 3
    return list(range(min(3, info.channel_count())))


def parse_class_names(arg):
    if not arg:
        return {}
    mapping = {}
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    for p in parts:
        if ':' in p:
            k, v = p.split(':', 1)
            mapping[k.strip()] = v.strip()
    return mapping


def run(model_path='fbcsp_lda.joblib', sfreq=250.0, window_s=3.0, step_s=0.5, scale_to_uV=False, picks=None, vote_k=1, class_names=None, block=False):
    model = FBCSP.load(model_path)
    # Ensure sampling rate consistent
    model.sfreq = sfreq
    stream_info = find_eeg_stream()
    inlet = StreamInlet(stream_info)
    pick_idxs = resolve_picks(stream_info, picks)
    n_channels = len(pick_idxs)
    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    # ring buffer
    buffer = np.zeros((n_channels, window_samples))
    buf_pos = 0

    print(f'Listening to stream "{stream_info.name()}" using channels {pick_idxs}, window {window_s}s')

    # voting/smoothing buffers
    proba_buf = deque(maxlen=max(1, int(vote_k)))
    classes_order = getattr(model.lda, 'classes_', np.array([0, 1]))
    name_map = parse_class_names(class_names)
    def lab2name(lbl):
        return name_map.get(str(lbl), str(lbl))

    # continuously pull samples
    while True:
        sample, timestamp = inlet.pull_sample()
        if sample is None:
            time.sleep(0.001)
            continue
        arr = np.asarray(sample, dtype=float)
        if scale_to_uV:
            arr = arr * 1e6
        # push into buffer (assume channels in sample order)
        buffer[:, buf_pos] = arr[pick_idxs]
        buf_pos += 1
        if buf_pos >= window_samples:
            # full window -> classify
            epoch = np.copy(buffer)
            pred = model.predict(epoch)
            proba = model.predict_proba(epoch)
            proba_buf.append(proba)
            # smoothing
            if len(proba_buf) == 1:
                voted_proba = proba
            else:
                voted_proba = np.mean(np.vstack(proba_buf), axis=0)
            voted_label = classes_order[int(np.argmax(voted_proba))]
            print(f'Prediction: {lab2name(pred)}  Probabilities: {proba} | Voted: {lab2name(voted_label)}  VotedProb: {voted_proba}')
            if block:
                # Non-overlapping windows: clear buffer and start fresh
                buffer[:] = 0.0
                buf_pos = 0
            else:
                # Sliding window: shift by step
                buffer = np.roll(buffer, -step_samples, axis=1)
                buf_pos = window_samples - step_samples


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='fbcsp_lda.joblib')
    ap.add_argument('--sfreq', type=float, default=250.0)
    ap.add_argument('--window', type=float, default=3.0)
    ap.add_argument('--step', type=float, default=0.5)
    ap.add_argument('--scale-to-uv', action='store_true', help='Multiply incoming samples (volts) by 1e6 to convert to uV')
    ap.add_argument('--picks', type=str, default='Cz,C3,C4', help='Channel picks as names or indices CSV (e.g., "Cz,C3,C4" or "0,1,2")')
    ap.add_argument('--vote-k', type=int, default=5, help='Majority/probability vote over last K windows (default 5). Use 1 to disable smoothing')
    ap.add_argument('--block', action='store_true', help='Use non-overlapping windows (wait full window each time)')
    ap.add_argument('--class-names', type=str, default='0:rest,1:hand_mi', help='Mapping for label display, e.g., "0:rest,1:hand_mi"')
    args = ap.parse_args()

    run(model_path=args.model, sfreq=args.sfreq, window_s=args.window, step_s=args.step, scale_to_uV=args.scale_to_uv, picks=args.picks, vote_k=args.vote_k, class_names=args.class_names, block=args.block)

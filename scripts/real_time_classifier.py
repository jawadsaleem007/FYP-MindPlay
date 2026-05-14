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

try:
    from scripts.command_cooldown import cooldown_status, resolve_state_file
except ImportError:
    from command_cooldown import cooldown_status, resolve_state_file

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


def resolve_numeric_picks(idxs, channel_count):
    if all(0 <= idx < channel_count for idx in idxs):
        return idxs
    if len(idxs) == channel_count:
        fallback = list(range(channel_count))
        print(
            f"Warning: requested source channel picks {idxs} are outside this {channel_count}-channel stream; "
            f"assuming the stream is already narrowed and using indices {fallback}"
        )
        return fallback
    raise ValueError(f"Channel picks {idxs} are out of range for stream with {channel_count} channels")


def resolve_picks(info, picks_arg, default_names=('Cz', 'C3', 'C4')):
    labels = get_channel_labels(info)
    idxs, names = parse_picks(picks_arg) if picks_arg else (None, None)
    if idxs is not None:
        return resolve_numeric_picks(idxs, info.channel_count())
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


def resolve_state_class_indexes(classes_order, lab2name):
    hand_idx = None
    rest_idx = None
    for idx, lbl in enumerate(classes_order):
        display = lab2name(lbl).strip().lower()
        raw = str(lbl).strip().lower()
        if hand_idx is None and (display in ('hand_mi', 'hand-mi', 'handmi') or raw in ('1', 'hand_mi', 'hand-mi', 'handmi')):
            hand_idx = idx
        if rest_idx is None and (display.startswith('rest') or raw in ('0', 'rest')):
            rest_idx = idx

    if hand_idx is None and len(classes_order) > 1:
        hand_idx = 1
    if rest_idx is None:
        rest_idx = 0
    return rest_idx, hand_idx


def run(
    model_path='fbcsp_lda.joblib',
    sfreq=250.0,
    window_s=3.0,
    step_s=0.5,
    scale_to_uV=False,
    picks='2,3,4',
    vote_k=1,
    class_names=None,
    block=False,
    hand_mi_threshold=0.97,
    hand_mi_consecutive=3,
    cooldown_state_file='gamepad_state.json',
):
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

    threshold = float(hand_mi_threshold)
    if threshold > 1.0:
        threshold = threshold / 100.0
    threshold = min(1.0, max(0.0, threshold))
    required_streak = max(1, int(hand_mi_consecutive))

    _rest_idx, hand_idx = resolve_state_class_indexes(classes_order, lab2name)
    hand_high_streak = 0
    cooldown_state_path = resolve_state_file(cooldown_state_file)

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
            proba = np.asarray(model.predict_proba(epoch), dtype=float).reshape(-1)
            proba_buf.append(proba)
            # smoothing
            if len(proba_buf) == 1:
                voted_proba = proba
            else:
                voted_proba = np.mean(np.vstack(proba_buf), axis=0)
            voted_label = classes_order[int(np.argmax(voted_proba))]

            if hand_idx is not None and 0 <= hand_idx < len(voted_proba):
                hand_prob = float(voted_proba[hand_idx])
            else:
                hand_prob = 0.0

            if hand_prob > threshold:
                hand_high_streak += 1
            else:
                hand_high_streak = 0

            realtime_state = 'hand_mi' if hand_high_streak >= required_streak else 'rest(0)'
            cooldown_note = ''
            if realtime_state == 'hand_mi':
                blocked, remaining, source = cooldown_status(cooldown_state_path)
                if blocked:
                    realtime_state = 'rest(0)'
                    hand_high_streak = 0
                    src = f' after {source}' if source else ''
                    cooldown_note = f' | SuppressedByGyroCooldown{src}: {remaining:.1f}s remaining'

            print(
                f'Prediction: {lab2name(pred)}  Probabilities: {proba} | '
                f'Voted: {lab2name(voted_label)}  VotedProb: {voted_proba} | '
                f'RealTimeState: {realtime_state} '
                f'(hand_mi_prob={hand_prob:.4f}, streak={hand_high_streak}/{required_streak}, threshold>{threshold:.2f})'
                f'{cooldown_note}'
            )
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
    ap.add_argument('--picks', type=str, default='2,3,4', help='Channel picks as names or indices CSV (current Smarting setup uses "2,3,4")')
    ap.add_argument('--vote-k', type=int, default=5, help='Majority/probability vote over last K windows (default 5). Use 1 to disable smoothing')
    ap.add_argument('--block', action='store_true', help='Use non-overlapping windows (wait full window each time)')
    ap.add_argument('--class-names', type=str, default='0:rest,1:hand_mi', help='Mapping for label display, e.g., "0:rest,1:hand_mi"')
    ap.add_argument('--hand-mi-threshold', type=float, default=0.97, help='Minimum hand_mi probability required. Values >1 are treated as percent (e.g., 97 => 0.97).')
    ap.add_argument('--hand-mi-consecutive', type=int, default=3, help='Consecutive windows above threshold required before emitting hand_mi.')
    ap.add_argument('--cooldown-state-file', type=str, default='gamepad_state.json', help='Shared state JSON containing gyro cooldown info; blank disables suppression')
    args = ap.parse_args()

    run(
        model_path=args.model,
        sfreq=args.sfreq,
        window_s=args.window,
        step_s=args.step,
        scale_to_uV=args.scale_to_uv,
        picks=args.picks,
        vote_k=args.vote_k,
        class_names=args.class_names,
        block=args.block,
        hand_mi_threshold=args.hand_mi_threshold,
        hand_mi_consecutive=args.hand_mi_consecutive,
        cooldown_state_file=args.cooldown_state_file,
    )

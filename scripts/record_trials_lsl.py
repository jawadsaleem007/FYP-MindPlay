"""Record labeled MI vs Rest epochs from an LSL EEG stream.

This script connects to the first LSL stream with type='EEG', selects channels
(Cz,C3,C4 by default), and records labeled trials (epochs) for training.

Outputs:
- epochs.npy : shape (n_trials_total, n_channels, n_samples) in microvolts (uV)
- labels.npy : shape (n_trials_total,) with 0 = MI (hand imagery), 1 = Rest

Notes:
- Ensure your Smarting 24 stream is active over LSL.
- Units: If your stream outputs volts, use --scale-to-uv to convert to microvolts.
- Channel mapping: If your stream exposes channel labels, we will pick by name;
  otherwise we will fall back to indices.
"""
from __future__ import annotations
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_byprop


def find_eeg_stream(timeout: float = 5.0):
    streams = resolve_byprop('type', 'EEG', timeout=timeout)
    if not streams:
        raise RuntimeError('No EEG LSL stream found (type="EEG"). Start your device stream.')
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
        # Fallback to empty labels list if metadata incomplete
        labels = [''] * info.channel_count()
    return labels


def parse_picks(picks_arg: str) -> Tuple[bool, List[str]]:
    # returns (is_index, values)
    parts = [p.strip() for p in picks_arg.split(',') if p.strip()]
    is_index = all(p.isdigit() for p in parts)
    return is_index, parts


def resolve_picks(labels: List[str], picks_arg: str, default_names=('Cz', 'C3', 'C4')) -> List[int]:
    is_index, vals = parse_picks(picks_arg) if picks_arg else (False, list(default_names))
    if is_index:
        idxs = [int(v) for v in vals]
        return idxs
    # name-based
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    idxs = []
    for name in vals:
        i = lower_map.get(name.lower())
        if i is not None:
            idxs.append(i)
    if len(idxs) == len(vals) and len(idxs) > 0:
        return idxs
    # fallback to defaults by name
    fallback = []
    for name in default_names:
        i = lower_map.get(name.lower())
        if i is not None:
            fallback.append(i)
    if len(fallback) == len(default_names):
        print(f"Picked channels by names {default_names} -> indices {fallback}")
        return fallback
    # final fallback: first 3 channels
    fallback = list(range(min(3, len(labels))))
    print(f"Warning: Could not resolve requested channel names; using indices {fallback}")
    return fallback


def collect_epoch(inlet: StreamInlet, n_channels: int, picks: List[int], samples_needed: int, scale_to_uV: bool) -> np.ndarray:
    buf = np.zeros((n_channels, samples_needed), dtype=float)
    got = 0
    while got < samples_needed:
        chunk, _ = inlet.pull_chunk(timeout=2.0)
        if not chunk:
            continue
        arr = np.asarray(chunk, dtype=float)
        # arr shape: (n_new, n_all_channels)
        take = min(samples_needed - got, arr.shape[0])
        dat = arr[:take, :]
        if scale_to_uV:
            dat = dat * 1e6
        buf[:, got:got+take] = dat[:, picks].T
        got += take
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=str, default='S01', help='Subject ID tag for filenames')
    ap.add_argument('--out-dir', type=str, default='data', help='Output directory')
    ap.add_argument('--picks', type=str, default='Cz,C3,C4', help='Channel names or indices CSV (e.g., "Cz,C3,C4" or "0,1,2")')
    ap.add_argument('--trial-len', type=float, default=3.0, help='Trial epoch length (seconds). Default 3.0s for MI and Rest')
    ap.add_argument('--trials-per-class', type=int, default=40, help='Trials for each class (MI and Rest)')
    ap.add_argument('--prep-len', type=float, default=2.0, help='Preparation time before each trial (seconds)')
    ap.add_argument('--inter-trial', type=float, default=2.0, help='Rest period between trials (seconds)')
    ap.add_argument('--randomize', action='store_true', help='Randomize order of MI/Rest trials')
    ap.add_argument('--scale-to-uv', action='store_true', help='Multiply incoming samples (volts) by 1e6 to convert to uV')
    ap.add_argument('--sfreq', type=float, default=0.0, help='Override sampling rate; if 0 use stream nominal rate')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Resolving LSL EEG stream...')
    info = find_eeg_stream()
    inlet = StreamInlet(info, max_buflen=360)
    stream_sfreq = info.nominal_srate()
    sfreq = args.sfreq if args.sfreq > 0 else stream_sfreq
    if sfreq <= 0:
        raise RuntimeError('Sampling rate could not be determined; provide --sfreq explicitly')

    labels = get_channel_labels(info)
    picks = resolve_picks(labels, args.picks)
    n_channels = len(picks)
    samples_needed = int(round(args.trial_len * sfreq))

    print(f'Stream: {info.name()} | channels={info.channel_count()} | nominal_sfreq={stream_sfreq} Hz')
    if labels and any(labels):
        print('Channel labels:', [labels[i] for i in range(len(labels)) if labels[i]])
    print(f'Using picks indices: {picks}  -> recording {n_channels} channels')
    print(f'Epoch length: {args.trial_len}s ({samples_needed} samples) at sfreq={sfreq} Hz')

    # Build trial sequence: 0=MI, 1=Rest
    seq = [0] * args.trials_per_class + [1] * args.trials_per_class
    if args.randomize:
        rng = np.random.RandomState(42)
        rng.shuffle(seq)

    epochs = []
    labels_out = []

    print('\nStarting acquisition. Follow console cues.')
    time.sleep(1.0)

    def flush_inlet(inlet: StreamInlet):
        # Drain any buffered samples so next epoch starts from "now"
        while True:
            chunk, _ = inlet.pull_chunk(timeout=0.0)
            if not chunk:
                break

    for t_idx, lab in enumerate(seq, start=1):
        cue = 'MI (Hand imagery)' if lab == 0 else 'REST'
        print(f'\nTrial {t_idx}/{len(seq)}: Prepare for {cue} ...')
        time.sleep(args.prep_len)
        print(f'NOW: {cue} (recording {args.trial_len}s)')
        # Ensure we don't consume buffered backlog from previous periods
        flush_inlet(inlet)
        epoch = collect_epoch(inlet, n_channels=n_channels, picks=picks, samples_needed=samples_needed, scale_to_uV=args.scale_to_uv)
        epochs.append(epoch)
        labels_out.append(lab)
        print(f'Done. Inter-trial rest {args.inter_trial}s ...')
        time.sleep(args.inter_trial)

    epochs = np.stack(epochs, axis=0)
    labels_out = np.asarray(labels_out, dtype=int)

    ts = time.strftime('%Y%m%d_%H%M%S')
    epochs_path = out_dir / f'{args.subject}_epochs_{ts}.npy'
    labels_path = out_dir / f'{args.subject}_labels_{ts}.npy'

    np.save(epochs_path, epochs)
    np.save(labels_path, labels_out)

    print('\nSaved:')
    print(' -', epochs_path)
    print(' -', labels_path)
    print(f'epochs shape={epochs.shape}, labels shape={labels_out.shape}')
    print('You can now train:')
    print(f'  python .\\scripts\\train_fbcsp_lda.py --epochs "{epochs_path}" --labels "{labels_path}" --sfreq {sfreq} --out fbcsp_lda_{args.subject}.joblib')


if __name__ == '__main__':
    main()

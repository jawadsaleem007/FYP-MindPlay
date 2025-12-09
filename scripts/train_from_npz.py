"""Train FBCSP+LDA from an NPZ file with keys: data, labels, metadata.

- data: ndarray with shape (n_trials, n_channels, n_samples) or object array of ndarrays
- labels: ndarray shape (n_trials,) with binary classes {0,1}
- metadata: JSON-like string or dict with at least 'sampling_rate'

Options:
- --picks: channel indices CSV to select, default '0,1,2' (e.g., Cz,C3,C4 positions in your dataset)
- --crop-start/--crop-dur: optionally crop each trial to a subwindow in seconds (e.g., start=1.0 dur=3.0)
- --out: output model path (.joblib)

Units: Expected microvolts (uV). No scaling is applied.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Ensure project root is on sys.path to import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fbcsp import FBCSP


def to_numpy_epochs(obj_arr: np.ndarray) -> np.ndarray:
    # Convert possibly object-typed array to float ndarray
    if obj_arr.dtype != object:
        return obj_arr.astype(float, copy=False)
    n_trials = obj_arr.shape[0]
    # infer ch, samples from first trial
    first = obj_arr[0]
    arr = np.stack([np.asarray(obj_arr[i], dtype=float) for i in range(n_trials)], axis=0)
    return arr


def parse_picks(picks_csv: str, n_channels: int) -> np.ndarray:
    parts = [p.strip() for p in picks_csv.split(',') if p.strip()]
    idxs = [int(p) for p in parts]
    for i in idxs:
        if i < 0 or i >= n_channels:
            raise ValueError(f'Pick index {i} out of range [0,{n_channels-1}]')
    return np.asarray(idxs, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, required=True, help='Path to NPZ file with data/labels/metadata')
    ap.add_argument('--picks', type=str, default='0,1,2', help='Channel indices CSV to select (e.g., "0,1,2")')
    ap.add_argument('--crop-start', type=float, default=None, help='Crop start time (s) within each trial')
    ap.add_argument('--crop-dur', type=float, default=None, help='Crop duration (s) within each trial')
    ap.add_argument('--out', type=str, default='fbcsp_lda_from_npz.joblib')
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    data = npz['data']
    labels = npz['labels']
    metadata = npz.get('metadata')
    if metadata is not None and isinstance(metadata.item() if hasattr(metadata, 'item') else metadata, str):
        try:
            meta = json.loads(metadata.item() if hasattr(metadata, 'item') else metadata)
        except Exception:
            meta = {}
    else:
        meta = {}

    X = to_numpy_epochs(data)  # (n_trials, n_channels, n_samples)
    y = labels.astype(int)
    n_trials, n_channels, n_samples = X.shape

    # Sampling rate from metadata
    sfreq = float(meta.get('sampling_rate', 0)) if isinstance(meta, dict) else 0.0
    if not sfreq or sfreq <= 0:
        # best-effort inference from trial_duration if present
        trial_duration = float(meta.get('trial_duration', 0.0)) if isinstance(meta, dict) else 0.0
        if trial_duration and n_samples and trial_duration > 0:
            sfreq = n_samples / trial_duration
        else:
            raise RuntimeError('Could not determine sampling rate; ensure metadata contains sampling_rate or provide consistent trial_duration')

    # Picks
    picks = parse_picks(args.picks, n_channels)
    X = X[:, picks, :]

    # Crop
    if args.crop_start is not None and args.crop_dur is not None:
        start_samp = int(round(args.crop_start * sfreq))
        stop_samp = start_samp + int(round(args.crop_dur * sfreq))
        if start_samp < 0 or stop_samp > n_samples:
            raise ValueError('Crop window is out of range of trial samples')
        X = X[:, :, start_samp:stop_samp]

    # Train
    model = FBCSP(sfreq=sfreq)
    model.fit(X, y)
    model.save(args.out)
    print(f'Trained FBCSP+LDA on {X.shape[0]} trials, channels={X.shape[1]}, samples={X.shape[2]}, sfreq={sfreq:.2f} Hz')
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()

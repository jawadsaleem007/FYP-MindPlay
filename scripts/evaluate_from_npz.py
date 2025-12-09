"""Evaluate FBCSP+LDA via K-fold CV on NPZ data.

Usage:
  python scripts/evaluate_from_npz.py --npz data\raw_data_XXXX.npz --picks 0,3,4 --crop-start 0 --crop-dur 3.0 --folds 5
Prints mean accuracy and std across folds.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fbcsp import FBCSP


def to_numpy_epochs(obj_arr: np.ndarray) -> np.ndarray:
    if obj_arr.dtype != object:
        return obj_arr.astype(float, copy=False)
    return np.stack([np.asarray(obj_arr[i], dtype=float) for i in range(obj_arr.shape[0])], axis=0)


def parse_picks(picks_csv: str, n_channels: int) -> np.ndarray:
    parts = [p.strip() for p in picks_csv.split(',') if p.strip()]
    idxs = [int(p) for p in parts]
    for i in idxs:
        if i < 0 or i >= n_channels:
            raise ValueError(f'Pick index {i} out of range [0,{n_channels-1}]')
    return np.asarray(idxs, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--picks', type=str, default='0,1,2')
    ap.add_argument('--crop-start', type=float, default=None)
    ap.add_argument('--crop-dur', type=float, default=None)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    X = to_numpy_epochs(npz['data'])
    y = npz['labels'].astype(int)
    meta_raw = npz.get('metadata')
    meta = {}
    if meta_raw is not None:
        try:
            meta = json.loads(meta_raw.item() if hasattr(meta_raw, 'item') else meta_raw)
        except Exception:
            meta = {}

    n_trials, n_channels, n_samples = X.shape
    picks = parse_picks(args.picks, n_channels)
    X = X[:, picks, :]

    if args.crop_start is not None and args.crop_dur is not None:
        sfreq = float(meta.get('sampling_rate', 0))
        if not sfreq or sfreq <= 0:
            raise RuntimeError('Sampling rate missing in metadata for cropping. Provide a valid sampling_rate.')
        start = int(round(args.crop_start * sfreq))
        stop = start + int(round(args.crop_dur * sfreq))
        if start < 0 or stop > n_samples:
            raise ValueError('Crop window out of range')
        X = X[:, :, start:stop]
        n_samples = X.shape[2]

    sfreq = float(meta.get('sampling_rate', 0))
    if not sfreq or sfreq <= 0:
        # attempt inference
        trial_duration = float(meta.get('trial_duration', 0))
        sfreq = n_samples / trial_duration if trial_duration else 0
        if not sfreq:
            raise RuntimeError('Could not determine sampling rate')

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X, y):
        model = FBCSP(sfreq=sfreq)
        model.fit(X[tr], y[tr])
        preds = [model.predict(ep) for ep in X[te]]
        accs.append(accuracy_score(y[te], preds))

    print(f'CV ({args.folds}-fold) accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}')
    print(f'(trials={n_trials}, channels={X.shape[1]}, samples={n_samples}, sfreq={sfreq:.1f} Hz)')


if __name__ == '__main__':
    main()

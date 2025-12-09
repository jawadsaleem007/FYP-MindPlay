"""Evaluate a trained FBCSP+LDA model on provided epochs/labels with K-fold CV.

This re-computes CSP+LDA per fold (not just using the saved model) to estimate generalization.
Also reports the accuracy of the loaded model (single fit) on the whole dataset.

Usage:
  python scripts/evaluate_trained_model.py --model fbcsp_lda_S02.joblib \
      --epochs data\S02_epochs_20251130_172259.npy --labels data\S02_labels_20251130_172259.npy --sfreq 500 --folds 5

Options:
  --picks 0,1,2         (optional) subset channels
  --crop-start 0 --crop-dur 4.0  (optional) crop window in seconds
  --n-csp 2             (override number of CSP pairs per band)
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure project root path for src import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fbcsp import FBCSP


def load_epochs_labels(ep_path: str, lab_path: str):
    X = np.load(ep_path)
    y = np.load(lab_path).astype(int)
    if X.ndim != 3:
        raise ValueError('epochs array must be 3D (n_epochs, n_channels, n_samples)')
    return X, y


def parse_picks(picks_csv: str, n_channels: int):
    parts = [p.strip() for p in picks_csv.split(',') if p.strip()]
    idxs = [int(p) for p in parts]
    for i in idxs:
        if i < 0 or i >= n_channels:
            raise ValueError(f'Pick {i} out of range [0,{n_channels-1}]')
    return np.asarray(idxs, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=str)
    ap.add_argument('--epochs', required=True, type=str)
    ap.add_argument('--labels', required=True, type=str)
    ap.add_argument('--sfreq', required=True, type=float)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--picks', type=str, default=None)
    ap.add_argument('--crop-start', type=float, default=None)
    ap.add_argument('--crop-dur', type=float, default=None)
    ap.add_argument('--n-csp', type=int, default=None)
    args = ap.parse_args()

    X, y = load_epochs_labels(args.epochs, args.labels)
    n_epochs, n_channels, n_samples = X.shape

    # subset channels
    if args.picks:
        picks = parse_picks(args.picks, n_channels)
        X = X[:, picks, :]
        n_channels = X.shape[1]
    else:
        picks = None

    # crop window
    if args.crop_start is not None and args.crop_dur is not None:
        start = int(round(args.crop_start * args.sfreq))
        stop = start + int(round(args.crop_dur * args.sfreq))
        if start < 0 or stop > n_samples:
            raise ValueError('Crop window out of range')
        X = X[:, :, start:stop]
        n_samples = X.shape[2]

    # Load existing trained model (single fit performance)
    loaded = FBCSP.load(args.model)
    # Predict each epoch with loaded model (assuming same sfreq & channel order)
    try:
        loaded_preds = [loaded.predict(ep) for ep in X]
        loaded_acc = accuracy_score(y, loaded_preds)
    except Exception as e:
        loaded_acc = None
        print(f'Warning: could not evaluate loaded model directly: {e}')

    # Cross-validation re-training per fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    accs = []
    cms = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = FBCSP(sfreq=args.sfreq)
        if args.n_csp is not None:
            model.n_csp = args.n_csp
        model.fit(X[tr], y[tr])
        preds = [model.predict(ep) for ep in X[te]]
        acc = accuracy_score(y[te], preds)
        accs.append(acc)
        cms.append(confusion_matrix(y[te], preds, labels=np.unique(y)))
        print(f'Fold {fold_idx}: acc={acc:.3f}')

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    print('\nSummary:')
    print(f'CV {args.folds}-fold accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')
    if loaded_acc is not None:
        print(f'Loaded model (single fit) accuracy: {loaded_acc:.3f}')
    print(f'Dataset: epochs={n_epochs}, channels={n_channels}, samples={n_samples}, sfreq={args.sfreq:.1f} Hz')

    # Aggregate confusion matrix
    if cms:
        total_cm = sum(cms)
        print('Confusion matrix (summed over folds):')
        print(total_cm)
        # per-class metrics
        tp = np.diag(total_cm)
        support = total_cm.sum(axis=1)
        for i, cls in enumerate(np.unique(y)):
            prec = tp[i] / total_cm[:, i].sum() if total_cm[:, i].sum() else 0
            rec = tp[i] / support[i] if support[i] else 0
            print(f' Class {cls}: precision={prec:.3f} recall={rec:.3f} support={support[i]}')

if __name__ == '__main__':
    main()

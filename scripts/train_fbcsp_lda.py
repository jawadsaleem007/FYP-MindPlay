"""Train FBCSP+LDA on per-subject data.

Usage examples:
  - Provide pre-epoched data: arrays saved with numpy: `epochs.npy` shape (n_epochs, n_channels, n_samples) and `labels.npy` shape (n_epochs,)
  - Or run without args to train on synthetic data (for quick test).

The model is saved with joblib and contains CSP filters and the LDA classifier.

Units: This pipeline expects EEG data in microvolts (uV). If your recordings are in volts, multiply by 1e6 before training or set `scale_to_uV=True`.
"""
import argparse
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path to import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fbcsp import FBCSP


def generate_synthetic(n_epochs=100, n_channels=3, n_samples=250, sfreq=250):
    # simple synthetic: class 0 has stronger power in mu (8-12 Hz) on channel 1
    rng = np.random.RandomState(42)
    X = rng.randn(n_epochs, n_channels, n_samples) * 5.0  # baseline noise (microvolts)
    t = np.arange(n_samples) / float(sfreq)
    for i in range(n_epochs):
        if i % 2 == 0:
            # class 0: add 10 Hz sine to channel 1
            X[i, 0] += 10.0 * np.sin(2 * np.pi * 10.0 * t)
    y = np.array([0 if i % 2 == 0 else 1 for i in range(n_epochs)])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=str, help='Path to numpy .npy file with epochs (n_epochs,n_channels,n_samples)')
    p.add_argument('--labels', type=str, help='Path to numpy .npy file with labels (n_epochs,)')
    p.add_argument('--sfreq', type=float, default=250.0)
    p.add_argument('--out', type=str, default='fbcsp_lda.joblib')
    args = p.parse_args()

    if args.epochs and args.labels:
        epochs = np.load(args.epochs)
        labels = np.load(args.labels)
    else:
        print('No data files provided — generating synthetic dataset for demonstration')
        epochs, labels = generate_synthetic(n_epochs=120, n_channels=3, n_samples=int(args.sfreq))

    # Ensure shape
    assert epochs.ndim == 3

    model = FBCSP(sfreq=args.sfreq)
    model.fit(epochs, labels)
    model.save(args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()

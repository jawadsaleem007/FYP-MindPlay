"""Mock LSL EEG sender for testing real-time classification.

Streams trials from an NPZ file (keys: data, labels, metadata) as an LSL EEG stream.
Default units: microvolts (uV), matching the rest of the pipeline.

Example:
  # Terminal 1: start sender (C3,Cz,C4 indices 0,1,2 at 500 Hz)
  python .\scripts\mock_lsl_sender.py --npz data\raw_data_20251128_105528.npz --picks 0,1,2 --sfreq 500 --name MockEEG

  # Terminal 2: run the classifier (4s window)
  python .\scripts\real_time_classifier.py --model fbcsp_lda_from_npz_012_4s.joblib --sfreq 500 --window 4.0 --step 0.5 --picks 0,1,2 --scale-to-uv
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


def to_numpy_epochs(obj_arr: np.ndarray) -> np.ndarray:
    if obj_arr.dtype != object:
        return obj_arr.astype(float, copy=False)
    return np.stack([np.asarray(obj_arr[i], dtype=float) for i in range(obj_arr.shape[0])], axis=0)


def build_info(name: str, n_channels: int, sfreq: float, labels=None):
    info = StreamInfo(name=name, type='EEG', channel_count=n_channels, nominal_srate=sfreq, channel_format='float32', source_id='mock_eeg_sender')
    if labels is None:
        labels = [f'ch{i}' for i in range(n_channels)]
    chs = info.desc().append_child('channels')
    for lab in labels:
        ch = chs.append_child('channel')
        ch.append_child_value('label', str(lab))
        ch.append_child_value('unit', 'uV')
        ch.append_child_value('type', 'EEG')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, required=True)
    ap.add_argument('--picks', type=str, default='0,1,2', help='Channel indices CSV (e.g., 0,1,2)')
    ap.add_argument('--sfreq', type=float, default=0.0, help='Sampling rate; 0 uses metadata value')
    ap.add_argument('--name', type=str, default='MockEEG')
    ap.add_argument('--loop', action='store_true', help='Loop over trials indefinitely')
    ap.add_argument('--gap', type=float, default=0.1, help='Seconds of gap between trials')
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
    sfreq = float(args.sfreq or meta.get('sampling_rate', 0) or 0)
    if sfreq <= 0:
        raise RuntimeError('Sampling rate missing; pass --sfreq or include sampling_rate in metadata')

    picks = [int(p.strip()) for p in args.picks.split(',') if p.strip()]
    X = X[:, picks, :]
    n_trials, n_channels, n_samples = X.shape

    # Labels for stream metadata
    labels = [f'ch{idx}' for idx in picks]

    info = build_info(args.name, n_channels=n_channels, sfreq=sfreq, labels=labels)
    outlet = StreamOutlet(info)

    print(f'Streaming {n_trials} trials | channels={n_channels} picks={picks} | sfreq={sfreq} Hz | name={args.name}')

    try:
        while True:
            for i in range(n_trials):
                trial = X[i]
                t0 = local_clock()
                for s in range(n_samples):
                    outlet.push_sample(trial[:, s].astype(np.float32).tolist())
                    # naive pacing
                    time.sleep(1.0 / sfreq)
                print(f'sent trial {i+1}/{n_trials} (label={y[i]}) in {local_clock()-t0:.2f}s')
                if args.gap > 0:
                    time.sleep(args.gap)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass

    print('Mock sender finished.')


if __name__ == '__main__':
    main()

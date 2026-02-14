**FBCSP+LDA EEG pipeline**

- **Purpose**: Train an individual FBCSP+LDA classifier for Hand motor imagery vs Rest using Cz,C3,C4 channels (or any 3-channel montage). The pipeline expects microvolts (uV) units throughout.

- **Files**:
- `src/fbcsp.py`: FBCSP implementation + LDA training, save/load.
- `scripts/train_fbcsp_lda.py`: Training script. Accepts `.npy` epoch and label files or can generate synthetic data.
- `scripts/real_time_classifier.py`: Connects to LSL EEG stream (type='EEG'), buffers windows, classifies in real-time using a saved model.
- `scripts/record_trials_lsl.py`: Records labeled MI vs Rest epochs from an LSL EEG stream (default picks Cz,C3,C4) and saves `epochs.npy` and `labels.npy` in microvolts.
- `requirements.txt`: Python dependencies.
- `tests/test_pipeline.py`: Synthetic data unit test for quick verification.

- **Data format for training**:
  - `epochs.npy`: numpy array of shape `(n_epochs, n_channels, n_samples)` in microvolts (uV).
  - `labels.npy`: numpy array of shape `(n_epochs,)` with values `0` (Hand MI) or `1` (Rest), binary.

- **Train** (example):

```powershell
python .\scripts\train_fbcsp_lda.py --epochs epochs.npy --labels labels.npy --sfreq 250 --out my_model.joblib
```

- **Real-time** (example):

```powershell
python .\scripts\real_time_classifier.py --model my_model.joblib --sfreq 250 --window 3.0 --step 0.5 --scale-to-uv
```

## GUI app (menu + full flow)

Built with **PyQt6** for a modern desktop UI.

Run:

```powershell
python .\scripts\gui_app.py
```

GUI includes:
- Start Training
- Real Time Classification
- Exit

Training flow in GUI:
1. Record trials (`record_trials_lsl.py`)
2. Train model (`train_fbcsp_lda.py`) with message: *Its Training Wait Please*
3. Evaluate model (`evaluate_trained_model.py`)

If evaluation accuracy is below 60%, GUI asks **Retry** or **Continue**.

If your Smarting24 LSL stream provides values in volts, pass `--scale-to-uv` to convert to microvolts. The pipeline expects channels ordered as Cz, C3, C4 or similar mapping; ensure channel ordering matches how you record.

**Record labeled data** (MI vs Rest)
- Start your Smarting LSL stream, then run:

```powershell
python .\scripts\record_trials_lsl.py --subject S01 --picks Cz,C3,C4 --trial-len 3.0 --trials-per-class 40 --prep-len 2.0 --inter-trial 2.0 --randomize --scale-to-uv
```

- This saves `epochs.npy` and `labels.npy` under `data/` with timestamps. Use the printed training command to train your per-subject model.

**Notes & next steps**:
- Before using with real subject data, collect epoched trials (e.g., 3s windows aligned to trial onset) and save as `epochs.npy`/`labels.npy` in uV.
- The real-time script uses a sliding window; tune `--window` and `--step` to your application.
- If you want channel selection or referencing (e.g., average reference), preprocess the raw stream to match training data.

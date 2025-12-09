import numpy as np
from src.fbcsp import FBCSP


def generate_synthetic(n_epochs=100, n_channels=3, n_samples=250, sfreq=250):
    rng = np.random.RandomState(0)
    X = rng.randn(n_epochs, n_channels, n_samples) * 5.0
    t = np.arange(n_samples) / float(sfreq)
    for i in range(n_epochs):
        if i % 2 == 0:
            # class 0: stronger 10 Hz on channel 0
            X[i, 0] += 8.0 * np.sin(2 * np.pi * 10.0 * t)
    y = np.array([0 if i % 2 == 0 else 1 for i in range(n_epochs)])
    return X, y


def test_fbcsp_train_and_predict():
    X, y = generate_synthetic(n_epochs=80, n_channels=3, n_samples=250, sfreq=250)
    model = FBCSP(sfreq=250)
    model.fit(X, y)
    # evaluate on held-out synthetic
    X_test, y_test = generate_synthetic(n_epochs=20, n_channels=3, n_samples=250, sfreq=250)
    preds = [model.predict(epoch) for epoch in X_test]
    acc = np.mean(np.array(preds) == y_test)
    print('Synthetic accuracy:', acc)
    assert acc >= 0.6

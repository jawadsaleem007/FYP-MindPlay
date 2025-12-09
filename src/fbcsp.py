import numpy as np
from scipy import signal, linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib


def bandpass_filter(data, sfreq, low, high, order=4):
    nyq = 0.5 * sfreq
    lowcut = low / nyq
    highcut = high / nyq
    b, a = signal.butter(order, [lowcut, highcut], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def epoch_covariance(epoch):
    # epoch: channels x samples
    cov = np.cov(epoch)
    # normalize
    return cov / np.trace(cov)


class FBCSP:
    """Filter Bank CSP + LDA pipeline.

    Notes:
    - Expects input epochs in microvolts (uV). Keep units consistent across training and realtime.
    - X shape for fit: (n_epochs, n_channels, n_samples)
    """

    def __init__(self, bands=None, n_csp=2, sfreq=250):
        # bands: list of (low,high) tuples
        if bands is None:
            # Typical motor imagery bands (Hz)
            bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 30)]
        self.bands = bands
        self.n_csp = n_csp  # number of pairs (selected per band will be 2*n_csp)
        self.sfreq = sfreq
        self.csp_filters = []  # list per band: filters shape (n_channels, 2*n_csp)
        self.lda = None

    def _fit_csp_for_band(self, X_band, y):
        # X_band: n_epochs x n_channels x n_samples (bandpassed)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError('CSP implementation requires exactly 2 classes')

        # compute mean covariance per class
        covs = {c: np.mean([epoch_covariance(epoch) for epoch, lab in zip(X_band, y) if lab == c], axis=0) for c in classes}
        composite = covs[classes[0]] + covs[classes[1]]
        # solve generalized eigenvalue problem
        eigvals, eigvecs = linalg.eigh(covs[classes[0]], composite)
        # sort eigenvalues descending
        ix = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, ix]
        # select n_csp from both sides
        selected = np.hstack([eigvecs[:, :self.n_csp], eigvecs[:, -self.n_csp:]])
        return selected

    def fit(self, X, y):
        """Fit CSP filters for each band and train an LDA on extracted features.

        X: n_epochs x n_channels x n_samples
        y: n_epochs
        """
        n_epochs, n_channels, n_samples = X.shape
        self.csp_filters = []
        features = []

        for band in self.bands:
            low, high = band
            X_band = np.array([bandpass_filter(epoch, self.sfreq, low, high) for epoch in X])
            W = self._fit_csp_for_band(X_band, y)
            self.csp_filters.append(W)
            # transform epochs
            feats_band = []
            for epoch in X_band:
                S = W.T @ epoch
                var = np.var(S, axis=1)
                # log-variance as features
                feats_band.append(np.log(var / np.sum(var)))
            features.append(np.array(feats_band))

        X_features = np.concatenate(features, axis=1)
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_features, y)

    def transform_epoch(self, epoch):
        """Transform a single epoch (n_channels x n_samples) into feature vector."""
        feats = []
        for band, W in zip(self.bands, self.csp_filters):
            low, high = band
            epoch_band = bandpass_filter(epoch, self.sfreq, low, high)
            S = W.T @ epoch_band
            var = np.var(S, axis=1)
            feats.append(np.log(var / np.sum(var)))
        return np.concatenate(feats)

    def predict(self, epoch):
        feats = self.transform_epoch(epoch)
        return self.lda.predict(feats.reshape(1, -1))[0]

    def predict_proba(self, epoch):
        feats = self.transform_epoch(epoch)
        return self.lda.predict_proba(feats.reshape(1, -1))[0]

    def save(self, path):
        joblib.dump({'bands': self.bands, 'n_csp': self.n_csp, 'sfreq': self.sfreq, 'csp_filters': self.csp_filters, 'lda': self.lda}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(bands=data['bands'], n_csp=data['n_csp'], sfreq=data['sfreq'])
        obj.csp_filters = data['csp_filters']
        obj.lda = data['lda']
        return obj

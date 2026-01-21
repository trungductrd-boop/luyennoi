import numpy as np
import librosa

def extract_mfcc_mean(path: str, n_mfcc: int = 13, sr: int = 16000, include_deltas: bool = True) -> np.ndarray:
    """Load audio from `path`, compute MFCCs and return a 1D numpy array of mean MFCCs.

    If include_deltas is True, the returned vector concatenates mean MFCC + mean delta + mean delta-delta.
    """
    # load audio
    y, sr = librosa.load(path, sr=sr)
    # compute mfcc (n_mfcc x frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mean across time axis -> shape (n_mfcc,)
    mfcc_mean = np.mean(mfcc, axis=1)
    if include_deltas:
        # compute first and second deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        delta_mean = np.mean(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)
        out = np.concatenate([mfcc_mean, delta_mean, delta2_mean])
    else:
        out = mfcc_mean
    return np.asarray(out)

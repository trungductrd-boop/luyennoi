"""
helpers_multimodal.py

Multimodal visual/mouth analysis removed.

This module previously provided visual mouth/tongue proxies and fusion code
used by an `/analyze-mouth` endpoint. That endpoint and the associated
visual analysis have been intentionally removed — keep this lightweight
stub to avoid import errors in helpers.py.
"""

def analyze_mouth_series_extended(samples, mar_threshold=0.18):
    # Disabled: return neutral analysis
    return {"count": 0, "mouth_score": 0.0, "avg_tongue_vis": 0.0, "jaw_std": 0.0, "speaking_fraction": 0.0}

def compute_audio_articulatory_features(path, sr=16000):
    return {"f1": None, "f2": None, "harmonic_ratio": 0.0, "centroid": 0.0, "rms": 0.0}

def fuse_multimodal_enhanced(*args, **kwargs):
    # return neutral combined score and empty details
    return 0.0, {"advice": [], "mouth_score": 0.0}

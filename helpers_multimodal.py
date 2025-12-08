from typing import List, Dict, Any, Tuple, Optional
import math
import statistics
import numpy as np
import librosa

# ---------------------
# Audio utilities
# ---------------------
def lpc_formants(y: np.ndarray, sr: int, order: int = 16) -> List[float]:
    """
    Estimate formants using LPC. Returns list of formant frequencies in Hz (sorted).
    Heuristic implementation; for production consider Praat/parselmouth.
    """
    if y is None or len(y) < 10:
        return []
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    # Apply hamming window on a central frame for stable vowels
    frame_len = min(len(y), int(0.03 * sr))  # 30ms
    start = max(0, len(y)//2 - frame_len//2)
    frame = y[start:start+frame_len] * np.hamming(frame_len)
    # LPC
    try:
        a = librosa.lpc(frame, order=order)
    except Exception:
        return []
    # find roots
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) >= 0.01]
    angz = np.angle(roots)
    freqs = sorted((angz * (sr / (2.0 * np.pi))).tolist())
    # Filter plausible formants range
    formants = [f for f in freqs if 90 < f < 5000]
    return formants


def estimate_formants_from_file(path: str, sr: int = 16000) -> Dict[str, Optional[float]]:
    """
    Load audio file and estimate first two formants F1, F2 (Hz).
    """
    try:
        y, s = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return {"f1": None, "f2": None, "sr": sr}
    formants = lpc_formants(y, sr, order=18)
    f1 = formants[0] if len(formants) > 0 else None
    f2 = formants[1] if len(formants) > 1 else None
    return {"f1": f1, "f2": f2, "sr": s}


def harmonic_ratio_proxy(y: np.ndarray) -> float:
    """
    Simple proxy for harmonic-to-noise ratio: compute energy ratio of harmonic component
    (via librosa.effects.harmonic) to total energy.
    """
    try:
        harm = librosa.effects.harmonic(y)
        e_total = float(np.sum(y * y) + 1e-9)
        e_harm = float(np.sum(harm * harm) + 1e-9)
        return max(0.0, min(1.0, e_harm / e_total))
    except Exception:
        return 0.0


def compute_audio_articulatory_features(path: str, sr: int = 16000) -> Dict[str, Any]:
    """
    Compute F1/F2, harmonic ratio, spectral centroid and rms as articulatory proxies.
    Returns dict: {f1, f2, harmonic_ratio, centroid, rms}
    """
    try:
        y, s = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return {"f1": None, "f2": None, "harmonic_ratio": 0.0, "centroid": 0.0, "rms": 0.0}
    # formants
    formants = lpc_formants(y, s, order=18)
    f1 = float(formants[0]) if len(formants) > 0 else None
    f2 = float(formants[1]) if len(formants) > 1 else None
    # harmonic ratio proxy
    hr = harmonic_ratio_proxy(y)
    # spectral centroid & rms
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=s)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
    except Exception:
        centroid = 0.0
        rms = 0.0
    return {"f1": f1, "f2": f2, "harmonic_ratio": hr, "centroid": centroid, "rms": rms}


# ---------------------
# Visual mouth/tongue proxies
# ---------------------
def compute_visual_proxies_from_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Snapshot expected to contain pixel coordinates or normalized coords for named points:
      - center (mouth center), left_corner, right_corner, upper_inner, lower_inner, chin
    Returns proxies:
      - inner_opening, mouth_width, jaw_disp, tongue_proxy_vis
    """
    def dist(a, b):
        if not a or not b:
            return 0.0
        try:
            return math.hypot(a.get("x", 0) - b.get("x", 0), a.get("y", 0) - b.get("y", 0))
        except Exception:
            return 0.0

    center = snap.get("center")
    left = snap.get("left_corner")
    right = snap.get("right_corner")
    upper = snap.get("upper_inner")
    lower = snap.get("lower_inner")
    chin = snap.get("chin")

    mouth_width = dist(left, right)
    inner_opening = dist(upper, lower)
    jaw_disp = dist(chin, center)  # proxy
    tongue_proxy_vis = 0.0
    if lower and center:
        tongue_proxy_vis = max(0.0, (center.get("y", 0) - lower.get("y", 0)))
    return {
        "mouth_width": mouth_width,
        "inner_opening": inner_opening,
        "jaw_disp": jaw_disp,
        "tongue_proxy_vis": tongue_proxy_vis
    }


def analyze_mouth_series_extended(samples: List[Dict[str, Any]], mar_threshold: float = 0.18) -> Dict[str, Any]:
    """
    Extended mouth analysis that also computes visual proxies for tongue tip and throat inference.
    Returns a dict with mouth_score, avg_tongue_vis, jaw_std, peaks_per_sec, etc.
    """
    if not samples:
        return {"count": 0, "mouth_score": 0.0, "avg_tongue_vis": 0.0, "jaw_std": 0.0, "speaking_fraction": 0.0}

    mar_vals = []
    tongue_vis_vals = []
    jaw_vals = []
    inner_openings = []
    widths = []
    times = []
    for s in samples:
        try:
            mar_vals.append(float(s.get("normalized_mar", 0.0)))
        except Exception:
            mar_vals.append(0.0)
        proxies = compute_visual_proxies_from_snapshot(s)
        tongue_vis_vals.append(proxies.get("tongue_proxy_vis", 0.0))
        jaw_vals.append(proxies.get("jaw_disp", 0.0))
        inner_openings.append(proxies.get("inner_opening", 0.0))
        widths.append(proxies.get("mouth_width", 0.0))
        ts = s.get("ts")
        if isinstance(ts, (int, float)):
            times.append(float(ts))

    count = len(mar_vals)
    avg_mar = float(statistics.mean(mar_vals)) if mar_vals else 0.0
    mar_std = float(statistics.pstdev(mar_vals)) if mar_vals else 0.0
    speaking_fraction = sum(1 for v in mar_vals if v > mar_threshold) / count if count > 0 else 0.0

    avg_tongue_vis = float(statistics.mean(tongue_vis_vals)) if tongue_vis_vals else 0.0
    jaw_std = float(statistics.pstdev(jaw_vals)) if len(jaw_vals) > 1 else 0.0
    avg_inner_open = float(statistics.mean(inner_openings)) if inner_openings else 0.0
    avg_width = float(statistics.mean(widths)) if widths else 0.0

    # duration estimate
    duration_sec = 0.0
    if len(times) >= 2:
        duration_sec = max(0.001, (max(times) - min(times)) / 1000.0)
    else:
        duration_sec = max(0.001, count / 30.0)

    # detect peaks for mar
    peaks = 0
    for i in range(1, count - 1):
        if mar_vals[i] > mar_vals[i - 1] and mar_vals[i] > mar_vals[i + 1]:
            if mar_vals[i] > (avg_mar + 0.5 * mar_std):
                peaks += 1
    peaks_per_sec = peaks / duration_sec if duration_sec > 0 else 0.0

    p_norm = max(0.0, min(1.0, peaks_per_sec / 8.0))
    mar_std_norm = max(0.0, min(1.0, mar_std / (avg_mar + 1e-6))) if avg_mar > 0 else 0.0
    tongue_vis_norm = max(0.0, min(1.0, avg_tongue_vis / (avg_width + 1e-6))) if avg_width > 0 else 0.0
    jaw_std_norm = max(0.0, min(1.0, jaw_std / (avg_width + 1e-6))) if avg_width > 0 else 0.0

    mouth_score = 0.45 * speaking_fraction + 0.25 * p_norm + 0.15 * mar_std_norm + 0.10 * tongue_vis_norm + 0.05 * jaw_std_norm
    mouth_score = max(0.0, min(1.0, mouth_score))

    return {
        "count": count,
        "avg_mar": avg_mar,
        "mar_std": mar_std,
        "speaking_fraction": speaking_fraction,
        "peaks": peaks,
        "peaks_per_sec": peaks_per_sec,
        "mouth_score": mouth_score,
        "avg_tongue_vis": avg_tongue_vis,
        "jaw_std": jaw_std,
        "avg_inner_open": avg_inner_open,
        "avg_width": avg_width,
        "duration_sec": duration_sec
    }


# ---------------------
# Fusion: include tongue + throat
# ---------------------
def normalize_mfcc_score(mfcc_dist: float, max_dist: float = 120.0) -> float:
    if mfcc_dist is None:
        return 0.0
    d = float(mfcc_dist)
    score = max(0.0, 1.0 - (d / max_dist))
    return min(1.0, score)


def fuse_multimodal_enhanced(audio_mfcc_dist: float,
                             pitch_diff: float,
                             tempo_diff: float,
                             audio_artic: Dict[str, Any],
                             mouth_analysis: Dict[str, Any],
                             weights: Dict[str, float] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Fuse audio + mouth (extended) and return combined score plus tongue/throat specific metrics.
    audio_artic expected keys: f1, f2, harmonic_ratio, centroid, rms
    mouth_analysis from analyze_mouth_series_extended
    """
    if weights is None:
        weights = {"audio": 0.6, "mouth": 0.25, "tongue_vis": 0.15}

    audio_score = normalize_mfcc_score(audio_mfcc_dist)
    mouth_score = float(mouth_analysis.get("mouth_score", 0.0))

    # tongue audio score based on F1/F2 similarity (heuristic)
    f1 = audio_artic.get("f1")
    f2 = audio_artic.get("f2")
    if f2 is not None:
        tongue_audio_score = 0.5 + 0.5 * (max(0.0, min(1.0, (f2 - 600.0) / (2400.0))))
    else:
        tongue_audio_score = 0.5

    # throat audio score: harmonic_ratio and spectral centroid/rms used as proxy
    hr = float(audio_artic.get("harmonic_ratio", 0.0))
    centroid = float(audio_artic.get("centroid", 0.0))
    rms = float(audio_artic.get("rms", 0.0))
    cent_norm = max(0.0, min(1.0, centroid / 4000.0)) if centroid else 0.0
    throat_audio_score = 0.6 * hr + 0.3 * (1.0 - cent_norm) + 0.1 * min(1.0, rms * 10.0)

    # tongue visual score from avg_tongue_vis (normalized by mouth width)
    tongue_vis = float(mouth_analysis.get("avg_tongue_vis", 0.0))
    avg_width = float(mouth_analysis.get("avg_width", 0.0)) or 1.0
    tongue_vis_score = max(0.0, min(1.0, tongue_vis / (avg_width + 1e-9)))

    # throat visual proxy: jaw_std small might indicate rigid jaw -> higher throat tension
    jaw_std = float(mouth_analysis.get("jaw_std", 0.0))
    jaw_score = max(0.0, min(1.0, 1.0 - (jaw_std / (avg_width + 1e-9))))

    # Combine tongue and throat scores
    tongue_combined = 0.65 * tongue_audio_score + 0.35 * tongue_vis_score
    throat_combined = 0.7 * throat_audio_score + 0.3 * jaw_score

    # base combined overall score (audio + mouth)
    base_combined = weights.get("audio", 0.6) * audio_score + weights.get("mouth", 0.25) * mouth_score
    # incorporate tongue and throat as minor modifiers
    combined = 0.85 * base_combined + 0.10 * tongue_combined + 0.05 * throat_combined

    # apply pitch/tempo penalties
    pitch_penalty = max(0.0, min(0.5, abs(float(pitch_diff)) / 100.0)) if pitch_diff is not None else 0.0
    tempo_penalty = max(0.0, min(0.5, abs(float(tempo_diff)) / 100.0)) if tempo_diff is not None else 0.0
    combined = combined * (1.0 - pitch_penalty) * (1.0 - tempo_penalty)
    combined = max(0.0, min(1.0, combined))

    # build advice
    advice = []
    if tongue_combined < 0.4:
        advice.append("Đầu lưỡi có thể chưa đúng vị trí — thử kéo lưỡi lên/ra hoặc quan sát mẫu khi phát âm.")
    elif tongue_combined < 0.7:
        advice.append("Đầu lưỡi khá tốt nhưng chưa chính xác hoàn toàn.")
    else:
        advice.append("Đầu lưỡi trông hợp lý so với mẫu.")

    if throat_combined < 0.4:
        advice.append("Có dấu hiệu căng họng hoặc giọng bị mất độ hài hòa — thử thư giãn cổ/hít thở sâu trước khi nói.")
    elif throat_combined < 0.7:
        advice.append("Throat/tension trung bình — để ý hơi và rung thanh quản.")
    else:
        advice.append("Không thấy dấu hiệu căng họng rõ rệt.")

    # overall advice based on combined score
    if combined > 0.8:
        advice.insert(0, "Phát âm rất giống mẫu. Giữ phong độ!")
    elif combined > 0.6:
        advice.insert(0, "Tương đối giống. Chỉnh nhẹ về cao độ/tốc độ để tốt hơn.")
    elif combined > 0.4:
        advice.insert(0, "Cần luyện: nghe lại mẫu và thử bắt chước kỹ hơn.")
    else:
        advice.insert(0, "Khác khá nhiều — tập từng âm tiết, chú ý miệng mở/đóng theo mẫu.")

    details = {
        "audio_score": audio_score,
        "mouth_score": mouth_score,
        "tongue_audio_score": tongue_audio_score,
        "tongue_vis_score": tongue_vis_score,
        "tongue_combined": tongue_combined,
        "throat_audio_score": throat_audio_score,
        "jaw_score": jaw_score,
        "throat_combined": throat_combined,
        "combined_score": combined,
        "pitch_penalty": pitch_penalty,
        "tempo_penalty": tempo_penalty,
        "raw_audio_artic": audio_artic,
        "mouth_analysis": mouth_analysis,
        "advice": advice
    }
    return combined, details

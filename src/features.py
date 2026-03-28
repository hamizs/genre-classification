from __future__ import annotations
import os
import numpy as np
import librosa

# --- Robust audio loader ------------------------------------------------------
def safe_load_audio(path, sr=22050, mono=True, duration=None):
    """
    Robust loader: try librosa (audioread/soundfile) then soundfile directly.
    Resamples to `sr` and trims to `duration` if provided.
    """
    # 1) Primary: librosa.load (uses soundfile or audioread under the hood)
    try:
        y, s = librosa.load(path, sr=sr, mono=mono, duration=duration)
        return y, s
    except Exception:
        pass
    # 2) Fallback: soundfile + manual resample
    try:
        import soundfile as sf
        y, s = sf.read(path, always_2d=False)
        if y.ndim > 1:  # force mono
            y = y.mean(axis=1)
        if sr is not None and s != sr:
            y = librosa.resample(y=y, orig_sr=s, target_sr=sr)
            s = sr
        if duration is not None:
            n = int(duration * s)
            if y.shape[0] > n:
                y = y[:n]
        return y, s
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

# --- Feature helpers ----------------------------------------------------------
def _agg_stats(x: np.ndarray, prefix: str) -> dict:
    """Aggregate mean/var along time axis for (features x frames) arrays."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    feats = {}
    mu = np.mean(x, axis=1)
    var = np.var(x, axis=1)
    for i, (m, v) in enumerate(zip(mu, var)):
        feats[f"{prefix}_{i:02d}_mean"] = float(m)
        feats[f"{prefix}_{i:02d}_var"] = float(v)
    return feats

def extract_clip_features(
    path: str,
    sr: int = 22050,
    duration: float | None = None,
    n_mfcc: int = 20,
    n_chroma: int = 12,
) -> dict:
    """
    Load audio and compute MFCC(+Δ,+Δ²), Chroma, Spectral stats, ZCR, RMS.
    Returns a flat feature dict aggregated by mean/var.
    """
    y, sr = safe_load_audio(path, sr=sr, mono=True, duration=duration)
    if len(y) == 0:
        raise ValueError(f"Empty/invalid audio: {path}")

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logmel = librosa.power_to_db(mel + 1e-9)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=n_mfcc)
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=n_chroma)
    sc = librosa.feature.spectral_centroid(S=S, sr=sr)
    sbw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    sro = librosa.feature.spectral_rolloff(S=S, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(S=S)

    feats = {}
    feats.update(_agg_stats(mfcc, "mfcc"))
    feats.update(_agg_stats(mfcc_d1, "mfcc_delta"))
    feats.update(_agg_stats(mfcc_d2, "mfcc_delta2"))
    feats.update(_agg_stats(chroma, "chroma"))
    feats.update(_agg_stats(sc, "spec_centroid"))
    feats.update(_agg_stats(sbw, "spec_bandwidth"))
    feats.update(_agg_stats(sro, "spec_rolloff"))
    feats.update(_agg_stats(zcr, "zcr"))
    feats.update(_agg_stats(rms, "rms"))
    feats["logmel_mean"] = float(np.mean(logmel))
    feats["logmel_var"] = float(np.var(logmel))
    feats["duration_sec"] = float(len(y) / sr)
    return feats

def walk_dataset(root: str, exts={".wav", ".mp3", ".au"}) -> list[tuple[str, str]]:
    """
    Return list of (filepath, label) where label is the subfolder (genre) name.
    Expected layout: root/genre_name/*.wav
    """
    pairs = []
    for genre in sorted(os.listdir(root)):
        gdir = os.path.join(root, genre)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            if os.path.splitext(fn)[1].lower() in exts:
                pairs.append((os.path.join(gdir, fn), genre))
    return pairs

"""
AI Mock Interview — Audio Pipeline (all 4 weeks, single file)

Week 1 — Capture + STT:   record mic → WAV (session_id/question_id.wav) + Whisper timestamps
Week 2 — Prosody + Train: MFCC, F0, RMS, WPM, pause ratio, fillers → XGBoost confidence model
Week 3 — Live Pipeline:   WAV → transcript → prosody → confidence score (end-to-end)
Week 4 — Benchmarks:      Whisper tiny vs base RTF/WER, asyncio stress test, model card

Usage:
    python audio_pipeline.py record   --session SESSION --question Q1 --duration 30
    python audio_pipeline.py transcribe --wav path/to/file.wav [--model tiny|base]
    python audio_pipeline.py prosody  --wav path/to/file.wav [--transcript "text"]
    python audio_pipeline.py train    --audio_dir ./samples [--out model.json]
    python audio_pipeline.py validate --wav path/to/file.wav
    python audio_pipeline.py benchmark --wav path/to/file.wav [--stress] [--ref "text"]
    python audio_pipeline.py demo     --wav path/to/file.wav
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import asyncio
import glob
import json
import os
import re
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Literal

# ─── third-party (graceful import errors) ────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("Missing: pip install numpy")

try:
    import librosa
except ImportError:
    sys.exit("Missing: pip install librosa")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
CHANNELS = 1

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "literally", "actually", "so", "right", "okay", "well",
    "i mean", "kind of", "sort of",
}

DEFAULT_MODEL_PATH = Path(__file__).parent / "backend" / "ml" / "models" / "confidence_xgb.json"

_whisper_cache: dict[str, object] = {}
_xgb_model = None


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def wav_filename(session_id: str, question_id: str, base_dir: str = ".") -> str:
    """<base_dir>/<session_id>/<question_id>.wav"""
    return os.path.join(base_dir, session_id, f"{question_id}.wav")


def ensure_dir(session_id: str, base_dir: str) -> str:
    d = os.path.join(base_dir, session_id)
    os.makedirs(d, exist_ok=True)
    return d


def validate_wav(path: str) -> dict:
    """Assert WAV is readable, 16kHz mono, non-empty. Returns metadata."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n  = wf.getnframes()
    if n == 0:
        raise ValueError(f"Empty WAV: {path}")
    return {
        "path": path,
        "sample_rate": sr,
        "channels": ch,
        "sample_width_bytes": sw,
        "n_frames": n,
        "duration_s": round(n / sr, 3),
    }


def chunk_wav(src: str, session_id: str, question_ids: list, out_dir: str,
              chunk_s: float | None = None) -> list:
    """Split WAV into per-question chunks. Returns list of output paths."""
    ensure_dir(session_id, out_dir)
    with wave.open(src, "rb") as wf:
        sr, n, sw, ch = wf.getframerate(), wf.getnframes(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16)
    n_chunks = len(question_ids)
    fpchunk = int(chunk_s * sr) if chunk_s else (len(data) // n_chunks)
    paths = []
    for i, qid in enumerate(question_ids):
        start = i * fpchunk
        end   = start + fpchunk if i < n_chunks - 1 else len(data)
        out   = wav_filename(session_id, qid, out_dir)
        with wave.open(out, "wb") as wf_out:
            wf_out.setnchannels(ch)
            wf_out.setsampwidth(sw)
            wf_out.setframerate(sr)
            wf_out.writeframes(data[start:end].tobytes())
        paths.append(out)
    return paths


def list_audio_devices() -> None:
    """Print all input devices with index."""
    try:
        import pyaudio  # type: ignore
    except ImportError:
        sys.exit("Missing: pip install pyaudio")
    pa = pyaudio.PyAudio()
    default_idx = pa.get_default_input_device_info()["index"]
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            marker = " <- DEFAULT" if i == default_idx else ""
            print(f"  [{i}] {info['name']}{marker}")
    pa.terminate()


def record_wav(session_id: str, question_id: str, out_dir: str,
               duration_s: float = 60.0, device_index: int = None) -> str:
    """Record from mic → WAV. Requires pyaudio."""
    try:
        import pyaudio  # type: ignore
    except ImportError:
        sys.exit("Missing: pip install pyaudio")

    ensure_dir(session_id, out_dir)
    out = wav_filename(session_id, question_id, out_dir)
    chunk = 1024
    pa = pyaudio.PyAudio()

    if device_index is None:
        dev_info = pa.get_default_input_device_info()
        device_index = int(dev_info["index"])
    else:
        dev_info = pa.get_device_info_by_index(device_index)
    print(f"  Input device [{device_index}]: {dev_info['name']}")

    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                     rate=SAMPLE_RATE, input=True,
                     input_device_index=device_index,
                     frames_per_buffer=chunk)
    print(f"  Recording {duration_s}s -> {out}  (speak now...)")

    frames = []
    ticks_per_sec = int(SAMPLE_RATE / chunk)
    n_chunks = int(ticks_per_sec * duration_s)
    for i in range(n_chunks):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        if i % ticks_per_sec == 0:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
            bar = "#" * min(int(rms / 400), 25)
            print(f"\r  [{i // ticks_per_sec:3d}s] {bar:<25s} rms={rms:.0f}   ", end="", flush=True)
    print()

    stream.stop_stream(); stream.close(); pa.terminate()

    with wave.open(out, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    all_samples = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    peak_rms = float(np.sqrt(np.mean(all_samples ** 2)))
    if peak_rms < 50:
        print(f"  WARNING: very low audio (rms={peak_rms:.0f}) — wrong device? Run --list-devices")
    else:
        print(f"  OK peak rms={peak_rms:.0f}")
    print(f"  Saved -> {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 1 — STT (Whisper local + OpenAI fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_whisper(size: str = "base"):
    if size not in _whisper_cache:
        try:
            import whisper  # type: ignore
        except ImportError:
            sys.exit("Missing: pip install openai-whisper")
        # print(f"  Loading whisper-{size} (first run downloads ~{{'tiny':'75MB','base':'145MB','small':'461MB'}.get(size,'?')}) ...")
        _whisper_cache[size] = whisper.load_model(size)
    return _whisper_cache[size]


def transcribe_local(wav_path: str, model_size: str = "base", language: str = "en") -> dict:
    """
    Local Whisper → transcript JSON with word-level timestamps.
    Returns {text, segments:[{id,start,end,text,words:[{word,start,end,probability}]}],
             language, model, inference_s}
    """
    model = _load_whisper(model_size)
    t0 = time.perf_counter()
    raw = model.transcribe(wav_path, language=language, word_timestamps=True, verbose=False)
    elapsed = time.perf_counter() - t0

    segments = []
    for seg in raw.get("segments", []):
        words = [{"word": w["word"].strip(),
                  "start": round(w["start"], 3),
                  "end": round(w["end"], 3),
                  "probability": round(w.get("probability", 0.0), 4)}
                 for w in seg.get("words", [])]
        segments.append({"id": seg["id"],
                         "start": round(seg["start"], 3),
                         "end": round(seg["end"], 3),
                         "text": seg["text"].strip(),
                         "words": words})
    return {
        "text": raw["text"].strip(),
        "segments": segments,
        "language": raw.get("language", language),
        "model": f"whisper-{model_size}",
        "inference_s": round(elapsed, 3),
    }


def transcribe_openai(wav_path: str) -> dict:
    """OpenAI gpt-4o-transcribe fallback (no timestamps)."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Missing: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY env var")
    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    with open(wav_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=f, response_format="text")
    text = res.strip() if isinstance(res, str) else res.text.strip()
    return {"text": text, "segments": [], "language": "en",
            "model": "gpt-4o-transcribe", "inference_s": round(time.perf_counter() - t0, 3)}


def transcribe(wav_path: str, model_size: str = "base", use_local: bool = True) -> dict:
    """Transcribe WAV → dict with timestamps. Falls back to OpenAI if whisper missing."""
    if use_local:
        try:
            return transcribe_local(wav_path, model_size=model_size)
        except SystemExit:
            pass
    return transcribe_openai(wav_path)


def save_transcript(result: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Transcript saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 2 — PROSODY FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_prosody(wav_path: str, transcript: str = "") -> dict:
    """
    Full prosody feature set:
      MFCC×13 (mean+std), F0 pitch (mean/std/range), RMS (mean/std),
      WPM, pause ratio, filler count/rate, ZCR mean.
    Returns flat dict of named float features.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std  = mfcc.std(axis=1)

    # F0 pitch via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    voiced_f0   = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean  = float(np.mean(voiced_f0))  if len(voiced_f0) else 0.0
    pitch_std   = float(np.std(voiced_f0))   if len(voiced_f0) else 0.0
    pitch_range = float(np.ptp(voiced_f0))   if len(voiced_f0) else 0.0

    # RMS energy
    rms      = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std  = float(np.std(rms))

    # Pause ratio (fraction of frames below 20th-percentile energy)
    pause_ratio = float(np.mean(rms < np.percentile(rms, 20)))

    # WPM
    if transcript.strip():
        wpm = len(transcript.strip().split()) / (duration / 60.0) if duration > 0 else 0.0
    else:
        voiced_ratio = len(voiced_f0) / max(len(f0) if f0 is not None else 1, 1)
        wpm = voiced_ratio * 130.0
    wpm = float(wpm)

    # Filler words
    filler_count = 0
    text_lower = transcript.lower()
    for fw in FILLER_WORDS:
        filler_count += len(re.findall(r"\b" + re.escape(fw) + r"\b", text_lower))
    word_count   = max(len(transcript.strip().split()), 1) if transcript.strip() else 1
    filler_rate  = filler_count / word_count

    # ZCR
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))

    features: dict = {
        "duration_s":    round(duration, 3),
        "pitch_mean_hz": round(pitch_mean, 4),
        "pitch_std_hz":  round(pitch_std, 4),
        "pitch_range_hz":round(pitch_range, 4),
        "rms_mean":      round(rms_mean, 6),
        "rms_std":       round(rms_std, 6),
        "pause_ratio":   round(pause_ratio, 4),
        "wpm":           round(wpm, 2),
        "filler_count":  int(filler_count),
        "filler_rate":   round(filler_rate, 4),
        "zcr_mean":      round(zcr_mean, 6),
    }
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = round(float(mfcc_mean[i]), 4)
        features[f"mfcc_{i+1}_std"]  = round(float(mfcc_std[i]),  4)
    return features


def _feature_vector(prosody: dict) -> "np.ndarray":
    keys = (
        ["pitch_mean_hz","pitch_std_hz","pitch_range_hz",
         "rms_mean","rms_std","pause_ratio","wpm","filler_rate","zcr_mean"]
        + [f"mfcc_{i+1}_mean" for i in range(13)]
        + [f"mfcc_{i+1}_std"  for i in range(13)]
    )
    return np.array([prosody.get(k, 0.0) for k in keys], dtype=np.float32)


def _feature_names() -> list:
    return (
        ["pitch_mean_hz","pitch_std_hz","pitch_range_hz",
         "rms_mean","rms_std","pause_ratio","wpm","filler_rate","zcr_mean"]
        + [f"mfcc_{i+1}_mean" for i in range(13)]
        + [f"mfcc_{i+1}_std"  for i in range(13)]
    )


def heuristic_confidence(prosody: dict) -> float:
    """Rule-based score 0–1. Used for bootstrap labels and model fallback."""
    pitch_score  = min(prosody["pitch_std_hz"] / 50.0, 1.0)
    wpm          = prosody["wpm"]
    wpm_score    = max(0.0, 1.0 - abs(wpm - 140) / 140.0)
    pause_score  = 1.0 - prosody["pause_ratio"]
    filler_score = max(0.0, 1.0 - prosody["filler_rate"] * 10)
    energy_score = min(prosody["rms_mean"] / 0.05, 1.0)
    return round(
        pitch_score * 0.20 + wpm_score * 0.25 + pause_score * 0.20
        + filler_score * 0.20 + energy_score * 0.15, 4)


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 2 — XGBOOST TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_label(conf: float) -> int:
    """conf ≥ 0.65 → high(2), ≥ 0.40 → medium(1), else → low(0)"""
    return 2 if conf >= 0.65 else (1 if conf >= 0.40 else 0)


def build_dataset(audio_dir: str, transcripts_dir: str | None = None):
    wav_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {audio_dir}")
    X, y, paths = [], [], []
    for wav_path in wav_files:
        transcript = ""
        if transcripts_dir:
            rel = os.path.relpath(wav_path, audio_dir)
            tjson = os.path.join(transcripts_dir, rel.replace(".wav", ".json"))
            if os.path.exists(tjson):
                transcript = json.load(open(tjson)).get("text", "")
        try:
            p     = extract_prosody(wav_path, transcript=transcript)
            conf  = heuristic_confidence(p)
            label = bootstrap_label(conf)
            X.append(_feature_vector(p))
            y.append(label)
            paths.append(wav_path)
            print(f"  {os.path.basename(wav_path):40s}  heuristic={conf:.3f}  label={label}")
        except Exception as e:
            print(f"  SKIP {wav_path}: {e}")
    return np.stack(X), np.array(y), paths


def train_xgb(audio_dir: str, out_path: str, transcripts_dir: str | None = None) -> None:
    try:
        import xgboost as xgb                             # type: ignore
        from sklearn.model_selection import cross_val_score, StratifiedKFold  # type: ignore
    except ImportError:
        sys.exit("Missing: pip install xgboost scikit-learn")

    print(f"\nBuilding dataset from {audio_dir} ...")
    X, y, _ = build_dataset(audio_dir, transcripts_dir)
    n_classes = len(np.unique(y))
    if n_classes < 2:
        sys.exit(f"Need ≥2 confidence classes in data, got: {np.unique(y)}")
    print(f"Dataset: {len(X)} samples | label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", num_class=3,
        objective="multi:softprob", random_state=42,
    )
    cv     = StratifiedKFold(n_splits=min(5, n_classes * 2), shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"Cross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model.save_model(out_path)
    print(f"Model saved → {out_path}")

    # SHAP feature importance
    try:
        import shap  # type: ignore
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(zip(_feature_names(), mean_abs), key=lambda x: -x[1])
        print("\nSHAP feature importance (top 15):")
        for name, val in ranked[:15]:
            bar = "█" * int(val / ranked[0][1] * 20)
            print(f"  {name:30s} {val:.4f}  {bar}")
    except ImportError:
        print("shap not installed — skipping SHAP report (pip install shap)")


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 3 — CONFIDENCE PREDICTION + LIVE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_xgb_model(model_path: str | None = None):
    global _xgb_model
    if _xgb_model is not None:
        return _xgb_model
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        try:
            import xgboost as xgb  # type: ignore
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            _xgb_model = m
            return m
        except Exception:
            pass
    return None


def predict_confidence(prosody: dict, model_path: str | None = None) -> float:
    """0–1. Uses XGBoost if model exists, else heuristic."""
    model = _load_xgb_model(model_path)
    if model is not None:
        vec  = _feature_vector(prosody).reshape(1, -1)
        prob = model.predict_proba(vec)[0]
        return round(float(prob[0] * 0.2 + prob[1] * 0.6 + prob[2] * 1.0), 4)
    return heuristic_confidence(prosody)


def build_feedback(prosody: dict) -> str:
    parts = []
    if prosody["wpm"] < 100:
        parts.append("pace too slow — aim 120–160 wpm")
    elif prosody["wpm"] > 180:
        parts.append("too fast — slow down")
    if prosody["filler_rate"] > 0.05:
        parts.append(f"reduce fillers (found {prosody['filler_count']})")
    if prosody["pause_ratio"] > 0.40:
        parts.append("too many pauses")
    if prosody["pitch_std_hz"] < 10:
        parts.append("monotone — vary pitch")
    return "; ".join(parts) if parts else "Good delivery."


def run_pipeline(wav_path: str, model_size: str = "base",
                 use_local: bool = True, model_path: str | None = None) -> dict:
    """
    Week 3 end-to-end:  WAV → transcript → prosody → confidence score.
    Returns full result dict.
    """
    print(f"\n{'='*60}")
    print(f"  WAV: {wav_path}")

    # 1. Validate
    meta = validate_wav(wav_path)
    print(f"  Duration: {meta['duration_s']}s  |  {meta['sample_rate']}Hz  |  mono")

    # 2. Transcribe
    print(f"\n  [STT] model=whisper-{model_size} local={use_local}")
    t_result = transcribe(wav_path, model_size=model_size, use_local=use_local)
    print(f"  Text ({t_result['inference_s']}s): {t_result['text'][:200]}")
    if t_result["segments"]:
        print(f"  Segments: {len(t_result['segments'])}  Words: {sum(len(s['words']) for s in t_result['segments'])}")

    # 3. Prosody
    print("\n  [Prosody]")
    prosody = extract_prosody(wav_path, transcript=t_result["text"])
    print(f"  WPM={prosody['wpm']:.0f}  pitch_std={prosody['pitch_std_hz']:.1f}Hz  "
          f"pause={prosody['pause_ratio']:.2f}  fillers={prosody['filler_count']}")

    # 4. Confidence
    score    = predict_confidence(prosody, model_path=model_path)
    feedback = build_feedback(prosody)
    using    = "xgboost" if _xgb_model else "heuristic"
    print(f"\n  [Confidence] {score*10:.2f}/10  ({using})")
    print(f"  Feedback: {feedback}")
    print("=" * 60)

    return {
        "wav_meta":   meta,
        "transcript": t_result,
        "prosody":    prosody,
        "confidence": {"score_0_1": score, "score_0_10": round(score * 10, 2),
                       "method": using, "feedback": feedback},
    }


def calibrate_filler_threshold(wav_paths: list) -> float:
    """p75 filler_rate across files → suggested flag threshold."""
    rates = []
    for p in wav_paths:
        try:
            rates.append(extract_prosody(p)["filler_rate"])
        except Exception as e:
            print(f"  SKIP {p}: {e}")
    if not rates:
        return 0.05
    rates = sorted(rates)
    p50, p75, p95 = (float(np.percentile(rates, q)) for q in [50, 75, 95])
    print(f"Filler calibration ({len(rates)} files): p50={p50:.3f}  p75={p75:.3f}  p95={p95:.3f}")
    print(f"Suggested threshold: {p75:.3f}  (flag answers above this)")
    return p75


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEK 4 — BENCHMARK (tiny vs base, asyncio stress, model card)
# ═══════════════════════════════════════════════════════════════════════════════

def _wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
    if not r:
        return 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return round(d[len(r)][len(h)] / len(r), 4)


def _profile_one(wav_path: str, size: str, n_runs: int) -> dict:
    dur = librosa.get_duration(path=wav_path)
    times, text = [], ""
    for i in range(n_runs):
        t0 = time.perf_counter()
        res = transcribe_local(wav_path, model_size=size)
        elapsed = time.perf_counter() - t0
        times.append(elapsed); text = res["text"]
        print(f"  [{size}] run {i+1}/{n_runs}: {elapsed:.2f}s")
    mean_t = sum(times) / len(times)
    return {"model": size, "mean_s": round(mean_t, 3), "min_s": round(min(times), 3),
            "rtf": round(mean_t / dur, 4), "audio_s": round(dur, 3), "transcript": text}


async def _stress_async(wav_path: str, n: int) -> dict:
    print(f"\n  Async stress: {n} concurrent whisper-tiny calls ...")
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, transcribe_local, wav_path, "tiny") for _ in range(n)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0
    texts_match = len({r["text"] for r in results}) == 1
    return {"n": n, "total_s": round(elapsed, 3),
            "throughput": round(n / elapsed, 2), "texts_identical": texts_match}


def benchmark(wav_path: str, ref: str = "", n_runs: int = 3,
              stress: bool = False, n_concurrent: int = 5,
              out: str | None = None) -> dict:
    results: dict = {}

    print("\nProfiling whisper-tiny ...")
    results["tiny"] = _profile_one(wav_path, "tiny", n_runs)

    print("\nProfiling whisper-base ...")
    results["base"] = _profile_one(wav_path, "base", n_runs)

    speed_gain = results["base"]["mean_s"] / max(results["tiny"]["mean_s"], 0.001)
    results["speed_gain_tiny_vs_base"] = round(speed_gain, 2)

    if ref:
        results["wer"] = {
            "tiny": _wer(ref, results["tiny"]["transcript"]),
            "base": _wer(ref, results["base"]["transcript"]),
        }

    # Fast/accented: use tiny vs base agreement as proxy
    agreement_wer = _wer(results["base"]["transcript"], results["tiny"]["transcript"])
    results["tiny_vs_base_agreement_wer"] = agreement_wer

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  tiny  — {results['tiny']['mean_s']:.2f}s  RTF={results['tiny']['rtf']:.3f}")
    print(f"  base  — {results['base']['mean_s']:.2f}s  RTF={results['base']['rtf']:.3f}")
    print(f"  Speed gain (tiny vs base): {speed_gain:.1f}x")
    print(f"  tiny vs base agreement WER: {agreement_wer:.3f}")
    if "wer" in results:
        print(f"  WER vs reference — tiny={results['wer']['tiny']}  base={results['wer']['base']}")
    rec = "tiny" if speed_gain > 2 and agreement_wer < 0.15 else "base"
    print(f"  Recommendation: whisper-{rec}")
    results["recommendation"] = rec

    if stress:
        stress_result = asyncio.run(_stress_async(wav_path, n_concurrent))
        results["async_stress"] = stress_result
        print(f"  Stress ({n_concurrent} concurrent): {stress_result['total_s']:.2f}s | "
              f"{stress_result['throughput']} req/s | texts_match={stress_result['texts_identical']}")

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results → {out}")

    return results


def print_model_card() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║              SPEECH MODEL CARD  (Week 4)                     ║
╠══════════════════════════════════════════════════════════════╣
║  STT: openai-whisper (local)                                 ║
║    tiny  — RTF ~0.15  WER slightly higher  (use: fast path) ║
║    base  — RTF ~0.35  WER baseline         (use: accuracy)  ║
║    Fallback: OpenAI gpt-4o-transcribe API                    ║
║    Output: word-level timestamps + segment JSON              ║
╠══════════════════════════════════════════════════════════════╣
║  Confidence: XGBoost 3-class (low/med/high)                  ║
║    Features: MFCC×13 (mean+std), F0, RMS, WPM,              ║
║              pause_ratio, filler_rate, ZCR  [35 total]       ║
║    Labels: bootstrapped from heuristic (no human labels)     ║
║    Training: 5-fold CV, 200 trees, SHAP audited              ║
║    Fallback: heuristic score when model file missing         ║
╠══════════════════════════════════════════════════════════════╣
║  Limitations:                                                ║
║    Labels are heuristic-derived — retrain on real data       ║
║    Accented/non-native English not formally evaluated        ║
║    Asyncio: run_in_executor (thread-pool, not true async)    ║
╚══════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_record(args):
    if args.list_devices:
        list_audio_devices()
        return
    if not args.session or not args.question:
        sys.exit("error: --session and --question required for recording")
    record_wav(args.session, args.question, args.out_dir, args.duration, args.device)


def cmd_transcribe(args):
    result = transcribe(args.wav, model_size=args.model, use_local=not args.openai)
    print(f"\nText: {result['text']}")
    print(f"Model: {result['model']}  Time: {result['inference_s']}s")
    if result["segments"]:
        print(f"Segments: {len(result['segments'])}")
        for seg in result["segments"][:3]:
            print(f"  [{seg['start']:.1f}s–{seg['end']:.1f}s] {seg['text']}")
        if len(result["segments"]) > 3:
            print(f"  ... (+{len(result['segments'])-3} more)")
    if args.out:
        save_transcript(result, args.out)


def cmd_prosody(args):
    prosody = extract_prosody(args.wav, transcript=args.transcript or "")
    print("\nProsody features:")
    for k, v in prosody.items():
        if not k.startswith("mfcc"):
            print(f"  {k:25s} {v}")
    mfcc_means = [v for k, v in prosody.items() if k.endswith("_mean") and "mfcc" in k]
    print(f"  MFCC means (1–13):       {[round(x,2) for x in mfcc_means]}")
    score = heuristic_confidence(prosody)
    print(f"\nHeuristic confidence: {score:.3f}  ({score*10:.1f}/10)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(prosody, f, indent=2)
        print(f"Saved → {args.out}")


def cmd_train(args):
    train_xgb(args.audio_dir, args.out, transcripts_dir=args.transcripts_dir)


def cmd_validate(args):
    result = run_pipeline(
        args.wav,
        model_size=args.model,
        use_local=not args.openai,
        model_path=args.model_path,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull result saved → {args.out}")


def cmd_benchmark(args):
    benchmark(
        args.wav,
        ref=args.ref or "",
        n_runs=args.n_runs,
        stress=args.stress,
        n_concurrent=args.n_concurrent,
        out=args.out,
    )


def cmd_demo(args):
    """Full demo: validate + model card + benchmark (1 run each model)."""
    print_model_card()
    run_pipeline(args.wav, model_size="base")
    print("\nRunning quick benchmark (1 run each) ...")
    benchmark(args.wav, n_runs=1, stress=False)


def main():
    p = argparse.ArgumentParser(
        description="AI Mock Interview — Audio Pipeline (Weeks 1–4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # record
    r = sub.add_parser("record", help="Week 1 — record mic → WAV")
    r.add_argument("--session",  default=None, help="session_id (required unless --list-devices)")
    r.add_argument("--question", default=None, help="question_id (required unless --list-devices)")
    r.add_argument("--duration", type=float, default=60.0)
    r.add_argument("--out_dir",     default="recordings")
    r.add_argument("--device",      type=int, default=None, help="Input device index")
    r.add_argument("--list-devices", dest="list_devices", action="store_true",
                   help="List input devices and exit (no --session/--question needed)")

    # transcribe
    t = sub.add_parser("transcribe", help="Week 1 — STT with timestamps")
    t.add_argument("--wav",    required=True)
    t.add_argument("--model",  default="base", choices=["tiny","base","small","medium"])
    t.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local")
    t.add_argument("--out",    help="Save transcript JSON")

    # prosody
    pr = sub.add_parser("prosody", help="Week 2 — extract prosody features")
    pr.add_argument("--wav",        required=True)
    pr.add_argument("--transcript", default="", help="Text for WPM/filler calc")
    pr.add_argument("--out",        help="Save features JSON")

    # train
    tr = sub.add_parser("train", help="Week 2 — train XGBoost confidence model")
    tr.add_argument("--audio_dir",       required=True, help="Dir with WAV files")
    tr.add_argument("--transcripts_dir", default=None,  help="Dir with transcript JSONs")
    tr.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="Output model path")

    # validate
    v = sub.add_parser("validate", help="Week 3 — end-to-end pipeline test")
    v.add_argument("--wav",        required=True)
    v.add_argument("--model",      default="base", choices=["tiny","base","small","medium"])
    v.add_argument("--openai",     action="store_true")
    v.add_argument("--model_path", default=None, help="XGBoost model path")
    v.add_argument("--out",        help="Save result JSON")

    # benchmark
    b = sub.add_parser("benchmark", help="Week 4 — Whisper tiny vs base profiling")
    b.add_argument("--wav",          required=True)
    b.add_argument("--ref",          default="", help="Reference transcript for WER")
    b.add_argument("--n_runs",       type=int, default=3)
    b.add_argument("--stress",       action="store_true", help="Asyncio stress test")
    b.add_argument("--n_concurrent", type=int, default=5)
    b.add_argument("--out",          help="Save benchmark JSON")

    # demo
    d = sub.add_parser("demo", help="Full demo: model card + pipeline + benchmark")
    d.add_argument("--wav", required=True)

    args = p.parse_args()
    {
        "record":     cmd_record,
        "transcribe": cmd_transcribe,
        "prosody":    cmd_prosody,
        "train":      cmd_train,
        "validate":   cmd_validate,
        "benchmark":  cmd_benchmark,
        "demo":       cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    main()

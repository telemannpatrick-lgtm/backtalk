"""Voice-ID: speaker identification, Phase 1 (recognition only, no
permission gating yet -- see Second Brain Build Plan Phase 3 in the vault).

Voiceprints are Resemblyzer embeddings (256-d vectors), one .npy file per
enrolled person, stored in backtalk/voiceprints/. identify() compares a
captured utterance's embedding against every enrolled voiceprint by cosine
similarity and returns the best match above a threshold, else "unknown".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

VOICEPRINTS_DIR = Path(__file__).parent / "voiceprints"
MATCH_THRESHOLD = 0.75  # cosine similarity; tune once real data exists

_encoder: VoiceEncoder | None = None


def _get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def embed(pcm_int16: np.ndarray, rate: int) -> np.ndarray:
    """int16 mono PCM at `rate` Hz -> a 256-d speaker embedding."""
    wav = preprocess_wav(pcm_int16.astype(np.float32) / 32768.0, source_sr=rate)
    return _get_encoder().embed_utterance(wav)


def enroll(name: str, pcm_int16: np.ndarray, rate: int) -> Path:
    """Save `name`'s voiceprint from a single captured sample. Overwrites
    any existing enrollment for that name."""
    return enroll_multi(name, [(pcm_int16, rate)])


def enroll_multi(name: str, samples: list[tuple[np.ndarray, int]]) -> Path:
    """Same as enroll(), but averages embeddings from several short clips
    instead of one long one -- matches real conversational utterance
    length better than a single deliberate long recording does (a single
    12s clean sample scored 0.93 similarity in testing, but real short
    hands-free utterances from the same enrolled voice scored 0.40-0.57,
    well under the 0.75 threshold -- length/style mismatch, not a bad
    mic or overlapping speech). Overwrites any existing enrollment."""
    VOICEPRINTS_DIR.mkdir(exist_ok=True)
    vecs = [embed(pcm, rate) for pcm, rate in samples]
    avg = np.mean(vecs, axis=0)
    avg = avg / np.linalg.norm(avg)  # re-normalize after averaging
    path = VOICEPRINTS_DIR / f"{name.lower()}.npy"
    np.save(path, avg)
    return path


def identify(pcm_int16: np.ndarray, rate: int) -> tuple[str, float]:
    """Returns (name, similarity) for the closest enrolled voiceprint, or
    ("unknown", best_similarity) if nothing clears MATCH_THRESHOLD."""
    if not VOICEPRINTS_DIR.exists():
        return "unknown", 0.0
    vec = embed(pcm_int16, rate)
    best_name, best_sim = "unknown", 0.0
    for f in VOICEPRINTS_DIR.glob("*.npy"):
        ref = np.load(f)
        sim = float(np.dot(vec, ref) / (np.linalg.norm(vec) * np.linalg.norm(ref)))
        if sim > best_sim:
            best_name, best_sim = f.stem, sim
    if best_sim < MATCH_THRESHOLD:
        return "unknown", best_sim
    return best_name, best_sim


def _record_clip_like_live(seconds: float) -> np.ndarray:
    """Capture `seconds` of audio through the EXACT same path the live
    pipeline uses (backtalk.ears._open_mic -- native rate + the same
    decimate-average downsample if the device needs it), not a naive
    direct sd.rec() call. A mismatch here was a real bug: sd.rec() at
    16kHz either fails outright or goes through PortAudio's own
    auto-convert, which ears.py's own comments already document as
    measured unreliable (a peak that hit 11% natively came back at 0.4%
    through auto-convert) -- enrollment and live identification must go
    through the same capture path or they're not comparable."""
    from backtalk.ears import _open_mic, FRAME_LEN, RATE as EARS_RATE
    n_frames = int(seconds * EARS_RATE / FRAME_LEN)
    frames = []
    with _open_mic() as stream:
        for _ in range(n_frames):
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
    return np.concatenate(frames)


if __name__ == "__main__":
    # Manual test: python -m backtalk.voiceid enroll <name>
    #              python -m backtalk.voiceid identify
    from backtalk.ears import RATE

    CLIP_S = 3
    N_CLIPS = 6
    PASSPHRASE = "Jarvis, activate profile"

    if len(sys.argv) >= 3 and sys.argv[1] == "enroll":
        name = sys.argv[2]
        samples = []
        for i in range(N_CLIPS):
            print(f"Clip {i+1}/{N_CLIPS}: say \"{PASSPHRASE}\" "
                  f"(~{CLIP_S}s)...")
            pcm = _record_clip_like_live(CLIP_S)
            samples.append((pcm, RATE))
        path = enroll_multi(name, samples)
        print(f"Saved voiceprint (averaged from {N_CLIPS} passphrase "
              f"repetitions, captured the same way as live): {path}")
    elif len(sys.argv) >= 2 and sys.argv[1] == "identify":
        print(f"Recording {CLIP_S}s -- talk naturally now (short, like a "
              f"real utterance)...")
        pcm = _record_clip_like_live(CLIP_S)
        name, sim = identify(pcm, RATE)
        print(f"Identified: {name} (similarity {sim:.3f})")
    else:
        print("usage: python -m backtalk.voiceid enroll <name>")
        print("       python -m backtalk.voiceid identify")

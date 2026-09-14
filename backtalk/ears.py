# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ears — mic capture with VAD endpointing, transcribed in-process
by faster-whisper. Local, free, no server, no API key.

record_held() is the hold-to-talk capture (the button is the VAD).
Ears.listen_once() is the legacy open-mic mode: blocks until one
complete utterance is heard, then returns its transcript. Endpointing:
an utterance opens after ~120ms of sustained speech, closes after
`silence_ms` of trailing quiet. A `gate` callable can suppress
listening (so the open mic ignores the speakers unless barge-in is on).
"""
import platform
import re
import sys
import threading

import numpy as np
import sounddevice as sd
import webrtcvad

from backtalk import signals
from backtalk.config import CFG
from backtalk.vlog import log

RATE = 16000
FRAME_MS = 30
FRAME_LEN = RATE * FRAME_MS // 1000  # samples per frame
OPEN_FRAMES = 4        # ~120ms speech to open an utterance
# Pre-roll ring, kept so the syllables spoken BEFORE the VAD opens survive.
# OPEN_FRAMES of this is the speech that did the opening, so the true
# lead-in is (PREROLL_FRAMES - OPEN_FRAMES) -- ~360ms here. At 8 frames it
# was 120ms, which reliably ate the wake word: "Jarvis" takes ~400ms and
# opens on a soft affricate the aggressiveness-3 VAD sleeps through, so
# transcripts arrived as "Can you hear me?" with the name already gone.
PREROLL_FRAMES = 16
MAX_UTTER_S = 30

_NONSPEECH = re.compile(r"[\[(][^\])]*[\])]")

_model = None
_model_lock = threading.Lock()
_backend = None          # "mlx" once the GPU path loads, else "faster-whisper"


def _apple_gpu_available() -> bool:
    """Apple Silicon only. CTranslate2, the runtime under faster-whisper,
    has no Metal backend, so on every Mac it transcribes on the CPU while
    the GPU sits idle. mlx-whisper runs the SAME model on the GPU.

    Measured on an M4 Max, small.en, a 6.5s clip, warm: 0.88s on the CPU
    path against 0.12s on the GPU, with a character-identical transcript
    on three of four test clips and a two-comma difference on the fourth.

    Not a second product and not a user-facing choice: same model name
    from the same config key, same text out, one platform finally running
    it properly. Anything that is not an Apple Silicon Mac keeps
    faster-whisper, which already uses CUDA wherever it exists."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper                       # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_repo(model_name: str) -> str:
    """A faster-whisper model name -> its MLX conversion on the Hub."""
    return f"mlx-community/whisper-{model_name}-mlx"


_mic_checked = False


_mic_device_warned = False


def _mic_index():
    """Resolve mic_device (a device NAME) to an index, or None for the default.

    A NAME and never an index, because indices shift every time a device
    connects or disconnects, which is the exact event this setting exists
    to survive. Measured on a real machine: plugging a USB microphone in
    moved the default pair from [-1, 1] to [1, 3], silently changing the
    OUTPUT device too.

    Re-resolved on every stream open rather than cached at startup, for
    the same reason. Exact name wins, then the first case-insensitive
    substring, so a precise name can never be beaten by a loose one.
    """
    global _mic_device_warned
    want = str(CFG.get("mic_device", "") or "").strip()
    if not want:
        return None
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        log(f"[ears] could not list audio devices ({e}) -- using the "
            f"default mic")
        return None
    ins = [(i, d) for i, d in enumerate(devices)
           if d.get("max_input_channels", 0) > 0]
    # Windows lists the same physical mic once per audio backend (MME,
    # DirectSound, WASAPI, WDM-KS). WASAPI is the modern, correct one;
    # measured on a real device, MME's own format conversion produced
    # near-silent captures (0.1-0.6% peak) while WASAPI captured the
    # same voice cleanly (11%+ peak) seconds apart with nothing else
    # changed. When several entries share a name, try the WASAPI one
    # first.
    def _is_wasapi(idx: int) -> bool:
        return hostapis[devices[idx]["hostapi"]]["name"] == "Windows WASAPI"
    ins = sorted(ins, key=lambda pair: not _is_wasapi(pair[0]))
    for i, d in ins:
        if d["name"] == want:
            _mic_device_warned = False
            return i
    low = want.lower()
    for i, d in ins:
        if low in d["name"].lower():
            _mic_device_warned = False
            return i
    if not _mic_device_warned:          # once per disappearance, not per press
        _mic_device_warned = True
        log(f"[ears] mic_device {want!r} not found -- using the system "
            f"default. Inputs I can see: {[d['name'] for _, d in ins]}")
    return None


class _ResamplingStream:
    """Wraps a stream opened at the device's OWN rate and presents reads
    in RATE (16kHz) frames via a plain decimate-average downsample.

    Some backends -- Windows WASAPI in particular -- refuse to open at
    16kHz outright rather than resampling for us (PaErrorCode -9997,
    "Invalid sample rate"). PortAudio does offer an auto-convert path
    for WASAPI, but it measured unreliable on a real device: the exact
    same recording that peaked at 11%+ captured natively came back at
    0.4% through auto-convert. Decimate-average from the true native
    rate measured lossless by comparison (peak survived a 48kHz -> 16kHz
    pass byte-for-byte), so that is what this does instead of trusting
    the OS to do it.
    """

    def __init__(self, device, native_rate: int):
        self._factor = max(1, round(native_rate / RATE))
        self._native_frame_len = FRAME_LEN * self._factor
        self._stream = sd.InputStream(device=device, samplerate=native_rate,
                                      channels=1, dtype="int16",
                                      blocksize=self._native_frame_len)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        return self._stream.__exit__(*exc)

    def read(self, frames):
        block, overflow = self._stream.read(frames * self._factor)
        arr = block[:, 0].astype(np.float64)
        down = arr.reshape(frames, self._factor).mean(axis=1).astype(np.int16)
        return down.reshape(-1, 1), overflow


def _open_mic():
    """Open the capture stream on the configured mic.

    Degrades to the system default if that device will not open --
    unplugged between the lookup and the open, busy, or refusing the
    sample rate. The mic gets worse; it never goes mute.
    """
    dev = _mic_index()
    opts = dict(samplerate=RATE, channels=1, dtype="int16",
                blocksize=FRAME_LEN)
    try:
        return sd.InputStream(device=dev, **opts)
    except Exception as e:
        if dev is not None:
            try:
                native_rate = int(sd.query_devices(dev)["default_samplerate"])
                if native_rate != RATE:
                    log(f"[ears] mic_device wants {native_rate}Hz, not "
                        f"{RATE}Hz -- capturing native and downsampling")
                    return _ResamplingStream(dev, native_rate)
            except Exception:
                pass
            log(f"[ears] could not open mic_device {CFG.get('mic_device')!r} "
                f"({e}) -- using the system default")
            try:
                return sd.InputStream(**opts)
            except Exception:
                pass                   # fall through to the rebuild below
        return _reopen_after_device_change(opts)


def _reopen_after_device_change(opts):
    """Last resort: rebuild the audio system, then open the mic once more.

    PortAudio caches the device list when it initialises, so a device that
    disappears afterwards leaves a stale entry behind. A Bluetooth headset
    flipping between listening and call modes does this every time the mic
    opens, and from then on EVERY capture fails while the voice line looks
    perfectly healthy and simply never hears another word.

    Rebuilding refreshes the list. It also closes every open stream, the
    speaking one included, which is why Mouth._get_out rebuilds a stream
    it finds dead rather than trusting the one it is holding. Do not
    remove that guard without removing this.
    """
    log("[ears] the audio devices changed -- rebuilding and reopening")
    try:
        sd._terminate()
    except Exception:
        pass                           # already down; re-initialising is the point
    sd._initialize()
    return sd.InputStream(**opts)


_mic_warned = False

# Substrings PortAudio uses when the problem is the DEVICE rather than the
# audio. Matched on the message because the exception TYPE is the same
# PortAudioError whether a device vanished or a stream merely glitched.
_DEVICE_ERROR_HINTS = ("error querying device", "invalid device",
                       "device unavailable", "no default input",
                       "invalid number of channels", "device not found")


def _mic_message(detail: str) -> list[str]:
    """The one explanation, so startup and mid-session say the same thing."""
    return [
        "[ears] NO WORKING MICROPHONE. Nothing can be recorded on this "
        "machine, so the talk key will have nothing to send.",
        f"[ears] the audio system said: {detail}",
        "[ears] plug one in and start the voice line again. If one IS "
        "plugged in, check it is allowed in this system's microphone "
        "privacy settings -- and if you have several, put part of the "
        "one you want in \"mic_device\" in backtalk.json.",
    ]


def explain_audio_failure(exc) -> bool:
    """Turn a device-level audio failure into plain words. Returns True
    when it handled the message, so the caller can skip the raw repr.

    The startup pre-flight cannot cover a microphone that is unplugged or
    dies MID-SESSION, and that person gets the worst version of this:
    no warning at all, and a raw PortAudioError on every single press,
    forever. The key hook keeps working throughout, so it still looks
    like it is listening. This says the same sentences the pre-flight
    would have said, at the moment it becomes true.

    Said in full once, then briefly, because a message repeated on every
    key press stops being information and becomes noise.
    """
    global _mic_warned
    text = str(exc).lower()
    # THE TWO HALVES OF THIS TEST ARE NOT DOING THE SAME JOB. Do not
    # simplify it to one. Measured on Windows: the SAME missing microphone
    # produces "Error querying device -1" when it is absent at startup and
    # "A device ID has been used that is out of range for your system
    # [MME error 2]" when it is unplugged mid-stream. The second matches
    # not one hint below, and was caught only by the type check -- on the
    # very first real test of the case this function exists for. The
    # hints catch device failures raised as something other than a
    # PortAudioError; the type catches PortAudio wording nobody predicted.
    if type(exc).__name__ != "PortAudioError" and \
            not any(h in text for h in _DEVICE_ERROR_HINTS):
        return False
    if _mic_warned:
        log("[ears] still no working microphone.")
        return True
    _mic_warned = True
    for line in _mic_message(f"{type(exc).__name__}: {exc}"):
        log(line)
    return True


def check_microphone() -> bool:
    """Say whether recording is possible at all, BEFORE the greeting.

    Without this the voice line boots on a machine with no microphone,
    warms, speaks its greeting and presents a working push-to-talk
    prompt. The key hook works perfectly throughout, so the user is
    given every impression it is listening -- and the only sign of
    trouble is a raw PortAudioError AFTER they have held the key and
    spoken. It then repeats forever, because holding a key again cannot
    conjure a device.
    """
    global _mic_checked
    if _mic_checked:
        return True
    _mic_checked = True
    dev = _mic_index()
    try:
        sd.check_input_settings(device=dev, channels=1,
                                samplerate=RATE, dtype="int16")
        return True
    except Exception as e:
        # Same fallback _open_mic() actually uses: some backends (WASAPI
        # in particular) refuse a non-native rate outright. Check at the
        # device's own rate before declaring it dead -- otherwise this
        # cries wolf on a mic that works fine (see _ResamplingStream).
        if dev is not None:
            try:
                native_rate = int(sd.query_devices(dev)["default_samplerate"])
                sd.check_input_settings(device=dev, channels=1,
                                        samplerate=native_rate, dtype="int16")
                return True
            except Exception:
                pass
        global _mic_warned
        _mic_warned = True      # said it here; do not repeat on first press
        for line in _mic_message(str(e)):
            log(line)
        return False


def _probe(model):
    """Run a tenth of a second of silence through the real path.

    faster-whisper is lazy: transcribe() returns a generator and does no
    work until it is iterated, so the list() is what actually exercises
    the backend and is not redundant.
    """
    segments, _ = model.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                   language="en")
    list(segments)


def warm():
    """Load the STT model (first call downloads it to the HF cache).
    Called at startup while the greeting plays, so the first real
    utterance doesn't pay the load."""
    global _model, _backend
    check_microphone()
    with _model_lock:
        if _model is None:
            if _apple_gpu_available():
                import mlx_whisper
                repo = _mlx_repo(CFG["stt_model"])
                log(f"[ears] loading {CFG['stt_model']} on the Apple GPU...")
                # This API has no separate load call: the first transcribe
                # pulls and caches the weights. Warm on a beat of silence so
                # the first real utterance does not pay for it.
                mlx_whisper.transcribe(np.zeros(RATE // 10, dtype=np.float32),
                                       path_or_hf_repo=repo, language="en",
                                       verbose=None)
                _model, _backend = repo, "mlx"
            else:
                from faster_whisper import WhisperModel
                want = CFG["stt_device"]
                log(f"[ears] loading {CFG['stt_model']} "
                    f"({want}/{CFG['stt_compute']})...")
                _model = WhisperModel(CFG["stt_model"], device=want,
                                      compute_type=CFG["stt_compute"])
                # PROVE the device before the greeting, not at the first
                # spoken sentence. WhisperModel CONSTRUCTS perfectly well
                # against a GPU it cannot actually use: "auto" picks CUDA
                # on any NVIDIA machine, and the CUDA runtime is not
                # loaded until the first inference. So warm-up logged
                # "model ready", startup reported healthy, and a missing
                # cublas DLL only surfaced when the user finally spoke --
                # long after the greeting, in a place they could not
                # connect to a setting. The Apple-GPU branch above has
                # always done this; this one never did.
                try:
                    _probe(_model)
                except Exception as e:
                    if want == "cpu":
                        raise
                    log(f"[ears] {want!r} does not work on this machine "
                        f"({type(e).__name__}: {e}).")
                    log("[ears] falling back to the CPU. Set "
                        "\"stt_device\": \"cpu\" in backtalk.json to skip "
                        "this check in future.")
                    _model = WhisperModel(CFG["stt_model"], device="cpu",
                                          compute_type=CFG["stt_compute"])
                    _probe(_model)
                _backend = "faster-whisper"
            log(f"[ears] model ready ({_backend})")
    return _model


def _log_voice_id(pcm: np.ndarray) -> None:
    """Phase 1 of voice-ID (see Second Brain Build Plan): identify-and-log
    only, nothing gates on this yet. Never allowed to break the actual
    voice pipeline -- any failure here is swallowed and logged, not raised.
    """
    try:
        from backtalk import voiceid
        name, sim = voiceid.identify(pcm, RATE)
        log(f"[voiceid] {name} (similarity {sim:.3f})")
    except Exception as e:
        log(f"[voiceid] identification failed (non-fatal): {e!r}")


def transcribe(pcm: np.ndarray) -> str:
    """int16 mono 16kHz -> text. Bracketed non-speech markers that
    whisper emits ([BLANK_AUDIO], [SIGHS], (coughs)...) are stripped;
    if nothing remains, it was silence."""
    model = warm()
    audio = pcm.astype(np.float32) / 32768.0
    lang = "en" if CFG["stt_model"].endswith(".en") else None
    if _backend == "mlx":
        import mlx_whisper
        text = mlx_whisper.transcribe(audio, path_or_hf_repo=model,
                                      temperature=0.0, language=lang,
                                      verbose=None)["text"].strip()
    else:
        segments, _ = model.transcribe(audio, temperature=0.0, language=lang)
        text = "".join(s.text for s in segments).strip()
    return _NONSPEECH.sub("", text).strip()


class Ears:
    def __init__(self, aggressiveness: int = 3, silence_ms: int = 1500):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames = silence_ms // FRAME_MS

    def listen_once(self, gate=None, timeout_s: float | None = None,
                    abort=None) -> str | None:
        """Block until one utterance completes; return transcript
        (or None on timeout). An `abort` callable is checked every
        frame; returning True closes the mic and returns None, which
        is how a live switch back to push-to-talk shuts the open mic
        down promptly instead of after one more utterance."""
        frames: list[np.ndarray] = []
        ring: list[np.ndarray] = []   # pre-roll so the first syllable survives
        speech_run = 0
        silence_run = 0
        speech_total = 0
        in_utterance = False
        elapsed = 0.0

        with _open_mic() as stream:
            while True:
                block, _ = stream.read(FRAME_LEN)
                elapsed += FRAME_MS / 1000
                if abort and abort():
                    return None
                if timeout_s and elapsed > timeout_s and not in_utterance:
                    return None
                mono = block[:, 0].copy()
                if gate and gate():
                    # speakers are talking and barge-in isn't on: ignore
                    ring.clear()
                    continue
                is_speech = self.vad.is_speech(mono.tobytes(), RATE)
                if not in_utterance:
                    ring.append(mono)
                    if len(ring) > PREROLL_FRAMES:
                        ring.pop(0)
                    speech_run = speech_run + 1 if is_speech else 0
                    if speech_run >= OPEN_FRAMES:
                        in_utterance = True
                        frames = ring[:]
                        silence_run = 0
                        signals.set_state("listening")
                else:
                    frames.append(mono)
                    if is_speech:
                        speech_total += 1
                        silence_run = 0
                    else:
                        silence_run += 1
                    if silence_run >= self.silence_frames or \
                       len(frames) * FRAME_MS / 1000 > MAX_UTTER_S:
                        if speech_total < 8:
                            # <240ms of actual speech: a noise blip, not
                            # a sentence — keep listening
                            in_utterance = False
                            frames, ring = [], []
                            speech_run = speech_total = 0
                            continue
                        pcm = np.concatenate(frames)
                        _log_voice_id(pcm)
                        return transcribe(pcm)


def record_held(is_held, max_s: float = 60.0, min_s: float = 0.25) -> str | None:
    """Hold-to-talk capture: record raw audio while is_held() is True,
    then transcribe. The button is the VAD — no endpointing. Returns
    None for taps shorter than min_s (accidental presses)."""
    frames: list[np.ndarray] = []
    with _open_mic() as stream:
        while is_held() and len(frames) * FRAME_MS / 1000 < max_s:
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
        # a small tail so the last word isn't clipped at release
        for _ in range(6):
            block, _ = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
    if len(frames) * FRAME_MS / 1000 < min_s:
        return None
    pcm = np.concatenate(frames)
    _log_voice_id(pcm)
    return transcribe(pcm)


if __name__ == "__main__":
    import time
    print("[ears] listening — say something...", flush=True)
    ears = Ears()
    start = time.time()
    while time.time() - start < 30:
        text = ears.listen_once(timeout_s=30 - (time.time() - start))
        if text:
            print(f"[ears] heard: {text!r}", flush=True)
            break
        if text is None:
            print("[ears] timed out with no speech", flush=True)
            break
        print("[ears] (noise/empty — still listening)", flush=True)

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
"""The signal bus — tiny files any other program can watch.

The voice line leaves notes; faces read the notes. That one dumb trick
is the whole integration surface:

  .voice_state        idle | listening | thinking | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_loading_pid  exists while the thinking sound is playing
  .voice_rate_limits  JSON {window: {utilization, resets_at}} — only
                      written when show_usage is on

Written to signals_dir (default: the repo root). Visualizers built on
this contract just work.

THE BAREHANDS SEAM: set barehands_state_dir in backtalk.json to a
barehands checkout's state/ folder and the same signals are mirrored in
its format (state/state as a bare word, state/wave.json normalized
0..1) — the on-screen ring becomes your agent's face with zero glue.

Every write is wrapped: the bus must never crash the voice line.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

from backtalk.config import CFG

_DIR = CFG["signals_dir"]
_STATE_FILE = os.path.join(_DIR, ".voice_state")
_WAVEFORM_FILE = os.path.join(_DIR, ".voice_waveform")
_LOADING_PID_FILE = os.path.join(_DIR, ".voice_loading_pid")
_DIRECTION_FILE = os.path.join(_DIR, ".voice_direction")
_REPLY_DONE_FILE = os.path.join(_DIR, ".voice_reply_done")
_RATE_LIMIT_FILE = os.path.join(_DIR, ".voice_rate_limits")

_BH = CFG.get("barehands_state_dir") or ""
_BH_STATE = os.path.join(_BH, "state") if _BH else ""
_BH_WAVE = os.path.join(_BH, "wave.json") if _BH else ""

_THINKING_SOUND = CFG.get("thinking_sound") or ""

_WAVEFORM_MIN_INTERVAL = 1.0 / 15   # ~15 writes/sec is plenty for 60fps reads
_last_waveform_write = 0.0
_static_proc: subprocess.Popen | None = None
_static_thread: threading.Thread | None = None
_static_stop_event = threading.Event()


def set_state(name: str):
    """Write the state. Never raises — the show must go on."""
    try:
        with open(_STATE_FILE, "w") as f:
            f.write(name)
    except OSError:
        pass
    if _BH_STATE:
        try:
            with open(_BH_STATE, "w") as f:
                f.write(name)
        except OSError:
            pass


def feed_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) — throttled, downsampled to 64 points.

    Also re-asserts state="speaking" on the same throttle: this only runs
    while the mouth is audibly playing, so the bus self-heals within
    ~70ms if a stray writer stomps the state mid-speech. (That self-heal
    rule once closed a bug that took a whole evening to find.)"""
    global _last_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
        if _BH_WAVE:
            norm = np.clip(np.abs(raw) / 32768.0, 0.0, 1.0)
            with open(_BH_WAVE, "w") as f:
                f.write(json.dumps({"ts": now, "samples": norm.tolist()}))
    except (OSError, ValueError):
        pass
    set_state("speaking")


def direction(items):
    """Stage directions the agent wrote into its reply, published at the
    moment the audio carrying them starts playing.

    Your agent can emit `<<anything>>` inline and backtalk will never speak
    it. What the tag MEANS is deliberately not backtalk's business: it
    publishes the raw strings and something else decides. That is the whole
    reason this is a file and not a plugin API.

    The timing is the point, and it is the one part a watcher cannot do for
    itself: these fire when the sentence becomes AUDIBLE, not when the model
    generated it. A screen cue lands on the spoken word instead of seconds
    early. Never raises."""
    if not items:
        return
    try:
        with open(_DIRECTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "directions": list(items)}))
    except OSError:
        pass


def reply_done():
    """One reply has finished speaking and its audio has fully drained.

    Distinct from the state going idle, which also happens in the gaps
    BETWEEN sentences of the same reply. Anything waiting for the agent to
    genuinely stop talking wants this rather than a state flicker. Never
    raises."""
    try:
        with open(_REPLY_DONE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time()}))
    except OSError:
        pass


_rate_limits: dict = {}


def set_rate_limit(window: str, utilization, resets_at):
    """One usage window's reading — how much of the plan is spent.

    Merged rather than replaced, because the reading arrives one window
    at a time and a face wants to draw both at once. `utilization` is a
    0..1 fraction (or None when the window has not reported a number
    yet, which is a real state and not an error); `resets_at` is a unix
    epoch.

    NOTHING CALLS THIS UNLESS show_usage IS ON. That is a privacy
    default, not a performance one: this is the account holder's own
    spend, and it renders on a face that may well be pointed at a
    camera. It never appears without being asked for. (Community fix,
    ai-visualizer issue #1.)

    Never raises."""
    if not window:
        return
    _rate_limits[window] = {"utilization": utilization,
                            "resets_at": resets_at}
    try:
        with open(_RATE_LIMIT_FILE, "w") as f:
            f.write(json.dumps(_rate_limits))
    except OSError:
        pass


def _quiet_copy(path: str, factor: float = 0.35) -> str:
    """Windows-only: SoundPlayer has no volume control, so scale the WAV
    samples down once and cache the result next to the original, rather
    than re-scaling on every single play. Returns the original path
    unchanged if anything about this goes wrong -- a full-volume thinking
    sound beats a silent one."""
    import wave
    quiet_path = path.rsplit(".", 1)[0] + f"_quiet{int(factor*100)}.wav"
    if os.path.exists(quiet_path):
        return quiet_path
    try:
        with wave.open(path, "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())
        if params.sampwidth != 2:
            return path  # only handling the common 16-bit case
        import array
        samples = array.array("h", frames)
        for i in range(len(samples)):
            samples[i] = int(samples[i] * factor)
        with wave.open(quiet_path, "wb") as dst:
            dst.setparams(params)
            dst.writeframes(samples.tobytes())
        return quiet_path
    except Exception:
        return path


def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["afplay", "-v", "0.35", path]
    if sys.platform == "win32":
        # None of afplay/ffplay/aplay/paplay exist on Windows by default,
        # so this branch was previously silently falling through to
        # returning None -- the thinking sound never played through the
        # local speakers at all, only via ai-visualizer's own separate
        # browser-side player watching the same .voice_loading_pid signal.
        # PowerShell's SoundPlayer needs no extra install, same fix
        # pattern already used for jarvis-dashboard's Windows TTS bug.
        # SoundPlayer has no volume control at all (always full system
        # volume, unlike afplay's "-v 0.35" and ffplay's "-volume 35"
        # above) -- tried Windows Media Player's COM object next since it
        # does expose volume, but on this machine it got permanently
        # stuck in the "Transitioning" playState and never actually
        # played anything, likely because classic Windows Media Player
        # itself isn't fully present on this debloated Windows build.
        # Real fix: pre-generate a volume-reduced copy of the sound file
        # once, and play THAT with the SoundPlayer approach that's
        # already confirmed working -- sidesteps needing any player to
        # support volume control at all.
        quiet_path = _quiet_copy(path)
        return ["powershell", "-NoProfile", "-Command",
                f"(New-Object Media.SoundPlayer '{quiet_path}').PlaySync()"]
    for cand in ("ffplay", "aplay", "paplay"):
        from shutil import which
        if which(cand):
            if cand == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", "-volume", "35", path]
            return [cand, path]
    return None


def _static_loop(cmd: list[str]):
    """Runs on a background thread: replays the thinking sound back-to-back
    until static_stop() sets the stop event. Fixes a real gap -- the sound
    used to play exactly once (~20-36s) and then go silent for however
    much longer the actual task took, which is most of them tonight."""
    global _static_proc
    while not _static_stop_event.is_set():
        try:
            _static_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(_LOADING_PID_FILE, "w") as f:
                f.write(str(_static_proc.pid))
        except OSError:
            return
        _static_proc.wait()
        if _static_stop_event.is_set():  # terminated by static_stop()
            return


def static_start():
    """Optional thinking sound — loops while the brain works."""
    global _static_thread
    if not _THINKING_SOUND or not os.path.exists(_THINKING_SOUND):
        return
    static_stop()
    cmd = _player_cmd(_THINKING_SOUND)
    if not cmd:
        return
    _static_stop_event.clear()
    _static_thread = threading.Thread(
        target=_static_loop, args=(cmd,), daemon=True)
    _static_thread.start()


def static_stop():
    global _static_proc, _static_thread
    _static_stop_event.set()
    if _static_proc is not None:
        try:
            _static_proc.terminate()
        except OSError:
            pass
        _static_proc = None
    if _static_thread is not None:
        _static_thread.join(timeout=1.0)
        _static_thread = None
    try:
        os.remove(_LOADING_PID_FILE)
    except OSError:
        pass


def is_static_playing() -> bool:
    """True while the thinking sound is actively looping. The open mic's
    gate needs this -- it only ever checked mouth.speaking, so once the
    thinking sound started looping continuously (instead of playing once
    for ~20-36s) it bled into the real mic for however long a task took,
    repeatedly triggering false VAD detections on its own audio."""
    return _static_thread is not None and _static_thread.is_alive()

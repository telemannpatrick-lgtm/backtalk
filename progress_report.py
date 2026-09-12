#!/usr/bin/env python3
"""Download a URL while reporting real progress to the ai-visualizer signal
bus (.voice_progress), so the face can show an actual percentage for tasks
that have one -- unlike thinking/idle, which can't express "62% of 1.2GB."

Not part of backtalk itself -- any script can write .voice_progress
directly; this is just the reusable one for "download a file." Streams the
download itself (urllib, not curl) so real byte counts drive the
percentage directly, instead of parsing a text progress meter.

Usage: python3 progress_report.py <url> <output_path> [label]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

BUS_DIR = Path(__file__).parent
PROGRESS_FILE = BUS_DIR / ".voice_progress"
CHUNK_SIZE = 1024 * 256  # 256KB


def _write_progress(percent: float, label: str):
    try:
        PROGRESS_FILE.write_text(json.dumps(
            {"ts": time.time(), "percent": round(percent, 1), "label": label}))
    except OSError:
        pass  # never let a signal-bus write crash the actual download


def _clear_progress():
    try:
        PROGRESS_FILE.unlink()
    except OSError:
        pass


def download(url: str, output_path: str, label: str = ""):
    label = label or Path(output_path).name
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_write = 0.0
        with open(output_path, "wb") as out:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if total and now - last_write > 0.3:  # ~3 updates/sec, plenty
                    _write_progress(downloaded / total * 100, label)
                    last_write = now
    _write_progress(100.0, label)
    time.sleep(0.5)  # let the last 100% actually get polled before clearing
    _clear_progress()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 progress_report.py <url> <output_path> [label]")
        sys.exit(1)
    try:
        download(sys.argv[1], sys.argv[2],
                 sys.argv[3] if len(sys.argv) > 3 else "")
    except Exception as e:
        _clear_progress()
        print(f"download failed: {e!r}")
        sys.exit(1)

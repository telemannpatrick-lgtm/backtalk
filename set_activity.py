#!/usr/bin/env python3
"""Write a one-line "what's happening right now" label to the signal bus,
so the face shows real current status even when no spoken reply has
landed in a while -- separate from .voice_progress (which is specifically
for tasks with a measurable percentage). Meant to be called by the agent
itself during a long working stretch, not by backtalk automatically --
backtalk has no visibility into what a tool call is actually doing.

Usage: python3 set_activity.py "editing signals.py"
       python3 set_activity.py --clear
"""
import json
import sys
import time
from pathlib import Path

ACTIVITY_FILE = Path(__file__).parent / ".voice_activity"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python3 set_activity.py "label" | --clear')
        sys.exit(1)
    if sys.argv[1] == "--clear":
        try:
            ACTIVITY_FILE.unlink()
        except OSError:
            pass
    else:
        ACTIVITY_FILE.write_text(
            json.dumps({"ts": time.time(), "label": sys.argv[1]}))

#!/bin/bash
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
# backtalk entrypoint — start a spoken conversation with your agent.
# Terminal-invoked (inherits the terminal's mic permission). Ctrl-C hangs up.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
# WSL2 self-heal. PortAudio's ALSA pulse plugin looks for the socket at
# the standard runtime path whatever $PULSE_SERVER says, and WSLg only
# creates it at /mnt/wslg/PulseServer. Without this link, playback does
# not fail cleanly: it CRASHES the process with a core dump, which reads
# as the voice line being broken rather than the audio path being
# unwired. That runtime folder is wiped on every reboot, so the link is
# remade on every launch rather than once at install.
#
# Guarded on the socket existing, so this is a no-op on every platform
# that is not WSL2.
if [ -S /mnt/wslg/PulseServer ] && [ ! -S "/run/user/$(id -u)/pulse/native" ]; then
  mkdir -p "/run/user/$(id -u)/pulse"
  ln -sf /mnt/wslg/PulseServer "/run/user/$(id -u)/pulse/native"
fi
# Single-instance guard: a stale voice session left in a background
# terminal answers the same mic alongside a fresh launch = two voices at
# once, and it sounds haunted. One body, one mouth.
if pkill -f "backtalk[.]main" 2>/dev/null; then
  echo "[backtalk] replaced a previous voice session"
  sleep 1   # let the old process release mic/speaker devices
fi
# pkill above only sees this shell's own MSYS process tree, so a backtalk
# instance launched from a *different* terminal (the normal case on
# Windows — you never relaunch from the same window) survives it. Sweep
# by command line system-wide instead, via a real .ps1 file rather than an
# inline -Command string (that was tried first and failed: bash's argv
# escaping through to a native exe is fragile enough that it silently
# never fired). No-op where powershell.exe doesn't exist (real Linux/Mac).
if command -v powershell.exe >/dev/null 2>&1; then
  killed=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "kill-stale.ps1" -Pattern "*backtalk.main*" 2>/dev/null | tr -d '\r')
  if [ -n "$killed" ] && [ "$killed" != "0" ]; then
    echo "[backtalk] replaced a previous voice session (cross-window)"
    sleep 1
  fi
fi
# Self-repair: reconcile the environment with the shipped package list
# before launching (sub-second when already current). --inexact keeps
# anything the person's agent added on purpose; a missing package
# (a half-finished install, a drifted env) heals here instead of
# crashing on import. If it fails (offline), launch anyway.
uv sync -q --inexact 2>/dev/null || true
exec uv run python -m backtalk.main "$@" 2> >(grep -vi "pkg_resources\|VIRTUAL_ENV" >&2)

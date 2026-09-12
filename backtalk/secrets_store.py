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
"""Cross-platform secret lookup: OS credential store first, env var
fallback. The same chain mouth.py uses for the ElevenLabs key
(_get_elevenlabs_key), generalized so remote.py's auth token can use it
too without a second copy of the lookup logic. A secret NEVER goes in a
file this repo writes — see each caller for the exact seeding command.
"""
import os
import subprocess
import sys

_cache: dict[str, str] = {}


def get_secret(key_slot: str, env_var: str) -> str:
    """The secret for `key_slot`, checked in this order:
      1. macOS Keychain, item `key_slot`:
         security add-generic-password -a "$USER" -s <key_slot> -T /usr/bin/security -w
      2. Linux secret-tool (libsecret), service `key_slot`:
         secret-tool store --label backtalk service <key_slot>
      3. the `env_var` environment variable — the last-resort fallback,
         and the only option on Windows for now. Know the tradeoff: an
         export line in a shell profile is a plaintext value on disk,
         which is exactly what the keychain path avoids.
    Cached in-process after the first lookup (successful or not — a
    missing secret doesn't get re-probed on every call)."""
    if key_slot in _cache:
        return _cache[key_slot]
    value = ""
    try:
        if sys.platform == "darwin":
            r = subprocess.run(
                ["security", "find-generic-password", "-s", key_slot, "-w"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                value = r.stdout.strip()
        elif sys.platform.startswith("linux"):
            from shutil import which
            if which("secret-tool"):
                r = subprocess.run(
                    ["secret-tool", "lookup", "service", key_slot],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    value = r.stdout.strip()
    except Exception:
        pass
    value = value or os.environ.get(env_var, "")
    _cache[key_slot] = value
    return value

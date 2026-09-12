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
"""Remote voice access — a phone (or anything speaking this small
WebSocket protocol) becomes another mouth/ears into THIS SAME live
backtalk session. Not a second assistant, not a second Claude session:
text/transcribed input from a remote client is pushed into the exact
queue main.py already drains for typed terminal input, so it goes
through the one shared handle() pipeline — same interrupt logic, same
permission gate, same session. Every sentence backtalk speaks (replies
AND permission questions) is mirrored to every connected client via
broadcast_tap, which main.py registers on the shared Mouth instance.

Broadcast is a party line, not per-client: every connected client hears
everything backtalk says, regardless of who asked. Fine for one
person's own devices; know that before handing the token to anyone else.

Security, modeled on jarvis-dashboard's companion server (see the vault
note "Agent Voice and Face Stack" for the comparison this grew out of),
adapted to backtalk's own conventions:
  - TLS via a self-signed CA generated on first use (openssl subprocess,
    same tool backtalk already assumes is available for git); no manual
    setup script needed, same "warm on first use" pattern as ears.warm()
    and mouth.warm().
  - A 64-char hex auth token, timing-safe compared, resolved through
    secrets_store.get_secret — NEVER generated into a file by this code.
  - Per-IP rate limiting and a connection cap, an idle timeout that
    closes a forgotten connection.

Protocol (JSON text frames + raw binary audio frames):
  client -> server:
    {"type": "text", "text": "..."}            one turn, text
    {"type": "audio_start"}                    begin an audio turn
    <binary frames: raw PCM16 mono 16kHz>       the audio itself
    {"type": "audio_end"}                       transcribe + send as text
    {"type": "ping"}
  server -> client:
    {"type": "connected"}
    {"type": "transcript", "text": "..."}       what audio_end transcribed
    {"type": "tts_audio", "data": "<b64 pcm16>", "sampleRate": N}
    {"type": "tts_end"}
    {"type": "error", "stage": "...", "message": "..."}
    {"type": "pong"}

No format conversion is needed for audio in either direction: backtalk's
ears.transcribe() takes raw PCM directly (unlike jarvis's whisper-cli
pipeline, which needs a WAV file and therefore ffmpeg) and mouth's
synth_stream() already yields raw PCM16 chunks. A client sends and
receives raw samples; ffmpeg is not a dependency of this module.
"""
import asyncio
import base64
import hmac
import json
import secrets as pysecrets
import shutil
import socket
import subprocess
import ssl
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets

from backtalk import secrets_store
from backtalk.config import CFG, REPO
from backtalk.vlog import log

CERTS_DIR = REPO / "certs"


def _ensure_certs(certs_dir: Path) -> tuple[str, str]:
    """Generate a self-signed CA + server certificate under certs_dir if
    one isn't already there. Returns (server_cert_path, server_key_path).

    Mirrors jarvis-dashboard's companion/setup.sh cert chain (CA ->
    signed server cert, SAN covering localhost/*.local/this hostname),
    but generated on demand here rather than via a separate setup
    script — remote access just works the first time it's enabled.
    """
    ca_key = certs_dir / "ca-key.pem"
    ca_cert = certs_dir / "ca.pem"
    server_key = certs_dir / "server-key.pem"
    server_cert = certs_dir / "server.pem"
    if server_cert.exists() and server_key.exists():
        return str(server_cert), str(server_key)

    openssl = shutil.which("openssl")
    if not openssl:
        raise RuntimeError(
            "openssl not found on PATH -- needed once to generate the "
            "TLS certificate for remote access. It ships with Git for "
            "Windows (…\\Git\\mingw64\\bin\\openssl.exe) or install it "
            "directly, then try again.")

    certs_dir.mkdir(parents=True, exist_ok=True)
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "localhost"

    def run(*args):
        subprocess.run([openssl, *args], check=True,
                       capture_output=True, timeout=30)

    run("genrsa", "-out", str(ca_key), "4096")
    run("req", "-x509", "-new", "-nodes", "-key", str(ca_key), "-sha256",
        "-days", "3650", "-out", str(ca_cert), "-subj",
        "/CN=backtalk Local CA")
    run("genrsa", "-out", str(server_key), "2048")
    csr = certs_dir / "server.csr"
    run("req", "-new", "-key", str(server_key), "-out", str(csr),
        "-subj", "/CN=backtalk-remote")
    ext_file = certs_dir / "server.ext"
    ext_file.write_text(
        "subjectAltName=DNS:localhost,DNS:*.local,DNS:"
        f"{hostname},IP:127.0.0.1\n")
    try:
        run("x509", "-req", "-in", str(csr), "-CA", str(ca_cert),
            "-CAkey", str(ca_key), "-CAcreateserial", "-out",
            str(server_cert), "-days", "730", "-sha256",
            "-extfile", str(ext_file))
    finally:
        csr.unlink(missing_ok=True)
        ext_file.unlink(missing_ok=True)
        (certs_dir / "ca.srl").unlink(missing_ok=True)
    log(f"[remote] generated a self-signed TLS certificate "
        f"(CA: {ca_cert}) -- install {ca_cert.name} as a trusted root "
        f"on any device that will connect, same as jarvis-dashboard's "
        f"certificate docs describe.")
    return str(server_cert), str(server_key)


def _build_ssl_context(cert_path: str, key_path: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)
    return ctx


class RemoteServer:
    def __init__(self, inbound_queue, loop: asyncio.AbstractEventLoop):
        self._queue = inbound_queue
        self._loop = loop
        self._clients: set = set()
        self._attempts: dict[str, list[float]] = {}
        self._server = None
        self._cfg = CFG.get("remote") or {}
        self._max_connections = int(self._cfg.get("max_connections") or 2)
        self._rate_limit = int(self._cfg.get("rate_limit_per_minute") or 10)
        self._idle_timeout = float(self._cfg.get("idle_timeout_s") or 300)
        self._token: str | None = None

    async def start(self) -> bool:
        """Resolve the token and certs, then bind. Returns False (and
        logs why) rather than raising: a remote-access misconfiguration
        must never take the local voice line down with it."""
        slot = self._cfg.get("token_key_slot") or "backtalk-remote"
        self._token = secrets_store.get_secret(slot, "BACKTALK_REMOTE_TOKEN")
        if not self._token:
            self._log_missing_token(slot)
            return False

        try:
            cert_path, key_path = await self._loop.run_in_executor(
                None, _ensure_certs, CERTS_DIR)
            ssl_ctx = _build_ssl_context(cert_path, key_path)
        except Exception as e:
            log(f"[remote] TLS certificate setup failed, remote access "
                f"stays off: {e}")
            return False

        host = self._cfg.get("host") or "0.0.0.0"
        port = int(self._cfg.get("port") or 7777)
        try:
            self._server = await websockets.serve(
                self._handler, host, port, ssl=ssl_ctx, max_size=2 ** 22)
        except OSError as e:
            log(f"[remote] couldn't bind {host}:{port} ({e}) -- remote "
                f"access stays off. Change remote.port in backtalk.json "
                f"if something else already owns that port.")
            return False

        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = "this machine"
        log(f"[remote] listening on wss://{hostname}:{port} "
            f"(token required, {self._max_connections} connection max)")
        return True

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _log_missing_token(self, slot: str):
        suggestion = pysecrets.token_hex(32)
        log("[remote] remote.enabled is true but no auth token is set "
            "-- refusing to listen unprotected.")
        log("[remote] backtalk will never write this to a file itself. "
            f'Set it yourself, in a FRESH terminal window: '
            f'setx BACKTALK_REMOTE_TOKEN "{suggestion}"')
        log(f"[remote] (macOS/Linux: prefer the OS keychain instead — "
            f"see secrets_store.get_secret's docstring for the seeding "
            f"command, slot name {slot!r})")
        log("[remote] setx only takes effect for processes started "
            "AFTER it runs — open a new terminal and relaunch backtalk.")

    # ---- Per-connection handling ----

    def _peer_ip(self, websocket) -> str:
        try:
            return str(websocket.remote_address[0])
        except Exception:
            return "unknown"

    def _extract_token(self, websocket) -> str:
        try:
            query = parse_qs(urlparse(websocket.path).query)
            return (query.get("token") or [""])[0]
        except Exception:
            return ""

    def _check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        recent = [t for t in self._attempts.get(ip, []) if now - t < 60]
        if len(recent) >= self._rate_limit:
            return False
        recent.append(now)
        self._attempts[ip] = recent
        return True

    async def _handler(self, websocket):
        ip = self._peer_ip(websocket)
        if not self._check_rate_limit(ip):
            log(f"[remote] rate limit hit from {ip}")
            await websocket.close(4029, "Too many attempts")
            return
        token = self._extract_token(websocket)
        if not token or not hmac.compare_digest(token, self._token):
            log(f"[remote] rejected connection from {ip}: bad token")
            await websocket.close(4001, "Unauthorized")
            return
        if len(self._clients) >= self._max_connections:
            log(f"[remote] rejected connection from {ip}: at max "
                f"connections ({self._max_connections})")
            await websocket.close(4003, "Too many connections")
            return

        self._clients.add(websocket)
        log(f"[remote] client connected from {ip} "
            f"({len(self._clients)}/{self._max_connections})")
        activity = [time.monotonic()]
        watchdog = asyncio.create_task(self._idle_watchdog(websocket, activity))
        await self._safe_send(websocket, {"type": "connected"})

        audio_buf = bytearray()
        in_audio = False
        try:
            async for message in websocket:
                activity[0] = time.monotonic()
                if isinstance(message, bytes):
                    if in_audio:
                        audio_buf += message
                    continue
                try:
                    msg = json.loads(message)
                except ValueError:
                    continue
                mtype = msg.get("type")
                if mtype == "ping":
                    await self._safe_send(websocket, {"type": "pong"})
                elif mtype == "text":
                    text = str(msg.get("text") or "").strip()
                    if text:
                        self._queue.put(text)
                elif mtype == "audio_start":
                    audio_buf = bytearray()
                    in_audio = True
                elif mtype == "audio_end":
                    in_audio = False
                    if audio_buf:
                        await self._transcribe_and_queue(websocket, bytes(audio_buf))
                    audio_buf = bytearray()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            watchdog.cancel()
            self._clients.discard(websocket)
            log(f"[remote] client disconnected ({ip})")

    async def _idle_watchdog(self, websocket, activity: list):
        try:
            while True:
                await asyncio.sleep(30)
                if time.monotonic() - activity[0] > self._idle_timeout:
                    log("[remote] closing idle connection")
                    await websocket.close(4000, "Idle timeout")
                    return
        except asyncio.CancelledError:
            return

    async def _transcribe_and_queue(self, websocket, audio_bytes: bytes):
        from backtalk import ears
        try:
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)
            text = await self._loop.run_in_executor(None, ears.transcribe, pcm)
        except Exception as e:
            log(f"[remote] transcription failed: {e}")
            await self._safe_send(websocket, {
                "type": "error", "stage": "transcription", "message": str(e)})
            return
        text = (text or "").strip()
        if not text:
            await self._safe_send(websocket, {
                "type": "error", "stage": "transcription",
                "message": "Empty transcription — please try again"})
            return
        await self._safe_send(websocket, {"type": "transcript", "text": text})
        self._queue.put(text)

    async def _safe_send(self, websocket, obj: dict):
        try:
            await websocket.send(json.dumps(obj))
        except Exception:
            pass

    # ---- Broadcast: called from Mouth's worker thread ----

    def broadcast_tap(self, sentence: str, directions=None):
        """Registered on Mouth.taps (main.py wires this up). Runs on
        Mouth's OWN worker thread, not the asyncio loop — the same
        thread that already does this exact kind of blocking synthesis
        work for local playback, so a second, independent synthesis
        pass here for remote clients costs CPU but never blocks the
        event loop or local audio. asyncio.run_coroutine_threadsafe
        hands each chunk to the loop without this thread waiting on it."""
        if not self._clients or not sentence or not sentence.strip():
            return
        from backtalk import mouth as mouth_module
        try:
            for rate, pcm in mouth_module.synth_stream(sentence):
                payload = json.dumps({
                    "type": "tts_audio",
                    "data": base64.b64encode(pcm.tobytes()).decode("ascii"),
                    "sampleRate": rate,
                })
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_raw(payload), self._loop)
        except Exception as e:
            log(f"[remote] broadcast synthesis failed: {e}")
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_raw(json.dumps({"type": "tts_end"})), self._loop)

    async def _broadcast_raw(self, payload: str):
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


def _insecure_client_ctx() -> ssl.SSLContext:
    """Self-signed server cert -> the test client must not verify it
    against a public CA. Only ever used by the __main__ self-test below,
    never by a real client (a real client should trust the generated
    CA cert, not disable verification)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


if __name__ == "__main__":
    # Self-test: start a real server on a throwaway port with a
    # throwaway token, then exercise it as a client would — wrong
    # token rejected, text message reaches the queue, broadcast_tap
    # actually synthesizes and delivers audio. No mobile client or
    # real backtalk.json changes required to run this.
    import os
    import queue as _queue

    TEST_PORT = 17778

    async def _self_test():
        os.environ.setdefault("BACKTALK_REMOTE_TOKEN", pysecrets.token_hex(32))
        CFG["remote"] = {**(CFG.get("remote") or {}),
                         "enabled": True, "host": "127.0.0.1",
                         "port": TEST_PORT}

        loop = asyncio.get_event_loop()
        q: "_queue.Queue[str]" = _queue.Queue()
        server = RemoteServer(q, loop)
        if not await server.start():
            print("[remote self-test] FAIL: server did not start")
            return
        token = server._token

        results = []

        def check(name, ok, detail=""):
            results.append(ok)
            print(f"[remote self-test] {'PASS' if ok else 'FAIL'}: {name}"
                  + (f" ({detail})" if detail else ""))

        # 1. Wrong token: the handshake succeeds (auth happens after,
        # in _handler), so the client sees the socket close immediately.
        try:
            async with websockets.connect(
                    f"wss://127.0.0.1:{TEST_PORT}/?token=wrong",
                    ssl=_insecure_client_ctx()) as ws:
                await ws.recv()
            check("wrong token rejected", False, "connection stayed open")
        except websockets.exceptions.ConnectionClosed as e:
            check("wrong token rejected", True, f"close code {e.code}")

        # 2. Correct token: connect, get `connected`, send text, confirm
        # it lands in the queue exactly like typed terminal input would.
        async with websockets.connect(
                f"wss://127.0.0.1:{TEST_PORT}/?token={token}",
                ssl=_insecure_client_ctx()) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
            check("connected message", hello.get("type") == "connected",
                  repr(hello))

            await ws.send(json.dumps({"type": "text", "text": "hello from test"}))
            await asyncio.sleep(0.2)
            try:
                got = q.get_nowait()
            except _queue.Empty:
                got = None
            check("text message reached inbound queue",
                  got == "hello from test", repr(got))

            # 3. Broadcast tap: real synthesis, real audio over the wire.
            server.broadcast_tap("This is a broadcast test.")
            chunks, got_end = 0, False
            for _ in range(500):
                msg = json.loads(await asyncio.wait_for(ws.recv(), 15))
                if msg["type"] == "tts_audio":
                    chunks += 1
                elif msg["type"] == "tts_end":
                    got_end = True
                    break
            check("broadcast_tap delivered audio", chunks > 0 and got_end,
                  f"{chunks} chunk(s), tts_end={got_end}")

        await server.stop()
        print(f"[remote self-test] {sum(results)}/{len(results)} passed")
        if not all(results):
            raise SystemExit(1)

    asyncio.run(_self_test())

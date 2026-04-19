"""Cactus local LLM + tools, optional Gemini fallback (google-genai)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable


def _ensure_cactus_python_on_path() -> None:
    """Make ``from src.cactus / src.downloads import ...`` work without requiring
    the caller to export PYTHONPATH. The Cactus FFI package lives in
    ``desert/cactus/python`` alongside the libcactus shared library; we add it
    to ``sys.path`` at import time so every worker process can load it.
    """
    candidate = Path(__file__).resolve().parent.parent / "cactus" / "python"
    if candidate.is_dir():
        p = str(candidate)
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_cactus_python_on_path()


log = logging.getLogger(__name__)

SendFn = Callable[[dict[str, Any]], Awaitable[None]]

MOCK = os.environ.get("DESERT_MOCK", "").lower() in ("1", "true", "yes")


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "")


def _cactus_cloud_handoff_enabled() -> bool:
    key = (os.environ.get("CACTUS_CLOUD_KEY") or os.environ.get("CACTUS_CLOUD_API_KEY") or "").strip()
    return bool(key) and _env_flag("DESERT_ENABLE_CACTUS_CLOUD", "1")


def _llm_inference_options_json() -> str:
    """Match InferenceOptions JSON for libcactus; optional parallel Cactus cloud if API key is set."""
    handoff = _cactus_cloud_handoff_enabled()
    return json.dumps(
        {
            "max_tokens": 512,
            "temperature": 0.6,
            "telemetry_enabled": False,
            "auto_handoff": handoff,
            "handoff_with_images": handoff and _env_flag("DESERT_CACTUS_CLOUD_IMAGES", "0"),
        }
    )


def _messages_to_gemini_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user")
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content)
        lines.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(lines)


def _gemini_cloud_sync(messages: list[dict[str, Any]], *, force: bool = False) -> str | None:
    """Blocking Gemini completion; used from asyncio.to_thread."""
    if not force and not _env_flag("DESERT_CLOUD_FALLBACK", "1"):
        return None
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return None
    model = (os.environ.get("DESERT_GEMINI_MODEL") or "gemini-3-flash-preview").strip()
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = _messages_to_gemini_prompt(messages)
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (getattr(resp, "text", None) or "").strip()
        return text or None
    except Exception:
        log.exception("gemini cloud" if force else "gemini fallback")
        return None


def _gemini_fallback_sync(messages: list[dict[str, Any]]) -> str | None:
    return _gemini_cloud_sync(messages, force=False)


def _repo_cactus_root() -> Path:
    """desert/worker -> desert -> repo root; sibling cactus/."""
    return Path(__file__).resolve().parent.parent.parent / "cactus"


def _resolve_llm_weights() -> Path:
    """Match ``examples/main.py``: ``ensure_model(MODEL_ID)`` or explicit ``DESERT_LLM_WEIGHTS``."""
    env = os.environ.get("DESERT_LLM_WEIGHTS")
    if env:
        return Path(env)
    from src.downloads import ensure_model

    model_id = os.environ.get("DESERT_LLM_MODEL_ID", "google/gemma-3-270m-it")
    return ensure_model(model_id)


def _asr_model_id() -> str:
    return os.environ.get("DESERT_ASR_MODEL", "openai/whisper-base")


def summarize_outbound_messages(buf: list[dict[str, Any]]) -> tuple[str, str, str | None]:
    """Derive (reply_text, inference_source, error) from pipeline outbound messages."""
    err_msg: str | None = None
    for m in buf:
        if m.get("type") == "error":
            err_msg = str(m.get("message") or "error")
    last_text = ""
    last_src = "local"
    for m in buf:
        if m.get("type") == "llm_response":
            last_text = str(m.get("text") or "")
            last_src = str(m.get("inference_source") or last_src)
        if m.get("type") == "inference_meta" and m.get("inference_source"):
            last_src = str(m.get("inference_source"))
    return last_text, last_src, err_msg


class CactusPipeline:
    def __init__(self, send: SendFn) -> None:
        self._send = send
        self._asr_model: int | None = None
        self._llm_model: int | None = None
        self._tools_json: str = ""
        self._ready = False
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def capture_outbound(self):
        buf: list[dict[str, Any]] = []
        prev = self._send

        async def _cap(m: dict[str, Any]) -> None:
            buf.append(m)

        self._send = _cap
        try:
            yield buf
        finally:
            self._send = prev

    async def ensure_loaded(self, tools_json: str) -> None:
        if tools_json:
            self._tools_json = tools_json
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        if MOCK:
            log.warning("DESERT_MOCK=1: skipping Cactus model load")
            self._ready = True
            return

        from src.cactus import cactus_destroy, cactus_init
        from src.downloads import ensure_model

        if self._asr_model is not None:
            cactus_destroy(self._asr_model)
            self._asr_model = None
        if self._llm_model is not None:
            cactus_destroy(self._llm_model)
            self._llm_model = None

        asr_path = ensure_model(_asr_model_id())
        llm_path = _resolve_llm_weights()
        if not llm_path.is_dir() or not (llm_path / "config.txt").exists():
            raise FileNotFoundError(f"LLM weights not found: {llm_path}")

        log.info("Loading ASR from %s", asr_path)
        self._asr_model = cactus_init(str(asr_path), None, False)
        log.info("Loading LLM from %s", llm_path)
        self._llm_model = cactus_init(str(llm_path), None, False)
        self._ready = True

    def close(self) -> None:
        if MOCK:
            return
        try:
            from src.cactus import cactus_destroy

            if self._asr_model is not None:
                cactus_destroy(self._asr_model)
            if self._llm_model is not None:
                cactus_destroy(self._llm_model)
        except Exception as e:
            log.debug("cactus destroy: %s", e)
        self._asr_model = None
        self._llm_model = None
        self._ready = False

    async def process_utterance(self, session_id: str, pcm_s16le: bytes) -> None:
        """Run full pipeline for one utterance (16-bit PCM mono 16kHz)."""
        await self.ensure_loaded(self._tools_json)
        if not pcm_s16le:
            await self._send({"type": "error", "session_id": session_id, "message": "empty audio"})
            return

        if MOCK:
            text = "[mock transcription] User spoke (mock)."
            await self._send({"type": "transcription", "session_id": session_id, "text": text, "partial": False})
            reply = "[mock] I heard you. Set DESERT_MOCK=0 and build Cactus for real replies."
            await self._send(
                {
                    "type": "llm_response",
                    "session_id": session_id,
                    "text": reply,
                    "partial": False,
                    "inference_source": "local",
                }
            )
            await self._send({"type": "inference_meta", "session_id": session_id, "inference_source": "local"})
            await self._stream_tts(session_id, reply)
            await self._send({"type": "complete", "session_id": session_id})
            return

        loop = asyncio.get_running_loop()
        text = await asyncio.to_thread(self._transcribe_pcm, pcm_s16le, session_id, loop)
        if not text.strip():
            await self._send({"type": "transcription", "session_id": session_id, "text": "", "partial": False})
            await self._send({"type": "complete", "session_id": session_id})
            return

        await self._send({"type": "transcription", "session_id": session_id, "text": text, "partial": False})
        await self._run_llm_tool_loop(session_id, text)

    async def process_text_prompt(self, session_id: str, text: str, *, force_cloud: bool = False) -> None:
        """LLM + tools + TTS from typed user text (skip STT)."""
        text = (text or "").strip()
        if not text:
            await self._send({"type": "error", "session_id": session_id, "message": "empty text"})
            await self._send({"type": "complete", "session_id": session_id})
            return

        await self.ensure_loaded(self._tools_json)

        if MOCK:
            await self._send({"type": "transcription", "session_id": session_id, "text": text, "partial": False})
            if force_cloud:
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": "You are a concise assistant. Answer directly.",
                    },
                    {"role": "user", "content": text},
                ]
                fb = await asyncio.to_thread(_gemini_cloud_sync, messages, force=True)
                if fb:
                    await self._send(
                        {
                            "type": "llm_response",
                            "session_id": session_id,
                            "text": fb,
                            "partial": False,
                            "inference_source": "gemini",
                        }
                    )
                    await self._send({"type": "inference_meta", "session_id": session_id, "inference_source": "gemini"})
                    await self._stream_tts(session_id, fb)
                else:
                    await self._send(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "message": "force cloud: set GEMINI_API_KEY or disable DESERT_MOCK",
                        }
                    )
                await self._send({"type": "complete", "session_id": session_id})
                return
            reply = "[mock] Text received. Set DESERT_MOCK=0 and build Cactus for real replies."
            await self._send(
                {
                    "type": "llm_response",
                    "session_id": session_id,
                    "text": reply,
                    "partial": False,
                    "inference_source": "local",
                }
            )
            await self._send({"type": "inference_meta", "session_id": session_id, "inference_source": "local"})
            await self._stream_tts(session_id, reply)
            await self._send({"type": "complete", "session_id": session_id})
            return

        await self._send({"type": "transcription", "session_id": session_id, "text": text, "partial": False})
        await self._run_llm_tool_loop(session_id, text, force_cloud=force_cloud)

    async def _run_llm_tool_loop(self, session_id: str, text: str, *, force_cloud: bool = False) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise voice assistant. "
                    "Use tools for YouTube, document/legal-style analysis, or web search when facts or "
                    "current information are needed. After tools, answer in short natural language."
                ),
            },
            {"role": "user", "content": text},
        ]

        if force_cloud:
            fb = await asyncio.to_thread(_gemini_cloud_sync, messages, force=True)
            if fb:
                await self._send(
                    {
                        "type": "llm_response",
                        "session_id": session_id,
                        "text": fb,
                        "partial": False,
                        "inference_source": "gemini",
                    }
                )
                await self._send({"type": "inference_meta", "session_id": session_id, "inference_source": "gemini"})
                await self._stream_tts(session_id, fb)
            else:
                await self._send(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "message": "force cloud failed: set GEMINI_API_KEY (and check DESERT_GEMINI_MODEL)",
                    }
                )
            await self._send({"type": "complete", "session_id": session_id})
            return

        options = _llm_inference_options_json()
        max_iters = 5

        from src.cactus import cactus_complete
        from worker.tools import run_tool

        for _iteration in range(max_iters):

            def run_complete() -> dict[str, Any]:
                try:
                    raw = cactus_complete(
                        self._llm_model,
                        json.dumps(messages),
                        options,
                        self._tools_json,
                        None,
                        None,
                    )
                    raw = (raw or "").strip()
                    if not raw:
                        return {"success": False, "error": "empty completion response"}
                    return json.loads(raw)
                except json.JSONDecodeError as e:
                    return {"success": False, "error": f"invalid completion json: {e}"}

            result = await asyncio.to_thread(run_complete)
            if not result.get("success"):
                err = result.get("error") or "completion failed"
                fb_text = await asyncio.to_thread(_gemini_fallback_sync, messages)
                if fb_text:
                    await self._send(
                        {
                            "type": "llm_response",
                            "session_id": session_id,
                            "text": fb_text,
                            "partial": False,
                            "inference_source": "gemini",
                        }
                    )
                    await self._send({"type": "inference_meta", "session_id": session_id, "inference_source": "gemini"})
                    await self._stream_tts(session_id, fb_text)
                else:
                    await self._send({"type": "error", "session_id": session_id, "message": err})
                break

            base_reply = (result.get("response") or "").strip()
            fcs = result.get("function_calls") or []

            reply_text = base_reply
            used_gemini = False
            if not fcs and not reply_text:
                fb = await asyncio.to_thread(_gemini_fallback_sync, messages)
                if fb:
                    reply_text = fb
                    used_gemini = True

            infer_src = "gemini" if used_gemini else "local"

            await self._send(
                {
                    "type": "llm_response",
                    "session_id": session_id,
                    "text": reply_text,
                    "partial": False,
                    "inference_source": infer_src,
                }
            )

            if not fcs:
                await self._send(
                    {"type": "inference_meta", "session_id": session_id, "inference_source": infer_src}
                )
                if reply_text:
                    await self._stream_tts(session_id, reply_text)
                break

            messages.append({"role": "assistant", "content": base_reply or "(tool call)"})

            for fc in fcs:
                name, args = _normalize_function_call(fc)
                await self._send({"type": "tool_call", "session_id": session_id, "tool": name, "args": args})
                tool_out = run_tool(name, args)
                await self._send({"type": "tool_result", "session_id": session_id, "result": tool_out})
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps({"name": name, "content": json.dumps(tool_out)}),
                    }
                )

        await self._send({"type": "complete", "session_id": session_id})

    def _transcribe_pcm(
        self,
        pcm: bytes,
        session_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> str:
        from src.cactus import (
            cactus_stream_transcribe_process,
            cactus_stream_transcribe_start,
            cactus_stream_transcribe_stop,
        )

        stream = cactus_stream_transcribe_start(self._asr_model, None)
        last_sent = ""
        try:
            step = 32000
            for i in range(0, len(pcm), step):
                chunk = pcm[i : i + step]
                partial_json = cactus_stream_transcribe_process(stream, chunk)
                try:
                    p = json.loads(partial_json)
                    txt = (p.get("confirmed") or "") + (p.get("pending") or "")
                    if txt and txt != last_sent:
                        last_sent = txt
                        loop.call_soon_threadsafe(
                            lambda t=txt: asyncio.create_task(
                                self._send(
                                    {
                                        "type": "transcription",
                                        "session_id": session_id,
                                        "text": t,
                                        "partial": True,
                                    }
                                )
                            )
                        )
                except json.JSONDecodeError:
                    pass
            final_json = cactus_stream_transcribe_stop(stream)
            data = json.loads(final_json)
            return (data.get("response") or "").strip()
        except Exception as e:
            log.exception("transcribe")
            try:
                cactus_stream_transcribe_stop(stream)
            except Exception:
                pass
            raise RuntimeError(str(e)) from e

    async def _stream_tts(self, session_id: str, text: str) -> None:
        if not text.strip():
            return
        if os.environ.get("DESERT_DISABLE_TTS", "1").lower() in ("1", "true", "yes"):
            return
        try:
            import edge_tts

            voice = os.environ.get("EDGE_TTS_VOICE", "en-US-AriaNeural")
            communicate = edge_tts.Communicate(text[:8000], voice)
            buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
                    if len(buf) >= 32000:
                        payload = base64.b64encode(bytes(buf)).decode("ascii")
                        buf.clear()
                        await self._send({"type": "tts_chunk", "session_id": session_id, "audio": payload})
            if buf:
                payload = base64.b64encode(bytes(buf)).decode("ascii")
                await self._send({"type": "tts_chunk", "session_id": session_id, "audio": payload})
        except Exception as e:
            log.exception("tts")
            await self._send({"type": "error", "session_id": session_id, "message": f"tts: {e}"})


def _normalize_function_call(fc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "function" in fc and isinstance(fc["function"], dict):
        fn = fc["function"]
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments") or "{}"
    else:
        name = str(fc.get("name") or "")
        raw_args = fc.get("arguments") or "{}"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
    return name, args


async def webm_to_pcm_s16le_16k(webm: bytes) -> bytes:
    """Decode browser MediaRecorder (webm/opus) to s16le mono 16kHz."""
    if not webm:
        return b""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — install ffmpeg for audio decode")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(webm)
    if proc.returncode != 0:
        log.warning("ffmpeg: %s", err.decode("utf-8", errors="ignore")[:400])
        raise RuntimeError("ffmpeg failed to decode audio")
    return out

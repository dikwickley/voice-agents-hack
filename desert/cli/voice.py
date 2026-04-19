"""Voice dictation for the orchestrator TUI.

Captures 16 kHz / mono / s16le PCM from the host microphone (PortAudio via
``sounddevice``) and transcribes it into the prompt. Two backends are
supported and selected via ``DESERT_VOICE_BACKEND``:

* ``gemini`` (default when ``GEMINI_API_KEY`` is set) — sends the captured
  PCM as a WAV blob to ``google-genai`` and asks for a verbatim
  transcription. Dramatically better on low-level laptop mics than the
  on-device Whisper.
* ``cactus`` — the original on-device Cactus/Whisper pipeline. No network,
  but sensitive to input level and prone to hallucinations on quiet audio.

Heavy deps (cactus + sounddevice + google-genai) are imported lazily and
surfaced as ``VoiceUnavailable`` with an actionable message, so the TUI
keeps working even if one of them isn't installed on this host.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_cactus_python_on_path() -> None:
    """Mirror ``worker.cactus_pipeline._ensure_cactus_python_on_path`` so the
    TUI can ``from src.cactus import ...`` without the caller exporting
    ``PYTHONPATH``.
    """
    candidate = Path(__file__).resolve().parent.parent / "cactus" / "python"
    if candidate.is_dir():
        p = str(candidate)
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_cactus_python_on_path()


log = logging.getLogger(__name__)

# Cactus' streaming ASR expects raw 16-bit PCM, mono, 16 kHz. Same contract the
# worker honours in ``worker.cactus_pipeline._transcribe_pcm``.
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
BYTES_PER_SAMPLE = 2


BACKEND_GEMINI = "gemini"
BACKEND_CACTUS = "cactus"


class VoiceUnavailable(RuntimeError):
    """sounddevice / PortAudio / Cactus / Gemini not usable on this host."""


def _resolve_backend(explicit: str | None) -> str:
    """Pick a transcription backend.

    Precedence:
      1. explicit constructor arg (``VoiceEngine(backend=...)``)
      2. ``DESERT_VOICE_BACKEND`` env (``gemini`` | ``cactus``)
      3. ``gemini`` if ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is set
      4. ``cactus`` otherwise
    """
    pick = (explicit or os.environ.get("DESERT_VOICE_BACKEND") or "").strip().lower()
    if pick in (BACKEND_GEMINI, BACKEND_CACTUS):
        return pick
    has_key = bool(
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    )
    return BACKEND_GEMINI if has_key else BACKEND_CACTUS


@dataclass(frozen=True)
class AudioStats:
    """Quick read on what we actually captured — useful to tell 'mic silent /
    wrong device' apart from 'mic fine, ASR returned nothing'."""

    duration_s: float
    peak_int16: int
    rms_int16: float

    @property
    def peak_db(self) -> float:
        """Peak level in dBFS (0 dBFS = int16 clipping). ``-inf`` for pure silence."""
        if self.peak_int16 <= 0:
            return float("-inf")
        return 20.0 * math.log10(self.peak_int16 / 32768.0)

    @property
    def looks_silent(self) -> bool:
        """Heuristic: RMS below ~-60 dBFS is effectively silence for Whisper."""
        return self.rms_int16 < 30.0  # ~ -60 dBFS


_WHISPER_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")


def _clean_transcript(text: str) -> str:
    """Strip Whisper's internal tags (``<|nn|>``, ``<|la|>``, ``<|en|>``, …)
    and collapse whitespace. Cactus surfaces these verbatim in ``response``."""
    stripped = _WHISPER_SPECIAL_TOKEN_RE.sub(" ", text or "")
    return " ".join(stripped.split()).strip()


def _pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> bytes:
    """Wrap raw s16le PCM in a minimal RIFF/WAVE header so we can hand it to
    any consumer that expects a real audio file (e.g. ``audio/wav`` parts on
    the Gemini API)."""
    byte_rate = sample_rate * channels * BYTES_PER_SAMPLE
    block_align = channels * BYTES_PER_SAMPLE
    data_size = len(pcm)
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


_GEMINI_TRANSCRIBE_PROMPT = (
    "Transcribe the spoken audio verbatim. "
    "Return ONLY the transcript as plain text — no quotes, no labels, "
    "no commentary, no timestamps. "
    "If the clip contains no intelligible speech, return an empty string."
)


def audio_stats(pcm: bytes) -> AudioStats:
    """Compute (duration, peak, rms) of a s16le mono 16 kHz buffer."""
    samples = len(pcm) // BYTES_PER_SAMPLE
    if samples == 0:
        return AudioStats(0.0, 0, 0.0)
    duration = samples / SAMPLE_RATE
    try:
        import numpy as np

        arr = np.frombuffer(pcm, dtype=np.int16)
        peak = int(np.max(np.abs(arr.astype(np.int32))))
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
    except ImportError:
        # Fallback without numpy — slow but functional.
        import struct

        values = struct.unpack(f"<{samples}h", pcm)
        peak = max(abs(v) for v in values)
        rms = math.sqrt(sum(v * v for v in values) / samples)
    return AudioStats(duration_s=duration, peak_int16=peak, rms_int16=rms)


def list_input_devices() -> list[tuple[int, str, int]]:
    """Return ``[(device_id, name, max_input_channels)]`` for every input
    device PortAudio can see. Used by ``/voice devices`` in the TUI."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise VoiceUnavailable(str(e)) from e
    out: list[tuple[int, str, int]] = []
    for idx, dev in enumerate(sd.query_devices()):
        inputs = int(dev.get("max_input_channels") or 0)
        if inputs > 0:
            out.append((idx, str(dev.get("name") or f"device{idx}"), inputs))
    return out


def _resolve_input_device() -> int | str | None:
    """Resolve ``DESERT_AUDIO_INPUT_DEVICE`` (int index or name substring) to
    a value PortAudio will accept. ``None`` → use the system default."""
    raw = (os.environ.get("DESERT_AUDIO_INPUT_DEVICE") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw  # sounddevice matches by substring


class VoiceEngine:
    """Owns the transcription backend + live microphone stream.

    Lifecycle from the TUI's perspective:

        eng = VoiceEngine()
        eng.load()                     # blocking: mmaps weights (cactus) or
                                       # verifies Gemini credentials (gemini)
        eng.start_recording()          # opens PortAudio stream (callback fills buffer)
        ...user speaks...
        pcm = eng.stop_recording()     # closes stream, returns accumulated PCM
        text = eng.transcribe(pcm)     # dispatch to the active backend
        eng.close()                    # releases resources
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        backend: str | None = None,
        gemini_model: str | None = None,
    ) -> None:
        self.backend = _resolve_backend(backend)
        self.model_id = model_id or os.environ.get("DESERT_ASR_MODEL", "openai/whisper-base")
        # Deliberately *not* reusing ``DESERT_GEMINI_MODEL`` (that's the LLM
        # fallback model — e.g. ``gemini-3-flash-preview``, which hallucinates
        # lengthy paragraphs on short audio clips instead of transcribing).
        self.gemini_model = (
            gemini_model
            or os.environ.get("DESERT_VOICE_GEMINI_MODEL")
            or "gemini-2.5-flash"
        )
        self._asr_handle: int | None = None
        self._gemini_ready = False
        self._stream = None
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.last_device_name: str | None = None
        self.last_raw_json: str | None = None

    @property
    def backend_label(self) -> str:
        """Human-readable tag for the TUI (e.g. ``gemini: gemini-2.5-flash``)."""
        if self.backend == BACKEND_GEMINI:
            return f"gemini: {self.gemini_model}"
        return f"cactus: {self.model_id}"

    @property
    def is_loaded(self) -> bool:
        if self.backend == BACKEND_GEMINI:
            return self._gemini_ready
        return self._asr_handle is not None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def recorded_seconds(self) -> float:
        with self._buffer_lock:
            n = len(self._buffer)
        return n / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    def load(self) -> None:
        """Prepare the active backend. Idempotent; safe to call from a worker
        thread.

        * Gemini: verifies ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is set and
          imports ``google-genai`` (fast, no network I/O).
        * Cactus: downloads (if needed) and mmaps the ASR weights.
        """
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            if self.backend == BACKEND_GEMINI:
                self._load_gemini()
            else:
                self._load_cactus()

    def _load_gemini(self) -> None:
        api_key = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        ).strip()
        if not api_key:
            raise VoiceUnavailable(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set — add it to "
                "desert/.env, or run `/voice cactus` to use the on-device model."
            )
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise VoiceUnavailable(
                "google-genai is not installed. Add it to the env "
                "(`uv add google-genai`) and retry."
            ) from e
        self._gemini_ready = True
        log.info("voice: gemini backend ready (model=%s)", self.gemini_model)

    def _load_cactus(self) -> None:
        try:
            from src.cactus import cactus_init
            from src.downloads import ensure_model
        except ImportError as e:
            raise VoiceUnavailable(
                "Cactus Python FFI not found. Run "
                "`cactus build --python` in desert/cactus/, then retry."
            ) from e
        try:
            asr_path = ensure_model(self.model_id)
        except Exception as e:
            raise VoiceUnavailable(f"failed to fetch ASR model {self.model_id}: {e}") from e
        log.info("voice: loading ASR from %s", asr_path)
        self._asr_handle = cactus_init(str(asr_path), None, False)

    def start_recording(self) -> None:
        """Open a 16 kHz mono s16le input stream. Callback appends to
        ``self._buffer`` until :py:meth:`stop_recording` is called. Honours
        ``DESERT_AUDIO_INPUT_DEVICE`` (int index or name substring) for picking
        a specific mic; falls back to PortAudio's system default."""
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as e:
            raise VoiceUnavailable(
                "sounddevice is not installed. Add it to the env "
                "(`uv add sounddevice`) and install PortAudio "
                "(macOS: `brew install portaudio`)."
            ) from e
        except OSError as e:
            raise VoiceUnavailable(
                f"PortAudio not available: {e}. "
                "On macOS run `brew install portaudio` and retry."
            ) from e

        device = _resolve_input_device()

        # Resolve device → a real name so the TUI can surface it.
        try:
            info = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
            self.last_device_name = str(info.get("name") or "default")
            if int(info.get("max_input_channels") or 0) <= 0:
                raise VoiceUnavailable(
                    f"selected audio device has no input channels: {self.last_device_name!r}"
                )
        except VoiceUnavailable:
            raise
        except Exception as e:
            raise VoiceUnavailable(f"could not inspect audio device {device!r}: {e}") from e

        with self._buffer_lock:
            self._buffer.clear()

        def _callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            if status:
                log.debug("voice: stream status %s", status)
            with self._buffer_lock:
                self._buffer.extend(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=device,
                callback=_callback,
            )
            self._stream.start()
            log.info("voice: recording from %s (device=%r)", self.last_device_name, device)
        except Exception as e:
            self._stream = None
            raise VoiceUnavailable(
                f"could not open microphone ({self.last_device_name!r}): {e}"
            ) from e

    def stop_recording(self) -> bytes:
        """Stop the input stream and return the accumulated PCM. Safe to call
        even if no stream is open (returns ``b""``)."""
        if self._stream is None:
            return b""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            log.debug("voice: stop_recording swallowed: %s", e)
        finally:
            self._stream = None
        with self._buffer_lock:
            pcm = bytes(self._buffer)
            self._buffer.clear()
        return pcm

    def transcribe(self, pcm: bytes) -> str:
        """Transcribe ``pcm`` (s16le / 16 kHz / mono) via the active backend."""
        if not self.is_loaded:
            self.load()
        if not pcm:
            return ""
        if self.backend == BACKEND_GEMINI:
            return self._transcribe_gemini(pcm)
        return self._transcribe_cactus(pcm)

    def _transcribe_gemini(self, pcm: bytes) -> str:
        """Upload the capture as an inline WAV part and ask Gemini for a
        verbatim transcript. Much more robust on quiet laptop mics than the
        on-device Whisper — the model handles VAD / normalization internally.
        """
        # Skip the round-trip entirely on obviously silent buffers — Gemini
        # will otherwise fill the void with a filler token ("Mmm", "Uh").
        if audio_stats(pcm).looks_silent:
            self.last_raw_json = json.dumps({"model": self.gemini_model, "skipped": "silent"})
            log.info("voice: skipping gemini call (silent buffer, %d bytes)", len(pcm))
            return ""

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise VoiceUnavailable("google-genai import failed") from e

        api_key = (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        ).strip()
        if not api_key:
            raise VoiceUnavailable("GEMINI_API_KEY is not set")

        wav = _pcm_to_wav(pcm)
        client = genai.Client(api_key=api_key)
        audio_part = types.Part.from_bytes(data=wav, mime_type="audio/wav")
        try:
            resp = client.models.generate_content(
                model=self.gemini_model,
                contents=[_GEMINI_TRANSCRIBE_PROMPT, audio_part],
            )
        except Exception as e:
            # Surface the upstream error to the TUI verbatim — this is usually
            # a bad key, wrong model id, or a transient 5xx.
            self.last_raw_json = json.dumps({"error": str(e)})
            raise VoiceUnavailable(f"gemini transcribe failed: {e}") from e

        text = (getattr(resp, "text", None) or "").strip()
        # Some models like to wrap short utterances in quotes — strip a single
        # matched pair so "hello" doesn't land in the prompt as \"hello\".
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        self.last_raw_json = json.dumps({"model": self.gemini_model, "response": text})
        log.info("voice: gemini transcribe (%d bytes pcm) = %r", len(pcm), text)
        return text

    def _transcribe_cactus(self, pcm: bytes) -> str:
        """Run Cactus one-shot ``cactus_transcribe`` over ``pcm``.

        Cactus has two quirks we steer around:

        1. ``cactus_stream_transcribe_{start,process,stop}`` only returns
           *confirmed* tokens on ``_stop``; a short "ctrl+v, speak, ctrl+v"
           capture comes back as ``{"confirmed": ""}`` without ever decoding.
           So we use the one-shot ``cactus_transcribe`` instead.
        2. ``cactus_transcribe`` gates on Silero VAD by default (options
           ``use_vad=true``). On MacBook built-in mics the input level is
           often low enough that VAD classifies real speech as silence and
           the engine short-circuits with 0 decoded tokens. If VAD returns
           nothing we retry with ``use_vad=false``, which forces Whisper to
           decode the whole buffer.
        """
        try:
            from src.cactus import cactus_transcribe
        except ImportError as e:
            raise VoiceUnavailable("Cactus Python FFI import failed") from e

        # audio_path=None tells Cactus to use ``pcm_data`` directly; passing an
        # empty string here triggers "Both audio_file_path and pcm_buffer
        # provided" in the engine.
        def _run(options_json: str) -> tuple[str, dict[str, Any]]:
            raw = cactus_transcribe(self._asr_handle, None, None, options_json, None, pcm)
            try:
                return raw, json.loads(raw)
            except json.JSONDecodeError:
                return raw, {}

        raw, data = _run('{"use_vad":true}')
        text = _clean_transcript(data.get("response") or "")
        if not text:
            # VAD probably mis-classified the speech as silence. Retry without
            # it so Whisper actually decodes the buffer.
            raw, data = _run('{"use_vad":false}')
            text = _clean_transcript(data.get("response") or "")
        self.last_raw_json = raw
        log.info("voice: cactus transcribe raw = %s", raw)
        return text

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._asr_handle is not None:
            try:
                from src.cactus import cactus_destroy

                cactus_destroy(self._asr_handle)
            except Exception:
                pass
            self._asr_handle = None
        self._gemini_ready = False

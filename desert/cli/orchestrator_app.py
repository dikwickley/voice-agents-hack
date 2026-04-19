"""Desert orchestrator TUI — Gemini-CLI inspired single-column layout."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
from typing import Any

import httpx
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

from cli.voice import VoiceEngine, VoiceUnavailable, audio_stats, list_input_devices

log = logging.getLogger(__name__)

DEFAULT_BASE = "http://127.0.0.1:8000"


def _public_endpoint() -> str:
    """Best-effort label for the orchestrator's reachable address.

    Prefers DESERT_P2P_ANNOUNCE_ADDR (workers dial this), falls back to the
    host's primary LAN IP, and finally returns '' so the UI can hide the
    field instead of showing the misleading loopback bind address.
    """
    announce = (os.environ.get("DESERT_P2P_ANNOUNCE_ADDR") or "").strip()
    if announce:
        m = re.match(r"^/(?:ip4|ip6|dns4|dns6|dns)/([^/]+)/", announce)
        if m:
            host = m.group(1)
            if host and host not in ("0.0.0.0", "::", "::0"):
                return host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return ""


# ── Banner (block-letter "DESERT"), shaded line-by-line for a gradient feel ──
_BANNER = [
    " ██████   ██████   ██████   ██████   ██████   ████████",
    " ██   ██  ██       ██       ██       ██   ██     ██   ",
    " ██   ██  █████    ███████  █████    ██████      ██   ",
    " ██   ██  ██            ██  ██       ██   ██     ██   ",
    " ██████   ███████  ███████  ███████  ██   ██     ██   ",
]
# Pastel wash: cactus green (top 3) → desert yellow (bottom 2).
_BANNER_COLORS = ["#6fbf7d", "#8ed189", "#b8dc86", "#e8d289", "#f2c977"]

_TIPS = [
    "[dim]Tips for getting started:[/]",
    "[dim] 1. Describe a task; workers split it [italic]map → reduce[/] and answer together.[/]",
    "[dim] 2. Type [#f0d890]/help[/] for commands, or [#f0d890]/workers[/] to list connected nodes.[/]",
    "[dim] 3. [#f0d890]/parallel 4[/] to fan out manually · [#f0d890]/cloud on[/] to force Gemini.[/]",
    "[dim] 4. [#f0d890]/voice[/] to dictate into the prompt · [#f0d890]ctrl+v[/] toggles recording.[/]",
]

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.08


_HELP = [
    "[bold #e2e8f0]commands[/]",
    "[#94a3b8]  /help[/]                    this reference",
    "[#94a3b8]  /workers[/]                 list registered libp2p nodes",
    "[#94a3b8]  /parallel <N|auto>[/]       fan-out for the next job",
    "[#94a3b8]  /cloud on|off|toggle[/]     force Gemini on every sub-agent",
    "[#94a3b8]  /voice on|off|toggle[/]     enable dictation (ctrl+v to start/stop)",
    "[#94a3b8]  /voice devices[/]           list input devices (set DESERT_AUDIO_INPUT_DEVICE)",
    "[#94a3b8]  /clear[/]                   clear the session",
    "[#94a3b8]  /quit[/]                    exit",
    "",
    "[dim]Anything else is sent as a challenge. Enter submits; Esc or Ctrl+R focuses the prompt.[/]",
]


class OrchestratorApp(App[None]):
    """Single-column agentic CLI: banner → session → prompt → status."""

    ALLOW_SELECT = False

    BINDINGS = [
        Binding("ctrl+q", "quit", "quit", show=True),
        # Textual disables its default ctrl+c → quit when stdin is a real TTY
        # (so it doesn't shadow copy). Re-bind it explicitly and mark priority
        # so the focused Input doesn't swallow the key. SIGINT from a broken
        # terminal still exits via the runtime's own handler.
        Binding("ctrl+c", "quit", "quit", show=False, priority=True),
        Binding("ctrl+l", "clear_session", "clear", show=True),
        Binding("pageup", "scroll_session('page_up')", "scroll", show=True),
        Binding("pagedown", "scroll_session('page_down')", "scroll", show=False),
        Binding("ctrl+up", "scroll_session('up')", "scroll", show=False),
        Binding("ctrl+down", "scroll_session('down')", "scroll", show=False),
        Binding("ctrl+home", "scroll_session('home')", "scroll", show=False),
        Binding("ctrl+end", "scroll_session('end')", "scroll", show=False),
        Binding("escape", "focus_prompt", "prompt", show=False),
        Binding("ctrl+r", "focus_prompt", "prompt", show=False),
        # priority=True so the Input widget's built-in ctrl+v (paste) doesn't
        # swallow it while voice mode is on. When voice mode is off we fall
        # through to a no-op that also blocks paste, so we show a hint.
        Binding("ctrl+v", "toggle_record", "voice", show=False, priority=True),
    ]

    CSS = """
    $bg: #000000;
    $bg-soft: #000000;
    $bg-input: #0a0a0a;
    $border: #262626;
    $border-focus: #8ed189;
    $fg: #e2e8f0;
    $muted: #64748b;
    $accent: #f2c977;

    Screen { background: $bg; }

    #frame {
        width: 100%;
        height: 100%;
        padding: 1 2 1 2;
    }

    #banner {
        height: auto;
        margin-bottom: 1;
        content-align: left top;
    }

    #tips {
        height: auto;
        margin-bottom: 1;
        color: $muted;
    }

    #session {
        height: 1fr;
        min-height: 4;
        background: $bg;
        border: none;
        padding: 0;
        overflow-y: scroll;
        scrollbar-size-vertical: 1;
        scrollbar-background: $bg;
        scrollbar-background-hover: $bg;
        scrollbar-background-active: $bg;
        scrollbar-color: $border;
        scrollbar-color-hover: $border-focus;
        scrollbar-color-active: $border-focus;
    }

    #prompt-wrap {
        height: 3;
        margin: 1 0 0 0;
        padding: 0 1;
        background: $bg-input;
        border: round $border;
    }
    #prompt-wrap:focus-within {
        border: round $border-focus;
    }
    #prompt-caret {
        width: 3;
        color: $accent;
        content-align: left middle;
    }
    #prompt {
        width: 1fr;
        height: 1;
        border: none;
        background: $bg-input;
        padding: 0;
        color: $fg;
    }

    #voice-banner {
        height: 1;
        padding: 0 1;
        color: $muted;
        background: $bg;
        margin: 1 0 0 0;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $muted;
        background: $bg;
        margin: 1 0 0 0;
    }

    #hint {
        height: 1;
        padding: 0 1;
        color: $muted;
        background: $bg;
    }
    """

    def __init__(self, base_url: str = DEFAULT_BASE) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.public_endpoint = _public_endpoint()
        self._client = httpx.Client(timeout=httpx.Timeout(600.0))
        self._job_id: str | None = None
        self._poll_job = False
        self._force_cloud = False
        self._parallel: int | None = None
        self._workers_online = 0
        self._last_status = "idle"
        self._sub_snapshot: dict[str, str] = {}
        self._voice_mode = False
        self._voice_recording = False
        self._voice_engine: VoiceEngine | None = None
        # Ephemeral one-line status for recording / transcribing. Animated
        # via ``_voice_banner_timer`` which cycles through ``_SPINNER_FRAMES``.
        self._voice_banner_text: str = ""
        self._voice_banner_spin: bool = False
        self._voice_banner_frame: int = 0
        self._voice_banner_timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static(self._render_banner(), id="banner")
            yield Static("\n".join(_TIPS), id="tips", markup=True)
            yield RichLog(
                id="session",
                markup=True,
                highlight=False,
                wrap=True,
                auto_scroll=True,
            )
            yield Static("", id="voice-banner", markup=True)
            with Horizontal(id="prompt-wrap"):
                yield Static("❯", id="prompt-caret")
                yield Input(
                    placeholder="Describe a task · or /help for commands",
                    id="prompt",
                )
            yield Static("", id="status")
            # yield Static(
            #     "[dim]enter[/] submit  ·  [dim]ctrl+v[/] voice  ·  "
            #     "[dim]pgup/pgdn[/] scroll  ·  [dim]ctrl+l[/] clear  ·  "
            #     "[dim]ctrl+q[/] quit",
            #     id="hint",
            # )

    def on_mount(self) -> None:
        self.title = "Desert"
        self.sub_title = self.public_endpoint or ""
        self._refresh_status()
        self.set_interval(2.5, self._refresh_cluster_tick)
        self.set_interval(0.85, self._tick_job_poll)
        self.query_one("#voice-banner", Static).display = False
        self.query_one("#prompt", Input).focus()

    def on_key(self, event: events.Key) -> None:
        """Any printable keystroke (e.g. '/') goes to the prompt."""
        prompt = self.query_one("#prompt", Input)
        if self.focused is prompt:
            return
        ch = event.character
        if ch and ch.isprintable():
            event.stop()
            prompt.focus()
            prompt.insert_text_at_cursor(ch)

    # ── rendering helpers ─────────────────────────────────────────────────

    def _render_banner(self) -> Text:
        text = Text()
        for line, color in zip(_BANNER, _BANNER_COLORS):
            text.append(line + "\n", style=f"bold {color}")
        return text

    def _refresh_status(self) -> None:
        status = self.query_one("#status", Static)
        cloud = "[#8ed189]gemini on[/]" if self._force_cloud else "[dim]gemini off[/]"
        par = "auto" if self._parallel is None else str(self._parallel)
        dot = "[#7cc78a]●[/]" if self._workers_online > 0 else "[#f87171]●[/]"
        parts: list[str] = [dot]
        if self.public_endpoint:
            parts.append(f"[#e2e8f0]{self.public_endpoint}[/]")
        parts.extend(
            [
                f"[#e2e8f0]{self._workers_online}[/] nodes online",
                f"parallel [#e2e8f0]{par}[/]",
                cloud,
            ]
        )
        if self._voice_recording:
            parts.append("[#e88e5a]● recording[/]")
        elif self._voice_mode:
            parts.append("[#8ed189]voice on[/]")
        parts.append(self._last_status)
        sep = "   [dim]·[/]   "
        status.update(parts[0] + " " + sep.join(parts[1:]))

    def _log(self, line: str | Text = "") -> None:
        self.query_one("#session", RichLog).write(line)

    # ── ephemeral voice banner (recording / transcribing) ─────────────────

    def _set_voice_banner(self, text: str | None, *, spin: bool = False) -> None:
        """Show, animate, or clear the one-line voice status banner above the
        prompt.

        ``text=None`` hides the banner entirely (collapsing its row, so the
        prompt snaps back up). ``spin=True`` prefixes ``text`` with a braille
        spinner frame that ticks at ``_SPINNER_INTERVAL``; otherwise the
        prefix is a static orange ``●`` recording dot.
        """
        banner = self.query_one("#voice-banner", Static)
        if text is None:
            if self._voice_banner_timer is not None:
                self._voice_banner_timer.stop()
                self._voice_banner_timer = None
            self._voice_banner_text = ""
            self._voice_banner_spin = False
            banner.display = False
            banner.update("")
            return
        self._voice_banner_text = text
        self._voice_banner_spin = spin
        banner.display = True
        self._render_voice_banner()
        if spin and self._voice_banner_timer is None:
            self._voice_banner_timer = self.set_interval(
                _SPINNER_INTERVAL, self._tick_voice_banner
            )
        elif (not spin) and self._voice_banner_timer is not None:
            self._voice_banner_timer.stop()
            self._voice_banner_timer = None

    def _render_voice_banner(self) -> None:
        if not self._voice_banner_text:
            return
        if self._voice_banner_spin:
            frame = _SPINNER_FRAMES[self._voice_banner_frame % len(_SPINNER_FRAMES)]
            prefix = f"[#f2c977]{frame}[/]"
        else:
            prefix = "[#e88e5a]●[/]"
        self.query_one("#voice-banner", Static).update(
            f"{prefix} {self._voice_banner_text}"
        )

    def _tick_voice_banner(self) -> None:
        self._voice_banner_frame = (self._voice_banner_frame + 1) % len(_SPINNER_FRAMES)
        self._render_voice_banner()

    # ── periodic refreshes ────────────────────────────────────────────────

    def _refresh_cluster_tick(self) -> None:
        try:
            r = self._client.get(f"{self.base_url}/v1/workers", timeout=2.5)
            r.raise_for_status()
            data = r.json()
            self._workers_online = int(data.get("count") or 0)
        except Exception:
            self._workers_online = 0
        self._refresh_status()

    def _tick_job_poll(self) -> None:
        if not (self._poll_job and self._job_id):
            return
        jid = self._job_id
        try:
            r = self._client.get(f"{self.base_url}/v1/jobs/{jid}")
            r.raise_for_status()
            job = r.json()
        except Exception as e:
            self._log(f"[#f87171]poll error:[/] {e}")
            self._poll_job = False
            return

        subs = job.get("sub_results") or []
        for i, s in enumerate(subs, start=1):
            tid = s.get("task_id") or f"idx{i}"
            st = s.get("status") or "pending"
            prev = self._sub_snapshot.get(tid)
            if prev == st:
                continue
            self._sub_snapshot[tid] = st
            wid = (s.get("worker_id") or "—")[:12]
            if st == "assigned":
                self._log(
                    f"  [#f4c977]●[/] [dim]agent {i}[/] [#f0d890]{wid}[/] [dim]picked up task[/]"
                )
            elif st == "done":
                snippet = ((s.get("text") or "").strip()[:140]).replace("\n", " ")
                suffix = "…" if s.get("text") and len(s["text"]) > 140 else ""
                self._log(
                    f"  [#7cc78a]✓[/] [dim]agent {i}[/] [#f0d890]{wid}[/] [dim]→[/] {snippet}{suffix}"
                )
            elif st == "failed" or s.get("error"):
                self._log(
                    f"  [#f87171]✗[/] [dim]agent {i}[/] {s.get('error') or 'failed'}"
                )

        st = job.get("status")
        done = sum(1 for x in subs if x.get("status") == "done")
        self._last_status = (
            f"[#94a3b8]{st}[/] [dim]·[/] {done}/{len(subs)} map"
            if subs
            else f"[#94a3b8]{st}[/]"
        )
        self._refresh_status()

        if st == "done" and job.get("final_answer") is not None:
            self._log("")
            self._log("[bold #8ed189]── final answer ──[/]")
            self._log(f"[#e2e8f0]{job.get('final_answer') or ''}[/]")
            self._log("")
            self._poll_job = False
            self._job_id = None
            self._last_status = "done"
            self._refresh_status()
        elif st == "failed":
            self._log(f"[#f87171]job failed:[/] {job.get('error') or ''}")
            self._poll_job = False
            self._job_id = None
            self._last_status = "failed"
            self._refresh_status()

    # ── actions ───────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        if self._voice_engine is not None:
            try:
                self._voice_engine.close()
            except Exception:
                pass
        self.exit()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_clear_session(self) -> None:
        self.query_one("#session", RichLog).clear()

    def action_scroll_session(self, direction: str) -> None:
        log = self.query_one("#session", RichLog)
        log.auto_scroll = direction in ("end", "down", "page_down")
        {
            "up": log.scroll_up,
            "down": log.scroll_down,
            "page_up": log.scroll_page_up,
            "page_down": log.scroll_page_down,
            "home": log.scroll_home,
            "end": log.scroll_end,
        }[direction]()

    # ── input handling ────────────────────────────────────────────────────

    @on(Input.Submitted, "#prompt")
    def _on_submit(self, event: Input.Submitted) -> None:
        value = (event.value or "").strip()
        event.input.value = ""
        if not value:
            return
        if value.startswith("/"):
            self._handle_slash(value)
        else:
            self._submit_job(value)

    def _handle_slash(self, raw: str) -> None:
        parts = raw.strip().split()
        cmd, args = parts[0][1:].lower(), parts[1:]
        self._log(f"[#f0d890]❯[/] [dim]{raw}[/]")
        if cmd in ("help", "?"):
            for line in _HELP:
                self._log(line)
        elif cmd == "workers":
            self._cmd_workers()
        elif cmd == "parallel":
            self._cmd_parallel(args)
        elif cmd == "cloud":
            self._cmd_cloud(args)
        elif cmd == "voice":
            self._cmd_voice(args)
        elif cmd == "clear":
            self.action_clear_session()
        elif cmd in ("quit", "exit", "q"):
            self.action_quit()
        else:
            self._log(
                f"[#f87171]unknown command[/] [dim]{raw}[/] — try [#f0d890]/help[/]"
            )

    def _cmd_workers(self) -> None:
        try:
            r = self._client.get(f"{self.base_url}/v1/workers", timeout=3.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self._log(f"[#f87171]/workers failed:[/] {e}")
            return
        workers = data.get("workers") or []
        self._log(f"[#e2e8f0]{len(workers)}[/] nodes online")
        for w in workers:
            wid = w.get("worker_id") or (w.get("peer_id") or "?")[:12]
            short = wid if len(wid) <= 28 else wid[:12] + "…" + wid[-12:]
            pid = (w.get("peer_id") or "")[:12]
            ago = w.get("last_seen_sec_ago")
            self._log(
                f"  [#7cc78a]●[/] [#f0d890]{short}[/]  [dim]peer {pid}… · Δ{ago}s[/]"
            )

    def _cmd_parallel(self, args: list[str]) -> None:
        if not args:
            cur = "auto" if self._parallel is None else str(self._parallel)
            self._log(f"parallel = [#e2e8f0]{cur}[/]  [dim](use /parallel <N|auto>)[/]")
            return
        v = args[0].lower()
        if v in ("auto", "none", "null", ""):
            self._parallel = None
        else:
            try:
                n = int(v)
                if not 1 <= n <= 32:
                    raise ValueError
                self._parallel = n
            except ValueError:
                self._log("[#f87171]parallel must be 1–32 or [#f0d890]auto[/][/]")
                return
        self._log(
            f"[dim]parallel →[/] [#e2e8f0]{'auto' if self._parallel is None else self._parallel}[/]"
        )
        self._refresh_status()

    def _cmd_cloud(self, args: list[str]) -> None:
        if not args or args[0].lower() == "toggle":
            self._force_cloud = not self._force_cloud
        elif args[0].lower() in ("on", "1", "true", "yes"):
            self._force_cloud = True
        elif args[0].lower() in ("off", "0", "false", "no"):
            self._force_cloud = False
        else:
            self._log("[#f87171]usage:[/] /cloud on|off|toggle")
            return
        self._log(
            f"[dim]gemini →[/] [#e2e8f0]{'on' if self._force_cloud else 'off'}[/]"
        )
        self._refresh_status()

    # ── voice / dictation ─────────────────────────────────────────────────

    def _cmd_voice(self, args: list[str]) -> None:
        if args and args[0].lower() in ("devices", "device", "ls", "list"):
            self._cmd_voice_devices()
            return
        want: bool | None = None
        if not args or args[0].lower() == "toggle":
            want = not self._voice_mode
        elif args[0].lower() in ("on", "1", "true", "yes", "start"):
            want = True
        elif args[0].lower() in ("off", "0", "false", "no", "stop"):
            want = False
        else:
            self._log("[#f87171]usage:[/] /voice on|off|toggle | /voice devices")
            return
        self._set_voice_mode(want)

    def _cmd_voice_devices(self) -> None:
        try:
            devices = list_input_devices()
        except VoiceUnavailable as e:
            self._log(f"[#f87171]voice:[/] {e}")
            return
        if not devices:
            self._log("[#f87171]no input devices found[/]")
            return
        current = os.environ.get("DESERT_AUDIO_INPUT_DEVICE") or "system default"
        self._log(
            f"[dim]audio input devices[/] [dim]· current override:[/] [#e2e8f0]{current}[/]"
        )
        for idx, name, ch in devices:
            self._log(f"  [#f0d890]{idx:>2}[/]  [#e2e8f0]{name}[/]  [dim]({ch}ch)[/]")
        self._log(
            "[dim]pick one with[/] "
            "[#f0d890]DESERT_AUDIO_INPUT_DEVICE=<id|name>[/] "
            "[dim]in desert/.env, then restart[/]"
        )

    def _set_voice_mode(self, on: bool) -> None:
        if on and not self._voice_mode:
            if self._voice_engine is None:
                self._voice_engine = VoiceEngine()
            self._voice_mode = True
            self._log(
                "[dim]voice →[/] [#e2e8f0]on[/]  "
                f"[dim]· backend[/] [#e2e8f0]{self._voice_engine.backend_label}[/]  "
                "[dim]· press[/] [#f0d890]ctrl+v[/] [dim]to start/stop dictation[/]"
            )
            if not self._voice_engine.is_loaded:
                backend = self._voice_engine.backend
                if backend == "gemini":
                    self._log("[dim]warming up Gemini client…[/]")
                elif backend == "gemma4":
                    self._log("[dim]loading Gemma 4 weights (first time only)…[/]")
                else:
                    self._log("[dim]loading ASR model (first time only)…[/]")
                self.run_worker(self._voice_load_async(), exclusive=False)
        elif (not on) and self._voice_mode:
            self._voice_mode = False
            if self._voice_recording and self._voice_engine is not None:
                try:
                    self._voice_engine.stop_recording()
                except Exception:
                    pass
                self._voice_recording = False
            self._set_voice_banner(None)
            self._log("[dim]voice →[/] [#e2e8f0]off[/]")
        self._refresh_status()

    async def _voice_load_async(self) -> None:
        assert self._voice_engine is not None
        try:
            await asyncio.to_thread(self._voice_engine.load)
        except VoiceUnavailable as e:
            self._log(f"[#f87171]voice unavailable:[/] {e}")
            self._voice_mode = False
            self._refresh_status()
            return
        except Exception as e:
            self._log(f"[#f87171]voice load failed:[/] {e}")
            self._voice_mode = False
            self._refresh_status()
            return
        self._log("[#8ed189]voice ready[/]")

    def action_toggle_record(self) -> None:
        """Ctrl+V: start/stop dictation. No-op outside voice mode."""
        if not self._voice_mode or self._voice_engine is None:
            self._log(
                "[#94a3b8]voice mode is off — type [#f0d890]/voice[/] to enable dictation.[/]"
            )
            return
        if self._voice_recording:
            self._stop_recording_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        assert self._voice_engine is not None
        if not self._voice_engine.is_loaded:
            self._log("[#94a3b8]voice model still loading — try again in a moment.[/]")
            return
        try:
            self._voice_engine.start_recording()
        except VoiceUnavailable as e:
            self._log(f"[#f87171]voice:[/] {e}")
            return
        except Exception as e:
            self._log(f"[#f87171]voice start failed:[/] {e}")
            return
        self._voice_recording = True
        dev = self._voice_engine.last_device_name or "default"
        self._set_voice_banner(
            f"[dim]recording from[/] [#e2e8f0]{dev}[/] "
            f"[dim]· press[/] [#f0d890]ctrl+v[/] [dim]to stop[/]",
            spin=False,
        )
        self._refresh_status()

    def _stop_recording_and_transcribe(self) -> None:
        assert self._voice_engine is not None
        try:
            pcm = self._voice_engine.stop_recording()
        except Exception as e:
            self._voice_recording = False
            self._set_voice_banner(None)
            self._refresh_status()
            self._log(f"[#f87171]voice stop failed:[/] {e}")
            return
        self._voice_recording = False
        stats = audio_stats(pcm)
        # Capture-level diagnostics (duration / peak / rms) go to the debug log
        # only — they clutter the TUI for normal dictation. Silent-capture
        # hints are still surfaced inline in ``_transcribe_async`` on failure.
        log.debug(
            "voice: captured %.1fs peak_db=%.1f rms=%.0f",
            stats.duration_s,
            stats.peak_db if stats.peak_db != float("-inf") else -120.0,
            stats.rms_int16,
        )
        dev = self._voice_engine.last_device_name or "default"
        # Swap the static ● recording banner for an animated spinner on the
        # same row; it stays visible until transcription completes/fails.
        self._set_voice_banner(
            f"[dim]transcribing from[/] [#e2e8f0]{dev}[/]",
            spin=True,
        )
        self._refresh_status()
        self.run_worker(self._transcribe_async(pcm, stats), exclusive=True)

    async def _transcribe_async(self, pcm: bytes, stats: Any) -> None:
        assert self._voice_engine is not None
        try:
            text = await asyncio.to_thread(self._voice_engine.transcribe, pcm)
        except VoiceUnavailable as e:
            self._set_voice_banner(None)
            self._log(f"[#f87171]voice unavailable:[/] {e}")
            return
        except Exception as e:
            self._set_voice_banner(None)
            self._log(f"[#f87171]transcribe failed:[/] {e}")
            return
        self._set_voice_banner(None)
        if not text:
            if stats.looks_silent:
                dev = self._voice_engine.last_device_name or "default"
                self._log(
                    f"[dim]no speech detected — is[/] [#e2e8f0]{dev}[/] "
                    "[dim]the right mic? try[/] [#f0d890]/voice devices[/]"
                )
            else:
                self._log("[dim]no speech detected[/]")
            return
        prompt = self.query_one("#prompt", Input)
        existing = prompt.value
        new_value = (existing.rstrip() + " " + text).lstrip() if existing else text
        prompt.value = new_value
        prompt.cursor_position = len(new_value)
        prompt.focus()

    # ── job submission ────────────────────────────────────────────────────

    def _submit_job(self, challenge: str) -> None:
        try:
            r = self._client.post(
                f"{self.base_url}/v1/jobs",
                json={
                    "challenge": challenge,
                    "parallel": self._parallel,
                    "force_cloud_fallback": self._force_cloud,
                },
            )
            r.raise_for_status()
            jid = r.json()["job_id"]
        except Exception as e:
            self._log(f"[#f87171]job error:[/] {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    self._log(f"[dim]{e.response.text}[/]")
                except Exception:
                    pass
            return

        par_label = "auto" if self._parallel is None else str(self._parallel)
        cloud_label = "on" if self._force_cloud else "off"
        self._job_id = jid
        self._poll_job = True
        self._sub_snapshot = {}
        self._last_status = "running"
        self._log("")
        self._log(f"[bold #8ed189]❯[/] [#e2e8f0]{challenge}[/]")
        self._log(
            f"  [dim]job[/] [#f0d890]{jid}[/] [dim]· parallel[/] {par_label} [dim]· gemini[/] {cloud_label}"
        )
        self._refresh_status()


def run_tui(base_url: str = DEFAULT_BASE) -> None:
    OrchestratorApp(base_url=base_url).run()

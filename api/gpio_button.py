"""Push-to-talk GPIO button.

One press starts a turn: chime, mic icon on, record. A second press while
listening stops the recording early. The state machine (`ButtonFlow`) is
plain Python with no hardware dependency, so it also backs
`POST /voice/press` -- a way to exercise the whole chime/listen/cancel flow
from a dev machine with no button wired up.

`start()` wires a real `gpiozero.Button` on top of that when it can; on
anything that isn't a Pi with a button on the pin, it logs and leaves the
flow reachable only through `/voice/press`.
"""

import logging
import os
import threading
from typing import Callable, Optional

from dotenv import load_dotenv

from api import listen, state
from flow.agent import AgentRateLimited
from flow.schemas import MirrorAgentResponse
from flow.voice import play_chime

# Loaded independently of llm/lib/model.py's own load_dotenv() call: this
# module's env vars (below) are read at import time, which can happen before
# that one runs depending on import order, so .env.local overrides would
# otherwise be silently missed. load_dotenv() never overwrites an already-set
# environment variable, so calling it again here is harmless.
load_dotenv(".env.local")

log = logging.getLogger("api.gpio_button")

BUTTON_PIN = int(os.environ.get("MIRROR_BUTTON_PIN", "26"))
LISTEN_TIMEOUT = float(os.environ.get("MIRROR_LISTEN_TIMEOUT", "5.0"))
PHRASE_LIMIT = float(os.environ.get("MIRROR_PHRASE_LIMIT", "20.0"))

AgentTurn = Callable[[str], MirrorAgentResponse]
SpeakSafely = Callable[[str], None]


class ButtonFlow:
    """Owns the idle -> listening -> busy state machine for one button."""

    IDLE, LISTENING, BUSY = "idle", "listening", "busy"

    def __init__(self, agent_turn: AgentTurn, speak_safely: SpeakSafely):
        self._agent_turn = agent_turn
        self._speak_safely = speak_safely
        self._status = self.IDLE
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()

    def press(self) -> str:
        """Handle one button press. Returns "started", "stopped" (a
        recording was cut short), or "busy" (a turn is already thinking or
        speaking, so the press was ignored)."""
        with self._status_lock:
            if self._status == self.IDLE:
                # Claimed immediately so a fast double press can't start two
                # turns before the background thread has a chance to run.
                self._status = self.BUSY
            elif self._status == self.LISTENING:
                self._stop_event.set()
                return "stopped"
            else:
                return "busy"

        threading.Thread(target=self._run_turn, daemon=True).start()
        return "started"

    def _run_turn(self) -> None:
        try:
            try:
                play_chime()
            except Exception as exc:
                log.warning("chime playback failed: %s", exc)

            with self._status_lock:
                self._status = self.LISTENING
            self._stop_event.clear()
            state.emit("listening_start")

            # Recording can end well before the turn is anywhere near done --
            # Whisper still has to transcribe it, which on a cold model load
            # can take much longer than the recording itself. Told only
            # `on_capture_done` (fired the instant the mic itself stops,
            # rather than after transcription too), the mirror would
            # otherwise keep showing "listening" through that whole wait,
            # including after a second press was used to cut the recording
            # short.
            capture_ended = False

            def _on_capture_done() -> None:
                nonlocal capture_ended
                capture_ended = True
                state.emit("listening_stop")

            try:
                transcript = listen.listen_interruptible(
                    self._stop_event,
                    timeout=LISTEN_TIMEOUT,
                    phrase_limit=PHRASE_LIMIT,
                    on_capture_done=_on_capture_done,
                )
            except Exception as exc:
                log.warning("mic capture failed: %s", exc)
                transcript = ""
                if not capture_ended:
                    # Failed before ever reaching on_capture_done (e.g. the
                    # mic device failed to open) -- the mirror never heard
                    # about it otherwise.
                    state.emit("listening_stop")

            if not transcript:
                # Nobody said anything before the mic timed out -- tell the
                # mirror the cycle is over so it drops the loading spinner
                # instead of showing it forever with no reply ever coming.
                state.emit("turn_idle")
                return

            state.emit("user_message", {"text": transcript})

            with self._status_lock:
                self._status = self.BUSY
            # Emitted before the (potentially slow) agent call returns, so the
            # mirror can show a "thinking" placeholder immediately instead of
            # leaving the log looking stalled until the reply is ready.
            state.emit("assistant_thinking")
            try:
                result = self._agent_turn(transcript)
            except AgentRateLimited as exc:
                log.warning("agent turn rate limited: %s", exc)
                state.emit(
                    "assistant_message",
                    {
                        "text": "I'm getting too many requests right now -- give me a few seconds and try again.",
                        "display_card": None,
                    },
                )
                return
            except Exception as exc:
                log.warning("agent turn failed: %s", exc)
                state.emit(
                    "assistant_message",
                    {"text": "Sorry, I'm having trouble with that right now.", "display_card": None},
                )
                return

            state.emit(
                "assistant_message",
                {
                    "text": result.voice_response,
                    "display_card": result.display_card.model_dump() if result.display_card else None,
                },
            )
            if result.voice_response:
                self._speak_safely(result.voice_response)
        finally:
            with self._status_lock:
                self._status = self.IDLE


_flow: Optional[ButtonFlow] = None
_button = None  # gpiozero.Button, only set when real hardware is wired up


def start(agent_turn: AgentTurn, speak_safely: SpeakSafely) -> None:
    """Build the button flow and, if this looks like a Pi, arm a real GPIO
    button on top of it. Safe to call on a dev machine with no GPIO."""
    global _flow, _button

    _flow = ButtonFlow(agent_turn, speak_safely)

    try:
        from gpiozero import Button
    except Exception as exc:
        log.info("gpiozero not available (%s); use POST /voice/press to test the flow", exc)
        return

    try:
        _button = Button(BUTTON_PIN, bounce_time=0.05)
        _button.when_pressed = _flow.press
    except Exception as exc:
        log.warning(
            "could not arm GPIO button on pin %s (%s); use POST /voice/press to test the flow",
            BUTTON_PIN,
            exc,
        )
        _button = None
        return

    log.info("GPIO button armed on pin %s", BUTTON_PIN)


def press() -> Optional[str]:
    """Simulate a physical button press. Returns None if the flow was never
    started (i.e. `start()` was not called yet), otherwise one of
    "started" / "stopped" / "busy" -- see `ButtonFlow.press`."""
    if _flow is None:
        return None
    return _flow.press()


def is_armed() -> bool:
    """Whether a real GPIO button is wired up, as opposed to `/voice/press`-only."""
    return _button is not None

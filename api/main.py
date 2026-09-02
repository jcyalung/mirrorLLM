"""HTTP front end for the mirror agent.

Run from the repo root (`npm run api`):
    python -m uvicorn api.main:app --port 8000

Avoid --reload unless you are editing this file: a reload builds a new agent and
drops the conversation history.

Endpoints are defined with `def` rather than `async def` on purpose: the agent
does blocking network and tool work, so FastAPI hands each request to a worker
thread instead of stalling the event loop.
"""

import logging
import threading
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import gpio_button, listen, state
from flow.agent import AgentRateLimited, MirrorAgent
from flow.schemas import MirrorAgentResponse
from flow.tools.calendar import CalendarNotAuthorized, list_upcoming_events
from flow.tools.spotify import (
    SpotifyNotAuthorized,
    SpotifyNotConfigured,
    get_now_playing,
)
from flow.voice import speak
from llm.lib.model import MODEL_NAME

log = logging.getLogger("api")
# Nothing in this codebase calls logging.basicConfig, so "api" has no handler
# of its own -- INFO records would otherwise vanish silently (only WARNING+
# reaches Python's stderr fallback handler). voice_events (below) needs INFO
# to actually show, so give this logger its own handler explicitly.
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    log.addHandler(handler)


_NOISY_POLL_PATHS = (
    # Polled by the chat module every `pollInterval` ms; voice_events below
    # logs in its place, but only when a poll actually turns up a new event.
    "/voice/events",
    # Polled by the spotify module every `fetchInterval` ms (default 1s);
    # there's no "new" now-playing state worth calling out the way a voice
    # event is, so this one is just dropped outright.
    "/spotify/now-playing",
)


class _NoisyPollFilter(logging.Filter):
    """Drops uvicorn's access-log line for the front end's steady polling
    endpoints -- left alone they fire every second or faster regardless of
    whether anything happened, drowning out everything else in the log."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in _NOISY_POLL_PATHS)


logging.getLogger("uvicorn.access").addFilter(_NoisyPollFilter())

app = FastAPI(title="mirrorLLM agent API", version="1.0.0")

# The mirror front end is served from port 8080, a different origin, so browser
# calls need CORS even though everything is on the same machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One long-lived agent so multi-turn memory survives across requests. Keeping
# this process alive independently of MagicMirror is the whole point of the
# service: the mirror restarts constantly in development, this does not.
_agent = MirrorAgent()
_agent_lock = threading.Lock()

# Edge-TTS playback goes through a single pygame mixer, so only one line can be
# spoken at a time.
_speech_lock = threading.Lock()


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, description="What the user said or typed.")
    speak: bool = Field(default=True, description="Read the reply aloud via TTS.")


class ListenRequest(BaseModel):
    speak: bool = True
    timeout: float = Field(default=8.0, description="Seconds to wait for speech to start.")
    phrase_limit: float = Field(default=15.0, description="Maximum seconds of speech.")
    device_index: Optional[int] = Field(
        default=None,
        description="Microphone to record from; omit to use the G321 headset.",
    )


class TurnResponse(BaseModel):
    transcript: Optional[str] = Field(
        default=None, description="Recognised speech, when the turn came from the mic."
    )
    response: MirrorAgentResponse


def _speak_safely(text: str) -> None:
    """Play `text`, swallowing audio failures.

    Runs as a background task after the response is sent, so a machine with no
    working output device still gets a usable API.
    """
    with _speech_lock:
        try:
            speak(text)
        except Exception as exc:
            log.warning("TTS playback failed: %s", exc)


def _agent_turn(prompt: str) -> MirrorAgentResponse:
    """Run one turn through the shared agent. Shared with the GPIO button flow
    (api/gpio_button.py) so both paths go through the same lock and history."""
    with _agent_lock:
        return _agent.send(prompt)


def _run_turn(prompt: str, say: bool, background: BackgroundTasks) -> MirrorAgentResponse:
    try:
        result = _agent_turn(prompt)
    except AgentRateLimited as exc:
        raise HTTPException(
            status_code=429, detail=f"too many requests, try again shortly: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"agent failed: {exc}") from exc

    # Speak after responding, so the mirror can render the card immediately
    # instead of waiting out the audio.
    if say and result.voice_response:
        background.add_task(_speak_safely, result.voice_response)

    return result


# The GPIO button flow (and its dev-machine stand-in, POST /voice/press)
# reuses the same agent turn and TTS lock as the HTTP endpoints above.
gpio_button.start(agent_turn=_agent_turn, speak_safely=_speak_safely)


@app.get("/calendar/events")
def calendar_events():
    """Upcoming Google Calendar events for today and the rest of this week."""
    try:
        return {"events": list_upcoming_events()}
    except CalendarNotAuthorized as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"calendar failed: {exc}") from exc


@app.get("/spotify/now-playing")
def spotify_now_playing():
    """What the user is listening to right now, if anything."""
    try:
        return get_now_playing()
    except (SpotifyNotConfigured, SpotifyNotAuthorized) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"spotify failed: {exc}") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "turns": sum(1 for m in _agent.messages if m.get("role") == "user"),
        "stt_loaded": listen.is_loaded(),
        "button_armed": gpio_button.is_armed(),
    }


@app.post("/agent/ask", response_model=TurnResponse)
def agent_ask(req: AskRequest, background: BackgroundTasks):
    """Run one agent turn from text. This is the endpoint the mirror calls."""
    return TurnResponse(response=_run_turn(req.prompt, req.speak, background))


@app.post("/agent/reset")
def agent_reset():
    """Start a fresh conversation, dropping all history."""
    global _agent
    with _agent_lock:
        _agent = MirrorAgent()
    return {"status": "reset"}


@app.get("/voice/devices")
def voice_devices():
    """List real microphones, so `device_index` can be pinned to the right one."""
    return {"devices": listen.list_devices()}


@app.get("/voice/events")
def voice_events(since: int = 0):
    """Events from the GPIO button flow, for the mirror module to poll.

    Types: listening_start, listening_stop, user_message ({text}),
    assistant_thinking, assistant_message ({text, display_card}), turn_idle,
    calendar_updated ({html_link}). Pass the highest `id` seen so far as
    `since` to get only what's new.
    """
    events = state.since(since)
    if events:
        log.info("voice/events: %d new event(s) since id %d", len(events), since)
    return {"events": events}


@app.post("/voice/press")
def voice_press():
    """Simulate a physical button press.

    Runs the exact same chime/listen/cancel state machine as the real GPIO
    button (api/gpio_button.py), so this is how the flow gets exercised on a
    dev machine with no button wired up. A press while a turn is already
    listening stops that recording early, same as a real second press.
    """
    outcome = gpio_button.press()
    if outcome is None:
        raise HTTPException(status_code=503, detail="button flow not initialized")
    if outcome == "busy":
        raise HTTPException(status_code=409, detail="already thinking or speaking")
    return {"status": outcome}


@app.post("/voice/turn", response_model=TurnResponse)
def voice_turn(req: ListenRequest, background: BackgroundTasks):
    """Record one utterance, transcribe it, and run it through the agent.

    A fixed-timeout, non-interruptible capture for scripted/manual use. The
    GPIO button (api/gpio_button.py) uses `listen.listen_interruptible`
    instead, so a second press can cut the recording short.
    """
    try:
        transcript = listen.transcribe_once(
            timeout=req.timeout,
            phrase_limit=req.phrase_limit,
            device_index=req.device_index,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"capture failed: {exc}") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="no speech detected")

    return TurnResponse(
        transcript=transcript,
        response=_run_turn(transcript, req.speak, background),
    )

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

from api import listen
from flow.agent import MirrorAgent
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


def _run_turn(prompt: str, say: bool, background: BackgroundTasks) -> MirrorAgentResponse:
    with _agent_lock:
        try:
            result = _agent.send(prompt)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"agent failed: {exc}") from exc

    # Speak after responding, so the mirror can render the card immediately
    # instead of waiting out the audio.
    if say and result.voice_response:
        background.add_task(_speak_safely, result.voice_response)

    return result


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


@app.post("/voice/turn", response_model=TurnResponse)
def voice_turn(req: ListenRequest, background: BackgroundTasks):
    """Record one utterance, transcribe it, and run it through the agent.

    This is what the GPIO button will call.
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

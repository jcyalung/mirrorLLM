import json
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

from openai import RateLimitError
from pydantic import ValidationError

from llm.lib.model import client, MODEL_NAME
from flow.schemas import MirrorAgentResponse
from flow.tools import ACTION_TOOLS, EMIT_TOOL, run_tool

MAX_TOOL_ROUNDS = 4

# Matches flow/tools/calendar.py's default -- the model needs to reason about
# "today"/"tomorrow"/"next Friday" in the same zone create_calendar_event
# stamps events in, or a relative date it resolves can land on the wrong day.
MIRROR_TIME_ZONE = "America/Los_Angeles"


class AgentRateLimited(RuntimeError):
    """Raised when the upstream LLM API rejects a request as rate-limited
    (HTTP 429) -- i.e. too many requests in too short a window."""


def _current_context() -> str:
    """A fresh "what day/time is it" line, injected into the system prompt on
    every turn (not baked in once at startup, since one agent stays alive for
    days). Without this the model has no way to know the real date and
    silently guesses one whenever asked for anything relative ("tomorrow",
    "in an hour", "next Friday") -- so create_calendar_event gets called with
    a plausible but wrong start_iso/end_iso: the event is created, just not on
    any day the user is actually looking."""
    now = datetime.now(ZoneInfo(MIRROR_TIME_ZONE))
    return (
        f"Current date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')} "
        f"({MIRROR_TIME_ZONE}). Resolve relative dates and times ('tomorrow', "
        "'next Friday', 'in an hour') against this, and always include the "
        "correct year in start_iso/end_iso."
    )


SYSTEM_PROMPT = (
    "You are the voice assistant for a smart mirror. Keep spoken replies short, "
    "casual, and conversational -- like a friend talking to you at the bathroom "
    "mirror, not a customer service bot. "
    "The mirror screen only shows glanceable, high-contrast cards, never long "
    "text, so keep display_card items short. "
    "Use search_web for recipes, current facts, or anything that should come "
    "from the internet. If asked for recipes, search and offer up to 3 options "
    "with titles and URLs. "
    "Use get_wikipedia for encyclopedia-style summaries. "
    "Use create_calendar_event to schedule things on the user's Google Calendar. "
    "For music, use list_spotify_playlists to see the user's library, "
    "play_spotify_song to play a specific track, and shuffle_spotify_playlist "
    "to shuffle one of their playlists. Playback starts immediately, so only "
    "call these when the user actually asks for music. "
    "Use notify_discord or send_email only when the user explicitly asks you to "
    "send, text, or notify them (or someone) -- never do it silently or as a "
    "side effect. "
    "Base factual answers on tool results, not memory. "
    "When you're ready to answer, call emit_response exactly once with a short "
    "voice_response and, if there's something worth glancing at (a recipe, an "
    "event, search results, a confirmation), a display_card. Skip display_card "
    "for plain chit-chat."
)


def _assistant_to_message(assistant_message) -> dict:
    # NIM rejects assistant messages with empty (or whitespace-only, once
    # trimmed) content -- a plain " " placeholder passes on the round it's
    # produced but gets rejected the next time it's replayed as history.
    payload = {"role": "assistant", "content": assistant_message.content or "..."}
    if assistant_message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ]
    return payload


def _parse_emit(arguments: dict) -> MirrorAgentResponse:
    try:
        return MirrorAgentResponse.model_validate(arguments)
    except ValidationError:
        voice = arguments.get("voice_response") or (
            "Sorry, I'm having trouble with that right now."
        )
        return MirrorAgentResponse(voice_response=str(voice))


class MirrorAgent:
    """Multi-turn chat agent with tool calling and structured mirror output.

    Each `send()` call runs a full tool-use round trip and returns a
    `MirrorAgentResponse` -- a short voice line plus an optional display card,
    ready to hand off to TTS and the MagicMirror module.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self._base_prompt = system_prompt
        self.messages: List[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, user_text: str) -> MirrorAgentResponse:
        # Rewritten every turn, not just set once at construction, since one
        # agent instance stays alive (and keeps taking turns) across days.
        self.messages[0]["content"] = f"{self._base_prompt}\n\n{_current_context()}"
        self.messages.append({"role": "user", "content": user_text})

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            force_final = round_index == MAX_TOOL_ROUNDS
            tool_choice = (
                {"type": "function", "function": {"name": "emit_response"}}
                if force_final
                else "auto"
            )

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=self.messages,
                    tools=ACTION_TOOLS + [EMIT_TOOL],
                    tool_choice=tool_choice,
                )
            except RateLimitError as exc:
                raise AgentRateLimited(str(exc)) from exc
            assistant_message = response.choices[0].message
            self.messages.append(_assistant_to_message(assistant_message))

            if not assistant_message.tool_calls:
                # Model answered in plain text instead of calling emit_response.
                text = (assistant_message.content or "").strip()
                return MirrorAgentResponse(
                    voice_response=text or "Sorry, I didn't catch that."
                )

            emitted = None
            for tool_call in assistant_message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if name == "emit_response":
                    emitted = _parse_emit(args)
                    result = "ok"
                else:
                    result = run_tool(name, args)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            if emitted is not None:
                return emitted

        return MirrorAgentResponse(
            voice_response="Sorry, I'm having trouble with that right now."
        )

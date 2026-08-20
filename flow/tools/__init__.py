import json

from llm.lib.web import TOOLS as _WEB_TOOLS, run_tool as _run_web_tool

from flow.tools.calendar import CALENDAR_TOOL, create_calendar_event
from flow.tools.notify import (
    NOTIFY_DISCORD_TOOL,
    SEND_EMAIL_TOOL,
    notify_discord,
    send_email,
)

# Tools the model can call while it's still working the problem.
ACTION_TOOLS = [*_WEB_TOOLS, CALENDAR_TOOL, NOTIFY_DISCORD_TOOL, SEND_EMAIL_TOOL]

# The model must call this exactly once, last, to close out a turn. Forcing
# every reply through one schema-shaped tool call is far more reliable than
# asking for free-form JSON in `content`.
EMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_response",
        "description": (
            "Finish the turn. Call this last, exactly once, after any other "
            "tools you needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "voice_response": {
                    "type": "string",
                    "description": "Short, casual sentence(s) to read aloud via TTS.",
                },
                "display_card": {
                    "type": "object",
                    "description": (
                        "Glanceable card for the mirror screen. Omit this field "
                        "entirely for plain chit-chat that doesn't need one."
                    ),
                    "properties": {
                        "card_type": {
                            "type": "string",
                            "enum": [
                                "recipe",
                                "calendar",
                                "weather",
                                "notification",
                                "wikipedia",
                                "web_results",
                            ],
                        },
                        "title": {
                            "type": "string",
                            "description": "High-contrast headline.",
                        },
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Bullet points, ingredients, or itinerary entries.",
                        },
                        "footer_note": {
                            "type": "string",
                            "description": "Small subtext/status, e.g. '📅 Added to your calendar'.",
                        },
                    },
                    "required": ["card_type", "title"],
                },
            },
            "required": ["voice_response"],
        },
    },
}


def run_tool(name: str, arguments: dict) -> str:
    try:
        if name in {"search_web", "get_wikipedia"}:
            return _run_web_tool(name, arguments)
        if name == "create_calendar_event":
            link = create_calendar_event(**arguments)
            return json.dumps({"status": "created", "html_link": link})
        if name == "notify_discord":
            return json.dumps(notify_discord(arguments["message"]))
        if name == "send_email":
            return json.dumps(
                send_email(
                    arguments["subject"], arguments["body"], arguments.get("recipient")
                )
            )
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

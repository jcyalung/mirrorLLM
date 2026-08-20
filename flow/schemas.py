from typing import List, Literal, Optional

from pydantic import BaseModel, Field

CardType = Literal[
    "recipe", "calendar", "weather", "notification", "wikipedia", "web_results"
]


class DisplayCard(BaseModel):
    """Glanceable widget rendered on the mirror glass. Keep it short -- this is
    read at a glance, not read in full."""

    card_type: CardType = Field(description="Category of card shown on the mirror.")
    title: str = Field(description="High-contrast main headline for the mirror glass.")
    items: List[str] = Field(
        default_factory=list,
        description="Bullet points, ingredients, or itinerary entries.",
    )
    footer_note: Optional[str] = Field(
        default=None,
        description="Small subtext/status, e.g. '📅 Added to your calendar'.",
    )


class MirrorAgentResponse(BaseModel):
    """Final, structured turn output: what gets spoken vs. what gets shown."""

    voice_response: str = Field(
        description="Conversational sentence read aloud via TTS. Short and casual."
    )
    display_card: Optional[DisplayCard] = Field(
        default=None,
        description="Structured widget for the mirror screen. Omit for plain chit-chat.",
    )

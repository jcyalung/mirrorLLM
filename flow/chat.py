"""Interactive CLI for the mirror agent flow.

Run from the repo root:
    python -m flow.chat
    python -m flow.chat --mute   # skip TTS playback, text only
"""

import sys

from flow.agent import MirrorAgent
from flow.voice import speak
from llm.lib.model import MODEL_NAME


def _print_card(card):
    if not card:
        return
    print(f"  ┌─ [{card.card_type}] {card.title}")
    for item in card.items:
        print(f"  │  • {item}")
    print(f"  └─ {card.footer_note}" if card.footer_note else "  └─")


def main():
    muted = "--mute" in sys.argv[1:]
    agent = MirrorAgent()
    print(f"Mirror chat with {MODEL_NAME}. Type quit or exit to stop.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text.lower() in {"quit", "exit"}:
            break

        result = agent.send(user_text)
        print(f"Mirror (voice): {result.voice_response}")
        _print_card(result.display_card)

        if not muted:
            try:
                speak(result.voice_response)
            except Exception as exc:
                print(f"  (voice playback failed, continuing text-only: {exc})")
                muted = True

        print()


if __name__ == "__main__":
    main()

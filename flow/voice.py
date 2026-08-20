"""Text-to-speech output for the mirror flow.

Deliberately independent of speech/tts.py's Whisper (STT) setup -- chat.py
only needs to speak replies, not transcribe audio, so there's no reason to
load a speech-recognition model just to play a voice line.
"""

import asyncio
import io
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import edge_tts
import pygame

VOICE = "en-US-GuyNeural"
RATE = "+15%"


async def _generate_and_play(text: str, voice: str, rate: str) -> None:
    communicate = edge_tts.Communicate(text, voice, rate=rate)

    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    audio_stream.seek(0)

    if not pygame.mixer.get_init():
        pygame.mixer.init()

    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def speak(text: str, voice: str = VOICE, rate: str = RATE) -> None:
    """Synthesize `text` with Edge-TTS and play it through the default output device.

    `rate` is an Edge-TTS relative-speed string, e.g. "+20%" or "-10%".
    """
    if not text:
        return
    asyncio.run(_generate_and_play(text, voice, rate))

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
import numpy as np
import pygame

VOICE = "en-US-GuyNeural"
RATE = "+15%"

# Two-note chime played before the mic starts listening -- synthesized on the
# fly so the button flow doesn't depend on shipping/finding a sound asset.
CHIME_NOTES = (880.0, 1318.5)  # A5 then E6
CHIME_NOTE_DURATION = 0.09
CHIME_GAP = 0.02


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


def _tone(freq: float, duration: float, sample_rate: int, volume: float) -> np.ndarray:
    n = max(1, int(sample_rate * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)

    # Fade the edges a few milliseconds in/out so the tone doesn't click.
    fade = min(n // 2, int(sample_rate * 0.01))
    if fade > 0:
        envelope = np.ones(n)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        wave *= envelope

    return wave * volume


def play_chime(volume: float = 0.4) -> None:
    """Play a short two-note beep, blocking until it finishes.

    Called right before the mic starts listening, so the button flow can
    simply play the chime and then start recording once this returns.
    """
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    sample_rate = pygame.mixer.get_init()[0]

    gap = np.zeros(int(sample_rate * CHIME_GAP))
    notes = [_tone(freq, CHIME_NOTE_DURATION, sample_rate, volume) for freq in CHIME_NOTES]
    tones = np.concatenate([notes[0], gap, notes[1]])
    samples = np.clip(tones * 32767, -32768, 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([samples, samples]))

    sound = pygame.sndarray.make_sound(stereo)
    channel = sound.play()
    while channel is not None and channel.get_busy():
        pygame.time.Clock().tick(60)

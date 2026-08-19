import asyncio
import io
import queue
import tempfile
import numpy as np
import sounddevice as sd
import edge_tts
import pygame
import whisper

# Audio recording configuration
SAMPLE_RATE = 16000
CHANNELS = 1

# Select an Edge TTS Neural Voice (e.g., 'en-US-GuyNeural', 'en-US-AriaNeural', 'en-US-ChristopherNeural')
VOICE = "en-US-GuyNeural"

# 1. Initialize Whisper STT & Pygame Audio Mixer
print("Loading Whisper STT model...")
whisper_model = whisper.load_model("base")

pygame.mixer.init()


async def generate_and_play_tts(text: str, voice: str = VOICE):
    """Synthesizes text using edge-tts and plays it directly."""
    print(f"\n[Speaking]: \"{text}\"")
    communicate = edge_tts.Communicate(text, voice)

    # Accumulate audio data in memory
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])

    audio_stream.seek(0)

    # Play via pygame mixer
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def speak(text: str):
    """Synchronous wrapper to run the async edge-tts player."""
    asyncio.run(generate_and_play_tts(text))


def record_and_transcribe():
    """Captures microphone audio via Enter key toggle and transcribes it."""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    input("\n[Press ENTER to START speaking]")
    print(" Recording... Speak your prompt.")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    with stream:
        input("[Press ENTER to STOP recording]")

    print(" Transcribing speech...")

    # Combine recorded chunks
    audio_chunks = []
    while not q.empty():
        audio_chunks.append(q.get())

    if not audio_chunks:
        return ""

    audio_data = np.concatenate(audio_chunks, axis=0).flatten()
    result = whisper_model.transcribe(audio_data, fp16=False)
    return result["text"].strip()


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    speak("Voice system initialized. Press enter whenever you're ready to speak.")

    while True:
        transcription = record_and_transcribe()

        if transcription:
            print(f"\n[Transcribed Input]: \"{transcription}\"")

            # 1. Read back what you said
            speak(f"You said: {transcription}")

            # 2. (Optional) Chain to your Agent loop:
            # agent_reply = run_agent(transcription)
            # speak(agent_reply)
        else:
            print("No speech detected.")

        cont = input("\nRecord another? (y/n): ").strip().lower()
        if cont != "y":
            speak("Goodbye!")
            break
import io
import json
import speech_recognition as sr
import whisper

# 1. Load Local Speech-to-Text Model (tiny, base, small, or medium)
print("Loading Whisper ASR model...")
whisper_model = whisper.load_model("base")

def listen_and_transcribe():
    """Captures microphone input using Voice Activity Detection (VAD)

    and returns the transcribed text.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone(sample_rate=16000) as source:
        print("\nAdjusting for ambient noise... (1 sec)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Listening... (Speak your request)")
        # Automatically detects when you start and stop talking
        audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        print("Processing audio...")

    # Write WAV buffer directly to memory (avoiding temp files on disk)
    wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    wav_buffer = io.BytesIO(wav_bytes)

    # Whisper can transcribe directly from a file-like or byte stream
    # or you can use speech_recognition's built-in whisper wrapper:
    result = recognizer.recognize_whisper(audio_data, model="base")
    return result.strip()

# ---------------------------------------------------------
# 2. Integrate with the Agentic Loop
# ---------------------------------------------------------
if __name__ == "__main__":
    # Prompt the user via voice
    user_spoken_prompt = listen_and_transcribe()
    print(f"\n[You Said]: \"{user_spoken_prompt}\"")

    if user_spoken_prompt:
        # Pass the transcribed voice input into the agent
        # (run_agent function from the previous step)
        # run_agent(user_spoken_prompt)
        print("Dispatching to LLM agent...")
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import tempfile
import os

SAMPLE_RATE = 16000
CHANNELS = 1

# Load Whisper model (local ASR)
print("Loading Whisper model...")
model = whisper.load_model("base")

def record_toggle():
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """Audio stream callback pushing raw chunks to the queue."""
        if status:
            print(status)
        q.put(indata.copy())

    input("\n[Press ENTER to START recording]")
    print(" Recording... Speak now.")

    # Start audio recording stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback
    )

    with stream:
        input("[Press ENTER to STOP recording]\n")

    print(" Stopped recording. Processing audio...")

    # Drain queue into single continuous numpy array
    audio_chunks = []
    while not q.empty():
        audio_chunks.append(q.get())

    if not audio_chunks:
        print("No audio captured.")
        return ""

    audio_data = np.concatenate(audio_chunks, axis=0)

    # Whisper expects normalized 16kHz float32 audio
    # Directly transcribe numpy array
    audio_flat = audio_data.flatten()
    result = model.transcribe(audio_flat, fp16=False)
    
    return result["text"].strip()

if __name__ == "__main__":
    transcription = record_toggle()
    print(f"\n[Transcribed]: \"{transcription}\"")
    # Feed into your agent: run_agent(transcription)
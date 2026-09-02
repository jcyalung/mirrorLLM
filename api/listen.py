"""Push-to-talk speech capture for the API.

Whisper is loaded lazily on first use. Importing this module has to stay cheap
so the service can boot -- and serve text-only requests -- on a machine with no
microphone and without paying the model load up front.
"""

import os
import threading
from typing import Callable, List, Optional

from dotenv import load_dotenv

# Loaded independently of llm/lib/model.py's own load_dotenv() call, same
# reasoning as api/gpio_button.py: PREFERRED_MIC (below) is read at import
# time, which can happen before that one runs.
load_dotenv(".env.local")

# Recording owns the microphone, so serialise it. Two overlapping captures
# would fight over the input device and both come back garbled.
_mic_lock = threading.Lock()
_recognizer = None
_loaded = False

# Windows lists the same headset through several host APIs plus Steam/Voicemod
# aliases, hence a name match rather than just taking the default device --
# and on the Pi that turns out to matter too: ALSA's "default" pseudo-device
# (also flagged is_default) is a 32-channel virtual mixer that PortAudio
# negotiates inconsistently for capture, so falling back to is_default there
# picks the flaky path instead of the real hardware one. Override via
# MIRROR_MIC_NAME in .env.local for whatever's actually plugged in.
PREFERRED_MIC = os.environ.get("MIRROR_MIC_NAME", "G321")

_JUNK_MARKERS = (
    "microsoft sound mapper",
    "primary sound capture driver",
    "steam streaming",
    "voicemod",
    "@system32",
    "bthhfenum",
)

# MME is what Windows already marks as default for this headset, and it is
# the path speech_recognition has actually succeeded on.
_HOST_API_RANK = {
    "MME": 0,
    "Windows WASAPI": 1,
    "Windows DirectSound": 2,
    "Windows WDM-KS": 3,
}


def is_loaded() -> bool:
    """Whether the Whisper model has been loaded into memory yet."""
    return _loaded


def _is_junk(name: str) -> bool:
    lowered = name.lower().strip()
    if not lowered or lowered in ("input ()", "input"):
        return True
    return any(marker in lowered for marker in _JUNK_MARKERS)


def _pick_preferred(group: List[dict]) -> dict:
    """One physical device is listed once per host API; keep the best copy."""
    default = next((device for device in group if device["is_default"]), None)
    if default is not None:
        return default
    return min(group, key=lambda d: _HOST_API_RANK.get(d["host_api"], 99))


def list_devices() -> List[dict]:
    """Real microphones, one entry each, with the index to pass as `device_index`."""
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        default_index = None
        try:
            default_index = audio.get_default_input_device_info()["index"]
        except Exception:
            # Plenty of Windows machines have no default input configured.
            pass

        raw = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info["maxInputChannels"] < 1:
                continue
            name = info["name"]
            if _is_junk(name):
                continue
            raw.append(
                {
                    "index": index,
                    "name": name,
                    "channels": int(info["maxInputChannels"]),
                    "sample_rate": int(info["defaultSampleRate"]),
                    "host_api": audio.get_host_api_info_by_index(info["hostApi"])["name"],
                    "is_default": index == default_index,
                    "preferred": PREFERRED_MIC.lower() in name.lower(),
                }
            )

        by_name: dict[str, List[dict]] = {}
        for device in raw:
            by_name.setdefault(device["name"], []).append(device)

        devices = []
        for group in by_name.values():
            chosen = _pick_preferred(group)
            chosen["preferred"] = chosen["preferred"] or any(d["preferred"] for d in group)
            chosen["is_default"] = chosen["is_default"] or any(d["is_default"] for d in group)
            devices.append(chosen)

        devices.sort(key=lambda d: (not d["preferred"], not d["is_default"], d["index"]))
        return devices
    finally:
        audio.terminate()


def resolve_device_index(device_index: Optional[int] = None) -> Optional[int]:
    """Pin to the G321 when the caller does not name a device."""
    if device_index is not None:
        return device_index

    devices = list_devices()
    for device in devices:
        if device["preferred"]:
            return device["index"]
    for device in devices:
        if device["is_default"]:
            return device["index"]
    return None


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        _recognizer = recognizer
    return _recognizer


def _preflight(device_index: Optional[int], sample_rate: int) -> None:
    """Open the device and close it again, purely to surface a real error.

    `Microphone.__enter__` catches open() failures without re-raising and hands
    back a source whose stream is None; the failure then resurfaces from
    `__exit__` as `'NoneType' object has no attribute 'close'`, which says
    nothing about the actual cause. Failing here instead keeps the message
    actionable.
    """
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(
            input_device_index=device_index,
            channels=1,
            format=pyaudio.paInt16,
            rate=sample_rate,
            frames_per_buffer=1024,
            input=True,
        )
        stream.close()
    except Exception as exc:
        raise RuntimeError(
            f"could not open microphone (device_index={device_index}, "
            f"{sample_rate} Hz): {exc}. Call GET /voice/devices to list inputs."
        ) from exc
    finally:
        audio.terminate()


def transcribe_once(
    timeout: float = 8.0,
    phrase_limit: float = 15.0,
    device_index: Optional[int] = None,
    model: str = "base",
) -> str:
    """Record one utterance from the microphone and return its transcript.

    Uses voice-activity detection, so the caller only has to signal the start --
    recording stops on its own once speech ends. That maps directly onto a
    single GPIO button press.

    Returns an empty string if nobody spoke before `timeout` seconds.
    """
    global _loaded

    import speech_recognition as sr

    recognizer = _get_recognizer()
    device_index = resolve_device_index(device_index)

    # sample_rate is left unset so each device opens at its native rate.
    # Forcing 16 kHz is rejected by many inputs, and recognize_whisper
    # resamples to what Whisper wants anyway.
    try:
        microphone = sr.Microphone(device_index=device_index)
    except Exception as exc:
        raise RuntimeError(
            f"no usable microphone (device_index={device_index}): {exc}. "
            "Call GET /voice/devices to list inputs."
        ) from exc

    _preflight(device_index, microphone.SAMPLE_RATE)

    with _mic_lock:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
            except sr.WaitTimeoutError:
                return ""

        # recognize_whisper caches the model on the Recognizer, so only the
        # first call pays the load cost.
        text = recognizer.recognize_whisper(audio, model=model)
        _loaded = True

    return text.strip()


def _rms(data: bytes) -> float:
    import numpy as np

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def listen_interruptible(
    stop_event: threading.Event,
    timeout: float = 5.0,
    silence_duration: float = 1.0,
    phrase_limit: float = 20.0,
    device_index: Optional[int] = None,
    model: str = "base",
    on_capture_done: Optional[Callable[[], None]] = None,
) -> str:
    """Record from the mic until the speaker goes quiet, `timeout` seconds pass
    with nobody talking, or `stop_event` is set -- whichever happens first.

    Unlike `transcribe_once`, this reads the stream in small (~50ms) chunks
    and checks `stop_event` between reads, so a second button press can cut a
    recording short almost immediately instead of waiting for
    `speech_recognition`'s blocking `listen()` to return on its own. Whatever
    was captured before the interruption is still transcribed. Returns "" if
    nothing was said before `timeout`, or if `stop_event` fires before anyone
    started talking.

    `on_capture_done`, if given, fires the moment the mic itself stops
    recording -- before the (potentially much slower, especially on a cold
    Whisper model load) transcription step below runs. Without this, a caller
    that only finds out once the whole function returns has no way to tell
    the difference between "still recording" and "done recording, still
    transcribing", so a UI driven off that alone would keep showing the
    "recording" state through the transcription wait too.
    """
    global _loaded

    import pyaudio
    import speech_recognition as sr

    device_index = resolve_device_index(device_index)
    audio = pyaudio.PyAudio()

    try:
        info = (
            audio.get_device_info_by_index(device_index)
            if device_index is not None
            else audio.get_default_input_device_info()
        )
    except Exception as exc:
        audio.terminate()
        raise RuntimeError(
            f"no usable microphone (device_index={device_index}): {exc}. "
            "Call GET /voice/devices to list inputs."
        ) from exc

    rate = int(info["defaultSampleRate"])
    chunk = max(1, rate // 20)  # ~50ms per read, so a stop press lands quickly

    with _mic_lock:
        try:
            stream = audio.open(
                input_device_index=device_index,
                channels=1,
                format=pyaudio.paInt16,
                rate=rate,
                frames_per_buffer=chunk,
                input=True,
            )
        except Exception as exc:
            audio.terminate()
            raise RuntimeError(
                f"could not open microphone (device_index={device_index}, {rate} Hz): {exc}. "
                "Call GET /voice/devices to list inputs."
            ) from exc

        frames: List[bytes] = []
        timed_out = False
        try:
            # A short ambient sample sets the bar for "someone is talking",
            # the same idea as `recognizer.adjust_for_ambient_noise`.
            ambient = [
                _rms(stream.read(chunk, exception_on_overflow=False)) for _ in range(4)
            ]
            threshold = max(500.0, (sum(ambient) / len(ambient)) * 3)

            speaking = False
            silence_chunks = 0
            silence_needed = max(1, int(silence_duration * rate / chunk))
            timeout_chunks = max(1, int(timeout * rate / chunk))
            max_chunks = max(1, int(phrase_limit * rate / chunk))
            waited_chunks = 0

            for _ in range(max_chunks):
                if stop_event.is_set():
                    break

                data = stream.read(chunk, exception_on_overflow=False)
                loud = _rms(data) > threshold

                if not speaking:
                    if loud:
                        speaking = True
                        frames.append(data)
                    else:
                        waited_chunks += 1
                        if waited_chunks >= timeout_chunks:
                            timed_out = True  # nobody spoke within `timeout`
                            break
                else:
                    frames.append(data)
                    silence_chunks = 0 if loud else silence_chunks + 1
                    if silence_chunks >= silence_needed:
                        break
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

        # The mic has physically stopped recording at this point, regardless
        # of which branch above got us here -- say so now, not after
        # transcription (below) has also had its turn.
        if on_capture_done is not None:
            on_capture_done()

        if timed_out or not frames:
            return ""

        recognizer = _get_recognizer()
        audio_data = sr.AudioData(b"".join(frames), rate, 2)
        text = recognizer.recognize_whisper(audio_data, model=model)
        _loaded = True

    return text.strip()

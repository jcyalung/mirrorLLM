"""Push-to-talk speech capture for the API.

Whisper is loaded lazily on first use. Importing this module has to stay cheap
so the service can boot -- and serve text-only requests -- on a machine with no
microphone and without paying the model load up front.
"""

import threading
from typing import List, Optional

# Recording owns the microphone, so serialise it. Two overlapping captures
# would fight over the input device and both come back garbled.
_mic_lock = threading.Lock()
_recognizer = None
_loaded = False

# Windows lists the same headset through several host APIs plus Steam/Voicemod
# aliases. Pin the G321 for now; the Pi will only have one input, so this
# preference becomes a no-op there.
PREFERRED_MIC = "G321"

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

"""Text-to-speech (voice output) with a male voice.

Order of engines, the first that works is used:
  1. Edge TTS (Microsoft neural voices, free, no key, needs internet):
     Albanian  sq-AL-IlirNeural       (male)
     Macedonian mk-MK-AleksandarNeural (male)
     These are the only male neural voices that exist for the two languages.
  2. gTTS (Google Translate voice): Albanian only, and the voice is female.
  3. The browser's own speech engine (Web Speech API), only if the computer has a voice
     for the language. The app shows this as a button.
"""
from __future__ import annotations

import asyncio
import io
import json
import threading

EDGE_VOICES = {
    "sq": {"male": "sq-AL-IlirNeural", "female": "sq-AL-AnilaNeural"},
    "mk": {"male": "mk-MK-AleksandarNeural", "female": "mk-MK-MarijaNeural"},
}
GTTS_LANG = {"sq": "sq"}


def edge_voice(lang: str, gender: str = "male") -> str | None:
    return EDGE_VOICES.get(lang, {}).get(gender)


async def _edge_async(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    buf = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return bytes(buf)


def _run(coro):
    """Run a coroutine also when the caller already has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    out: dict = {}

    def target():
        try:
            out["v"] = asyncio.run(coro)
        except Exception as e:  # noqa: BLE001
            out["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in out:
        raise out["e"]
    return out["v"]


def edge_mp3(text: str, lang: str, gender: str = "male", rate: str = "-5%") -> bytes | None:
    voice = edge_voice(lang, gender)
    if not voice or not text.strip():
        return None
    try:
        audio = _run(_edge_async(text, voice, rate))
        return audio or None
    except Exception:  # noqa: BLE001 - no internet, package missing or service changed
        return None


def gtts_mp3(text: str, lang: str) -> bytes | None:
    if lang not in GTTS_LANG or not text.strip():
        return None
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=GTTS_LANG[lang]).write_to_fp(buf)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def synthesize(text: str, lang: str, gender: str = "male") -> tuple[bytes | None, str | None]:
    """(mp3 bytes, engine) from the first engine that works, or (None, None).

    engine is the Edge voice name (e.g. "sq-AL-IlirNeural") or "gtts".
    """
    audio = edge_mp3(text, lang, gender)
    if audio:
        return audio, edge_voice(lang, gender)
    audio = gtts_mp3(text, lang)
    if audio:
        return audio, "gtts"
    return None, None


def mp3(text: str, lang: str) -> bytes | None:
    """MP3 bytes (male voice when possible), or None. Kept for older callers."""
    return synthesize(text, lang)[0]


def browser_speech_html(text: str, lang: str, label: str) -> str:
    """A small button that reads the text with the browser's speech engine (prefers a male voice)."""
    code = {"sq": "sq-AL", "mk": "mk-MK"}[lang]
    return f"""
<button style="padding:6px 12px;border-radius:6px;border:1px solid #888;cursor:pointer"
 onclick='const u=new SpeechSynthesisUtterance({json.dumps(text)});u.lang="{code}";
 const v=speechSynthesis.getVoices().filter(x=>x.lang.startsWith("{code[:2]}"));
 const m=v.find(x=>/male|ilir|aleksandar/i.test(x.name)&&!/female/i.test(x.name));
 if(m||v[0])u.voice=m||v[0];speechSynthesis.cancel();speechSynthesis.speak(u);'>🔊 {label}</button>"""

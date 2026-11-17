"""Text-to-speech for the repair steps.

Albanian: gTTS (Google Text-to-Speech, needs internet) returns an MP3 that the app plays.
Macedonian: gTTS has no Macedonian voice, so the app uses the browser's own speech
engine (Web Speech API, lang mk-MK); it works only if the computer has a Macedonian voice.
"""
from __future__ import annotations

import io
import json

GTTS_LANG = {"sq": "sq"}


def mp3(text: str, lang: str) -> bytes | None:
    """MP3 bytes, or None when this language has no gTTS voice or there is no internet."""
    if lang not in GTTS_LANG:
        return None
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=GTTS_LANG[lang]).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def browser_speech_html(text: str, lang: str, label: str) -> str:
    """A small button that reads the text with the browser's speech engine."""
    code = {"sq": "sq-AL", "mk": "mk-MK"}[lang]
    return f"""
<button style="padding:6px 12px;border-radius:6px;border:1px solid #888;cursor:pointer"
 onclick='const u=new SpeechSynthesisUtterance({json.dumps(text)});u.lang="{code}";
 speechSynthesis.cancel();speechSynthesis.speak(u);'>🔊 {label}</button>"""

"""Voice input: speech-to-text in Albanian / Macedonian, and matching a spoken answer to a button.

Speech-to-text engines (the first that works is used):
  1. OpenAI Whisper API (gpt-4o-transcribe, then whisper-1), only if an OpenAI API key is given.
     Best quality for Albanian.
  2. Google Web Speech (free, no key, needs internet) through the SpeechRecognition package,
     language sq-AL / mk-MK. This is the default, so the app works without any extra key.

st.audio_input gives a WAV file, which both engines accept directly (no ffmpeg needed).
"""
from __future__ import annotations

import io
import re

import auto_answer

GOOGLE_LANG = {"sq": "sq-AL", "mk": "mk-MK"}
WHISPER_MODELS = ("gpt-4o-transcribe", "whisper-1")
OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"


class SpeechError(Exception):
    """No engine could turn the recording into text."""


def whisper(wav: bytes, lang: str, api_key: str) -> str:
    import httpx  # installed together with the anthropic package

    last = None
    for model in WHISPER_MODELS:
        r = httpx.post(OPENAI_URL, headers={"Authorization": f"Bearer {api_key}"},
                       data={"model": model, "language": lang, "response_format": "json"},
                       files={"file": ("voice.wav", wav, "audio/wav")}, timeout=60)
        if r.status_code == 200:
            return r.json().get("text", "").strip()
        last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
    raise SpeechError(last)


def google_free(wav: bytes, lang: str) -> str:
    import speech_recognition as sr

    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav)) as src:
        audio = rec.record(src)
    try:
        return rec.recognize_google(audio, language=GOOGLE_LANG[lang]).strip()
    except sr.UnknownValueError:
        return ""


def transcribe(wav: bytes, lang: str, openai_key: str | None = None) -> tuple[str, str]:
    """(text, engine). Raises SpeechError when every engine fails. Empty text = nothing understood."""
    errors = []
    if openai_key:
        try:
            return whisper(wav, lang, openai_key), "whisper"
        except Exception as e:  # noqa: BLE001 - fall back to the free engine
            errors.append(f"Whisper: {e}")
    try:
        return google_free(wav, lang), "google"
    except Exception as e:  # noqa: BLE001
        errors.append(f"Google: {e}")
    raise SpeechError("; ".join(errors))


# ------------------------------------------------------------------ spoken answer -> button
YES = {"po", "patjetër", "sigurisht", "posi", "ja", "да", "секако", "точно"}
NO = {"jo", "nuk", "s'ka", "aspak", "не", "нема", "никако"}
ORDINALS = {  # exact numbers, or word stems ("parë", "para", "pari" all start with "par")
    "sq": [("1", "një", "par"), ("2", "dy", "dyt"), ("3", "tre", "tri", "tret"), ("4", "katër")],
    "mk": [("1", "еден", "една", "прв"), ("2", "два", "две", "втор"), ("3", "три", "трет"), ("4", "четири", "четвр")],
}


def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", (text or "").lower())


def yes_no(text: str) -> int | None:
    """0 for yes, 1 for no, None if unclear. 'No' wins over 'yes' ("po, jo..." is unclear -> None)."""
    w = set(_words(text))
    y, n = bool(w & YES), bool(w & NO)
    if y and not n:
        return 0
    if n and not y:
        return 1
    return None


def ordinal(text: str, n_options: int, lang: str) -> int | None:
    w = _words(text)
    if len(w) > 4:  # a long sentence is a description, not "the second one"
        return None
    def said(form):
        return any(x == form or (len(form) >= 3 and not form.isdigit() and x.startswith(form)) for x in w)
    hits = {i for i, forms in enumerate(ORDINALS.get(lang, [])[:n_options]) if any(said(f) for f in forms)}
    return hits.pop() if len(hits) == 1 else None


def match_option(text: str, options: list[str], lang: str, yes_no_question: bool = False) -> int | None:
    """Index of the option the user said, or None when it is not clear (the app then asks again).

    1. yes/no questions: "po"/"jo" ("да"/"не").
    2. "the first / second / 2": ordinals.
    3. word overlap with the option text (same stemming as auto_answer), unique best only.
    """
    if not text.strip():
        return None
    if yes_no_question:
        return yes_no(text)
    i = ordinal(text, len(options), lang)
    if i is not None:
        return i
    answers = [{"text": {"sq": o, "mk": o}} for o in options]
    return auto_answer.choose_answer(text, answers, lang, min_overlap=1)

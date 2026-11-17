"""LLM + vision part of the pipeline (Claude API).

Two jobs, both optional: the app works without an API key and falls back to
TF-IDF on the text and retrieval-only manual search.

1. analyze_photo: a vision model looks at the user's photo and returns structured
   JSON: which object it is, brand/model if visible, text read from labels (OCR),
   visible damage, and whether it involves electricity/gas/water.
2. answer_from_manual: RAG generation step. The answer must use only the retrieved
   manual passages and cite them as [1], [2], ...

Set the key with the environment variable ANTHROPIC_API_KEY (or in the app sidebar).
"""
from __future__ import annotations

import base64
import json
import os

MODEL = "claude-opus-5-5"
OBJECTS = ["door_handle", "cabinet_door", "chair_table", "drawer", "wall_shelf", "other"]
LANG_NAME = {"sq": "Albanian", "mk": "Macedonian"}

PHOTO_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "enum": OBJECTS},
        "confidence": {"type": "number"},
        "brand": {"type": "string"},
        "model": {"type": "string"},
        "ocr_text": {"type": "string"},
        "damage": {"type": "string"},
        "broken_parts": {"type": "boolean"},
        "involves_electricity_gas_or_water": {"type": "boolean"},
    },
    "required": ["object", "confidence", "brand", "model", "ocr_text", "damage", "broken_parts",
                 "involves_electricity_gas_or_water"],
    "additionalProperties": False,
}


def available(api_key: str | None = None) -> bool:
    if not (api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _client(api_key: str | None = None):
    import anthropic
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _create(client, **kwargs):
    """messages.create with server-side refusal fallback; returns the response."""
    return client.beta.messages.create(
        model=MODEL,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **kwargs,
    )


def _text(response) -> str:
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    return next((b.text for b in response.content if b.type == "text"), "")


def analyze_photo(image_bytes: bytes, media_type: str, description: str, lang: str,
                  api_key: str | None = None) -> dict:
    prompt = (
        "You are the vision module of a home-repair assistant. Look at the photo of a broken household object.\n"
        f"Allowed objects: {', '.join(OBJECTS)} (chair_table = chair or table; use 'other' if none fits).\n"
        "Return: the object; your confidence 0-1; brand and model if a label, sticker or logo shows them "
        "(empty string if not visible, never guess); ocr_text = every piece of text you can read in the photo, "
        "exactly as written; damage = what looks broken, one short sentence in "
        f"{LANG_NAME[lang]}; broken_parts = true only if a part is visibly broken, cracked or missing; "
        "involves_electricity_gas_or_water = true if the repair touches wiring, sockets, "
        "gas or water pipes.\n"
        f"The user's own description: {description or '(none)'}"
    )
    response = _create(
        _client(api_key),
        max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": PHOTO_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.standard_b64encode(image_bytes).decode()}},
            {"type": "text", "text": prompt},
        ]}],
    )
    data = json.loads(_text(response))
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    return data


def answer_from_manual(question: str, passages: list[dict], lang: str, api_key: str | None = None) -> str:
    """RAG generation: answer only from the retrieved passages, with [n] citations."""
    context = "\n\n".join(f"[{i}] ({p['source']})\n{p['text']}" for i, p in enumerate(passages, 1))
    prompt = (
        f"Answer the user's repair question in {LANG_NAME[lang]}, in at most 8 short numbered steps.\n"
        "Use ONLY the manual passages below and cite them like [1]. If they do not answer the question, "
        "say so. If the repair involves electricity or gas, tell the user to switch it off and call a "
        "professional instead of giving steps.\n\n"
        f"<passages>\n{context}\n</passages>\n\nQuestion: {question}"
    )
    response = _create(
        _client(api_key),
        max_tokens=4000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )
    return _text(response)


def manual_guidance(manual: list[dict], problem: str, repair_title: str, lang: str,
                    api_key: str | None = None) -> str:
    """Read the user's own manual (PDF, photo of a page, or pasted text, in ANY language) and
    return the parts relevant to this repair, translated into the user's language.

    manual: list of {"kind": "pdf" | "image" | "text", "data": bytes | str, "media_type": str, "name": str}
    """
    content = []
    for m in manual:
        if m["kind"] == "pdf":
            content.append({"type": "document", "title": m["name"], "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.standard_b64encode(m["data"]).decode()}})
        elif m["kind"] == "image":
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": m["media_type"],
                "data": base64.standard_b64encode(m["data"]).decode()}})
        else:
            content.append({"type": "text", "text": f"<manual name=\"{m['name']}\">\n{m['data']}\n</manual>"})
    content.append({"type": "text", "text": (
        "Above is the manufacturer's manual of the user's furniture (PDF, photo of a page, or text). "
        "It may be written in ANY language.\n"
        f"The user's problem: {problem}\nThe repair we planned: {repair_title}\n\n"
        f"Write in {LANG_NAME[lang]}, translating from the manual's language:\n"
        "1. Model name / article number if the manual shows it.\n"
        "2. The manual's steps and part codes that matter for this repair, as short numbered steps.\n"
        "3. The manual's safety warnings for this repair.\n"
        "For each point, say where it is in the manual (page, section or step number). "
        "Use only what the manual says. If the manual does not cover this problem, say so in one sentence."
    )})
    response = _create(
        _client(api_key),
        max_tokens=4000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
    )
    return _text(response)

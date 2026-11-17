"""Answer clarifying questions automatically when the text or the photo already answers them.

The decision tree stays the same; before showing a question we try to infer its answer:
- safety question: answered "No" only when the vision model looked at the photo and saw no
  electricity, gas or water (the text safety gate has already run before this point)
- broken parts: "Yes" if the text or the photo analysis mentions broken/cracked/missing parts,
  "No" if the vision model says nothing is broken; otherwise ask
- object questions: word overlap (same stems as TF-IDF) between the user's text and each answer;
  an answer is chosen only if it shares at least 2 words and clearly beats the others
- tools: always asked, only the user knows what they have

Every inferred answer is shown to the user with its source, so it can be checked.
"""
from __future__ import annotations

from text_recognizer import tokens

BROKEN = ["thyer", "thyhe", "carë", "care", "çarë", "mungo", "këput", "keput", "скрш", "напук", "недос", "искрш", "пукна"]
STOP = {stem[:5] for stem in ["nuk", "është", "eshte", "dhe", "por", "ose", "nga", "ka", "të", "te", "një", "sirtarit",
                              "не", "се", "од", "или", "но", "на", "и"]}


def _stems(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOP}


def mentions_broken(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in BROKEN)


def safety(vision: dict | None):
    """Return 'no' when the photo analysis rules out electricity/gas/water, else None (ask)."""
    if vision and vision.get("involves_electricity_gas_or_water") is False:
        return "no"
    return None


def broken(text: str, vision: dict | None):
    if mentions_broken(text) or (vision and mentions_broken(vision.get("damage", ""))):
        return "yes"
    if vision and vision.get("broken_parts") is False:
        return "no"
    return None


def choose_answer(text: str, answers: list[dict], lang: str, min_overlap: int = 2):
    """Index of the answer whose words the user already used, or None."""
    user = _stems(text)
    overlaps = [len(user & (_stems(a["text"]["sq"]) | _stems(a["text"]["mk"]))) for a in answers]
    best = max(range(len(answers)), key=lambda i: overlaps[i])
    others = [o for i, o in enumerate(overlaps) if i != best]
    if overlaps[best] >= min_overlap and all(overlaps[best] > o for o in others):
        return best
    return None


def question_nodes(tree: dict) -> list[dict]:
    """All question nodes of a decision tree (depth-first)."""
    if "q" not in tree:
        return []
    out = [tree]
    for a in tree["answers"]:
        out += question_nodes(a["next"])
    return out

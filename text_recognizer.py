"""Text recognizer: which object is the user describing? (TF-IDF + cosine similarity)

Written from scratch (no scikit-learn) so every step can be explained:
1. normalize: lowercase, ë->e, ç->c (users often type without these letters)
2. tokenize into words, drop very short words
3. stem: keep the first 5 letters (dollapi, dollapit -> dolla; фиоката, фиока -> фиока)
4. TF-IDF: a word is important if it is frequent in one object's texts and rare in others
5. cosine similarity between the user's text and each object's vector -> best object

Works for Albanian and Macedonian at the same time.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

KB = Path(__file__).parent / "knowledge"
STEM = 5
NORMALIZE = str.maketrans({"ë": "e", "ç": "c"})


def tokens(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower().translate(NORMALIZE))
    return [w[:STEM] for w in words if len(w) > 2 and not w.isdigit()]


class TfidfRecognizer:
    def __init__(self, docs: dict[str, list[str]], min_score: float = 0.05):
        """docs: object id -> list of training texts."""
        self.min_score = min_score
        tf = {obj: Counter(t for text in texts for t in tokens(text)) for obj, texts in docs.items()}
        n = len(tf)
        df = Counter(t for counts in tf.values() for t in counts)
        # smoothed idf: a word in every object still gets a small weight
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = {obj: self._vector(counts) for obj, counts in tf.items()}

    def _vector(self, counts: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, text: str) -> dict[str, float]:
        q = self._vector(Counter(tokens(text)))
        return {obj: sum(w * vec.get(t, 0.0) for t, w in q.items()) for obj, vec in self.vectors.items()}

    def predict(self, text: str):
        """Return (object id or None, scores)."""
        s = self.scores(text)
        best = max(s, key=s.get)
        return (best if s[best] >= self.min_score else None), s


def training_docs(exclude: int | None = None) -> dict[str, list[str]]:
    diag = json.loads((KB / "diagnosis.json").read_text(encoding="utf-8"))
    examples = json.loads((KB / "examples.json").read_text(encoding="utf-8"))["examples"]
    docs = {oid: [" ".join(o["keywords"]), o["name"]["sq"], o["name"]["mk"]] for oid, o in diag["objects"].items()}
    for i, ex in enumerate(examples):
        if i != exclude:
            docs[ex["object"]].append(ex["text"])
    return docs


def build() -> TfidfRecognizer:
    return TfidfRecognizer(training_docs())


def keyword_baseline(text: str):
    """Baseline: count keyword prefixes per object (no learning)."""
    diag = json.loads((KB / "diagnosis.json").read_text(encoding="utf-8"))
    words = re.findall(r"\w+", text.lower())
    s = {oid: sum(1 for w in words for k in o["keywords"] if w.startswith(k)) for oid, o in diag["objects"].items()}
    best = max(s, key=s.get)
    return (best if s[best] > 0 else None), s


def baseline_accuracy() -> float:
    examples = json.loads((KB / "examples.json").read_text(encoding="utf-8"))["examples"]
    return sum(keyword_baseline(e["text"])[0] == e["object"] for e in examples) / len(examples)


def leave_one_out() -> tuple[float, list[tuple[str, str, str | None]]]:
    """Accuracy when each example is predicted by a model trained without it."""
    examples = json.loads((KB / "examples.json").read_text(encoding="utf-8"))["examples"]
    errors = []
    for i, ex in enumerate(examples):
        pred, _ = TfidfRecognizer(training_docs(exclude=i)).predict(ex["text"])
        if pred != ex["object"]:
            errors.append((ex["text"], ex["object"], pred))
    return 1 - len(errors) / len(examples), errors


if __name__ == "__main__":
    print(f"Keyword baseline accuracy: {baseline_accuracy():.0%}")
    acc, errors = leave_one_out()
    print(f"Leave-one-out accuracy: {acc:.0%}")
    for text, true, pred in errors:
        print(f"  ✗ '{text}': duhej {true}, doli {pred}")

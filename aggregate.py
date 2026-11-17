"""Aggregation: combine the evidence from the photo and from the text into one decision.

Each source gives a score per object. Scores are first normalized into a probability-like
distribution (sum = 1), then combined with one of three methods:

- weighted average:  P(o) = w_photo * P_photo(o) + w_text * P_text(o)
- product (naive Bayes style, sources assumed independent):  P(o) ∝ P_photo(o) * P_text(o)
- majority vote: each source votes for its best object; ties go to the more confident source

A source that is missing (no photo, no API key, empty text) is simply left out.
"""
from __future__ import annotations

EPS = 1e-3


def normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        return {k: 1 / len(scores) for k in scores}
    return {k: max(v, 0.0) / total for k, v in scores.items()}


def photo_distribution(objects: list[str], predicted: str, confidence: float) -> dict[str, float]:
    """The vision model gives one label + confidence; spread the rest evenly."""
    rest = (1 - confidence) / max(len(objects) - 1, 1)
    return {o: (confidence if o == predicted else rest) for o in objects}


def weighted_average(sources: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    objects = next(iter(sources.values())).keys()
    total_w = sum(weights[s] for s in sources)
    return {o: sum(weights[s] * sources[s][o] for s in sources) / total_w for o in objects}


def product(sources: dict[str, dict[str, float]]) -> dict[str, float]:
    objects = next(iter(sources.values())).keys()
    raw = {o: 1.0 for o in objects}
    for dist in sources.values():
        for o in objects:
            raw[o] *= dist[o] + EPS
    return normalize(raw)


def majority_vote(sources: dict[str, dict[str, float]]) -> str:
    votes, best_conf = {}, {}
    for dist in sources.values():
        o = max(dist, key=dist.get)
        votes[o] = votes.get(o, 0) + 1
        best_conf[o] = max(best_conf.get(o, 0), dist[o])
    return max(votes, key=lambda o: (votes[o], best_conf[o]))


def combine(objects: list[str], text_scores: dict[str, float] | None, photo: tuple[str, float] | None,
            method: str = "weighted", w_photo: float = 0.6, w_text: float = 0.4):
    """Return (best object or None, combined distribution, sources used)."""
    sources = {}
    if text_scores and max(text_scores.values()) > 0:
        sources["text"] = normalize(text_scores)
    if photo and photo[0] in objects:
        sources["photo"] = photo_distribution(objects, photo[0], photo[1])
    if not sources:
        return None, {}, []
    if method == "product":
        dist = product(sources)
    elif method == "vote":
        best = majority_vote(sources)
        return best, weighted_average(sources, {"photo": w_photo, "text": w_text}), list(sources)
    else:
        dist = weighted_average(sources, {"photo": w_photo, "text": w_text})
    return max(dist, key=dist.get), dist, list(sources)

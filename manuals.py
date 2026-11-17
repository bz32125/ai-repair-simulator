"""Manual search (the retrieval part of RAG).

The library has two kinds of documents:
1. built-in guides, generated from scenarios.json and the safety tips (sq + mk)
2. manuals the user uploads as PDF (e.g. the IKEA/manufacturer manual of their chair)

Every document is cut into chunks (passages). A question is matched against all chunks
with TF-IDF + cosine similarity, and the top k passages are returned. With an API key,
llm.answer_from_manual() then writes an answer that cites those passages (generation).
Without a key the app shows the passages themselves.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from urllib.parse import quote_plus

from text_recognizer import TfidfRecognizer

ROOT = Path(__file__).parent
LIBRARY = ROOT / "manuals"
CHUNK_CHARS = 900


def builtin_chunks() -> list[dict]:
    scen = json.loads((ROOT / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    tips = json.loads((ROOT / "knowledge" / "tips.json").read_text(encoding="utf-8"))["tips"]
    chunks = []
    for s in scen:
        for lang in ("sq", "mk"):
            steps = "\n".join(f"- {a[f'label_{lang}']}" for a in s["actions"] if not a.get("safety"))
            warn = "\n".join(f"! {a[f'safety_msg_{lang}']}" for a in s["actions"] if a.get(f"safety_msg_{lang}"))
            chunks.append({"source": f"Udhëzues: {s[f'title_{lang}']}", "lang": lang,
                           "text": f"{s[f'title_{lang}']}\n{s[f'description_{lang}']}\n{steps}\n{warn}"})
    for t in tips:
        for lang in ("sq", "mk"):
            chunks.append({"source": f"Këshillë: {t['title'][lang]}", "lang": lang,
                           "text": f"{t['title'][lang]}\n{t['body'][lang]}"})
    return chunks


def pdf_chunks(file_bytes: bytes, name: str) -> list[dict]:
    from pypdf import PdfReader
    chunks = []
    for page_no, page in enumerate(PdfReader(io.BytesIO(file_bytes)).pages, 1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        for start in range(0, len(text), CHUNK_CHARS):
            piece = text[start:start + CHUNK_CHARS]
            if len(piece) > 40:
                chunks.append({"source": f"{name}, faqja {page_no}", "lang": "any", "text": piece})
    return chunks


def _models_line(text: str) -> list[str]:
    m = re.search(r"models:\s*(.+?)(?:-->|$)", text, re.IGNORECASE | re.MULTILINE)
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []


def library_chunks(folder: Path = LIBRARY) -> list[dict]:
    """Manuals saved in the manuals/ folder (.md, .txt split by sections; .pdf by pages)."""
    chunks = []
    if not folder.exists():
        return chunks
    for path in sorted(folder.iterdir()):
        if path.name.lower() == "readme.md":
            continue
        if path.suffix.lower() == ".pdf":
            new = pdf_chunks(path.read_bytes(), path.name)
            models = []
        elif path.suffix.lower() in (".md", ".txt"):
            text = path.read_text(encoding="utf-8")
            models = _models_line(text)
            body = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
            title = body.splitlines()[0].lstrip("# ").strip() if body else path.stem
            parts = [p.strip() for p in re.split(r"\n(?=#+ )|\n\s*\n", body) if len(p.strip()) > 40]
            new = [{"source": f"{path.name}, {p.splitlines()[0].lstrip('# ').strip()[:60]}", "lang": "any",
                    "text": f"{title}\n{p}"} for p in parts]
        else:
            continue
        for c in new:
            c["file"], c["models"] = path.name, models
        chunks += new
    return chunks


def match_manuals(chunks: list[dict], *texts: str) -> list[str]:
    """Manual files whose brand/model/article number appears in the OCR text, brand or model."""
    seen = " ".join(t for t in texts if t).lower()
    files = []
    for c in chunks:
        ids = c.get("models", [])[1:]  # first entry is the brand: a brand alone is too general
        if c.get("file") and c["file"] not in files and any(m.lower() in seen for m in ids):
            files.append(c["file"])
    return files


class ManualIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.model = TfidfRecognizer({str(i): [c["text"]] for i, c in enumerate(chunks)}, min_score=0.0)

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        scores = self.model.scores(query)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [dict(self.chunks[int(i)], score=scores[i]) for i in best if scores[i] >= min_score]


def manufacturer_search_url(brand: str, model: str) -> str | None:
    """A web search link for the official manual, when the photo showed a brand or model."""
    q = " ".join(x for x in (brand, model) if x).strip()
    return f"https://www.google.com/search?q={quote_plus(q + ' manual pdf')}" if q else None

# AI Repair Assistant
Introduction to AI (ECS-609, SEEU) — individual project, Besnik Zeqiri

A bilingual (**Albanian / Macedonian**) assistant that helps ordinary people fix simple broken
things at home (furniture, doors, cabinets, drawers, shelves) that would usually need a handyman.
The user describes the problem (and optionally uploads a photo); the assistant recognizes the
object, asks short clarifying questions, lists the tools and gives safe, ordered repair steps.

Inspired by iFixit FixBot (see [ROADMAP.md](ROADMAP.md)), but built as an **explainable AI pipeline**
instead of a large language model: every decision can be shown and tested.

## How it works
```
text + photo ─► safety gate ─► recognition ─────────► aggregation ─► 3-5 questions ─► tools ─► A* planner ─► steps ─► speech
                (rules)        text: TF-IDF            weighted avg /   (decision tree,  (rules)  (BFS for              (TTS)
                               photo: vision LLM + OCR product / vote   buttons)                  comparison)
                               or CNN (MobileNetV3)
manuals: built-in guides + uploaded PDFs ─► TF-IDF retrieval ─► LLM answer with citations (RAG)
voice:   microphone ─► speech-to-text (Google free / Whisper) ─► same pipeline; answers matched to buttons
         questions + steps ─► male neural voice (Edge TTS: sq-AL-Ilir, mk-MK-Aleksandar)
```
A detailed explanation of every AI step is in [docs/how_ai_works_sq.md](docs/how_ai_works_sq.md) (Albanian)
and [docs/how_ai_works_mk.md](docs/how_ai_works_mk.md) (Macedonian), also shown in the app.
| Component | AI technique | File |
|---|---|---|
| Safety gate | Rule-based system | `app.py`, `knowledge/diagnosis.json` |
| Text recognition | TF-IDF + cosine similarity (from scratch), keyword baseline | `text_recognizer.py` |
| Photo recognition | CNN, transfer learning (MobileNetV3-Small) | `train_classifier.py`, `photo_classifier.py` |
| Photo understanding, OCR, brand/model | Vision LLM with a JSON schema (structured output) | `llm.py` |
| Combining photo + text | Aggregation: weighted average, product (Naive Bayes), majority vote | `aggregate.py` |
| Clarifying questions | Decision tree, 3–5 questions | `knowledge/diagnosis.json` |
| Manual search | RAG: TF-IDF retrieval over guides and uploaded PDFs + LLM answer with citations | `manuals.py`, `llm.py` |
| Voice input | Speech-to-text: Google Web Speech (free, no key) or OpenAI Whisper (optional key); spoken answer → button by yes/no, ordinals and word overlap | `voice.py` |
| Voice output | Text-to-speech, male neural voice: Edge TTS `sq-AL-IlirNeural` / `mk-MK-AleksandarNeural`; fallback gTTS (female, sq) and browser speech | `tts.py` |
| Repair steps | State-space search: BFS and A* (admissible heuristic) | `planner.py`, `scenarios.json` |

## Current results
| Measure | Result |
|---|---|
| Text recognition, TF-IDF (leave-one-out, 40 sq/mk sentences) | **98%** (keyword baseline: 88%) |
| A* vs BFS, states expanded (cabinet door) | **13 vs 38**, same plan length |
| Unsafe actions in generated plans | 0 (tested) |
| Automated tests | 31 passing |
| Photo classifier | waiting for the photo dataset (30–50 photos per object) |

## Voice assistant
The microphone (**🎙️ Fol me mikrofon**, on by default) lets the user describe the problem in Albanian (or Macedonian)
and answer the clarifying questions by voice ("po", "jo", "e dyta", or in your own words).
The assistant answers with a voice **only if the user turns on 🔊 Asistenti flet me zë** in the sidebar (off by default);
then it reads every question and the final steps aloud with a male voice. Buttons still work at any time.
Voice needs internet (both speech services are online) and microphone permission in the browser.

## Scope
- Objects: door handle, cabinet door, chair/table (bolted or glued legs), drawer off its runners, wall shelf
- Sections: Assistant, Tips & Tricks, Categories, Tools glossary (Albanian, Macedonian, English names)
- Electrical, gas and electronics repairs are **out of scope**: the assistant gives safety information only
  and tells the user to call a professional

## Run
```
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...        # optional (Windows; use export on Mac/Linux): vision, OCR, manual answers
set OPENAI_API_KEY=sk-...               # optional: Whisper speech-to-text (better Albanian); without it the free engine is used
streamlit run app.py
python planner.py                      # BFS vs A* on every scenario
python text_recognizer.py              # TF-IDF accuracy and errors
python -m unittest discover tests      # automated tests
pip install torch torchvision && python train_classifier.py --data dataset   # once photos exist
```

## Files
- `app.py` — Streamlit app (language switch sq/mk, voice assistant)
- `voice.py`, `tts.py` — speech-to-text and the male text-to-speech voice
- `planner.py` — BFS, A*, step checker (supervisor)
- `text_recognizer.py` — TF-IDF recognizer and keyword baseline
- `train_classifier.py`, `photo_classifier.py` — photo classifier (training and use)
- `scenarios.json` — repair knowledge base: actions, preconditions, effects, safety rules (sq/mk/en)
- `knowledge/` — decision trees, tips, tools, categories, example sentences
- `PEAS.md`, `DECISIONS.md`, `ROADMAP.md` — specification, decision log, plan

All repair content is a draft and is being verified against manufacturer instructions.
AI assistance is documented in the AI Use Log.

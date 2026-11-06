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
text + photo ─► safety gate (rules) ─► object recognition ─► decision tree ─► tools (rules) ─► A* planner ─► steps
                 electrical/gas →       photo: CNN (transfer    clarifying       from the        STRIPS-style
                 refuse + refer         learning, MobileNetV3)  questions as     planned         actions with
                                        text: TF-IDF + cosine   buttons          actions         preconditions
```
| Component | AI technique | File |
|---|---|---|
| Safety gate | Rule-based system | `app.py`, `knowledge/diagnosis.json` |
| Text recognition | TF-IDF + cosine similarity (from scratch), keyword baseline | `text_recognizer.py` |
| Photo recognition | CNN, transfer learning (MobileNetV3-Small) | `train_classifier.py`, `photo_classifier.py` |
| Clarifying questions | Decision tree | `knowledge/diagnosis.json` |
| Repair steps | State-space search: BFS and A* (admissible heuristic) | `planner.py`, `scenarios.json` |

## Current results
| Measure | Result |
|---|---|
| Text recognition, TF-IDF (leave-one-out, 40 sq/mk sentences) | **98%** (keyword baseline: 88%) |
| A* vs BFS, states expanded (cabinet door) | **13 vs 38**, same plan length |
| Unsafe actions in generated plans | 0 (tested) |
| Photo classifier | waiting for the photo dataset (30–50 photos per object) |

## Scope
- Objects: door handle, cabinet door, chair/table (bolted or glued legs), drawer off its runners, wall shelf
- Sections: Assistant, Tips & Tricks, Categories, Tools glossary (Albanian, Macedonian, English names)
- Electrical, gas and electronics repairs are **out of scope**: the assistant gives safety information only
  and tells the user to call a professional

## Run
```
pip install -r requirements.txt
streamlit run app.py
python planner.py                      # BFS vs A* on every scenario
python text_recognizer.py              # TF-IDF accuracy and errors
python -m unittest discover tests      # automated tests
pip install torch torchvision && python train_classifier.py --data dataset   # once photos exist
```

## Files
- `app.py` — Streamlit app (language switch sq/mk)
- `planner.py` — BFS, A*, step checker (supervisor)
- `text_recognizer.py` — TF-IDF recognizer and keyword baseline
- `train_classifier.py`, `photo_classifier.py` — photo classifier (training and use)
- `scenarios.json` — repair knowledge base: actions, preconditions, effects, safety rules (sq/mk/en)
- `knowledge/` — decision trees, tips, tools, categories, example sentences
- `PEAS.md`, `DECISIONS.md`, `ROADMAP.md` — specification, decision log, plan

All repair content is a draft and is being verified against manufacturer instructions.
AI assistance is documented in the AI Use Log.

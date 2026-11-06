# Roadmap: AI Repair Assistant

## Related work: iFixit FixBot
FixBot (iFixit) is a chat assistant for repairs: photo/text/voice input, a large language model
that searches iFixit's guide library (72,000+ products, mostly electronics), answers with cited
guide steps and photos. Our goal is similar, but we build it differently.

## What makes this project different
| | FixBot | AI Repair Assistant |
|---|---|---|
| Language | English | **Albanian and Macedonian** (tool names also in English) |
| Domain | Electronics (phones, laptops, consoles) | **Home repairs**: furniture, doors, cabinets, shelves |
| How it decides | Large language model (black box) | **Explainable pipeline**: classifier → decision tree → rules → A* planner; the app shows *why* |
| Safety | Not described on its page | **Hard safety rules**: electrical and gas repairs are refused; unsafe actions are excluded from plans; dangerous myths explained |
| Step checking | Text answer | **Step-by-step checklist** with preconditions (cannot skip a needed step) |
| Cost / offline | Paid tier, cloud | Free, runs locally, no API keys |

## Features
- **Assistant** (chat): photo + description → object → clarifying questions (buttons) → tools → ordered steps.
- **Tips & Tricks**: short safe tricks + safety info + "dangerous myths" (e.g. testing wires with the back of the hand).
- **Categories**: Furniture, Indoor, Outdoor, Electrical, Electronics/Technology, Tools.
  Only Furniture and Indoor get repair plans; the others are tips and safety information only.
- **Tools glossary**: Albanian, Macedonian and English names, what it is for (photos later).

## Steps to the final result (syllabus weeks)
| Week | Step | Evidence |
|---|---|---|
| 4–5 | ✅ Planner BFS + A*, Streamlit app with all sections, sq/mk, 6 scenarios, keyword baseline, TF-IDF (98%), safety gate, 10 tests | `python planner.py`, `python text_recognizer.py`, tests |
| 5 | Presentation; verify every step against a source (manufacturer guides) | sources in `scenarios.json` |
| 5–6 | Photo dataset 30–50 per object (cabinet door, door handle, chair/table, drawer); Data Card | dataset folder, Data Card |
| 6–7 | Train the photo classifier (`train_classifier.py`, ready) → **MVP v0.1** | validation accuracy |
| 7–8 | Combine photo + text scores; collect real user sentences; freeze test set | test cases |
| 9–10 | Error and risk analysis: wrong object, wrong diagnosis, unsafe input never passes the gate | error table |
| 11 | Prototype v0.2: photos in tips/tools, step photos, better UI | demo |
| 12 | Beta with 3–5 test users (Milestone 4) | user results |
| 13 | Defence | |

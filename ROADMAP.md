# Roadmap: AI Repair Assistant

## Related work: iFixit FixBot
FixBot (iFixit) is a chat assistant for repairs: photo/text/voice input, a large language model
that searches iFixit's guide library (72,000+ products, mostly electronics), answers with cited
guide steps and photos. Our goal is similar, but we build it differently.

## What makes this project different
| | FixBot | AI Repair Assistant |
|---|---|---|
| Language | English | **Albanian** (tool names in Albanian + English) |
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
- **Tools glossary**: Albanian + English name, what it is for (photos later).

## Steps to the final result (syllabus weeks)
| Week | Step | Evidence |
|---|---|---|
| 4–5 | Planner BFS + A* over `scenarios.json` (done: `planner.py`), Streamlit skeleton with all sections (done: `app.py`) | `python planner.py` output, app screenshots |
| 5 | Baseline: keyword recognizer on the text + rule safety gate; presentation | demo |
| 5–6 | Photo dataset 30–50 per object (cabinet door, door handle, chair/table, drawer); Data Card | dataset folder, Data Card |
| 6 | Add scenarios for chair/table and drawer; translate step labels to Albanian; verify every step against a source | `scenarios.json` |
| 6–7 | Image classifier with transfer learning (e.g. MobileNetV2), connected to the app → **MVP v0.1** | accuracy on validation set |
| 7–8 | TF-IDF text recognizer replaces keywords; combine photo + text scores; freeze test set | test cases |
| 9–10 | Error and risk analysis: wrong object, wrong diagnosis, unsafe input never passes the gate | error table |
| 11 | Prototype v0.2: photos in tips/tools, step photos, better UI | demo |
| 12 | Beta with 3–5 test users (Milestone 4) | user results |
| 13 | Defence | |

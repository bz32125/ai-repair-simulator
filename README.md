# AI Repair Assistant
Introduction to AI (ECS-609) — individual project, Besnik Zeqiri

A photo-based assistant that helps ordinary people fix simple broken things at home
(furniture, doors, cabinets, simple hand tools) that would usually need a handyman.

## How it works
1. **Photo upload** — the user uploads a photo of the broken object.
2. **Recognition** — an image classifier identifies the object type (e.g. cabinet door, door handle).
3. **Clarifying questions** — a decision tree asks short questions to pin down the exact problem
   (e.g. "Does the handle wobble or has it fallen off?").
4. **Tools** — rule-based recommendation of the tools and parts needed (screwdriver type, screws, wall plugs...).
5. **Repair steps** — a planner (BFS/A*) orders the repair actions from the scenario knowledge base,
   respecting preconditions and safety rules.
6. **Visual guidance** — each step is shown with arrows/markers on a reference image of the object.

## Architecture
Photo → Image classifier → Decision tree (questions) → Problem state → Planner (BFS/A*) → Tools + Steps → Annotated image

## Scope (MVP)
- 5 object types: sagging cabinet door, loose door handle, wobbly chair/table, drawer off its runners, loose hammer head
- Text + image interface in Python; markers drawn on prepared reference images (not yet on the user's photo)
- Unsafe repairs (electrical, gas, structural) are out of scope: the assistant tells the user to call a professional

## Files
- `scenarios.json` — repair knowledge base: actions, preconditions, effects, safety rules (draft, being verified)
- `PEAS.md` — agent and task-environment specification
- `DECISIONS.md` — decision log

## Status
- Week 3: PEAS, architecture, 3 draft repair scenarios
- Project direction changed from a game to a photo-based assistant (approved by the professor)
- Next (Week 4, Milestone 2): photo dataset, ~30–50 photos per object type

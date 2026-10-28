# AI Repair Simulator
Introduction to AI (ECS-609) — individual project, Besnik Zeqiri

An educational game where players learn home repairs that usually need a professional
(furniture, doors, wall mounting). An AI supervisor reviews every action, like a chess
game review: Good / Mistake / Big Mistake, with explanations and 3-level hints.

## Architecture
Player action → Environment (repair state) → Planner (BFS/A*) → Evaluator → Feedback + Hint

## Files
- `scenarios.json` — repair scenarios: actions, preconditions, effects, safety rules (draft, being verified)
- `PEAS.md` — agent and task-environment specification (Week 3)

## Status
Week 3: PEAS + 3 scenarios. Next: Python environment that loads scenarios and applies actions.

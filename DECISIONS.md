# Decision Log

| Date | Decision | Why |
|---|---|---|
| 2026-10-21 | Initial proposal: repair game with an AI supervisor (replaced on 2026-10-30) | Own idea; teaches home repairs and fits search/planning topics |
| 2026-10-28 | Repair knowledge stored as data (actions, preconditions, effects, safety rules) | One engine can handle any repair; still used in the assistant to plan the repair steps |
| 2026-10-28 | Start with 3 scenarios: door handle, cabinet door, wall shelf | Common, low-risk repairs; easy to verify |
| 2026-10-30 | Change direction from a game to a photo-based repair assistant (approved by the professor) | More practical for real users; the professor suggested photo input |
| 2026-10-30 | Build own pipeline: image classifier + decision tree + planner, not only an external LLM API | Each AI component can be explained, tested and modified |
| 2026-10-30 | MVP limited to 5 object types; markers on reference images | Keeps the scope feasible for the Week 7 MVP |

# Decision Log

| Date | Decision | Why |
|---|---|---|
| 2026-10-21 | Initial proposal: repair game with an AI supervisor (replaced on 2026-10-30) | Own idea; teaches home repairs and fits search/planning topics |
| 2026-10-28 | Repair knowledge stored as data (actions, preconditions, effects, safety rules) | One engine can handle any repair; still used in the assistant to plan the repair steps |
| 2026-10-28 | Start with 3 scenarios: door handle, cabinet door, wall shelf | Common, low-risk repairs; easy to verify |
| 2026-10-30 | Change direction from a game to a photo-based repair assistant (approved by the professor) | More practical for real users; the professor suggested photo input |
| 2026-10-30 | Build own pipeline: image classifier + decision tree + planner, not only an external LLM API | Each AI component can be explained, tested and modified |
| 2026-10-30 | MVP limited to 5 object types; markers on reference images | Keeps the scope feasible for the Week 7 MVP |
| 2026-11-05 | Use iFixit FixBot as related work; differ by Albanian language, home-repair domain, explainable pipeline and hard safety rules | Same goal as an existing product, so the project needs clear differences to present |
| 2026-11-05 | Clarifying questions as buttons inside a chat UI, free text only for the first description | Buttons map exactly to decision-tree answers; the chat keeps the conversational feel |
| 2026-11-05 | Add Tips & Tricks, Categories and Tools glossary; Electrical/Electronics sections give safety info only | Useful for users, but electrical/gas repairs stay out of scope because mistakes can be fatal |
| 2026-11-05 | App in two languages: Albanian and Macedonian | Users in North Macedonia speak both; a clear difference from English-only tools |
| 2026-11-05 | TF-IDF text recognizer written from scratch, keyword matching kept as baseline | Explainable and measurable: 98% vs 88% (leave-one-out, 40 sentences) |
| 2026-11-05 | Added chair/table (bolted, glued) and drawer scenarios; MobileNetV3-Small for photos | Covers the 4 chosen objects; small model trains on a laptop CPU |

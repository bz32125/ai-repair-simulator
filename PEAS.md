# PEAS Specification — AI Repair Assistant

**P — Performance Measure:**
- Recognition accuracy: % of test photos where the object type is identified correctly (target ≥ 80%).
- Diagnosis accuracy: % of test cases where the clarifying questions lead to the correct problem.
- Efficiency: number of questions needed to reach a diagnosis (fewer is better).
- Guidance quality: % of test users (3–5) who can complete the repair by following the steps.
- Safety: unsafe situations are always flagged and referred to a professional (target 100%).

**E — Environment:** Photos of broken household objects taken by users (different angles, lighting, backgrounds), the user's answers to questions, and a knowledge base of repair scenarios.

**A — Actuators:** Shows the recognised object and its confidence, asks clarifying questions, lists the tools and parts needed, displays the ordered repair steps, draws markers/arrows on a reference image, and gives safety warnings.

**S — Sensors:** The uploaded photo and the user's answers to the clarifying questions.

## Task environment properties
| Property | Value | Why |
|---|---|---|
| Observable | Partially observable | A photo does not show everything (hidden screws, what is broken inside), so the agent must ask questions to complete its picture of the problem. |
| Deterministic | Stochastic (recognition) / Deterministic (repair steps) | The classifier can be uncertain about a photo, but once the problem is known, the same action always has the same effect. |
| Episodic / Sequential | Sequential | Each question depends on the previous answers, and repair steps must follow an order (e.g. scan the wall before drilling). |
| Static / Dynamic | Static | The broken object does not change while the user answers. |
| Discrete / Continuous | Mostly discrete | Object types, questions, tools and steps are finite sets; only the photo input (pixels) is continuous. |
| Single / Multi-agent | Single-agent | Only the assistant acts as an agent; the user provides information and performs the repair. |

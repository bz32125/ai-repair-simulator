# PEAS Specification — AI Repair Simulator

**P — Performance Measure:** The agent is evaluated by:
- Classification accuracy: % of player actions correctly labelled
  Good / Mistake / Big Mistake on the 10 frozen test sequences (target ≥ 90%).
- Safety: number of Big Mistakes a player makes per scenario (should decrease over repeated play).
- Learning: number of hints used and extra steps compared with the optimal plan.

  
**E — Environment:** The agent operates in a virtual home repair environment with different repair scenarios, tools, objects, and possible states.

**A — Actuators:** The agent provides feedback, explanations, hints, and evaluations of the user's actions. It can also suggest the next appropriate step.

**S — Sensors:** The agent receives information about the current scenario, the state of the repair task, and the action selected by the user.


## Task environment properties
| Property | Value | Why |
|---|---|---|
| Observable | Fully observable (simulation) / Partially observable (photo extension) | Inside a scenario the agent knows the complete repair state. With a real photo, the agent must first infer the problem from the image, so the initial state is not fully known. |
| Deterministic | Deterministic | The same action in the same state always produces the same result. |
| Episodic / Sequential | Sequential | Earlier actions affect later ones, e.g. drilling without scanning the wall first becomes a Big Mistake. Some steps can be done in any order (left or right screw first), so more than one correct plan is accepted. |
| Static / Dynamic | Static | Nothing changes while the player is thinking. |
| Discrete / Continuous | Discrete | Each scenario has a finite set of actions and states. |
| Single / Multi-agent | Single-agent | Only the AI supervisor acts as an agent; the player is a human user, not a competing agent. |

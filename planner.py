"""Repair planner: BFS and A* over the STRIPS-style scenarios in scenarios.json.

A state is a frozenset of facts. An action is applicable when all its `pre`
facts are true. Actions whose `safety` facts are missing are never chosen by
the planner (they would damage the object), and `__never__` marks actions that
are always unsafe (e.g. hitting a handle with a hammer).
"""
from __future__ import annotations

import heapq
import json
from collections import deque
from pathlib import Path

NEVER = "__never__"
DATA = Path(__file__).parent / "scenarios.json"


def load_scenarios(path: Path = DATA) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["id"]: s for s in data["scenarios"]}


def is_safe(action: dict, state: frozenset) -> bool:
    safety = action.get("safety", [])
    return NEVER not in safety and all(f in state for f in safety)


def applicable(action: dict, state: frozenset) -> bool:
    return all(f in state for f in action.get("pre", []))


def apply(action: dict, state: frozenset) -> frozenset:
    return (state - set(action.get("del", []))) | set(action.get("add", []))


def successors(scenario: dict, state: frozenset):
    for a in scenario["actions"]:
        if applicable(a, state) and is_safe(a, state):
            nxt = apply(a, state)
            if nxt != state:
                yield a, nxt


def goal_reached(scenario: dict, state: frozenset) -> bool:
    return all(f in state for f in scenario["goal"])


def bfs(scenario: dict):
    """Breadth-first search: shortest plan in number of steps."""
    start = frozenset(scenario.get("initial", []))
    frontier = deque([(start, [])])
    visited = {start}
    expanded = 0
    while frontier:
        state, plan = frontier.popleft()
        expanded += 1
        if goal_reached(scenario, state):
            return plan, expanded
        for a, nxt in successors(scenario, state):
            if nxt not in visited:
                visited.add(nxt)
                frontier.append((nxt, plan + [a]))
    return None, expanded


def h_missing_goals(scenario: dict, state: frozenset) -> int:
    """Heuristic: how many goal facts are still missing."""
    return sum(1 for f in scenario["goal"] if f not in state)


def astar(scenario: dict, h=h_missing_goals):
    """A* search with step cost 1 (or `cost` from the action, if given)."""
    start = frozenset(scenario.get("initial", []))
    counter = 0  # tie-breaker so heapq never compares states
    frontier = [(h(scenario, start), counter, 0, start, [])]
    best_g = {start: 0}
    expanded = 0
    while frontier:
        _, _, g, state, plan = heapq.heappop(frontier)
        if g > best_g.get(state, float("inf")):
            continue
        expanded += 1
        if goal_reached(scenario, state):
            return plan, expanded
        for a, nxt in successors(scenario, state):
            g2 = g + a.get("cost", 1)
            if g2 < best_g.get(nxt, float("inf")):
                best_g[nxt] = g2
                counter += 1
                heapq.heappush(frontier, (g2 + h(scenario, nxt), counter, g2, nxt, plan + [a]))
    return None, expanded


def check_user_step(scenario: dict, state: frozenset, action_id: str):
    """Supervisor: judge a step the user wants to take.

    Returns (verdict, message, new_state) where verdict is one of
    'ok', 'not_possible', 'unsafe'.
    """
    action = next((a for a in scenario["actions"] if a["id"] == action_id), None)
    if action is None or not applicable(action, state):
        return "not_possible", "Ky hap nuk mund të bëhet ende.", state
    if not is_safe(action, state):
        return "unsafe", action.get("safety_msg", "Hap i rrezikshëm."), state
    return "ok", "", apply(action, state)


if __name__ == "__main__":
    for sid, sc in load_scenarios().items():
        p_bfs, n_bfs = bfs(sc)
        p_ast, n_ast = astar(sc)
        print(f"{sid}: BFS {len(p_bfs)} hapa ({n_bfs} nyje), A* {len(p_ast)} hapa ({n_ast} nyje)")
        for i, a in enumerate(p_ast, 1):
            print(f"   {i}. {a['label']}")

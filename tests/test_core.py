"""Run: python -m unittest discover tests"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import planner  # noqa: E402
import text_recognizer  # noqa: E402

DIAG = json.loads((ROOT / "knowledge" / "diagnosis.json").read_text(encoding="utf-8"))
SCEN = planner.load_scenarios()


def leaves(node):
    if "q" in node:
        for a in node["answers"]:
            yield from leaves(a["next"])
    else:
        yield node


class PlannerTests(unittest.TestCase):
    def test_every_scenario_is_solvable_and_optimal(self):
        for sid, sc in SCEN.items():
            p_bfs, _ = planner.bfs(sc)
            p_ast, _ = planner.astar(sc)
            self.assertIsNotNone(p_ast, sid)
            self.assertEqual(len(p_bfs), len(p_ast), f"A* not optimal on {sid}")

    def test_plans_never_contain_unsafe_actions(self):
        for sid, sc in SCEN.items():
            state = frozenset(sc["initial"])
            for a in planner.astar(sc)[0]:
                self.assertTrue(planner.is_safe(a, state), f"{sid}: {a['id']}")
                state = planner.apply(a, state)

    def test_heuristic_is_admissible(self):
        # each action adds at most one goal fact -> 'missing goals' never overestimates
        for sid, sc in SCEN.items():
            for a in sc["actions"]:
                self.assertLessEqual(len(set(a.get("add", [])) & set(sc["goal"])), 1, f"{sid}: {a['id']}")

    def test_supervisor_flags_unsafe_step(self):
        sc = SCEN["chair_glue"]
        state = frozenset(["joint_closed"])
        verdict, _, _ = planner.check_user_step(sc, state, "sit_on_chair")
        self.assertEqual(verdict, "unsafe")


class KnowledgeTests(unittest.TestCase):
    def test_tree_leaves_point_to_existing_scenarios(self):
        for oid, obj in DIAG["objects"].items():
            for leaf in leaves(obj["tree"]):
                if "scenario" in leaf:
                    self.assertIn(leaf["scenario"], SCEN, oid)

    def test_both_languages_everywhere(self):
        for sid, sc in SCEN.items():
            for lang in ("sq", "mk"):
                self.assertTrue(sc.get(f"title_{lang}"), sid)
                for a in sc["actions"]:
                    self.assertTrue(a.get(f"label_{lang}"), f"{sid}.{a['id']} {lang}")
                    if a.get("safety_msg"):
                        self.assertTrue(a.get(f"safety_msg_{lang}"), f"{sid}.{a['id']} {lang}")


class RecognizerTests(unittest.TestCase):
    def test_leave_one_out_accuracy(self):
        acc, _ = text_recognizer.leave_one_out()
        self.assertGreaterEqual(acc, 0.9)

    def test_tfidf_beats_keyword_baseline(self):
        acc, _ = text_recognizer.leave_one_out()
        self.assertGreater(acc, text_recognizer.baseline_accuracy())


class AppTests(unittest.TestCase):
    def run_app(self, lang, text):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py")).run()
        if lang == "mk":
            at.sidebar.radio[0].set_value("mk").run()
        at.chat_input[0].set_value(text).run()
        self.assertFalse(at.exception)
        return at

    def test_safety_gate_both_languages(self):
        for lang, text in [("sq", "teli i prizës nuk ka rrymë"), ("mk", "жицата во штекерот нема струја")]:
            at = self.run_app(lang, text)
            self.assertTrue(at.error, lang)

    def test_full_flow_macedonian(self):
        at = self.run_app("mk", "Фиоката излезе од шините")
        at.button(key="ans_0_0").click().run()
        at.button(key="ans_1_0").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.success[0].value, SCEN["drawer_runners"]["title_mk"])
        self.assertEqual(len(at.checkbox), 6)


if __name__ == "__main__":
    unittest.main()

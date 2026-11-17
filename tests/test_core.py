"""Run: python -m unittest discover tests"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import aggregate  # noqa: E402
import manuals  # noqa: E402
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
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
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
        for key in ["pre_0_no", "pre_1_no", "ans_2_0", "ans_3_0"]:
            at.button(key=key).click().run()
            self.assertFalse(at.exception)
        self.assertEqual(at.success[0].value, SCEN["drawer_runners"]["title_mk"])
        self.assertEqual(len(at.checkbox), 6)

    def test_three_to_five_questions(self):
        at = self.run_app("sq", "Karrigia lëkundet kur ulem")
        for key in ["pre_0_no", "pre_1_no", "ans_2_0", "tools_0"]:
            at.button(key=key).click().run()
        asked = [m.value for m in at.markdown if m.value.startswith("•")]
        self.assertTrue(3 <= len(asked) <= 5, asked)
        self.assertEqual(len(at.checkbox), len(planner.astar(SCEN["chair_bolts"])[0]))

    def test_safety_question_stops_the_flow(self):
        at = self.run_app("sq", "Dera e dollapit varet")
        at.button(key="pre_0_yes").click().run()
        self.assertTrue(at.error)
        self.assertFalse(at.checkbox)


class AggregationTests(unittest.TestCase):
    OBJ = ["door_handle", "cabinet_door", "drawer"]

    def test_photo_and_text_agree(self):
        text = {"door_handle": 0.4, "cabinet_door": 0.1, "drawer": 0.0}
        for method in ("weighted", "product", "vote"):
            best, dist, used = aggregate.combine(self.OBJ, text, ("door_handle", 0.8), method)
            self.assertEqual(best, "door_handle")
            self.assertEqual(set(used), {"text", "photo"})
            self.assertAlmostEqual(sum(dist.values()), 1.0)

    def test_confident_photo_overrides_weak_text(self):
        text = {"door_handle": 0.12, "cabinet_door": 0.10, "drawer": 0.0}
        best, _, _ = aggregate.combine(self.OBJ, text, ("drawer", 0.95), "weighted")
        self.assertEqual(best, "drawer")

    def test_no_evidence(self):
        self.assertIsNone(aggregate.combine(self.OBJ, {o: 0.0 for o in self.OBJ}, None)[0])


class ManualTests(unittest.TestCase):
    def test_retrieval_finds_the_right_guide(self):
        ix = manuals.ManualIndex(manuals.builtin_chunks())
        self.assertIn(SCEN["drawer_runners"]["title_sq"], ix.search("sirtari ka dalë nga shinat")[0]["source"])
        self.assertIn(SCEN["door_handle"]["title_mk"], ix.search("кваката се клати")[0]["source"])

    def test_manual_link_only_with_brand_or_model(self):
        self.assertIsNone(manuals.manufacturer_search_url("", ""))
        self.assertIn("MALM", manuals.manufacturer_search_url("IKEA", "MALM"))


if __name__ == "__main__":
    unittest.main()


class LlmRequestTests(unittest.TestCase):
    """Checks the vision request and JSON parsing without network or API key."""

    def test_analyze_photo_request_and_parsing(self):
        import anthropic
        import httpx2

        import llm
        sent = {}
        reply = {"object": "chair_table", "confidence": 1.4, "brand": "IKEA", "model": "TEODORES",
                 "ocr_text": "TEODORES 902.108.62", "damage": "Këmba lëviz",
                 "involves_electricity_gas_or_water": False}

        def handler(request):
            sent["body"] = json.loads(request.content)
            return httpx2.Response(200, json={
                "id": "msg", "type": "message", "role": "assistant", "model": llm.MODEL,
                "content": [{"type": "text", "text": json.dumps(reply)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}})

        original = llm._client
        llm._client = lambda key=None: anthropic.Anthropic(
            api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
        try:
            out = llm.analyze_photo(b"img", "image/png", "karrigia lëviz", "sq")
        finally:
            llm._client = original
        self.assertEqual(out["object"], "chair_table")
        self.assertEqual(out["confidence"], 1.0)  # clamped to [0, 1]
        content = sent["body"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(sent["body"]["output_config"]["format"]["type"], "json_schema")

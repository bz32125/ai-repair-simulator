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

    def test_no_spoken_answer_unless_user_turns_it_on(self):
        import tts
        calls, orig = [], tts.synthesize
        tts.synthesize = lambda text, lang, gender="male": (calls.append(text), (b"ID3", "edge"))[1]
        try:
            at = self.run_app("sq", "teli i prizës nuk ka rrymë dhe del shkëndijë")
            self.assertEqual(calls, [])  # voice answers are off by default
            self.assertFalse([e for e in at.main if e.type == "audio"])
            at.sidebar.toggle[1].set_value(True).run()  # "Asistenti flet me zë"
            self.assertFalse(at.exception)
            self.assertTrue(calls)
        finally:
            tts.synthesize = orig

    def test_full_flow_macedonian(self):
        at = self.run_app("mk", "Фиоката излезе од шините")
        for key in ["pre_0_no", "pre_1_no", "ans_2_0", "ans_3_0"]:
            at.button(key=key).click().run()
            self.assertFalse(at.exception)
        self.assertEqual(at.success[0].value, SCEN["drawer_runners"]["title_mk"])
        self.assertEqual(len(at.checkbox), 6)

    def test_questions_already_answered_by_text_are_skipped(self):
        at = self.run_app("sq", "Sirtari ka dalë nga shinat")
        for key in ["pre_0_no", "pre_1_no", "ans_3_0"]:  # "what happens to the drawer?" is answered from the text
            at.button(key=key).click().run()
            self.assertFalse(at.exception)
        asked = [m.value for m in at.markdown if m.value.startswith("•")]
        self.assertTrue(any("✔" in a for a in asked), asked)
        self.assertEqual(at.success[0].value, SCEN["drawer_runners"]["title_sq"])

    def test_broken_part_in_text_goes_to_replacement(self):
        at = self.run_app("sq", "Këmba e karriges është thyer")
        at.button(key="pre_0_no").click().run()
        self.assertTrue(at.info)  # replacement advice, no repair steps
        self.assertFalse(at.checkbox)

    def test_pasted_manual_is_shown_with_the_steps(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        manual = ("MALM chest of drawers\n\nDrawer runners: tilt the drawer 20 degrees, put the rear wheels "
                  "into the runner channel until it clicks, then push the drawer in.\n\n"
                  "Safety: fix the chest to the wall with the safety brackets.")
        at.text_area[0].set_value(manual).run()
        at.chat_input[0].set_value("Sirtari ka dalë nga shinat").run()
        for key in ["pre_0_no", "pre_1_no", "ans_3_0"]:
            at.button(key=key).click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("runner" in m.value for m in at.markdown))

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

    def test_library_manual_found_from_ocr(self):
        lib = manuals.library_chunks()
        self.assertEqual(manuals.match_manuals(lib, "IKEA MALM Art. 403.604.65"), ["ikea_malm_403.604.65_sample.md"])
        self.assertEqual(manuals.match_manuals(lib, "IKEA"), [])  # brand alone is not enough
        top = manuals.ManualIndex(manuals.builtin_chunks() + lib).search("sirtari fërkon anash dhe nuk mbyllet drejt")
        self.assertEqual(top[0]["file"], "ikea_malm_403.604.65_sample.md")

    def test_manual_link_only_with_brand_or_model(self):
        self.assertIsNone(manuals.manufacturer_search_url("", ""))
        self.assertIn("MALM", manuals.manufacturer_search_url("IKEA", "MALM"))


class VoiceTests(unittest.TestCase):
    """Voice assistant: spoken answers -> buttons, STT fallback, male TTS voice."""

    def setUp(self):
        import voice
        self.voice = voice
        self.tree = json.loads((ROOT / "knowledge" / "diagnosis.json").read_text(encoding="utf-8"))

    def test_yes_no(self):
        v = self.voice
        self.assertEqual(v.yes_no("Po"), 0)
        self.assertEqual(v.yes_no("jo, nuk e kam"), 1)
        self.assertEqual(v.yes_no("Да"), 0)
        self.assertEqual(v.yes_no("не"), 1)
        self.assertIsNone(v.yes_no("po... jo"))
        self.assertIsNone(v.yes_no("ndoshta"))

    def test_spoken_answer_matches_option(self):
        node = self.tree["objects"]["door_handle"]["tree"]
        opts = [a["text"]["sq"] for a in node["answers"]]
        self.assertEqual(self.voice.match_option("lëviz dhe varet poshtë", opts, "sq"), 0)
        self.assertEqual(self.voice.match_option("e dyta", opts, "sq"), 1)
        self.assertEqual(self.voice.match_option("brava nuk mbyllet", opts, "sq"), 2)
        self.assertIsNone(self.voice.match_option("", opts, "sq"))
        self.assertIsNone(self.voice.match_option("banana", opts, "sq"))

    def test_spoken_answer_macedonian(self):
        node = self.tree["objects"]["door_handle"]["tree"]
        opts = [a["text"]["mk"] for a in node["answers"]]
        self.assertEqual(self.voice.match_option("првата", opts, "mk"), 0)

    def test_whisper_falls_back_to_free_engine(self):
        v = self.voice
        orig = v.whisper, v.google_free
        try:
            v.whisper = lambda *a: (_ for _ in ()).throw(RuntimeError("401"))
            v.google_free = lambda wav, lang: "sirtari ka dalë"
            self.assertEqual(v.transcribe(b"wav", "sq", "sk-bad"), ("sirtari ka dalë", "google"))
            self.assertEqual(v.transcribe(b"wav", "sq"), ("sirtari ka dalë", "google"))
            v.google_free = lambda wav, lang: (_ for _ in ()).throw(OSError("offline"))
            with self.assertRaises(v.SpeechError):
                v.transcribe(b"wav", "sq")
        finally:
            v.whisper, v.google_free = orig

    def test_male_voice_first_then_fallbacks(self):
        import tts
        self.assertEqual(tts.edge_voice("sq"), "sq-AL-IlirNeural")
        self.assertEqual(tts.edge_voice("mk"), "mk-MK-AleksandarNeural")
        orig = tts.edge_mp3, tts.gtts_mp3
        try:
            tts.edge_mp3 = lambda text, lang, gender="male": b"edge"
            self.assertEqual(tts.synthesize("Hapi 1", "sq"), (b"edge", "sq-AL-IlirNeural"))
            tts.edge_mp3 = lambda text, lang, gender="male": None
            tts.gtts_mp3 = lambda text, lang: b"g" if lang == "sq" else None
            self.assertEqual(tts.synthesize("Hapi 1", "sq"), (b"g", "gtts"))
            self.assertEqual(tts.synthesize("Чекор 1", "mk"), (None, None))
        finally:
            tts.edge_mp3, tts.gtts_mp3 = orig

    def test_app_has_microphone(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertTrue(any(w.type == "audio_input" for w in at.main))




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

    def test_manual_guidance_sends_pdf_and_image_blocks(self):
        import anthropic
        import httpx2

        import llm
        sent = {}

        def handler(request):
            sent["body"] = json.loads(request.content)
            return httpx2.Response(200, json={
                "id": "msg", "type": "message", "role": "assistant", "model": llm.MODEL,
                "content": [{"type": "text", "text": "1. Fut rrotat në shinë (faqja 4)."}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}})

        original = llm._client
        llm._client = lambda key=None: anthropic.Anthropic(
            api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
        try:
            out = llm.manual_guidance([
                {"kind": "pdf", "data": b"%PDF-1.4", "media_type": "application/pdf", "name": "malm.pdf"},
                {"kind": "image", "data": b"img", "media_type": "image/png", "name": "page.png"},
                {"kind": "text", "data": "Schublade einsetzen", "media_type": "text/plain", "name": "de"},
            ], "sirtari ka dalë", "Ktheje sirtarin në shina", "sq")
        finally:
            llm._client = original
        self.assertIn("faqja 4", out)
        types = [b["type"] for b in sent["body"]["messages"][0]["content"]]
        self.assertEqual(types, ["document", "image", "text", "text"])
        self.assertIn("ANY language", sent["body"]["messages"][0]["content"][-1]["text"])


class LlmQuestionTests(unittest.TestCase):
    def test_answer_questions_keeps_only_valid_answers(self):
        import anthropic
        import httpx2

        import llm
        reply = {"answers": [
            {"id": "safety", "option": 1, "reason": "photo: wooden drawer, no wires"},
            {"id": "broken", "option": -1, "reason": "not visible"},
            {"id": "Çfarë ndodh me sirtarin?", "option": 0, "reason": "photo: wheels out of runner"},
            {"id": "unknown", "option": 0, "reason": "x"},
            {"id": "Shinat janë të plota, pa pjesë të thyera?", "option": 7, "reason": "x"}]}

        def handler(request):
            return httpx2.Response(200, json={
                "id": "msg", "type": "message", "role": "assistant", "model": llm.MODEL,
                "content": [{"type": "text", "text": json.dumps(reply)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}})

        original = llm._client
        llm._client = lambda key=None: anthropic.Anthropic(
            api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
        qs = [{"id": "safety", "question": "q", "options": ["Po", "Jo"]},
              {"id": "broken", "question": "q", "options": ["Po", "Jo"]},
              {"id": "Çfarë ndodh me sirtarin?", "question": "q", "options": ["a", "b"]},
              {"id": "Shinat janë të plota, pa pjesë të thyera?", "question": "q", "options": ["Po", "Jo"]}]
        try:
            out = llm.answer_questions(qs, "problem me fjoken", (b"img", "image/png"), None)
        finally:
            llm._client = original
        self.assertEqual(out, {"safety": 1, "Çfarë ndodh me sirtarin?": 0})


if __name__ == "__main__":
    unittest.main()

"""AI Repair Assistant: Streamlit prototype (skeleton).

Run:  pip install -r requirements.txt && streamlit run app.py

Pipeline: photo + text -> safety gate (rules) -> object recognition
(image classifier, not trained yet; keyword baseline on the text for now)
-> decision tree (clarifying questions as buttons) -> tools (rules)
-> repair steps (A* planner, BFS shown for comparison).
"""
import json
import re
from pathlib import Path

import streamlit as st

import planner

ROOT = Path(__file__).parent
KB = ROOT / "knowledge"


@st.cache_data
def load(name):
    return json.loads((KB / name).read_text(encoding="utf-8"))


DIAG = load("diagnosis.json")
TIPS = load("tips.json")["tips"]
TOOLS = load("tools.json")["tools"]
CATS = load("categories.json")["categories"]
SCENARIOS = planner.load_scenarios()


def tokens(text):
    return re.findall(r"\w+", text.lower())


def is_unsafe(text):
    """Safety gate: rule-based check before any AI step."""
    return any(t.startswith(k) for t in tokens(text) for k in DIAG["unsafe_keywords"])


def guess_object(text):
    """Baseline text recognizer: keyword score per object (to be replaced by TF-IDF)."""
    scores = {}
    for oid, obj in DIAG["objects"].items():
        scores[oid] = sum(1 for t in tokens(text) for k in obj["keywords"] if t.startswith(k))
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0 else None), scores


def classify_photo(photo):
    """Placeholder for the image classifier (transfer learning, Week 5-6)."""
    return None, 0.0


def tools_for(plan):
    """Rule: every 'get_*' action in the plan is a tool the user needs."""
    return [a["label"].replace("Get ", "").replace("a ", "", 1) for a in plan if a["id"].startswith("get_")]


def reset():
    for k in ("stage", "obj", "node", "path", "text"):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="AI Repair Assistant", page_icon="🔧")
st.title("🔧 AI Repair Assistant")
st.caption("Ndihmë hap pas hapi për riparime të thjeshta në shtëpi, në shqip. Prototip.")

tab_ai, tab_tips, tab_cats, tab_tools = st.tabs(["Asistenti", "Tips & Tricks", "Kategoritë", "Mjetet"])

with tab_ai:
    ss = st.session_state
    ss.setdefault("stage", "describe")

    if ss.stage == "describe":
        with st.chat_message("assistant"):
            st.write("Përshkruaje problemin dhe, nëse mundesh, ngarko një foto.")
        photo = st.file_uploader("Foto (opsionale)", type=["jpg", "jpeg", "png"])
        text = st.chat_input("P.sh. 'Doreza e derës lëviz dhe varet poshtë'")
        if text:
            ss.text = text
            if is_unsafe(text):
                ss.stage = "unsafe"
            else:
                obj, _ = classify_photo(photo)
                if obj is None:
                    obj, _ = guess_object(text)
                ss.obj = obj
                ss.node = DIAG["objects"][obj]["tree"] if obj else None
                ss.path = []
                ss.stage = "diagnose" if obj else "pick"
            st.rerun()

    if "text" in ss:
        with st.chat_message("user"):
            st.write(ss.text)

    if ss.stage == "unsafe":
        with st.chat_message("assistant"):
            st.error(DIAG["unsafe_message"])
        st.button("Fillo nga e para", on_click=reset)

    elif ss.stage == "pick":
        with st.chat_message("assistant"):
            st.write("Nuk e njoha objektin. Cili është?")
            for oid, obj in DIAG["objects"].items():
                if st.button(obj["name"], key=f"pick_{oid}"):
                    ss.obj, ss.node, ss.path, ss.stage = oid, obj["tree"], [], "diagnose"
                    st.rerun()

    elif ss.stage == "diagnose":
        obj = DIAG["objects"][ss.obj]
        with st.chat_message("assistant"):
            st.write(f"Objekti: **{obj['name']}**")
            for q, a in ss.path:
                st.write(f"• {q} → *{a}*")
            node = ss.node
            if "q" in node:
                st.write(f"**{node['q']}**")
                for ans, nxt in node["answers"].items():
                    if st.button(ans, key=f"ans_{len(ss.path)}_{ans}"):
                        ss.path.append((node["q"], ans))
                        ss.node = nxt
                        st.rerun()
            elif "refer" in node:
                st.info(node["refer"])
            else:
                sc = SCENARIOS[node["scenario"]]
                plan, n_astar = planner.astar(sc)
                _, n_bfs = planner.bfs(sc)
                st.success(sc["title"])
                st.write("**Mjetet që të duhen:**")
                for t in tools_for(plan):
                    st.write(f"- {t}")
                st.write("**Hapat:**")
                for i, a in enumerate(plan, 1):
                    st.checkbox(f"{i}. {a['label']}", key=f"step_{sc['id']}_{i}")
                warnings = [a for a in sc["actions"] if a.get("safety_msg")]
                if warnings:
                    st.write("**Mos bëj:**")
                    for a in warnings:
                        st.warning(f"{a['label']}: {a['safety_msg']}")
                with st.expander("Pse këta hapa? (si mendon AI)"):
                    st.write(
                        f"Pema e vendimit zgjodhi skenarin `{sc['id']}` nga përgjigjet e tua. "
                        f"Planner-i A* gjeti {len(plan)} hapa duke hapur {n_astar} gjendje; "
                        f"BFS do të hapte {n_bfs}. Hapat e rrezikshëm (safety) përjashtohen nga plani."
                    )
        st.button("Problem tjetër", on_click=reset)

with tab_tips:
    kinds = {"siguri": "🛡️ Siguri", "mit_i_rrezikshem": "⚠️ Mit i rrezikshëm", "truk": "💡 Truk"}
    for tip in TIPS:
        with st.expander(f"{kinds.get(tip['kind'], '')} · {tip['title']}"):
            if tip["kind"] == "mit_i_rrezikshem":
                st.error("Mos e provo këtë. Shpjegohet vetëm që ta kuptosh pse është i rrezikshëm.")
            st.write(tip["body"])
            st.caption(f"Burimi: {tip['source']}")

with tab_cats:
    for c in CATS:
        st.subheader(c["name"])
        if c.get("warning"):
            st.warning(c["warning"])
        st.write("Asistenti jep hapa riparimi." if c["mode"] == "assistant" else "Vetëm këshilla dhe informacion.")
        for tip in (t for t in TIPS if t["category"] == c["id"]):
            st.write(f"- {tip['title']}")

with tab_tools:
    q = st.text_input("Kërko mjet (shqip ose anglisht)")
    for t in TOOLS:
        if not q or q.lower() in (t["sq"] + t["en"]).lower():
            st.write(f"**{t['sq']}** ({t['en']}): {t['use']}")

"""AI Repair Assistant: Streamlit prototype, in Albanian (sq) and Macedonian (mk).

Run:  pip install -r requirements.txt && streamlit run app.py

Pipeline: photo + text -> safety gate (rules) -> object recognition
(image classifier when trained, TF-IDF on the text otherwise)
-> decision tree (clarifying questions as buttons) -> tools (rules)
-> repair steps (A* planner, BFS shown for comparison).
"""
import json
import re
from pathlib import Path

import streamlit as st

import photo_classifier
import planner
import text_recognizer

ROOT = Path(__file__).parent
KB = ROOT / "knowledge"

UI = {
    "sq": {
        "lang": "Gjuha", "subtitle": "Ndihmë hap pas hapi për riparime të thjeshta në shtëpi. Prototip.",
        "tabs": ["Asistenti", "Tips & Tricks", "Kategoritë", "Mjetet"],
        "describe": "Përshkruaje problemin dhe, nëse mundesh, ngarko një foto.",
        "photo": "Foto (opsionale)", "placeholder": "P.sh. 'Doreza e derës lëviz dhe varet poshtë'",
        "restart": "Fillo nga e para", "other": "Problem tjetër",
        "unknown": "Nuk e njoha objektin. Cili është?", "object": "Objekti",
        "tools": "Mjetet që të duhen:", "steps": "Hapat:", "dont": "Mos bëj:",
        "why": "Pse këta hapa? (si mendon AI)",
        "why_text": "Njohja e tekstit (TF-IDF) zgjodhi **{obj}** me ngjashmëri {score:.2f}. Pema e vendimit zgjodhi skenarin `{sid}` nga përgjigjet e tua. Planner-i A* gjeti {n} hapa duke hapur {na} gjendje; BFS do të hapte {nb}. Hapat e rrezikshëm përjashtohen nga plani.",
        "kinds": {"safety": "🛡️ Siguri", "dangerous_myth": "⚠️ Mit i rrezikshëm", "trick": "💡 Truk"},
        "myth": "Mos e provo këtë. Shpjegohet vetëm që ta kuptosh pse është i rrezikshëm.",
        "source": "Burimi", "mode_assistant": "Asistenti jep hapa riparimi.", "mode_info": "Vetëm këshilla dhe informacion.",
        "search": "Kërko mjet", "done": "Bravo! Riparimi përfundoi. ✅",
    },
    "mk": {
        "lang": "Јазик", "subtitle": "Помош чекор по чекор за едноставни поправки во домот. Прототип.",
        "tabs": ["Асистент", "Совети и трикови", "Категории", "Алати"],
        "describe": "Опиши го проблемот и, ако можеш, прикачи фотографија.",
        "photo": "Фотографија (опционално)", "placeholder": "На пр. „Кваката на вратата се клати и виси надолу“",
        "restart": "Почни одново", "other": "Друг проблем",
        "unknown": "Не го препознав предметот. Кој е?", "object": "Предмет",
        "tools": "Потребни алати:", "steps": "Чекори:", "dont": "Не прави:",
        "why": "Зошто овие чекори? (како размислува AI)",
        "why_text": "Препознавањето текст (TF-IDF) избра **{obj}** со сличност {score:.2f}. Дрвото на одлучување го избра сценариото `{sid}` според твоите одговори. Планерот A* најде {n} чекори отворајќи {na} состојби; BFS би отворил {nb}. Опасните чекори се исклучени од планот.",
        "kinds": {"safety": "🛡️ Безбедност", "dangerous_myth": "⚠️ Опасен мит", "trick": "💡 Трик"},
        "myth": "Не го пробувај ова. Се објаснува само за да разбереш зошто е опасно.",
        "source": "Извор", "mode_assistant": "Асистентот дава чекори за поправка.", "mode_info": "Само совети и информации.",
        "search": "Барај алат", "done": "Браво! Поправката е завршена. ✅",
    },
}


@st.cache_data
def load(name):
    return json.loads((KB / name).read_text(encoding="utf-8"))


@st.cache_resource
def recognizer():
    return text_recognizer.build()


DIAG = load("diagnosis.json")
TIPS = load("tips.json")["tips"]
TOOLS = {t["id"]: t for t in load("tools.json")["tools"]}
CATS = load("categories.json")["categories"]
SCENARIOS = planner.load_scenarios()


def is_unsafe(text):
    """Safety gate: rule-based check that runs before any other AI step."""
    words = re.findall(r"\w+", text.lower())
    return any(w.startswith(k) for w in words for k in DIAG["unsafe_keywords"])


def classify_photo(photo):
    """Image classifier (train_classifier.py); (None, 0.0) until a model is trained."""
    obj, conf = photo_classifier.predict(photo)
    return (obj, conf) if obj in DIAG["objects"] else (None, 0.0)


def field(item, name, lang):
    """Localized field of a scenario/action, falling back to English."""
    return item.get(f"{name}_{lang}") or item.get(name, "")


def tools_for(plan, lang):
    """Rule: the tools of every action in the plan, without duplicates."""
    ids = list(dict.fromkeys(t for a in plan for t in a.get("tools", [])))
    return [TOOLS[i]["name"][lang] + f" ({TOOLS[i]['en']})" for i in ids if i in TOOLS]


def reset():
    for k in ("stage", "obj", "node", "path", "text", "score"):
        st.session_state.pop(k, None)


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="AI Repair Assistant", page_icon="🔧")
lang = st.sidebar.radio("Gjuha / Јазик", ["sq", "mk"], format_func=lambda x: {"sq": "Shqip", "mk": "Македонски"}[x])
U = UI[lang]
st.title("🔧 AI Repair Assistant")
st.caption(U["subtitle"])

tab_ai, tab_tips, tab_cats, tab_tools = st.tabs(U["tabs"])

with tab_ai:
    ss = st.session_state
    ss.setdefault("stage", "describe")

    if ss.stage == "describe":
        with st.chat_message("assistant"):
            st.write(U["describe"])
        photo = st.file_uploader(U["photo"], type=["jpg", "jpeg", "png"])
        text = st.chat_input(U["placeholder"])
        if text:
            ss.text = text
            if is_unsafe(text):
                ss.stage = "unsafe"
            else:
                obj, conf = classify_photo(photo)
                if obj is None:
                    obj, scores = recognizer().predict(text)
                    conf = scores.get(obj, 0.0) if obj else 0.0
                ss.obj, ss.score, ss.path = obj, conf, []
                ss.node = DIAG["objects"][obj]["tree"] if obj else None
                ss.stage = "diagnose" if obj else "pick"
            st.rerun()

    if "text" in ss:
        with st.chat_message("user"):
            st.write(ss.text)

    if ss.stage == "unsafe":
        with st.chat_message("assistant"):
            st.error(DIAG["unsafe_message"][lang])
        st.button(U["restart"], on_click=reset)

    elif ss.stage == "pick":
        with st.chat_message("assistant"):
            st.write(U["unknown"])
            for oid, obj in DIAG["objects"].items():
                if st.button(obj["name"][lang], key=f"pick_{oid}"):
                    ss.obj, ss.node, ss.path, ss.score, ss.stage = oid, obj["tree"], [], 0.0, "diagnose"
                    st.rerun()

    elif ss.stage == "diagnose":
        obj = DIAG["objects"][ss.obj]
        with st.chat_message("assistant"):
            st.write(f"{U['object']}: **{obj['name'][lang]}**")
            for q, a in ss.path:
                st.write(f"• {q[lang]} → *{a[lang]}*")
            node = ss.node
            if "q" in node:
                st.write(f"**{node['q'][lang]}**")
                for i, ans in enumerate(node["answers"]):
                    if st.button(ans["text"][lang], key=f"ans_{len(ss.path)}_{i}"):
                        ss.path.append((node["q"], ans["text"]))
                        ss.node = ans["next"]
                        st.rerun()
            elif "refer" in node:
                st.info(node["refer"][lang])
            else:
                sc = SCENARIOS[node["scenario"]]
                plan, n_astar = planner.astar(sc)
                _, n_bfs = planner.bfs(sc)
                st.success(field(sc, "title", lang))
                st.write(U["tools"])
                for t in tools_for(plan, lang):
                    st.write(f"- {t}")
                st.write(U["steps"])
                done = [st.checkbox(f"{i}. {field(a, 'label', lang)}", key=f"step_{sc['id']}_{i}")
                        for i, a in enumerate(plan, 1)]
                if all(done):
                    st.balloons()
                    st.success(U["done"])
                warnings = [a for a in sc["actions"] if a.get("safety_msg")]
                if warnings:
                    st.write(U["dont"])
                    for a in warnings:
                        st.warning(f"{field(a, 'label', lang)}: {field(a, 'safety_msg', lang)}")
                with st.expander(U["why"]):
                    st.write(U["why_text"].format(obj=obj["name"][lang], score=ss.score, sid=sc["id"],
                                                  n=len(plan), na=n_astar, nb=n_bfs))
        st.button(U["other"], on_click=reset)

with tab_tips:
    for tip in TIPS:
        with st.expander(f"{U['kinds'].get(tip['kind'], '')} · {tip['title'][lang]}"):
            if tip["kind"] == "dangerous_myth":
                st.error(U["myth"])
            st.write(tip["body"][lang])
            st.caption(f"{U['source']}: {tip['source']}")

with tab_cats:
    for c in CATS:
        st.subheader(c["name"][lang])
        if c.get("warning"):
            st.warning(c["warning"][lang])
        st.write(U["mode_assistant"] if c["mode"] == "assistant" else U["mode_info"])
        for tip in (t for t in TIPS if t["category"] == c["id"]):
            st.write(f"- {tip['title'][lang]}")

with tab_tools:
    q = st.text_input(U["search"]).lower()
    for t in TOOLS.values():
        if not q or q in (t["name"]["sq"] + t["name"]["mk"] + t["en"]).lower():
            st.write(f"**{t['name'][lang]}** ({t['en']}): {t['use'][lang]}")

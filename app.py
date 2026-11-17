"""AI Repair Assistant: Streamlit prototype, in Albanian (sq) and Macedonian (mk).

Run:  pip install -r requirements.txt && streamlit run app.py
Optional: set ANTHROPIC_API_KEY (or paste it in the sidebar) to turn on the vision model,
OCR of labels and LLM answers from manuals.

Pipeline:
  text + photo -> safety gate (rules)
               -> object recognition: TF-IDF on the text + vision model / CNN on the photo
               -> aggregation (weighted average / product / majority vote)
               -> 3-5 clarifying questions (decision tree)
               -> tools (rules) -> repair steps (A*, BFS for comparison) -> text-to-speech
  manuals: built-in guides + uploaded PDFs -> TF-IDF retrieval -> LLM answer with citations (RAG)
"""
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import aggregate
import llm
import manuals
import photo_classifier
import planner
import text_recognizer
import tts

ROOT = Path(__file__).parent
KB = ROOT / "knowledge"

UI = {
    "sq": {
        "subtitle": "Ndihmë hap pas hapi për riparime të thjeshta në shtëpi. Prototip.",
        "tabs": ["Asistenti", "Manualet", "Tips & Tricks", "Kategoritë", "Mjetet", "Si punon AI"],
        "describe": "Përshkruaje problemin dhe, nëse mundesh, ngarko një foto.",
        "photo": "Foto e objektit (opsionale)", "placeholder": "P.sh. 'Doreza e derës lëviz dhe varet poshtë'",
        "restart": "Fillo nga e para", "other": "Problem tjetër",
        "unknown": "Nuk jam i sigurt për objektin. Cili është?", "object": "Objekti",
        "tools": "Mjetet që të duhen:", "steps": "Hapat:", "dont": "Mos bëj:",
        "how": "Si e njoha objektin?", "why": "Pse këta hapa? (si mendon AI)",
        "why_text": "Pyetjet diagnostikuese ({nq}) e çuan te skenari `{sid}`. Planner-i A* gjeti {n} hapa duke hapur {na} gjendje; BFS do të hapte {nb}. Hapat e rrezikshëm përjashtohen nga plani.",
        "kinds": {"safety": "🛡️ Siguri", "dangerous_myth": "⚠️ Mit i rrezikshëm", "trick": "💡 Truk"},
        "safe_only": "Shfaq vetëm këshillat e sigurta", "myth": "Mos e provo këtë. Shpjegohet vetëm që ta kuptosh pse është i rrezikshëm.",
        "source": "Burimi", "mode_assistant": "Asistenti jep hapa riparimi.", "mode_info": "Vetëm këshilla dhe informacion.",
        "search": "Kërko mjet", "done": "Bravo! Riparimi përfundoi. ✅", "listen": "Dëgjo hapat",
        "vision": "Analiza e fotos (vision model)", "brand": "Marka", "model_": "Modeli", "ocr": "Teksti i lexuar nga foto (OCR)",
        "damage": "Dëmtimi", "manual_found": "U gjet manuali në bibliotekë", "manual_link": "Kërko manualin zyrtar të këtij modeli", "no_key": "Pa API key: foto nuk analizohet nga vision model; njohja bëhet nga teksti.",
        "agg": "Bashkimi i rezultateve", "text_src": "Teksti (TF-IDF)", "photo_src": "Foto", "combined": "Rezultati",
        "man_intro": "Ngarko manualin PDF të pajisjes (opsionale) dhe bëj një pyetje. Kërkimi bëhet në udhëzuesit e aplikacionit dhe në manualet që ngarkon.",
        "man_upload": "Manuale (PDF, TXT, MD)", "man_q": "Pyetja jote", "man_answer": "Përgjigja e AI-së", "man_passages": "Pjesët e gjetura në manuale",
        "man_none": "Nuk u gjet asgjë e lidhur. Provo me fjalë të tjera ose ngarko manualin.",
        "settings": "Cilësimet e AI-së", "key": "Anthropic API key (opsionale)", "method": "Metoda e bashkimit",
        "methods": {"weighted": "Mesatare e peshuar", "product": "Prodhim (Naive Bayes)", "vote": "Votim i shumicës"},
        "ai_on": "Vision + LLM: aktiv", "ai_off": "Vision + LLM: joaktiv (vetëm algoritmet lokale)", "error": "Gabim nga AI",
    },
    "mk": {
        "subtitle": "Помош чекор по чекор за едноставни поправки во домот. Прототип.",
        "tabs": ["Асистент", "Прирачници", "Совети и трикови", "Категории", "Алати", "Како работи AI"],
        "describe": "Опиши го проблемот и, ако можеш, прикачи фотографија.",
        "photo": "Фотографија од предметот (опционално)", "placeholder": "На пр. „Кваката на вратата се клати и виси надолу“",
        "restart": "Почни одново", "other": "Друг проблем",
        "unknown": "Не сум сигурен за предметот. Кој е?", "object": "Предмет",
        "tools": "Потребни алати:", "steps": "Чекори:", "dont": "Не прави:",
        "how": "Како го препознав предметот?", "why": "Зошто овие чекори? (како размислува AI)",
        "why_text": "Дијагностичките прашања ({nq}) доведоа до сценариото `{sid}`. Планерот A* најде {n} чекори отворајќи {na} состојби; BFS би отворил {nb}. Опасните чекори се исклучени од планот.",
        "kinds": {"safety": "🛡️ Безбедност", "dangerous_myth": "⚠️ Опасен мит", "trick": "💡 Трик"},
        "safe_only": "Прикажи само безбедни совети", "myth": "Не го пробувај ова. Се објаснува само за да разбереш зошто е опасно.",
        "source": "Извор", "mode_assistant": "Асистентот дава чекори за поправка.", "mode_info": "Само совети и информации.",
        "search": "Барај алат", "done": "Браво! Поправката е завршена. ✅", "listen": "Слушни ги чекорите",
        "vision": "Анализа на фотографијата (vision модел)", "brand": "Марка", "model_": "Модел", "ocr": "Текст прочитан од фотографијата (OCR)",
        "damage": "Оштетување", "manual_found": "Пронајден прирачник во библиотеката", "manual_link": "Барај го официјалниот прирачник за овој модел", "no_key": "Без API клуч: фотографијата не се анализира со vision модел; препознавањето е од текстот.",
        "agg": "Спојување на резултатите", "text_src": "Текст (TF-IDF)", "photo_src": "Фотографија", "combined": "Резултат",
        "man_intro": "Прикачи PDF прирачник за уредот (опционално) и постави прашање. Пребарувањето е во упатствата на апликацијата и во прирачниците што ги прикачуваш.",
        "man_upload": "Прирачници (PDF, TXT, MD)", "man_q": "Твоето прашање", "man_answer": "Одговор од AI", "man_passages": "Пронајдени делови од прирачниците",
        "man_none": "Ништо поврзано не е пронајдено. Пробај со други зборови или прикачи прирачник.",
        "settings": "Поставки за AI", "key": "Anthropic API клуч (опционално)", "method": "Метод на спојување",
        "methods": {"weighted": "Пондериран просек", "product": "Производ (Naive Bayes)", "vote": "Мнозинско гласање"},
        "ai_on": "Vision + LLM: активно", "ai_off": "Vision + LLM: неактивно (само локални алгоритми)", "error": "Грешка од AI",
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
OBJECTS = list(DIAG["objects"])
PRE = DIAG["pre_questions"]


def is_unsafe(text):
    """Safety gate: rule-based check that runs before any other AI step."""
    words = re.findall(r"\w+", text.lower())
    return any(w.startswith(k) for w in words for k in DIAG["unsafe_keywords"])


def field(item, name, lang):
    return item.get(f"{name}_{lang}") or item.get(name, "")


def tools_for(plan, lang):
    ids = list(dict.fromkeys(t for a in plan for t in a.get("tools", [])))
    return [TOOLS[i]["name"][lang] + f" ({TOOLS[i]['en']})" for i in ids if i in TOOLS]


def reset():
    for k in ("stage", "obj", "node", "path", "text", "pre", "tools_ok", "vision", "agg", "ai_error"):
        st.session_state.pop(k, None)


@st.cache_data(show_spinner=False)
def speech_mp3(text, lang):
    return tts.mp3(text, lang)


def speak(text, lang, label):
    """gTTS audio when available, otherwise the browser's own speech engine."""
    audio = speech_mp3(text, lang)
    if audio:
        st.audio(audio, format="audio/mp3")
        return
    html = tts.browser_speech_html(text, lang, label)  # built from our own step texts only
    if hasattr(st, "iframe"):
        st.iframe(html, height=50)
    else:
        components.html(html, height=50)


# ---------------------------------------------------------------- sidebar
st.set_page_config(page_title="AI Repair Assistant", page_icon="🔧")
lang = st.sidebar.radio("Gjuha / Јазик", ["sq", "mk"], format_func=lambda x: {"sq": "Shqip", "mk": "Македонски"}[x])
U = UI[lang]
st.sidebar.subheader(U["settings"])
api_key = st.sidebar.text_input(U["key"], type="password") or None
method = st.sidebar.selectbox(U["method"], list(U["methods"]), format_func=lambda m: U["methods"][m])
AI_ON = llm.available(api_key)
st.sidebar.caption(U["ai_on"] if AI_ON else U["ai_off"])

st.title("🔧 AI Repair Assistant")
st.caption(U["subtitle"])
tab_ai, tab_man, tab_tips, tab_cats, tab_tools, tab_how = st.tabs(U["tabs"])

# ---------------------------------------------------------------- assistant
with tab_ai:
    ss = st.session_state
    ss.setdefault("stage", "describe")

    if ss.stage == "describe":
        with st.chat_message("assistant"):
            st.write(U["describe"])
        photo = st.file_uploader(U["photo"], type=["jpg", "jpeg", "png", "webp"])
        if photo and not AI_ON and not photo_classifier.available():
            st.caption(U["no_key"])
        text = st.chat_input(U["placeholder"])
        if text:
            ss.text, ss.path, ss.pre, ss.tools_ok = text, [], 0, None
            photo_pred, ss.vision, ss.ai_error = None, None, None
            if photo is not None:
                if AI_ON:
                    try:
                        ss.vision = llm.analyze_photo(photo.getvalue(), photo.type, text, lang, api_key)
                        if ss.vision["object"] != "other":
                            photo_pred = (ss.vision["object"], ss.vision["confidence"])
                    except Exception as e:  # noqa: BLE001 - show the error, keep the local pipeline working
                        ss.ai_error = str(e)
                if photo_pred is None:
                    obj, conf = photo_classifier.predict(photo)
                    photo_pred = (obj, conf) if obj in OBJECTS else None
            unsafe = is_unsafe(text) or bool(ss.vision and ss.vision["involves_electricity_gas_or_water"])
            if unsafe:
                ss.stage = "unsafe"
            else:
                text_scores = recognizer().scores(text)
                best, dist, used = aggregate.combine(OBJECTS, text_scores, photo_pred, method)
                ss.agg = {"text": aggregate.normalize(text_scores) if max(text_scores.values()) > 0 else None,
                          "photo": photo_pred, "combined": dist, "used": used}
                confident = best is not None and dist[best] >= 0.35
                ss.obj = best if confident else None
                ss.node = DIAG["objects"][best]["tree"] if confident else None
                ss.stage = "diagnose" if confident else "pick"
            st.rerun()

    if "text" in ss:
        with st.chat_message("user"):
            st.write(ss.text)
    if ss.get("ai_error"):
        st.warning(f"{U['error']}: {ss.ai_error}")

    if ss.stage == "unsafe":
        with st.chat_message("assistant"):
            st.error(DIAG["unsafe_message"][lang])
        st.button(U["restart"], on_click=reset)

    elif ss.stage == "pick":
        with st.chat_message("assistant"):
            st.write(U["unknown"])
            for oid, obj in DIAG["objects"].items():
                if st.button(obj["name"][lang], key=f"pick_{oid}"):
                    ss.obj, ss.node, ss.stage = oid, obj["tree"], "diagnose"
                    st.rerun()

    elif ss.stage == "diagnose":
        obj = DIAG["objects"][ss.obj]
        with st.chat_message("assistant"):
            st.write(f"{U['object']}: **{obj['name'][lang]}**")
            with st.expander(U["how"]):
                v = ss.get("vision")
                if v:
                    st.write(f"**{U['vision']}**")
                    st.write(f"{U['brand']}: {v['brand'] or '–'} · {U['model_']}: {v['model'] or '–'}")
                    st.write(f"{U['ocr']}: {v['ocr_text'] or '–'}")
                    st.write(f"{U['damage']}: {v['damage'] or '–'}")
                    lib = manuals.library_chunks()
                    found = manuals.match_manuals(lib, v["ocr_text"], v["brand"], v["model"])
                    if found:
                        st.success(f"📘 {U['manual_found']}: {', '.join(found)}")
                        own = [c for c in lib if c.get("file") in found]
                        for p in manuals.ManualIndex(own).search(ss.text + " " + obj["name"]["sq"], k=2, min_score=0.0):
                            st.caption(p["source"])
                            st.write(p["text"])
                    url = manuals.manufacturer_search_url(v["brand"], v["model"])
                    if url:
                        st.link_button(U["manual_link"], url)
                agg = ss.get("agg")
                if agg and agg["combined"]:
                    st.write(f"**{U['agg']}** ({U['methods'][method]})")
                    rows = []
                    for o in OBJECTS:
                        rows.append({U["object"]: DIAG["objects"][o]["name"][lang],
                                     U["text_src"]: round(agg["text"][o], 2) if agg["text"] else None,
                                     U["photo_src"]: (round(agg["photo"][1], 2) if agg["photo"] and agg["photo"][0] == o else None),
                                     U["combined"]: round(agg["combined"][o], 2)})
                    st.dataframe(rows, hide_index=True)
            for q, a in ss.path:
                st.write(f"• {q[lang]} → *{a[lang]}*")

            node = ss.node
            if ss.pre < len(PRE):  # general questions first: safety, broken parts
                pq = PRE[ss.pre]
                st.write(f"**{pq['q'][lang]}**")
                c1, c2 = st.columns(2)
                if c1.button(DIAG["yes_no"][0][lang], key=f"pre_{ss.pre}_yes"):
                    ss.path.append((pq["q"], DIAG["yes_no"][0]))
                    if pq["yes"].get("unsafe"):
                        ss.stage = "unsafe"
                    else:
                        ss.node, ss.pre = pq["yes"], len(PRE)
                    st.rerun()
                if c2.button(DIAG["yes_no"][1][lang], key=f"pre_{ss.pre}_no"):
                    ss.path.append((pq["q"], DIAG["yes_no"][1]))
                    ss.pre += 1
                    st.rerun()
            elif "q" in node:
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
                tool_names = tools_for(plan, lang)
                if ss.tools_ok is None and tool_names:  # last question: does the user have the tools?
                    st.write(f"**{DIAG['tools_question'][lang]}**")
                    for t in tool_names:
                        st.write(f"- {t}")
                    for i, ans in enumerate(DIAG["tools_answers"]):
                        if st.button(ans[lang], key=f"tools_{i}"):
                            ss.path.append((DIAG["tools_question"], ans))
                            ss.tools_ok = i == 0
                            st.rerun()
                else:
                    st.success(field(sc, "title", lang))
                    if ss.tools_ok is False:
                        st.info(DIAG["tools_missing"][lang])
                    st.write(U["tools"])
                    for t in tool_names:
                        st.write(f"- {t}")
                    st.write(U["steps"])
                    labels = [field(a, "label", lang) for a in plan]
                    done = [st.checkbox(f"{i}. {lab}", key=f"step_{sc['id']}_{i}") for i, lab in enumerate(labels, 1)]
                    if all(done):
                        st.balloons()
                        st.success(U["done"])
                    speak(field(sc, "title", lang) + ". " + ". ".join(labels), lang, U["listen"])
                    warnings = [a for a in sc["actions"] if a.get("safety_msg")]
                    if warnings:
                        st.write(U["dont"])
                        for a in warnings:
                            st.warning(f"{field(a, 'label', lang)}: {field(a, 'safety_msg', lang)}")
                    with st.expander(U["why"]):
                        st.write(U["why_text"].format(nq=len(ss.path), sid=sc["id"], n=len(plan), na=n_astar, nb=n_bfs))
        st.button(U["other"], on_click=reset)

# ---------------------------------------------------------------- manuals (RAG)
with tab_man:
    st.write(U["man_intro"])
    pdfs = st.file_uploader(U["man_upload"], type=["pdf", "txt", "md"], accept_multiple_files=True)
    chunks = manuals.builtin_chunks() + manuals.library_chunks()
    for f in pdfs or []:
        try:
            if f.name.lower().endswith(".pdf"):
                chunks += manuals.pdf_chunks(f.getvalue(), f.name)
            else:
                text = f.getvalue().decode("utf-8", errors="ignore")
                chunks += [{"source": f.name, "lang": "any", "text": p.strip()}
                           for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
        except Exception as e:  # noqa: BLE001
            st.warning(f"{f.name}: {e}")
    question = st.text_input(U["man_q"])
    if question:
        if is_unsafe(question):
            st.error(DIAG["unsafe_message"][lang])
        passages = manuals.ManualIndex(chunks).search(question, k=4)
        if not passages:
            st.info(U["man_none"])
        else:
            if AI_ON:
                try:
                    st.write(f"**{U['man_answer']}**")
                    st.write(llm.answer_from_manual(question, passages, lang, api_key))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"{U['error']}: {e}")
            st.write(f"**{U['man_passages']}**")
            for i, p in enumerate(passages, 1):
                with st.expander(f"[{i}] {p['source']} · {p['score']:.2f}"):
                    st.write(p["text"])

# ---------------------------------------------------------------- tips, categories, tools
with tab_tips:
    safe_only = st.toggle(U["safe_only"])
    for tip in TIPS:
        if safe_only and tip["kind"] == "dangerous_myth":
            continue
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

with tab_how:
    doc = ROOT / "docs" / f"how_ai_works_{lang}.md"
    if doc.exists():
        st.markdown(doc.read_text(encoding="utf-8"))

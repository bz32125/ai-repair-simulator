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
  voice assistant: microphone -> speech-to-text (Whisper / Google) -> same pipeline;
                   questions and steps are read aloud with a male voice (Edge TTS Ilir / Aleksandar)
  manuals: built-in guides + uploaded PDFs -> TF-IDF retrieval -> LLM answer with citations (RAG)
"""
import json
import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import aggregate
import auto_answer
import llm
import manuals
import photo_classifier
import planner
import text_recognizer
import tts
import voice

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
        "why_text": "Pyetjet diagnostikuese ({nq}, prej tyre {na_auto} të përgjigjura vetë nga teksti/foto) e çuan te skenari `{sid}`. Planner-i A* gjeti {n} hapa duke hapur {na} gjendje; BFS do të hapte {nb}. Hapat e rrezikshëm përjashtohen nga plani.",
        "kinds": {"safety": "🛡️ Siguri", "dangerous_myth": "⚠️ Mit i rrezikshëm", "trick": "💡 Truk"},
        "safe_only": "Shfaq vetëm këshillat e sigurta", "myth": "Mos e provo këtë. Shpjegohet vetëm që ta kuptosh pse është i rrezikshëm.",
        "source": "Burimi", "mode_assistant": "Asistenti jep hapa riparimi.", "mode_info": "Vetëm këshilla dhe informacion.",
        "search": "Kërko mjet", "done": "Bravo! Riparimi përfundoi. ✅", "listen": "Dëgjo hapat",
        "vision": "Analiza e fotos (vision model)", "brand": "Marka", "model_": "Modeli", "ocr": "Teksti i lexuar nga foto (OCR)",
        "damage": "Dëmtimi", "manual_add": "📘 Shto manualin (opsionale): PDF, foto e faqes ose tekst, në çdo gjuhë", "manual_paste": "…ose ngjit tekstin e manualit këtu", "from_manual": "📘 Nga manuali yt", "manual_no_ai": "Pa API key: tregoj pjesët më të ngjashme të manualit, por vetëm nëse manuali ka tekst dhe është në të njëjtën gjuhë me pyetjen.", "auto": "✔ nga teksti/foto, pa pyetur", "manual_found": "U gjet manuali në bibliotekë", "manual_link": "Kërko manualin zyrtar të këtij modeli", "no_key": "Pa API key: foto nuk analizohet nga vision model; njohja bëhet nga teksti.",
        "agg": "Bashkimi i rezultateve", "text_src": "Teksti (TF-IDF)", "photo_src": "Foto", "combined": "Rezultati",
        "man_intro": "Ngarko manualin PDF të pajisjes (opsionale) dhe bëj një pyetje. Kërkimi bëhet në udhëzuesit e aplikacionit dhe në manualet që ngarkon.",
        "man_upload": "Manuale (PDF, TXT, MD)", "man_q": "Pyetja jote", "man_answer": "Përgjigja e AI-së", "man_passages": "Pjesët e gjetura në manuale",
        "man_none": "Nuk u gjet asgjë e lidhur. Provo me fjalë të tjera ose ngarko manualin.",
        "settings": "Cilësimet e AI-së", "key": "Anthropic API key (opsionale)", "method": "Metoda e bashkimit",
        "methods": {"weighted": "Mesatare e peshuar", "product": "Prodhim (Naive Bayes)", "vote": "Votim i shumicës"},
        "ai_on": "Vision + LLM: aktiv", "ai_off": "Vision + LLM: joaktiv (vetëm algoritmet lokale)", "error": "Gabim nga AI",
        "voice": "🎙️ Asistenti me zë", "voice_on": "🎙️ Fol me mikrofon", "voice_out": "🔊 Përgjigju me zë (zë burri)", "autoplay": "Lexoji pyetjet me zë automatikisht",
        "openai_key": "OpenAI API key (opsionale, për Whisper)", "stt_whisper": "Njohja e zërit: Whisper (OpenAI)",
        "stt_google": "Njohja e zërit: Google Web Speech (falas, pa key)",
        "mic_describe": "🎙️ Fol: përshkruaje problemin", "mic_answer": "🎙️ Përgjigju me zë",
        "heard": "🎙️ Dëgjova", "not_understood": "Nuk të kuptova mirë. Provo përsëri, ose shtyp një nga butonat.",
        "silence": "Nuk dëgjova asgjë. Fol pak më afër mikrofonit dhe provo përsëri.",
        "stt_error": "Njohja e zërit nuk punoi (duhet internet)", "voice_edge": "🔊 Zëri: {v} (mashkull, Edge TTS)",
        "voice_gtts": "🔊 Edge TTS nuk u arrit; po përdor gTTS (zë femre).",
        "voice_browser": "🔊 Asnjë shërbim zëri nuk u arrit; përdor zërin e browser-it.",
        "say": "Dëgjo", "found_obj": "E gjeta: {obj}.", "steps_intro": "Ja hapat.", "q_tools_say": "A i ke këto mjete: {t}?",
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
        "why_text": "Дијагностичките прашања ({nq}, од нив {na_auto} одговорени автоматски од текстот/фотографијата) доведоа до сценариото `{sid}`. Планерот A* најде {n} чекори отворајќи {na} состојби; BFS би отворил {nb}. Опасните чекори се исклучени од планот.",
        "kinds": {"safety": "🛡️ Безбедност", "dangerous_myth": "⚠️ Опасен мит", "trick": "💡 Трик"},
        "safe_only": "Прикажи само безбедни совети", "myth": "Не го пробувај ова. Се објаснува само за да разбереш зошто е опасно.",
        "source": "Извор", "mode_assistant": "Асистентот дава чекори за поправка.", "mode_info": "Само совети и информации.",
        "search": "Барај алат", "done": "Браво! Поправката е завршена. ✅", "listen": "Слушни ги чекорите",
        "vision": "Анализа на фотографијата (vision модел)", "brand": "Марка", "model_": "Модел", "ocr": "Текст прочитан од фотографијата (OCR)",
        "damage": "Оштетување", "manual_add": "📘 Додај прирачник (опционално): PDF, фотографија од страница или текст, на кој било јазик", "manual_paste": "…или залепи го текстот на прирачникот тука", "from_manual": "📘 Од твојот прирачник", "manual_no_ai": "Без API клуч: ги прикажувам најсличните делови од прирачникот, но само ако има текст и е на истиот јазик како прашањето.", "auto": "✔ од текстот/фотографијата, без прашање", "manual_found": "Пронајден прирачник во библиотеката", "manual_link": "Барај го официјалниот прирачник за овој модел", "no_key": "Без API клуч: фотографијата не се анализира со vision модел; препознавањето е од текстот.",
        "agg": "Спојување на резултатите", "text_src": "Текст (TF-IDF)", "photo_src": "Фотографија", "combined": "Резултат",
        "man_intro": "Прикачи PDF прирачник за уредот (опционално) и постави прашање. Пребарувањето е во упатствата на апликацијата и во прирачниците што ги прикачуваш.",
        "man_upload": "Прирачници (PDF, TXT, MD)", "man_q": "Твоето прашање", "man_answer": "Одговор од AI", "man_passages": "Пронајдени делови од прирачниците",
        "man_none": "Ништо поврзано не е пронајдено. Пробај со други зборови или прикачи прирачник.",
        "settings": "Поставки за AI", "key": "Anthropic API клуч (опционално)", "method": "Метод на спојување",
        "methods": {"weighted": "Пондериран просек", "product": "Производ (Naive Bayes)", "vote": "Мнозинско гласање"},
        "ai_on": "Vision + LLM: активно", "ai_off": "Vision + LLM: неактивно (само локални алгоритми)", "error": "Грешка од AI",
        "voice": "🎙️ Гласовен асистент", "voice_on": "🎙️ Зборувај преку микрофон", "voice_out": "🔊 Одговори со глас (машки глас)", "autoplay": "Читај ги прашањата гласно автоматски",
        "openai_key": "OpenAI API клуч (опционално, за Whisper)", "stt_whisper": "Препознавање говор: Whisper (OpenAI)",
        "stt_google": "Препознавање говор: Google Web Speech (бесплатно, без клуч)",
        "mic_describe": "🎙️ Зборувај: опиши го проблемот", "mic_answer": "🎙️ Одговори со глас",
        "heard": "🎙️ Слушнав", "not_understood": "Не те разбрав добро. Пробај повторно или притисни едно од копчињата.",
        "silence": "Не слушнав ништо. Зборувај поблиску до микрофонот и пробај повторно.",
        "stt_error": "Препознавањето говор не работеше (потребен е интернет)", "voice_edge": "🔊 Глас: {v} (машки, Edge TTS)",
        "voice_gtts": "🔊 Edge TTS не е достапен; нема резервен глас за македонски.",
        "voice_browser": "🔊 Ниеден говорен сервис не е достапен; се користи гласот на прелистувачот.",
        "say": "Слушни", "found_obj": "Го најдов: {obj}.", "steps_intro": "Еве ги чекорите.", "q_tools_say": "Дали ги имаш овие алати: {t}?",
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


def show_manual_guidance(manual, problem, sc, lang):
    """LLM reads the manual in any language; without a key, TF-IDF retrieval on its text."""
    if AI_ON:
        key = ("manual", problem, sc["id"], lang, tuple(m["name"] for m in manual))
        if key not in st.session_state:
            try:
                st.session_state[key] = llm.manual_guidance(manual, problem, field(sc, "title", lang), lang, api_key)
            except Exception as e:  # noqa: BLE001
                st.warning(f"{U['error']}: {e}")
                return
        st.write(st.session_state[key])
        return
    st.caption(U["manual_no_ai"])
    chunks = []
    for m in manual:
        if m["kind"] == "pdf":
            chunks += manuals.pdf_chunks(m["data"], m["name"])
        elif m["kind"] == "text":
            chunks += [{"source": m["name"], "text": p.strip()} for p in re.split(r"\n\s*\n", m["data"]) if len(p.strip()) > 40]
    if chunks:
        query = " ".join([problem, sc.get("title_sq", ""), sc.get("title_mk", ""), sc.get("title", "")]
                         + [a.get("label", "") for a in sc["actions"]])
        for p in manuals.ManualIndex(chunks).search(query, k=3, min_score=0.0):
            with st.expander(p["source"]):
                st.write(p["text"])


def reset():
    for k in ("stage", "obj", "node", "path", "text", "pre", "tools_ok", "vision", "agg", "ai_error", "manual", "photo", "llm_answers",
              "heard", "voice_miss", "spoken"):
        st.session_state.pop(k, None)


@st.cache_data(show_spinner=False)
def _speech_cached(text, lang):
    audio, engine = tts.synthesize(text, lang, "male")
    if audio is None:
        raise RuntimeError("no TTS engine")  # exceptions are not cached, so we retry next time
    return audio, engine


def speech_mp3(text, lang):
    try:
        return _speech_cached(text, lang)
    except RuntimeError:
        return None, None


def speak(text, lang, label, autoplay=False):
    """Male voice (Edge TTS) when available, then gTTS, then the browser's own speech engine.

    Nothing is spoken or shown unless the user turned on voice answers in the sidebar.
    autoplay plays a text only the first time it is shown, so a rerun does not repeat it.
    """
    if not VOICE_OUT:
        return
    audio, engine = speech_mp3(text, lang)
    if audio:
        spoken = st.session_state.setdefault("spoken", set())
        play = autoplay and AUTOPLAY and (lang, text) not in spoken
        spoken.add((lang, text))
        st.audio(audio, format="audio/mp3", autoplay=play)
        st.session_state.voice_engine = engine
        return
    st.session_state.voice_engine = "browser"
    html = tts.browser_speech_html(text, lang, label)  # built from our own texts and the user's own words only
    if hasattr(st, "iframe"):
        st.iframe(html, height=50)
    else:
        components.html(html, height=50)


def listen(label, key):
    """Microphone widget -> text. Returns the new transcript once per recording, else None."""
    if not VOICE_ON or not hasattr(st, "audio_input"):
        return None
    rec = st.audio_input(label, key=key)
    if rec is None:
        return None
    wav = rec.getvalue()
    done = st.session_state.setdefault("heard_ids", set())
    rid = (key, hash(wav))
    if rid in done:
        return None
    done.add(rid)
    try:
        with st.spinner("🎙️ …"):
            text, _ = voice.transcribe(wav, lang, openai_key)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{U['stt_error']}: {e}")
        return None
    if not text:
        st.session_state.voice_miss = U["silence"]
        return None
    return text


def voice_pick(key, options, yes_no=False):
    """Mic for one clarifying question: index of the option the user said, or None."""
    heard = listen(U["mic_answer"], key)
    if heard is None:
        return None
    st.session_state.heard = heard
    i = voice.match_option(heard, options, lang, yes_no)
    st.session_state.voice_miss = None if i is not None else f"«{heard}». {U['not_understood']}"
    return i


def voice_feedback():
    """Show (and say) what the microphone heard, or that it did not understand."""
    ss_ = st.session_state
    if ss_.get("voice_miss"):
        st.info(ss_.voice_miss)
        speak(U["not_understood"], lang, U["say"], autoplay=True)
        ss_.voice_miss = None
    elif ss_.get("heard"):
        st.caption(f"{U['heard']}: «{ss_.heard}»")


# ---------------------------------------------------------------- sidebar
st.set_page_config(page_title="AI Repair Assistant", page_icon="🔧")
lang = st.sidebar.radio("Gjuha / Јазик", ["sq", "mk"], format_func=lambda x: {"sq": "Shqip", "mk": "Македонски"}[x])
U = UI[lang]
st.sidebar.subheader(U["settings"])
api_key = st.sidebar.text_input(U["key"], type="password") or None
method = st.sidebar.selectbox(U["method"], list(U["methods"]), format_func=lambda m: U["methods"][m])
AI_ON = llm.available(api_key)
st.sidebar.caption(U["ai_on"] if AI_ON else U["ai_off"])
st.sidebar.subheader(U["voice"])
VOICE_ON = st.sidebar.toggle(U["voice_on"], value=True)  # microphone input
VOICE_OUT = st.sidebar.toggle(U["voice_out"], value=False)  # spoken answers only when the user chooses them
AUTOPLAY = st.sidebar.toggle(U["autoplay"], value=True, disabled=not VOICE_OUT)
openai_key = (st.sidebar.text_input(U["openai_key"], type="password", disabled=not VOICE_ON)
              or os.environ.get("OPENAI_API_KEY") or None)
if VOICE_ON:
    st.sidebar.caption(U["stt_whisper"] if openai_key else U["stt_google"])
if VOICE_OUT:
    eng = st.session_state.get("voice_engine")
    if eng == "gtts":
        st.sidebar.caption(U["voice_gtts"])
    elif eng == "browser":
        st.sidebar.caption(U["voice_browser"])
    else:
        st.sidebar.caption(U["voice_edge"].format(v=tts.edge_voice(lang).split("-")[2].replace("Neural", "")))

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
        manual_files = st.file_uploader(U["manual_add"], type=["pdf", "jpg", "jpeg", "png", "webp", "txt", "md"],
                                        accept_multiple_files=True)
        manual_text = st.text_area(U["manual_paste"], height=80)
        if photo and not AI_ON and not photo_classifier.available():
            st.caption(U["no_key"])
        text = st.chat_input(U["placeholder"])
        spoken_text = listen(U["mic_describe"], "mic_describe")
        if spoken_text:
            text, ss.heard = spoken_text, spoken_text
        voice_feedback()
        if text:
            ss.text, ss.path, ss.pre, ss.tools_ok = text, [], 0, None
            ss.manual = []
            for f in manual_files or []:
                kind = "pdf" if f.type == "application/pdf" else "image" if f.type.startswith("image/") else "text"
                data = f.getvalue() if kind != "text" else f.getvalue().decode("utf-8", errors="ignore")
                ss.manual.append({"kind": kind, "data": data, "media_type": f.type, "name": f.name})
            if manual_text.strip():
                ss.manual.append({"kind": "text", "data": manual_text, "media_type": "text/plain", "name": "tekst"})
            photo_pred, ss.vision, ss.ai_error, ss.llm_answers = None, None, None, None
            ss.photo = (photo.getvalue(), photo.type) if photo is not None else None
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
            speak(DIAG["unsafe_message"][lang], lang, U["say"], autoplay=True)
        st.button(U["restart"], on_click=reset)

    elif ss.stage == "pick":
        with st.chat_message("assistant"):
            st.write(U["unknown"])
            names = [o["name"][lang] for o in DIAG["objects"].values()]
            speak(U["unknown"] + " " + ", ".join(names) + "?", lang, U["say"], autoplay=True)
            picked = None
            for i, (oid, obj) in enumerate(DIAG["objects"].items()):
                if st.button(obj["name"][lang], key=f"pick_{oid}"):
                    picked = i
            vi = voice_pick("mic_pick", names)
            voice_feedback()
            if picked is None:
                picked = vi
            if picked is not None:
                oid = OBJECTS[picked]
                ss.obj, ss.node, ss.stage = oid, DIAG["objects"][oid]["tree"], "diagnose"
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
            # answer automatically every question the text/photo/manual already answers
            evidence = ss.text + " " + ((ss.get("vision") or {}).get("damage") or "")
            if AI_ON and ss.get("llm_answers") is None and (ss.get("photo") or ss.get("manual")):
                qs = [{"id": pq["id"], "question": pq["q"]["sq"], "options": [y["sq"] for y in DIAG["yes_no"]]} for pq in PRE]
                qs += [{"id": n["q"]["sq"], "question": n["q"]["sq"], "options": [a["text"]["sq"] for a in n["answers"]]}
                       for n in auto_answer.question_nodes(obj["tree"])]
                try:
                    with st.spinner("…"):
                        ss.llm_answers = llm.answer_questions(qs, ss.text, ss.get("photo"), ss.get("manual"), api_key)
                except Exception as e:  # noqa: BLE001
                    ss.llm_answers = {}
                    st.warning(f"{U['error']}: {e}")
            llm_ans = ss.get("llm_answers") or {}
            while True:
                if ss.pre < len(PRE):
                    pq = PRE[ss.pre]
                    if pq["id"] in llm_ans:
                        guess = "yes" if llm_ans[pq["id"]] == 0 else "no"
                    else:
                        guess = (auto_answer.safety(ss.get("vision")) if pq["id"] == "safety"
                                 else auto_answer.broken(ss.text, ss.get("vision")))
                    if guess is None:
                        break
                    if guess == "yes" and pq["yes"].get("unsafe"):
                        ss.stage = "unsafe"
                        st.rerun()
                    ss.path.append((pq["q"], DIAG["yes_no"][0 if guess == "yes" else 1], "auto"))
                    if guess == "yes":
                        ss.node, ss.pre = pq["yes"], len(PRE)
                    else:
                        ss.pre += 1
                elif "q" in ss.node:
                    i = llm_ans.get(ss.node["q"]["sq"])
                    if i is None:
                        i = auto_answer.choose_answer(evidence, ss.node["answers"], lang)
                    if i is None:
                        break
                    ss.path.append((ss.node["q"], ss.node["answers"][i]["text"], "auto"))
                    ss.node = ss.node["answers"][i]["next"]
                else:
                    break

            for q, a, src in ss.path:
                tag = f" _({U['auto']})_" if src == "auto" else ""
                st.write(f"• {q[lang]} → *{a[lang]}*{tag}")
            voice_feedback()

            # the first spoken question also says which object was recognized
            intro = "" if any(p[2] is None for p in ss.path) else U["found_obj"].format(obj=obj["name"][lang]) + " "
            node = ss.node
            if ss.pre < len(PRE):  # general questions first: safety, broken parts
                pq = PRE[ss.pre]
                st.write(f"**{pq['q'][lang]}**")
                speak(intro + pq["q"][lang], lang, U["say"], autoplay=True)
                c1, c2 = st.columns(2)
                choice = 0 if c1.button(DIAG["yes_no"][0][lang], key=f"pre_{ss.pre}_yes") else None
                if c2.button(DIAG["yes_no"][1][lang], key=f"pre_{ss.pre}_no"):
                    choice = 1
                if choice is None:
                    choice = voice_pick(f"mic_pre_{ss.pre}", [y[lang] for y in DIAG["yes_no"]], yes_no=True)
                if choice == 0:
                    ss.path.append((pq["q"], DIAG["yes_no"][0], None))
                    if pq["yes"].get("unsafe"):
                        ss.stage = "unsafe"
                    else:
                        ss.node, ss.pre = pq["yes"], len(PRE)
                    st.rerun()
                if choice == 1:
                    ss.path.append((pq["q"], DIAG["yes_no"][1], None))
                    ss.pre += 1
                    st.rerun()
            elif "q" in node:
                st.write(f"**{node['q'][lang]}**")
                opts = [a["text"][lang] for a in node["answers"]]
                speak(intro + node["q"][lang] + " " + "; ".join(opts) + "?", lang, U["say"], autoplay=True)
                choice = None
                for i, ans in enumerate(node["answers"]):
                    if st.button(ans["text"][lang], key=f"ans_{len(ss.path)}_{i}"):
                        choice = i
                if choice is None:
                    choice = voice_pick(f"mic_ans_{len(ss.path)}", opts)
                if choice is not None:
                    ans = node["answers"][choice]
                    ss.path.append((node["q"], ans["text"], None))
                    ss.node = ans["next"]
                    st.rerun()
            elif "refer" in node:
                st.info(node["refer"][lang])
                speak(node["refer"][lang], lang, U["say"], autoplay=True)
            else:
                sc = SCENARIOS[node["scenario"]]
                plan, n_astar = planner.astar(sc)
                _, n_bfs = planner.bfs(sc)
                tool_names = tools_for(plan, lang)
                if ss.tools_ok is None and tool_names:  # last question: does the user have the tools?
                    st.write(f"**{DIAG['tools_question'][lang]}**")
                    for t in tool_names:
                        st.write(f"- {t}")
                    short = ", ".join(t.split(" (")[0] for t in tool_names)
                    speak(intro + U["q_tools_say"].format(t=short), lang, U["say"], autoplay=True)
                    choice = None
                    for i, ans in enumerate(DIAG["tools_answers"]):
                        if st.button(ans[lang], key=f"tools_{i}"):
                            choice = i
                    if choice is None:
                        choice = voice_pick("mic_tools", [a[lang] for a in DIAG["tools_answers"]], yes_no=True)
                    if choice is not None:
                        ss.path.append((DIAG["tools_question"], DIAG["tools_answers"][choice], None))
                        ss.tools_ok = choice == 0
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
                    numbered = " ".join(f"{i}. {lab}." for i, lab in enumerate(labels, 1))
                    speak(f"{field(sc, 'title', lang)}. {U['steps_intro']} {numbered}", lang, U["listen"], autoplay=True)
                    warnings = [a for a in sc["actions"] if a.get("safety_msg")]
                    if warnings:
                        st.write(U["dont"])
                        for a in warnings:
                            st.warning(f"{field(a, 'label', lang)}: {field(a, 'safety_msg', lang)}")
                    if ss.get("manual"):
                        st.subheader(U["from_manual"])
                        show_manual_guidance(ss.manual, ss.text, sc, lang)
                    with st.expander(U["why"]):
                        st.write(U["why_text"].format(nq=len(ss.path), na_auto=sum(1 for p in ss.path if p[2] == "auto"), sid=sc["id"], n=len(plan), na=n_astar, nb=n_bfs))
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

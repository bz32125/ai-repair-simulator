## Si punon AI-ja e AI Repair Assistant

Asistenti nuk është një chatbot i vetëm. Është një **pipeline** (varg hapash), ku çdo hap përdor një teknikë tjetër të AI-së dhe mund të shpjegohet e testohet veç e veç.

| # | Hapi | Teknika e AI-së | Input → Output |
|---|---|---|---|
| 1 | Porta e sigurisë | **Sistem me rregulla** (rule-based) | teksti + analiza e fotos → "e sigurt" / "thirr profesionist" |
| 2 | Njohja nga teksti | **TF-IDF + cosine similarity** (e shkruar nga zeroja) | përshkrimi → probabilitet për çdo objekt |
| 3 | Njohja nga fotoja | **Vision model (LLM multimodal)** ose **CNN me transfer learning** (MobileNetV3) | fotoja → objekti, besueshmëria, marka/modeli, teksti i lexuar (**OCR**), dëmtimi |
| 4 | Bashkimi i rezultateve | **Aggregation**: mesatare e peshuar, prodhim (Naive Bayes), votim i shumicës | dy shpërndarje → një vendim |
| 5 | Pyetjet diagnostikuese | **Pemë vendimi** (decision tree), 3–5 pyetje | përgjigjet me butona → skenari i saktë |
| 6 | Mjetet | **Rregulla** | veprimet e planit → lista e mjeteve |
| 7 | Hapat e riparimit | **Kërkim në hapësirën e gjendjeve: BFS dhe A\*** | gjendja fillestare + qëllimi → plani më i shkurtër dhe i sigurt |
| 8 | Manualet | **RAG** (Retrieval-Augmented Generation): TF-IDF për kërkim + LLM për përgjigje | pyetja + manualet PDF → përgjigje me burime [1], [2] |
| 9 | Zëri (asistenti me zë) | **Speech-to-text** (Google / Whisper) + **text-to-speech** me zë burri (Edge TTS, neural) | zëri → tekst; pyetjet dhe hapat → audio |

### 1. Porta e sigurisë
Para çdo hapi tjetër kontrollohet nëse problemi lidhet me rrymë, gaz ose ujë: me fjalë kyçe në shqip dhe maqedonisht, dhe me fushën `involves_electricity_gas_or_water` që kthen vision model-i. Nëse po, asistenti **refuzon** dhe e drejton te profesionisti. Kjo pjesë është me rregulla dhe jo me ML, sepse këtu duam siguri 100%, jo "zakonisht të saktë".

### 2. TF-IDF (njohja nga teksti)
- Teksti normalizohet (ë→e, ç→c) dhe çdo fjalë shkurtohet në 5 shkronjat e para (*stemming*), që "dollapi", "dollapit" të jenë e njëjta fjalë.
- **TF** (term frequency): sa shpesh del fjala te përshkrimet e një objekti. **IDF** (inverse document frequency): sa e rrallë është te objektet e tjera. Fjala "sirtar" ka peshë të madhe, fjala "nuk" pothuajse zero.
- Përshkrimi i përdoruesit kthehet në vektor dhe krahasohet me vektorin e çdo objekti me **ngjashmëri kosinusi**.
- Rezultati: **98%** saktësi me leave-one-out në 40 fjali (shqip + maqedonisht), kundrejt **88%** për baseline-in me fjalë kyçe.

### 3. Vision model dhe OCR
Fotoja i dërgohet një modeli multimodal (Claude) me një **skemë JSON të detyrueshme** (structured output), që përgjigja të jetë gjithmonë e lexueshme nga kodi:
`object`, `confidence`, `brand`, `model`, `ocr_text`, `damage`, `involves_electricity_gas_or_water`.
- **OCR**: modeli lexon tekstin në etiketa, ngjitëse ose logo (p.sh. emrin e modelit të karriges). Nuk lejohet të hamendësojë: nëse nuk shihet, kthen bosh.
- **Identifikimi i modelit**: nëse dalin marka/modeli, aplikacioni krijon kërkimin për manualin zyrtar.
- Pa API key, fotoja përdoret nga CNN-ja jonë (transfer learning) kur të trajnohet me dataset-in tonë; deri atëherë njohja bëhet nga teksti.

### 4. Aggregation
Teksti dhe fotoja japin secili një shpërndarje probabilitetesh mbi objektet. Ato bashkohen me:
- **Mesatare e peshuar**: `P = 0.6·P_foto + 0.4·P_tekst`
- **Prodhim (Naive Bayes)**: `P ∝ P_foto · P_tekst`, duke supozuar që burimet janë të pavarura; shpërblen kur të dy pajtohen.
- **Votim i shumicës**: secili burim voton objektin e vet më të mirë.
Nëse probabiliteti më i lartë është nën 0.35, asistenti nuk rrezikon: e pyet përdoruesin cili objekt është.

### 5. Pema e vendimit (3–5 pyetje)
1. A lidhet me rrymë, gaz ose ujë? (siguri)
2. A ka pjesë të thyera ose që mungojnë? (ndërrim pjese, jo riparim)
3–4. Pyetje specifike për objektin (p.sh. "Si lidhet këmba: me vida apo e ngjitur?")
5. A i ke mjetet e duhura?
Pyetjet bëhen me **butona**, sepse çdo përgjigje duhet të çojë saktë te një degë e pemës.

### 6–7. Planifikimi me BFS dhe A\*
Çdo riparim është i përshkruar si problem kërkimi (stil STRIPS): **gjendje** (fakte si `bolts_loose`), **veprime** me *parakushte* dhe *efekte*, dhe **qëllim**. Veprimet e rrezikshme (p.sh. "godite me çekiç") nuk zgjidhen kurrë.
- **BFS** gjen planin më të shkurtër, por hap shumë gjendje.
- **A\*** përdor heuristikën `h = numri i fakteve të qëllimit që mungojnë`. Është e **pranueshme** (admissible), sepse çdo veprim shton të shumtën një fakt qëllimi, prandaj plani mbetet optimal. Te dera e dollapit: **13 gjendje me A\*, 38 me BFS**, i njëjti plan me 6 hapa.

### 8. RAG për manualet
1. **Retrieval**: udhëzuesit e aplikacionit dhe manualet PDF që ngarkon përdoruesi priten në pjesë (~900 karaktere) dhe indeksohen me TF-IDF. Pyetja gjen 3–4 pjesët më të ngjashme.
2. **Generation**: LLM-ja shkruan përgjigjen **vetëm** nga ato pjesë dhe i citon [1], [2]. Nëse pjesët nuk e përmbajnë përgjigjen, e thotë hapur.
Kështu përgjigjet bazohen te manuali i vërtetë i pajisjes, jo te "kujtesa" e modelit.

**Manuali i përdoruesit, në çdo gjuhë:** te Asistenti, poshtë fotos së objektit, përdoruesi mund të shtojë manualin si PDF, si foto të faqes ose si tekst. LLM-ja multimodale e lexon direkt (edhe PDF të skanuar ose foto), kupton çdo gjuhë (anglisht, gjermanisht, suedisht…) dhe nxjerr vetëm hapat, kodet e pjesëve dhe paralajmërimet që lidhen me riparimin, **të përkthyera** në shqip/maqedonisht dhe me vendin në manual (faqja, hapi). Pa API key, sistemi bën kërkim TF-IDF në tekstin e manualit, që punon vetëm kur manuali dhe pyetja janë në të njëjtën gjuhë.

**Pyetjet që AI-ja i di vetë:** para çdo pyetjeje, sistemi kontrollon nëse përgjigja është tashmë te teksti ose te fotoja. Nëse po, përgjigjet vetë dhe e shënon me "✔ nga teksti/foto", që përdoruesi ta kontrollojë.

### 9. Asistenti me zë
**Hyrja me zë (speech-to-text, nga zëri në tekst):** përdoruesi shtyp mikrofonin dhe flet. Regjistrimi (WAV) kthehet në tekst:
- pa asnjë key: **Google Web Speech** (falas, gjuha `sq-AL`), përmes paketës SpeechRecognition;
- nëse jepet OpenAI API key: **Whisper** (`gpt-4o-transcribe`, rezervë `whisper-1`), që e njeh shqipen më mirë. Nëse Whisper dështon, kalon vetë te Google.
Teksti futet në të njëjtën rrjedhë si teksti i shkruar (porta e sigurisë, TF-IDF, pema e vendimit).

**Përgjigjet me zë te pyetjet:** fjalia e folur lidhet me një nga butonat: "po"/"jo" për pyetjet po/jo, numrat rendorë ("e dyta", "2"), ose mbivendosja e fjalëve me tekstin e butonit (të njëjtat rrënjë si te TF-IDF). Nëse nuk është e qartë cilin buton do, asistenti **nuk hamendëson**: thotë "Nuk të kuptova" dhe pyet përsëri, që pema e vendimit të mbetet e saktë.

**Dalja me zë (text-to-speech), zë burri:** çdo pyetje sqaruese dhe hapat përfundimtarë lexohen automatikisht me zërin neural **Ilir** (`sq-AL-IlirNeural`, Microsoft Edge TTS, falas). Për maqedonisht: **Aleksandar** (`mk-MK-AleksandarNeural`). Këta janë të vetmit zëra neuralë meshkuj që ekzistojnë për këto gjuhë. Nëse shërbimi nuk arrihet, rezerva është gTTS (zë femre, vetëm shqip) dhe pastaj zëri i browser-it.

**Kufizimet (thënë hapur):** të dy shërbimet e zërit kërkojnë internet; Google Web Speech është një shërbim falas jo-zyrtar dhe mund të gabojë me dialektet ose zhurmën; Ilir tingëllon i pastër për fjali të thjeshta, por ndonjëherë thekson gabim emrat e huaj të mjeteve.

### Pse kështu dhe jo vetëm një LLM?
- Çdo hap **shpjegohet** dhe **matet** veç e veç (98% TF-IDF, 13 vs 38 gjendje te A\*).
- **Siguria** nuk varet nga një model që mund të gabojë.
- Aplikacioni **punon edhe pa internet/API**, vetëm me algoritmet lokale; LLM-ja shton foto, OCR dhe përgjigje nga manualet.

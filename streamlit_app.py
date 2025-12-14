# streamlit_app.py
import os
import html as _html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

load_dotenv()

DISCLAIMER = "Disclaimer: This is not medical advice. If you're worried or symptoms are severe, seek professional care."

CHROMA_DIR = "assets/chroma"
COLLECTION = "medquad"
META_PATH = "assets/medquad_meta.parquet"
TOP_K = 4

# -------------------------
# Safety + triage rules
# -------------------------
EMERGENCY_TRIGGERS = [
    "chest pain", "pressure in chest", "can't breathe", "cannot breathe", "severe shortness of breath",
    "blue lips", "seizure", "unconscious", "fainted", "stroke", "face droop", "slurred speech",
    "severe bleeding", "anaphylaxis", "throat closing"
]

def classify_complaint(text: str) -> str:
    t = (text or "").lower()
    if any(x in t for x in ["chest pain", "pressure in chest", "shortness of breath", "can't breathe", "cannot breathe"]):
        return "cardio_respiratory"
    if any(x in t for x in ["cough", "sore throat", "runny nose", "congestion", "fever", "flu"]):
        return "respiratory"
    if any(x in t for x in ["abdominal", "stomach", "diarrhea", "vomit", "nausea"]):
        return "gi"
    if any(x in t for x in ["headache", "dizzy", "faint", "numb", "weakness", "confusion"]):
        return "neuro"
    if any(x in t for x in ["rash", "hives", "itch", "swelling"]):
        return "skin_allergy"
    if any(x in t for x in ["burning urination", "uti", "frequent urination", "blood in urine"]):
        return "urinary"
    if any(x in t for x in ["injury", "cut", "bleeding", "fracture", "burn"]):
        return "trauma"
    return "general"

FLOWS = {
    "respiratory": [
        {"key":"age", "q":"Age group?", "opts":["<1 year","1–12","13–17","18–64","65+"]},
        {"key":"duration", "q":"How long have symptoms been going on?", "opts":["<24 hours","1–3 days","4–7 days",">1 week"]},
        {"key":"fever", "q":"Fever level?", "opts":["None","Low (≤100.4F / 38C)","Moderate","High (≥103F / 39.4C)"]},
        {"key":"breath", "q":"Any trouble breathing?", "opts":["No","Mild","Moderate","Severe / can't breathe"]},
        {"key":"chest_pain", "q":"Any chest pain or tightness?", "opts":["No","Mild","Moderate","Severe"]},
        {"key":"risk", "q":"Any high-risk factors?", "opts":["None","Pregnant","Immunocompromised","Asthma/COPD","Heart disease","Other"]},
    ],
    "gi": [
        {"key":"age", "q":"Age group?", "opts":["<1 year","1–12","13–17","18–64","65+"]},
        {"key":"main", "q":"What is the main problem?", "opts":["Diarrhea","Vomiting","Abdominal pain","Nausea","Mixed"]},
        {"key":"duration", "q":"How long has it been going on?", "opts":["<24 hours","1–2 days","3–7 days",">1 week"]},
        {"key":"blood", "q":"Any blood in stool or vomit?", "opts":["No","Yes"]},
        {"key":"dehydration", "q":"Any dehydration signs (very thirsty, dizzy, little urine)?", "opts":["No","Yes (mild)","Yes (moderate/severe)"]},
        {"key":"pain_level", "q":"How severe is the pain?", "opts":["None","Mild","Moderate","Severe"]},
    ],
    "cardio_respiratory": [
        {"key":"age", "q":"Age group?", "opts":["13–17","18–64","65+"]},
        {"key":"breath", "q":"Shortness of breath?", "opts":["No","Mild","Moderate","Severe / can't breathe"]},
        {"key":"chest_pain", "q":"Chest pain/tightness?", "opts":["No","Mild","Moderate","Severe"]},
        {"key":"onset", "q":"When did it start?", "opts":["Just now","Today","1–3 days",">3 days"]},
        {"key":"neuro", "q":"Any stroke-like symptoms (face droop, weakness, trouble speaking)?", "opts":["No","Yes"]},
    ],
    "neuro": [
        {"key":"age", "q":"Age group?", "opts":["1–12","13–17","18–64","65+"]},
        {"key":"headache", "q":"Is this mainly a headache?", "opts":["No","Yes - mild/moderate","Yes - severe / worst ever"]},
        {"key":"confusion", "q":"Any confusion, fainting, seizure, or new weakness/numbness?", "opts":["No","Yes"]},
        {"key":"fever", "q":"Any fever with stiff neck?", "opts":["No","Yes"]},
    ],
    "skin_allergy": [
        {"key":"age", "q":"Age group?", "opts":["<1 year","1–12","13–17","18–64","65+"]},
        {"key":"swelling", "q":"Any swelling of lips/face/tongue or throat tightness?", "opts":["No","Yes"]},
        {"key":"breath", "q":"Any trouble breathing?", "opts":["No","Yes"]},
        {"key":"rash", "q":"Rash type?", "opts":["Itchy hives","Red patches","Blisters","Other/unsure"]},
    ],
    "urinary": [
        {"key":"age", "q":"Age group?", "opts":["1–12","13–17","18–64","65+"]},
        {"key":"pain", "q":"Burning/pain with urination?", "opts":["No","Yes"]},
        {"key":"fever", "q":"Fever or back/flank pain?", "opts":["No","Yes"]},
        {"key":"blood", "q":"Blood in urine?", "opts":["No","Yes"]},
    ],
    "trauma": [
        {"key":"age", "q":"Age group?", "opts":["<1 year","1–12","13–17","18–64","65+"]},
        {"key":"bleeding", "q":"Is there uncontrolled bleeding?", "opts":["No","Yes"]},
        {"key":"head_injury", "q":"Head injury with confusion/vomiting/fainting?", "opts":["No","Yes"]},
        {"key":"pain_level", "q":"Pain level?", "opts":["Mild","Moderate","Severe"]},
    ],
    "general": [
        {"key":"age", "q":"Age group?", "opts":["<1 year","1–12","13–17","18–64","65+"]},
        {"key":"duration", "q":"How long have symptoms been going on?", "opts":["<24 hours","1–3 days","4–7 days",">1 week"]},
        {"key":"severity", "q":"Overall severity?", "opts":["Mild","Moderate","Severe"]},
        {"key":"redflags", "q":"Any red flags (chest pain, can't breathe, fainting, severe bleeding)?", "opts":["No","Yes"]},
    ],
}

def triage_from_answers(complaint: str, flow: str, answers: dict) -> tuple[str, str]:
    text = (complaint or "").lower()
    if any(x in text for x in EMERGENCY_TRIGGERS):
        return ("EMERGENCY", "Your description includes emergency warning signs. Seek urgent medical care now.")

    if answers.get("breath","").startswith("Severe"):
        return ("EMERGENCY", "Severe breathing difficulty is an emergency warning sign. Seek urgent care now.")
    if answers.get("chest_pain","").startswith("Severe"):
        return ("EMERGENCY", "Severe chest pain can be serious. Seek urgent medical evaluation now.")
    if answers.get("neuro","") == "Yes":
        return ("EMERGENCY", "Possible stroke-like symptoms are an emergency. Seek urgent care now.")
    if answers.get("bleeding","") == "Yes":
        return ("EMERGENCY", "Uncontrolled bleeding can be life-threatening. Seek urgent care now.")
    if answers.get("swelling","") == "Yes" and answers.get("breath","") == "Yes":
        return ("EMERGENCY", "Swelling with breathing trouble may be a severe allergic reaction. Seek urgent care now.")
    if answers.get("head_injury","") == "Yes":
        return ("URGENT", "Head injury symptoms can be serious. Consider same-day urgent evaluation.")

    if answers.get("fever","").startswith("High"):
        return ("URGENT", "High fever can require medical evaluation, especially in children/older adults.")
    if answers.get("dehydration","").startswith("Yes (moderate"):
        return ("URGENT", "Moderate/severe dehydration signs may need medical attention.")
    if answers.get("blood","") == "Yes" and flow in ["gi","urinary"]:
        return ("URGENT", "Blood can be concerning. Consider prompt medical evaluation.")

    return ("ROUTINE", "Symptoms sound non-emergent based on provided information. Monitor and seek care if worsening.")

def self_care_advice(flow: str, triage: str) -> list[str]:
    if triage != "ROUTINE":
        return []

    base = [
        "Rest, stay hydrated, and monitor symptoms.",
        "If symptoms worsen, last longer than expected, or you have new red flags, contact a clinician.",
        "Avoid mixing multiple medicines with the same active ingredient; ask a pharmacist if unsure."
    ]

    if flow == "respiratory":
        base += [
            "For fever/aches: consider OTC pain/fever reducers (e.g., acetaminophen or ibuprofen) if safe for you.",
            "For congestion: saline spray/humidifier can help; some people use OTC decongestants (check contraindications).",
            "For cough: warm fluids; honey may help for adults/older children (do not give honey to infants)."
        ]
    elif flow == "gi":
        base += [
            "Use oral rehydration solutions / electrolyte fluids if you have vomiting/diarrhea.",
            "Eat bland foods (toast, rice, bananas) as tolerated.",
            "Seek care if you can’t keep fluids down or feel dizzy/very weak."
        ]
    elif flow == "skin_allergy":
        base += [
            "If mild itching/hives: avoid suspected triggers and consider OTC antihistamines if safe for you.",
            "Seek urgent care if swelling or breathing trouble develops."
        ]
    else:
        base += ["For mild pain/fever, OTC pain/fever reducers may help if safe for you."]

    return base

# -------------------------
# RAG: Chroma + embeddings
# -------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _looks_like_chroma_persistence_mismatch(err: str) -> bool:
    s = (err or "").lower()
    needles = [
        "persistence mismatch",
        "local_persistent_hnsw",
        "persist_data",
        "dimensionality",
        "segment/impl/vector/local_persistent",
        "has no attribute",
    ]
    return any(n in s for n in needles)

def load_chroma_collection_safe():
    """
    Returns (collection, error_string_or_None). Never raises.
    NOT cached on purpose: avoids stale clients after you rebuild assets/chroma.
    """
    if not os.path.exists(CHROMA_DIR):
        return None, f"Chroma directory not found at: {CHROMA_DIR}"

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            col = client.get_collection(COLLECTION)
            return col, None
        except Exception as get_err:
            return None, f"Collection '{COLLECTION}' not found: {str(get_err)}"
    except Exception as e:
        return None, str(e)

def rag_retrieve(query: str, k: int = TOP_K):
    embedder = load_embedder()
    col, err = load_chroma_collection_safe()

    if col is None:
        if err and _looks_like_chroma_persistence_mismatch(err):
            st.error(
                "Chroma persistence mismatch detected (often after upgrading `chromadb`).\n\n"
                "Fix (rebuild index):\n"
                "`rm -rf assets/chroma && python scripts/build_index.py`"
            )
        else:
            st.error(
                "Chroma DB not ready.\n\n"
                f"Details: {err}\n\n"
                "If you haven't built the index yet, run:\n"
                "`python scripts/build_index.py`"
            )
        return []

    try:
        if col.count() == 0:
            st.warning("Chroma collection exists but is empty. Run: `python scripts/build_index.py`")
            return []
    except Exception:
        pass

    v = embedder.encode([query], convert_to_numpy=True).astype("float32")
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    try:
        res = col.query(
            query_embeddings=[v[0].tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
    except Exception as e:
        msg = str(e)
        if _looks_like_chroma_persistence_mismatch(msg):
            st.error(
                "Chroma persistence mismatch detected while querying.\n\n"
                "Fix (rebuild index):\n"
                "`rm -rf assets/chroma && python scripts/build_index.py`"
            )
        else:
            st.error(f"Chroma query failed: {msg}")
        return []

    hits = []
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i] or {}
        hits.append({
            "rank": i + 1,
            "score": float(res["distances"][0][i]),
            "question": md.get("question", ""),
            "answer": md.get("answer", ""),
            "source": md.get("source", ""),
            "url": md.get("url", ""),
            "question_type": md.get("question_type", ""),
        })
    return hits

def llm_answer(question: str, hits: list[dict]) -> str:
    model_id = os.getenv("HF_MODEL_ID", "ruslanmv/Medical-Llama3-8B")
    token = os.getenv("HF_TOKEN")
    if not token:
        return "HF_TOKEN not set. Add it to your .env file (HF_TOKEN=hf_...)."

    sources_block = ""
    for h in hits:
        a = (h.get("answer") or "")
        sources_block += f"[{h['rank']}] {h.get('source','')} — {h.get('url','')}\n"
        sources_block += a[:1200] + "\n\n"

    sys_rules = (
        "You are a medical information assistant. You are NOT a doctor.\n"
        "Rules:\n"
        "1) Use ONLY the provided SOURCES. If insufficient, say you don't have enough information.\n"
        "2) Explain in plain English.\n"
        "3) Include citations like [1], [2] matching the sources.\n"
        f"4) End with this disclaimer exactly: {DISCLAIMER}\n"
    )

    prompt = (
        f"{sys_rules}\n\n"
        f"SOURCES:\n{sources_block}\n"
        f"QUESTION: {question}\n\n"
        "Write a short answer with citations like [1]. "
        f"End with: {DISCLAIMER}"
    )

    try:
        client = InferenceClient(model=model_id, token=token)
        return client.text_generation(
            prompt,
            max_new_tokens=240,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
        )
    except HfHubHTTPError as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            return "HF error: Unauthorized (401). Check your HF_TOKEN."
        if "429" in msg:
            return "HF error: Rate-limited (429). Try again later."
        if "404" in msg:
            return "HF error: Model not accessible. It may require accepting terms or may not support inference."
        if "410" in msg:
            return "HF error: 410 Gone. Try: `pip install -U huggingface_hub`"
        return "Unable to generate response. Please try again later."
    except Exception:
        return "Unable to generate response. Please try again later."

# -------------------------
# UI helpers (chips + accordions + printable summary)
# -------------------------
def render_triage_chips(active: str) -> str:
    active = (active or "").upper()
    def cls(name: str) -> str:
        base = f"chip {name.lower()}"
        return f"{base} active" if active == name else base

    return f"""
    <div class="triage-chips">
      <span class="{cls('ROUTINE')}"><span class="dot"></span>ROUTINE</span>
      <span class="{cls('URGENT')}"><span class="dot"></span>URGENT</span>
      <span class="{cls('EMERGENCY')}"><span class="dot"></span>EMERGENCY</span>
    </div>
    """

def render_accordion(title: str, body_html: str, open_by_default: bool = False) -> str:
    t = _html.escape(title)
    open_attr = " open" if open_by_default else ""
    return f"""
    <details class="acc"{open_attr}>
      <summary>
        <span class="acc-title">{t}</span>
        <span class="acc-icon">▾</span>
      </summary>
      <div class="acc-body">
        {body_html}
      </div>
    </details>
    """

def render_citations_html(hits: list[dict]) -> str:
    if not hits:
        return '<div class="muted">(no citations retrieved)</div>'

    items = []
    for h in hits:
        rank = int(h.get("rank", 0) or 0)
        src = _html.escape((h.get("source") or "").strip() or "Source")
        url = (h.get("url") or "").strip()
        url_safe = _html.escape(url)
        link_html = (
            f'<a class="cite-link" href="{url_safe}" target="_blank" rel="noopener noreferrer">{url_safe}</a>'
            if url else '<span class="muted">(no url)</span>'
        )
        items.append(f"""
          <div class="cite-item">
            <div class="cite-head"><span class="cite-num">[{rank}]</span> <span class="cite-src">{src}</span></div>
            <div class="cite-url">{link_html}</div>
          </div>
        """)
    return "\n".join(items)

def render_passages_html(hits: list[dict]) -> str:
    if not hits:
        return '<div class="muted">(no passages retrieved)</div>'

    blocks = []
    for h in hits:
        rank = int(h.get("rank", 0) or 0)
        qtype = _html.escape((h.get("question_type") or "").strip())
        q = _html.escape((h.get("question") or "").strip())
        a = (h.get("answer") or "").strip()
        a_snip = _html.escape(a[:900] + ("…" if len(a) > 900 else ""))
        blocks.append(f"""
          <div class="passage">
            <div class="passage-head">
              <span class="passage-rank">[{rank}]</span>
              <span class="passage-type">{qtype}</span>
            </div>
            <div class="passage-q">{q}</div>
            <div class="passage-a">{a_snip}</div>
          </div>
        """)
    return "\n".join(blocks)

def build_summary_markdown(complaint: str, flow: str, triage: str, triage_note: str, answers: dict, hits: list[dict]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Printable Triage Summary")
    lines.append(f"- Generated: {ts}")
    lines.append("")
    lines.append("## Complaint")
    lines.append((complaint or "").strip() or "(not provided)")
    lines.append("")
    lines.append("## Detected Pathway")
    lines.append(flow or "general")
    lines.append("")
    lines.append("## Triage")
    lines.append(f"**{triage or ''}**")
    lines.append("")
    lines.append("## Triage Note")
    lines.append(triage_note or "")
    lines.append("")
    lines.append("## Answers")
    if answers:
        for k, v in answers.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- (no answers recorded)")
    lines.append("")
    lines.append("## Citations")
    if hits:
        for h in hits:
            src = (h.get("source") or "").strip()
            url = (h.get("url") or "").strip()
            qtype = (h.get("question_type") or "").strip()
            lines.append(f"- [{h.get('rank')}] {src} ({qtype}) — {url}")
    else:
        lines.append("- (no citations retrieved)")
    lines.append("")
    lines.append(f"---\n{DISCLAIMER}")
    return "\n".join(lines)

def render_printable_summary_card(complaint: str, triage: str, triage_note: str, answers_lines: list[str], citations_lines: list[str]) -> str:
    complaint_html = _html.escape(complaint or "(not provided)")
    triage_html = _html.escape(triage or "")
    note_html = _html.escape(triage_note or "")

    answers_html = _html.escape("\n".join(answers_lines) if answers_lines else "(no answers recorded)")
    citations_html = _html.escape("\n".join(citations_lines) if citations_lines else "(no citations retrieved)")

    return f"""
    <div class="summary-card">
      <div class="summary-title">📄 Printable Summary</div>

      <div class="summary-grid">
        <div class="summary-box">
          <div class="summary-label">Complaint</div>
          <div class="summary-value">{complaint_html}</div>
        </div>
        <div class="summary-box">
          <div class="summary-label">Triage</div>
          <div class="summary-value">{triage_html}</div>
        </div>
      </div>

      <div class="summary-box" style="margin-top:12px;">
        <div class="summary-label">Triage Note</div>
        <div class="summary-value" style="font-weight:800; opacity:.92;">{note_html}</div>
      </div>

      <div class="summary-box" style="margin-top:12px;">
        <div class="summary-label">Answers</div>
        <pre class="summary-pre">{answers_html}</pre>
      </div>

      <div class="summary-box" style="margin-top:12px;">
        <div class="summary-label">Citations</div>
        <pre class="summary-pre">{citations_html}</pre>
      </div>
    </div>
    """

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Medical Triage + RAG",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ---------- App background ---------- */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px circle at 12% 10%, rgba(59,130,246,.25), transparent 60%),
    radial-gradient(1000px circle at 92% 18%, rgba(16,185,129,.18), transparent 55%),
    radial-gradient(900px circle at 50% 95%, rgba(99,102,241,.22), transparent 60%),
    linear-gradient(180deg, #070B14 0%, #0A1224 55%, #070B14 100%);
  color: #E5E7EB;
}
[data-testid="stAppViewContainer"] *{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.block-container{
  max-width: 1120px;
  padding-top: 2.1rem;
  padding-bottom: 2.2rem;
}
header { visibility: hidden; }

/* ---------- Prevent hidden tab panels from affecting layout ---------- */
div[data-baseweb="tab-panel"][aria-hidden="true"] { display: none !important; }
div[data-baseweb="tab-panel"][hidden] { display: none !important; }

/* ---------- Hide Streamlit "Show widget keys" overlay (key=...) ---------- */
div[data-testid="stWidgetKey"],
span[data-testid="stWidgetKey"],
p[data-testid="stWidgetKey"],
code[data-testid="stWidgetKey"],
label[data-testid="stWidgetLabel"] code,
div[data-testid="stExpander"] summary code,
.stExpander code[class*="code"],
[data-testid="stExpander"] code,
.stExpander [data-testid="stCaption"],
div[data-testid="stExpander"] + div[data-testid="stCaption"],
.element-container code:contains("key="),
[data-testid="stMarkdownContainer"] code,
/* Target the specific expander key overlays */
.stExpander summary::before,
.stExpander summary::after,
div[data-testid="stExpander"] summary span[style*="position: absolute"],
div[data-testid="stExpander"] summary > span:first-child,
div[data-testid="stExpander"] > div:first-child,
.stExpander > div > div[data-testid="stCaption"],
div[data-testid="stExpander"] div[data-testid="stCaption"]{
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  position: absolute !important;
  left: -9999px !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}

/* Hide any caption/code that appears above expanders */
.stExpander + [data-testid="stCaption"],
[data-testid="stExpander"] + [data-testid="stCaption"],
div[data-baseweb="popover"] [data-testid="stCaption"]{
  display: none !important;
}

/* ---------- Streamlit Expanders - Black theme ---------- */
.stExpander,
div[data-testid="stExpander"]{
  background: rgba(0,0,0,0.70) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 14px !important;
  box-shadow: 0 12px 26px rgba(0,0,0,0.50) !important;
  margin-bottom: 16px !important;
  backdrop-filter: blur(10px) !important;
}

.stExpander details > summary,
div[data-testid="stExpander"] details > summary{
  background: rgba(0,0,0,0.65) !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 12px 14px !important;
  margin: 0 !important;
  color: #E5E7EB !important;
  font-weight: 700 !important;
}

.stExpander details[open] > summary,
div[data-testid="stExpander"] details[open] > summary{
  background: rgba(0,0,0,0.75) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px 14px 0 0 !important;
}

.stExpander details > div[role="group"],
div[data-testid="stExpander"] details > div[role="group"]{
  padding-top: 10px !important;
  background: rgba(0,0,0,0.60) !important;
}

/* ---------- Text visibility ---------- */
.main h2, .main h3, .main h4, .main h5, .main h6 { color: #F9FAFB !important; }
.main p, .main span, .main div, .main label { color: #E5E7EB !important; }
.stMarkdown, .stMarkdown p { color: #E5E7EB !important; }
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label {
  color: #F9FAFB !important;
  font-weight: 650 !important;
}

/* ---------- Hero ---------- */
.hero{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.30);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  margin-bottom: 1.2rem;
}
.hero-badge{
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(59,130,246,0.14);
  border: 1px solid rgba(59,130,246,0.30);
  color: #BFDBFE;
  font-weight: 800;
  font-size: 0.85rem;
}
.hero h1{
  margin: 10px 0 6px 0;
  font-size: 2.1rem;
  line-height: 1.15;
  font-weight: 900;
  letter-spacing: -0.02em;
  color: #F9FAFB;
}
.hero-sub{
  margin: 0 0 4px 0;
  color: rgba(229,231,235,0.85);
  font-size: 1rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap: 10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 8px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.20);
}
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 10px 14px;
  font-weight: 800;
  color: rgba(229,231,235,0.85);
  border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [aria-selected="true"]{
  background: linear-gradient(135deg, rgba(59,130,246,0.70), rgba(99,102,241,0.65));
  border-color: rgba(255,255,255,0.18);
  color: #FFFFFF;
}

/* ---------- Inputs ---------- */
.stTextInput input, .stTextArea textarea{
  color: #111827 !important;
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus{
  outline: none !important;
  border-color: rgba(59,130,246,0.55) !important;
  box-shadow: 0 0 0 4px rgba(59,130,246,0.18) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder{
  color: rgba(17,24,39,0.45) !important;
}

/* ---------- Buttons ---------- */
.stButton>button,
.stDownloadButton>button{
  background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.55rem 1.2rem !important;
  font-weight: 900 !important;
  box-shadow: 0 10px 24px rgba(59,130,246,0.25) !important;
  transition: all .18s ease !important;
}
.stButton>button:hover,
.stDownloadButton>button:hover{
  transform: translateY(-1px);
  filter: brightness(1.05);
}

/* ---------- Radio group ---------- */
.stRadio > div{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  padding: 14px !important;
}
.stRadio > div > label {
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.2) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  margin: 4px 0 !important;
  transition: all 0.2s ease !important;
}
.stRadio > div > label:hover {
  background: white !important;
  border-color: rgba(59,130,246,0.5) !important;
  transform: translateX(4px) !important;
  box-shadow: 0 2px 8px rgba(59,130,246,0.2) !important;
}
.stRadio > div > label > div { 
  color: #000000 !important; 
  font-weight: 500 !important;
}
.stRadio > div > label > div > div {
  color: #000000 !important;
}
.stRadio > div > label > div p {
  color: #000000 !important;
}
.stRadio > div > label span {
  color: #000000 !important;
}

/* ---------- Animated triage chips ---------- */
.triage-chips{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin: 10px 0 6px 0;
}
.chip{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 900;
  letter-spacing: .08em;
  font-size: .78rem;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: rgba(229,231,235,.70);
}
.chip .dot{
  width: 8px; height: 8px; border-radius: 999px;
  background: rgba(229,231,235,.55);
}
.chip.active{ color:#fff; }
.chip.routine.active{
  background: rgba(34,197,94,.14);
  border-color: rgba(34,197,94,.35);
  animation: pulseGreen 1.6s ease-in-out infinite;
}
.chip.routine.active .dot{ background: rgba(34,197,94,.95); box-shadow: 0 0 12px rgba(34,197,94,.8); }
@keyframes pulseGreen{
  0%,100%{ box-shadow: 0 0 14px rgba(34,197,94,.22); }
  50%{ box-shadow: 0 0 28px rgba(34,197,94,.55); }
}
.chip.urgent.active{
  background: rgba(245,158,11,.14);
  border-color: rgba(245,158,11,.35);
  animation: pulseAmber 1.6s ease-in-out infinite;
}
.chip.urgent.active .dot{ background: rgba(245,158,11,.95); box-shadow: 0 0 12px rgba(245,158,11,.8); }
@keyframes pulseAmber{
  0%,100%{ box-shadow: 0 0 14px rgba(245,158,11,.22); }
  50%{ box-shadow: 0 0 28px rgba(245,158,11,.55); }
}
.chip.emergency.active{
  background: rgba(239,68,68,.14);
  border-color: rgba(239,68,68,.35);
  animation: pulseRed 1.2s ease-in-out infinite;
}
.chip.emergency.active .dot{ background: rgba(239,68,68,.95); box-shadow: 0 0 12px rgba(239,68,68,.85); }
@keyframes pulseRed{
  0%,100%{ box-shadow: 0 0 16px rgba(239,68,68,.25); }
  50%{ box-shadow: 0 0 34px rgba(239,68,68,.65); }
}

/* ---------- Custom accordions (no Streamlit expander => no "key=..." overlap) ---------- */
.acc{
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 14px;
  box-shadow: 0 12px 26px rgba(0,0,0,0.40);
  margin: 10px 0 16px 0;
  overflow: hidden;
}
.acc summary{
  cursor: pointer;
  padding: 12px 14px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 12px;
  list-style: none;
}
.acc summary::-webkit-details-marker{ display:none; }
.acc .acc-title{
  font-weight: 900;
  color: rgba(243,244,246,0.95);
}
.acc .acc-icon{
  opacity: .7;
  font-weight: 900;
  transform: rotate(0deg);
  transition: transform .18s ease;
}
.acc[open] .acc-icon{ transform: rotate(180deg); }
.acc .acc-body{
  padding: 14px;
  border-top: 1px solid rgba(255,255,255,0.10);
}
.cite-item{ padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.cite-item:last-child{ border-bottom: 0; }
.cite-head{ font-weight: 800; color: rgba(243,244,246,0.95); }
.cite-num{ opacity: .9; margin-right: 6px; }
.cite-url{ margin-top: 6px; }
.cite-link{ color: rgba(147,197,253,0.95); text-decoration: underline; word-break: break-word; }
.muted{ opacity: .75; }

.passage{ padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.passage:last-child{ border-bottom: 0; }
.passage-head{ display:flex; gap:10px; align-items:baseline; }
.passage-rank{ font-weight: 900; }
.passage-type{ opacity: .8; font-weight: 800; }
.passage-q{ margin-top: 6px; font-weight: 800; opacity: .95; }
.passage-a{ margin-top: 8px; opacity: .92; line-height: 1.45; }

/* ---------- Printable summary card ---------- */
.summary-card{
  margin-top: 18px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 14px 34px rgba(0,0,0,0.32);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
.summary-title{
  font-size: 1.25rem;
  font-weight: 950;
  color: #F9FAFB;
  margin-bottom: 10px;
}
.summary-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
@media (max-width: 900px){
  .summary-grid{ grid-template-columns: 1fr; }
}
.summary-box{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px;
}
.summary-label{
  opacity:.75;
  font-weight: 900;
  font-size: .85rem;
  margin-bottom: 6px;
}
.summary-value{
  font-weight: 900;
  color: #F9FAFB;
  word-break: break-word;
}
.summary-pre{
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  background: rgba(0,0,0,0.18);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 12px;
  margin: 0;
  color: rgba(243,244,246,0.95);
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="hero-badge">RAG-grounded • Citations • Triage MCQ</div>
  <h1>🩺 Medical Q&A & Triage Assistant</h1>
  <p class="hero-sub">Step-by-step symptom intake + evidence-grounded answers with sources.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏥 Triage Intake", "💬 Ask a Question", "📊 Dataset Explorer"])

# -------- Triage Intake --------
with tabs[0]:
    if "triage_state" not in st.session_state:
        st.session_state.triage_state = {
            "started": False,
            "complaint": "",
            "flow": "general",
            "step": 0,
            "answers": {},
            "done": False,
            "triage": None,
            "triage_note": ""
        }

    S = st.session_state.triage_state

    st.markdown("### 📋 Step-by-Step Symptom Assessment")
    st.markdown("Answer a series of questions to help determine the appropriate level of care.")
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    complaint = st.text_input(
        "What is the main problem / symptom? (free text)",
        value=S["complaint"],
        placeholder="e.g., cough and fever, stomach pain, rash, chest pain...",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Start / Update Intake"):
            S["complaint"] = complaint.strip()
            S["flow"] = classify_complaint(S["complaint"])
            S["started"] = True
            S["step"] = 0
            S["answers"] = {}
            S["done"] = False
            S["triage"] = None
            S["triage_note"] = ""
            st.rerun()

    with colB:
        if st.button("Restart Intake"):
            st.session_state.triage_state = {
                "started": False, "complaint":"", "flow":"general",
                "step":0, "answers":{}, "done":False, "triage":None, "triage_note":""
            }
            st.rerun()

    hits_for_summary: list[dict] = []

    if S["started"]:
        flow_steps = FLOWS.get(S["flow"], FLOWS["general"])
        st.write(f"**Detected pathway:** `{S['flow']}`")

        if not S["done"] and S["step"] < len(flow_steps):
            step_obj = flow_steps[S["step"]]
            st.markdown(f"### 📝 Question {S['step']+1}/{len(flow_steps)}")
            st.markdown(f"**{step_obj['q']}**")

            choice = st.radio("Choose one:", step_obj["opts"])
            extra = st.text_area("Other / extra details (optional):")

            if st.button("Next"):
                S["answers"][step_obj["key"]] = choice if not extra.strip() else f"{choice} | details: {extra.strip()}"

                triage, note = triage_from_answers(
                    S["complaint"],
                    S["flow"],
                    {k: v.split(" | ")[0] for k, v in S["answers"].items()},
                )

                if triage == "EMERGENCY":
                    S["done"] = True
                    S["triage"] = triage
                    S["triage_note"] = note
                else:
                    S["step"] += 1
                    if S["step"] >= len(flow_steps):
                        S["done"] = True
                        S["triage"] = triage
                        S["triage_note"] = note

                st.rerun()

        if S["done"]:
            triage = S["triage"]
            note = S["triage_note"]

            st.markdown("## Triage Result")
            st.markdown(render_triage_chips(triage), unsafe_allow_html=True)
            st.markdown(f"**{note}**")

            if triage == "EMERGENCY":
                st.error("If you are in immediate danger or have severe symptoms, call local emergency services now.")

            advice = self_care_advice(S["flow"], triage)
            if advice:
                st.markdown("### 💊 Self-Care & OTC Guidance")
                for a in advice:
                    st.markdown(f"• {a}")
                st.caption("OTC suggestions are general information, not a prescription. Check labels and contraindications; ask a pharmacist if unsure.")

            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            st.markdown("### 📚 Evidence-Based Information")

            query = f"{S['complaint']}. Answers: " + "; ".join([f"{k}={v}" for k, v in S["answers"].items()])
            hits = rag_retrieve(query, TOP_K)
            hits_for_summary = hits

            if hits:
                with st.expander("📖 Citations & Sources", expanded=True):
                    for h in hits:
                        st.markdown(f"**[{h['rank']}]** {h['source']}")
                        st.caption(h['url'])

                with st.expander("📄 Retrieved Passages", expanded=False):
                    for h in hits:
                        st.markdown(f"**[{h['rank']}] {h['question_type']}**")
                        st.markdown(f"*{h['question']}*")
                        st.write((h["answer"] or "")[:900] + "…")
                        st.markdown("---")

                st.markdown("### 🤖 AI-Generated Summary")
                st.write(llm_answer(query, hits))
            else:
                st.info("No passages retrieved (or Chroma needs rebuild).")

            # -------- Download Summary --------
            summary_md = build_summary_markdown(
                complaint=S["complaint"],
                flow=S["flow"],
                triage=triage,
                triage_note=note,
                answers=S["answers"],
                hits=hits_for_summary,
            )

            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            filename = f"triage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            st.download_button(
                "📥 Download Summary Report (.md)",
                data=summary_md.encode("utf-8"),
                file_name=filename,
                mime="text/markdown",
            )
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

            st.markdown(f"**{DISCLAIMER}**")

# -------- Ask a Question --------
with tabs[1]:
    st.markdown("### 💬 Ask Your Medical Question")
    st.markdown("Get evidence-based information from trusted medical sources.")
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    q = st.text_input("Your question", placeholder="Example: What should I do for a cough and mild fever for 3 days?")
    if st.button("🔍 Get Answer with Citations"):
        with st.spinner("Searching medical knowledge base..."):
            hits = rag_retrieve(q, TOP_K)

        if hits:
            st.markdown("### 📚 Sources")
            st.markdown(render_accordion("Sources (click to expand)", render_citations_html(hits), open_by_default=True), unsafe_allow_html=True)

            st.markdown("### 🤖 Answer")
            with st.spinner("Generating answer..."):
                answer = llm_answer(q, hits)
            st.info(answer)
        else:
            st.info("No hits retrieved.")
        st.caption(DISCLAIMER)

# -------- Dataset Explorer --------
with tabs[2]:
    st.markdown("### 📊 MedQuAD Dataset Explorer")
    st.markdown("Browse and search through the medical Q&A knowledge base.")
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    if os.path.exists(META_PATH):
        df = pd.read_parquet(META_PATH).fillna("")
        qtype_choices = sorted(df["question_type"].dropna().unique().tolist()) if "question_type" in df.columns else []
        qtype = st.selectbox("Filter by question_type (optional)", ["(any)"] + qtype_choices)
        keyword = st.text_input("Keyword (optional)", placeholder="e.g., diabetes, fever, asthma")

        if st.button("Search"):
            with st.spinner("Searching dataset..."):
                key = (keyword or "").lower().strip()
                out = []
                shown = 0
                for _, r in df.iterrows():
                    if qtype != "(any)" and str(r.get("question_type","")) != str(qtype):
                        continue
                    text = (str(r.get("question","")) + " " + str(r.get("answer",""))).lower()
                    if key and key not in text:
                        continue
                    out.append({
                        "question_type": r.get("question_type",""),
                        "question_focus": r.get("question_focus",""),
                        "question": r.get("question",""),
                        "url": r.get("document_url","")
                    })
                    shown += 1
                    if shown >= 30:
                        break

            if out:
                st.success(f"✅ Found **{len(out)}** results")
                st.dataframe(out, use_container_width=True)
            else:
                st.info("No results found. Try a different keyword or filter.")
    else:
        st.warning("assets/medquad_meta.parquet not found.")
import os, json, time
import gradio as gr
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------- Config ----------
INDEX_PATH = "assets/medquad.index"
META_PATH  = "assets/medquad_meta.parquet"
TOP_K = 4

MODEL_ID = "ruslanmv/Medical-Llama3-8B"  # remote inference (hosted)
DISCLAIMER = "Disclaimer: This is not medical advice. If you're worried or symptoms are severe, seek professional care."

SYS_RULES = (
    "You are a medical information assistant. You are NOT a doctor.\n"
    "Rules:\n"
    "1) Use ONLY the provided SOURCES. If insufficient, say you don't have enough information.\n"
    "2) Explain in plain English.\n"
    "3) Do NOT diagnose or confirm what condition the user has.\n"
    "4) Include citations like [1], [2] matching the sources.\n"
    f"5) End with this disclaimer exactly: {DISCLAIMER}\n"
)

# ---------- Load data ----------
df = pd.read_parquet(META_PATH)
index = faiss.read_index(INDEX_PATH)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _clean(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)

def _embed(text: str) -> np.ndarray:
    v = embedder.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(v)
    return v

def retrieve(query: str, k: int = TOP_K):
    qv = _embed(query)
    scores, idx = index.search(qv, k)
    hits = []
    for rank, i in enumerate(idx[0].tolist(), start=1):
        r = df.iloc[i].to_dict()
        hits.append({
            "rank": rank,
            "score": float(scores[0][rank-1]),
            "question": _clean(r.get("question")),
            "answer": _clean(r.get("answer")),
            "question_type": _clean(r.get("question_type")),
            "question_focus": _clean(r.get("question_focus")),
            "source": _clean(r.get("document_source")) or "Unknown source",
            "url": _clean(r.get("document_url")),
        })
    return hits

def triage_hint(text: str) -> str:
    t = (text or "").lower()
    emergency = [
        "chest pain", "can't breathe", "cannot breathe", "trouble breathing",
        "blue lips", "seizure", "unconscious", "stroke", "severe bleeding"
    ]
    urgent = [
        "high fever", "worsening", "severe pain", "persistent vomiting",
        "dehydration", "infant", "newborn", "pregnant"
    ]
    if any(x in t for x in emergency):
        return "🚨 **Triage hint: EMERGENCY** (seek urgent medical evaluation)"
    if any(x in t for x in urgent):
        return "🟠 **Triage hint: URGENT** (consider same-day/soon medical advice)"
    return "🟢 **Triage hint: ROUTINE** (informational guidance; monitor and consult as needed)"

def is_diagnosis_request(q: str) -> bool:
    """
    Detect questions that ask the model to diagnose/confirm a condition.
    We intentionally keep this conservative (only strong diagnosis phrases).
    """
    t = (q or "").lower()
    patterns = [
        "do i have ", "do you think i have", "am i having", "am i having a",
        "what do i have", "what's wrong with me", "what is wrong with me",
        "can you diagnose", "diagnose me", "give me a diagnosis",
        "is it pneumonia", "is it cancer", "is this pneumonia", "is this cancer",
        "tell me what i have", "do i have pneumonia", "do i have covid"
    ]
    return any(p in t for p in patterns)

def format_citations_and_passages(hits):
    citations_md = []
    passages_md = []
    for h in hits:
        citations_md.append(f"[{h['rank']}] **{h['source']}** — {h['url']}")
        snippet = (_clean(h.get("answer"))[:900]).replace("\n", " ")
        passages_md.append(
            f"### [{h['rank']}] {h.get('question_type','')} — {h.get('question','')}\n"
            f"{snippet}..."
        )
    return "\n".join(citations_md), "\n\n".join(passages_md)

def generate_answer(user_q: str, hits):
    sources_block = ""
    for h in hits:
        ans = _clean(h.get("answer"))
        src = _clean(h.get("source")) or "Unknown source"
        url = _clean(h.get("url"))
        sources_block += f"[{h['rank']}] {src} — {url}\n"
        sources_block += (ans[:1200] + "\n\n")

    prompt = (
        f"{SYS_RULES}\n\n"
        f"SOURCES:\n{sources_block}\n"
        f"QUESTION: {user_q}\n\n"
        "Write a short answer with citations like [1]. "
        "Do NOT diagnose. If the user asks for diagnosis, refuse and suggest a safer re-phrasing. "
        f"End with: {DISCLAIMER}"
    )

    token = os.getenv("HF_TOKEN")
    if not token:
        return (
            "⚠️ HF_TOKEN is missing. Add it in your Space Settings → Variables and secrets.\n\n"
            f"{DISCLAIMER}"
        )

    client = InferenceClient(model=MODEL_ID, token=token)

    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=260,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False
        )
        return out
    except Exception as e:
        return (
            f"⚠️ I couldn’t generate a response right now due to a model/API error ({type(e).__name__}).\n\n"
            "Try again in a moment. The retrieval/citations below still show relevant sources.\n\n"
            f"{DISCLAIMER}"
        )

def log_feedback(kind: str, user_q: str, answer: str, hits):
    rec = {
        "ts": time.time(),
        "kind": kind,
        "question": user_q,
        "answer": answer,
        "sources": [{"rank": h["rank"], "url": h["url"]} for h in hits]
    }
    with open("feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return f"Saved feedback: **{kind}**"

# ---------- UI ----------
with gr.Blocks(title="Medical Q&A + RAG + Triage") as demo:
    gr.Markdown(
        f"# Medical Q&A & Triage Assistant (RAG-grounded)\n\n"
        f"**{DISCLAIMER}**\n\n"
        f"✅ Uses retrieval (MedQuAD) + citations\n"
        f"✅ Gives triage hint (Emergency/Urgent/Routine)\n"
        f"✅ Refuses diagnosis requests\n"
    )

    state_last = gr.State({"q": "", "a": "", "hits": []})

    with gr.Tabs():
        with gr.Tab("Chat"):
            chat = gr.Chatbot(height=360)  # compatible across HF Spaces Gradio versions
            user_q = gr.Textbox(
                label="Ask a medical question (informational)",
                placeholder="Example: cough and mild fever for 3 days..."
            )
            triage = gr.Markdown()

            with gr.Accordion("Citations", open=True):
                citations = gr.Markdown()
            with gr.Accordion("Retrieved passages", open=False):
                passages = gr.Markdown()

            with gr.Row():
                btn_send = gr.Button("Send")
                btn_clear = gr.Button("Clear")

            with gr.Row():
                fb_up = gr.Button("👍 Helpful")
                fb_down = gr.Button("👎 Not helpful")
                fb_wrong = gr.Button("🚩 Wrong citation")
            fb_status = gr.Markdown()

            def on_send(history, q):
                q = (q or "").strip()
                if not q:
                    return history, "", "", "", {"q": "", "a": "", "hits": []}

                hits = retrieve(q, TOP_K)
                tri = triage_hint(q)
                c_md, p_md = format_citations_and_passages(hits)

                # 1) EMERGENCY OVERRIDE (no LLM)
                if "EMERGENCY" in tri:
                    ans = (
                        "🚨 **This may be an emergency.** If you have chest pain or trouble breathing, "
                        "seek emergency care right now (call 911 / your local emergency number).\n\n"
                        "If symptoms are severe or worsening, do not wait for online guidance.\n\n"
                        f"{DISCLAIMER}"
                    )

                # 2) DIAGNOSIS REFUSAL (no LLM)
                elif is_diagnosis_request(q):
                    ans = (
                        "I can’t diagnose or confirm what condition you have. "
                        "But I *can* help with **general information** and **what to ask a clinician**.\n\n"
                        "Try asking one of these instead:\n"
                        "- “What is pneumonia and what are common symptoms?”\n"
                        "- “What are red-flag symptoms that mean I should seek urgent care?”\n"
                        "- “How is COVID vs flu vs cold generally distinguished by symptoms?”\n\n"
                        "If you share **age range**, **how long symptoms have lasted**, **temperature**, and "
                        "**any red flags** (trouble breathing, chest pain, confusion, dehydration), I can help you "
                        "understand the *general possibilities* and when to seek care.\n\n"
                        f"{DISCLAIMER}"
                    )

                # 3) Normal informational answer (use LLM with strict rules)
                else:
                    ans = generate_answer(q, hits)

                # Chat history in messages dict format (required by your Space’s gradio)
                if history is None:
                    history = []
                history = history + [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans}
                ]

                st = {"q": q, "a": ans, "hits": hits}
                return history, tri, c_md, p_md, st

            btn_send.click(on_send, inputs=[chat, user_q], outputs=[chat, triage, citations, passages, state_last])
            user_q.submit(on_send, inputs=[chat, user_q], outputs=[chat, triage, citations, passages, state_last])

            def on_clear():
                return [], "", "", "", {"q": "", "a": "", "hits": []}

            btn_clear.click(on_clear, outputs=[chat, triage, citations, passages, state_last])

            def on_fb(kind, st):
                if not st.get("q"):
                    return "Ask a question first."
                return log_feedback(kind, st["q"], st.get("a", ""), st.get("hits", []))

            fb_up.click(lambda st: on_fb("helpful", st), inputs=[state_last], outputs=[fb_status])
            fb_down.click(lambda st: on_fb("not_helpful", st), inputs=[state_last], outputs=[fb_status])
            fb_wrong.click(lambda st: on_fb("wrong_citation", st), inputs=[state_last], outputs=[fb_status])

        with gr.Tab("Dataset Explorer"):
            gr.Markdown("## MedQuAD Explorer (quick demo view)")
            qtype = gr.Dropdown(
                label="Filter by question_type (optional)",
                choices=sorted([x for x in df["question_type"].dropna().unique().tolist() if str(x).strip() != ""])
            )
            keyword = gr.Textbox(label="Search keyword", placeholder="e.g., diabetes, fever, asthma")
            btn_find = gr.Button("Search")

            table = gr.Dataframe(
                headers=["question_type", "question_focus", "question", "url"],
                interactive=False,
                wrap=True
            )
            status = gr.Markdown()

            def do_search(qtype, keyword):
                key = (keyword or "").lower().strip()
                out = []
                shown = 0
                for _, r in df.iterrows():
                    if qtype and str(r.get("question_type", "")) != str(qtype):
                        continue
                    text = (str(r.get("question", "")) + " " + str(r.get("answer", ""))).lower()
                    if key and key not in text:
                        continue
                    out.append([
                        r.get("question_type", ""),
                        r.get("question_focus", ""),
                        r.get("question", ""),
                        r.get("document_url", "")
                    ])
                    shown += 1
                    if shown >= 30:
                        break
                return out, f"Showing **{len(out)}** rows."

            btn_find.click(do_search, inputs=[qtype, keyword], outputs=[table, status])

# Launch
try:
    demo.launch(ssr_mode=False)
except TypeError:
    demo.launch()

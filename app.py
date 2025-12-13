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

MODEL_ID = "ruslanmv/Medical-Llama3-8B"  # remote inference
DISCLAIMER = "Disclaimer: This is not medical advice. If you're worried or symptoms are severe, seek professional care."

SYS_RULES = (
    "You are a medical information assistant. You are NOT a doctor.\n"
    "Rules:\n"
    "1) Use ONLY the provided SOURCES. If insufficient, say you don't have enough information.\n"
    "2) Explain in plain English.\n"
    "3) Include citations like [1], [2] matching the sources.\n"
    f"4) End with this disclaimer exactly: {DISCLAIMER}\n"
)

# ---------- Load data ----------
df = pd.read_parquet(META_PATH)
index = faiss.read_index(INDEX_PATH)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
            "question": r.get("question",""),
            "answer": r.get("answer",""),
            "question_type": r.get("question_type",""),
            "question_focus": r.get("question_focus",""),
            "source": r.get("document_source",""),
            "url": r.get("document_url",""),
        })
    return hits

def triage_hint(text: str) -> str:
    t = (text or "").lower()
    emergency = ["chest pain","can't breathe","cannot breathe","blue lips","seizure","unconscious","stroke","severe bleeding"]
    urgent = ["high fever","worsening","severe pain","persistent vomiting","dehydration","infant","newborn","pregnant"]
    if any(x in t for x in emergency):
        return "🚨 **Triage hint: EMERGENCY** (consider urgent medical evaluation)"
    if any(x in t for x in urgent):
        return "🟠 **Triage hint: URGENT** (consider same-day/soon medical advice)"
    return "🟢 **Triage hint: ROUTINE** (informational guidance; monitor and consult as needed)"

def format_citations_and_passages(hits):
    citations_md = []
    passages_md = []
    for h in hits:
        citations_md.append(f"[{h['rank']}] **{h['source']}** — {h['url']}")
        snippet = (h["answer"] or "")[:900].replace("\n", " ")
        passages_md.append(
            f"### [{h['rank']}] {h['question_type']} — {h['question']}\n"
            f"{snippet}..."
        )
    return "\n".join(citations_md), "\n\n".join(passages_md)

def generate_answer(user_q: str, hits):
    sources_block = ""
    for h in hits:
        sources_block += f"[{h['rank']}] {h['source']} — {h['url']}\n"
        sources_block += (h["answer"][:1200] + "\n\n")

    prompt = (
        f"{SYS_RULES}\n\n"
        f"SOURCES:\n{sources_block}\n"
        f"QUESTION: {user_q}\n\n"
        "Write a short answer with citations like [1]. "
        f"End with: {DISCLAIMER}"
    )

    token = os.getenv("HF_TOKEN")
    client = InferenceClient(model=MODEL_ID, token=token)

    out = client.text_generation(
        prompt,
        max_new_tokens=230,
        temperature=0.3,
        top_p=0.9,
        return_full_text=False
    )
    return out

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
    gr.Markdown(f"# Medical Q&A & Triage Assistant (RAG-grounded)\n\n**{DISCLAIMER}**")

    state_last = gr.State({"q":"", "a":"", "hits":[]})

    with gr.Tabs():
        with gr.Tab("Chat"):
            chat = gr.Chatbot(height=360)
            user_q = gr.Textbox(label="Ask a medical question (informational)", placeholder="Example: cough and mild fever for 3 days...")
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
                hits = retrieve(q, TOP_K)
                tri = triage_hint(q)
                c_md, p_md = format_citations_and_passages(hits)
                ans = generate_answer(q, hits)
                history = history + [(q, ans)]
                st = {"q": q, "a": ans, "hits": hits}
                return history, tri, c_md, p_md, st

            btn_send.click(on_send, inputs=[chat, user_q], outputs=[chat, triage, citations, passages, state_last])
            user_q.submit(on_send, inputs=[chat, user_q], outputs=[chat, triage, citations, passages, state_last])

            def on_clear():
                return [], "", "", "", {"q":"", "a":"", "hits":[]}

            btn_clear.click(on_clear, outputs=[chat, triage, citations, passages, state_last])

            def on_fb(kind, st):
                if not st["q"]:
                    return "Ask a question first."
                return log_feedback(kind, st["q"], st["a"], st["hits"])

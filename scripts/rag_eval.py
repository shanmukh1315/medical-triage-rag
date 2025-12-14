import os
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR = "assets/chroma"
COLLECTION = "medquad"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10–20 test questions (mix easy + tricky + edge cases)
TEST_QUESTIONS = [
    "I have a cough for 3 days with mild fever. What should I do?",
    "What are common causes of chest pain and when is it an emergency?",
    "What does blood in urine usually mean?",
    "I have nausea and diarrhea after eating outside. How can I prevent dehydration?",
    "What are signs of stroke that require emergency care?",
    "I have a sore throat and runny nose. Could it be flu or common cold?",
    "A baby has fever and is not feeding well. What are danger signs?",
    "What are symptoms and treatment options for asthma?",
    "What is anaphylaxis and what should you do immediately?",
    "How do you treat a mild skin rash with itching?",
    "What causes abdominal pain in the lower right side?",
    "How long does a viral cough typically last?",
    "What should I do if I feel dizzy and fainted once?",
    "What are warning signs of dehydration in children?",
    "Can stress cause headaches and how is it managed?",
    "What are symptoms of urinary tract infection (UTI)?",
    "When should someone with fever see a doctor?",
    "What is the difference between COVID-19 and flu symptoms?",
]

def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_collection():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Chroma directory not found: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def query_chroma(col, embedder, query: str, k: int):
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    qv = normalize(qv)
    res = col.query(
        query_embeddings=[qv[0].tolist()],
        n_results=k,
        include=["metadatas", "distances"],
    )
    hits = []
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    mds = res.get("metadatas", [[]])[0]
    for i in range(len(ids)):
        md = mds[i] or {}
        hits.append({
            "rank": i + 1,
            "distance": float(dists[i]),
            # For Chroma cosine, distance is typically lower=more similar.
            # A rough similarity proxy is (1 - distance).
            "similarity_proxy": float(1.0 - dists[i]),
            "source": md.get("source", ""),
            "url": md.get("url", ""),
            "question_type": md.get("question_type", ""),
            "question": md.get("question", ""),
            "answer_snippet": (md.get("answer", "") or "")[:200].replace("\n", " "),
        })
    return hits

def save_md_table(df: pd.DataFrame, path: Path, title: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

def main():
    embedder = load_embedder()
    col = load_collection()

    if col.count() == 0:
        raise RuntimeError("Chroma collection is empty. Run: python scripts/build_index.py")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------------
    # Experiment 1: Relevance @ K=4
    # -----------------------------
    exp1_rows = []
    for q in TEST_QUESTIONS:
        hits = query_chroma(col, embedder, q, k=4)
        top_sources = ", ".join([h["source"] for h in hits if h["source"]][:3])
        best_sim = max([h["similarity_proxy"] for h in hits], default=0.0)
        exp1_rows.append({
            "Question": q,
            "Top Sources (first 3)": top_sources,
            "Best similarity (1-dist)": round(best_sim, 4),
            "Manual relevance (0/1/2)": "",  # fill after you inspect results
            "Notes": "",
        })

    exp1_df = pd.DataFrame(exp1_rows)
    exp1_csv = OUT_DIR / "exp1_relevance_k4.csv"
    exp1_md = OUT_DIR / "exp1_relevance_k4.md"
    exp1_df.to_csv(exp1_csv, index=False)
    save_md_table(exp1_df, exp1_md, f"Experiment 1 — Retrieval Relevance (K=4) — {ts}")

    # -------------------------------------------------
    # Experiment 2: Compare TOP_K = 2 vs 4 vs 8 quality
    # -------------------------------------------------
    ks = [2, 4, 8]
    exp2_rows = []
    for q in TEST_QUESTIONS:
        for k in ks:
            hits = query_chroma(col, embedder, q, k=k)
            distances = [h["distance"] for h in hits]
            sims = [h["similarity_proxy"] for h in hits]
            sources = [h["source"] for h in hits if h["source"]]

            exp2_rows.append({
                "Question": q,
                "K": k,
                "Avg distance": round(float(np.mean(distances)) if distances else 0.0, 4),
                "Best similarity (1-dist)": round(float(np.max(sims)) if sims else 0.0, 4),
                "Unique sources": len(set(sources)),
                "Top-1 source": hits[0]["source"] if hits else "",
                "Manual quality (0/1/2)": "",  # fill after checking if added noise helped/hurt
                "Notes": "",
            })

    exp2_df = pd.DataFrame(exp2_rows)
    exp2_csv = OUT_DIR / "exp2_topk_comparison.csv"
    exp2_md = OUT_DIR / "exp2_topk_comparison.md"
    exp2_df.to_csv(exp2_csv, index=False)
    save_md_table(exp2_df, exp2_md, f"Experiment 2 — TOP_K Sensitivity (2 vs 4 vs 8) — {ts}")

    print("✅ Done.")
    print(f"- {exp1_md}")
    print(f"- {exp2_md}")
    print(f"- {exp1_csv}")
    print(f"- {exp2_csv}")

if __name__ == "__main__":
    main()

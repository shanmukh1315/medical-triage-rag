import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

META_PATH = "assets/medquad_meta.parquet"
CHROMA_DIR = "assets/chroma"
COLLECTION = "medquad"
BATCH = 128

def main():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing {META_PATH}. Put your parquet in assets/ first.")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    df = pd.read_parquet(META_PATH)
    df = df.fillna("")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # avoid duplicating if rerun
    existing = set(col.get(include=[])["ids"])
    rows = []
    for i in range(len(df)):
        _id = str(i)
        if _id not in existing:
            rows.append(i)

    if not rows:
        print("✅ Chroma already built (no new rows).")
        return

    print(f"Building embeddings for {len(rows)} new rows...")

    for start in tqdm(range(0, len(rows), BATCH)):
        batch_idx = rows[start:start+BATCH]
        texts = []
        metadatas = []
        ids = []

        for i in batch_idx:
            r = df.iloc[i]
            q = str(r.get("question",""))
            a = str(r.get("answer",""))
            src = str(r.get("document_source",""))
            url = str(r.get("document_url",""))
            qtype = str(r.get("question_type",""))
            focus = str(r.get("question_focus",""))

            # what we embed
            emb_text = f"Q: {q}\nA: {a}"
            texts.append(emb_text)
            ids.append(str(i))

            metadatas.append({
                "question": q,
                "answer": a,
                "source": src,
                "url": url,
                "question_type": qtype,
                "question_focus": focus,
            })

        embs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
        # cosine space works best if normalized
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms

        col.add(
            ids=ids,
            embeddings=embs.tolist(),
            documents=[m["answer"] for m in metadatas],  # required by Chroma
            metadatas=metadatas,
        )

    print(f"✅ Done. Persisted Chroma to {CHROMA_DIR}")

if __name__ == "__main__":
    main()

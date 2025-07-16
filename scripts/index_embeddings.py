import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from create_embeddings import EMBEDDINGS_DIR, DATA_DIR


# Build index once
def build_and_save_index(embeddings, save_path="buffy_faiss_index"):
    index = faiss.IndexFlatIP(1536)
    index.add(embeddings.astype('float32'))

    # Save to disk
    faiss.write_index(index, f"{save_path}.index")
    print(f"index saved to {save_path}.index")
    return index


# Load existing index
def load_index(index_path=EMBEDDINGS_DIR / "buffy_faiss_index.index"):
    if Path(index_path).exists():
        print(f"Loading existing FAISS index from {index_path}")
        return faiss.read_index(index_path)
    else:
        print(f"No index found at {index_path}")
        return None


# Smart loading pattern
def get_or_create_index(embeddings, index_path="buffy_faiss_index.index"):
    # Try to load existing
    index = load_index(EMBEDDINGS_DIR / index_path)

    if index is None:
        # Build new one
        print("Building new FAISS index...")
        index = build_and_save_index(embeddings, index_path.replace('.index', ''))
    else:
        print("Using existing FAISS index")

    return index


if __name__ == "__main__":
    embeddings = np.load(EMBEDDINGS_DIR / "buffy_embeddings_vectors.npy")
    chunks_df = pd.read_csv(DATA_DIR / "buffy_chunks.csv")

    index = get_or_create_index(embeddings)
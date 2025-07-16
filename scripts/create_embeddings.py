import pandas as pd
import numpy as np
import openai
import time
import os
from pathlib import Path
from typing import List
import pickle
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / DATA_DIR / "embeddings"


def create_embeddings_batch(texts: List[str], model="text-embedding-3-small", batch_size=100):
    """
    Create embeddings for a list of texts using OpenAI API.
    Processes in batches to handle rate limits and large datasets.
    """
    all_embeddings = []
    total_cost = 0

    print(f"Creating embeddings for {len(texts)} chunks...")
    print(f"Using model: {model}")
    print(f"Estimated cost: ~${(len(' '.join(texts).split()) * 0.00000002):.3f}")

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        try:
            response = openai.embeddings.create(
                model=model,
                input=batch_texts
            )

            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Track usage/cost
            tokens_used = response.usage.total_tokens
            cost = tokens_used * 0.00000002  # $0.02 per 1M tokens for text-embedding-3-small
            total_cost += cost

            # Rate limiting - be nice to the API
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            print(batch_texts[0][:100])
            print(batch_texts[-1][:100])
            # Add empty embeddings for failed batch (you might want to retry instead)
            empty_embedding = [0.0] * 1536  # text-embedding-3-small dimension
            all_embeddings.extend([empty_embedding] * len(batch_texts))

    print(f"Embedding creation complete!")
    print(f"Total cost: ${total_cost:.4f}")
    np.save("../models/buffy_embeddings_raw.npy", all_embeddings)
    print(f"Saved {len(all_embeddings)} embeddings")
    print(f"Embedding dimension: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")


    return np.array(all_embeddings), total_cost


def save_embeddings(chunks_df, embeddings=None, embeddings_file=None, filename_base="buffy_embeddings"):
    """
    Save chunks and embeddings for later use.
    Can take either embeddings array directly or load from file.
    """

    # Load embeddings from file if provided
    if embeddings_file is not None:
        print(f"Loading embeddings from {EMBEDDINGS_DIR / embeddings_file}")
        embeddings = np.load(embeddings_file)
        print(f"Loaded embeddings shape: {embeddings.shape}")

    if embeddings is None:
        raise ValueError("Must provide either embeddings array or embeddings_file")

    # Check length match and handle mismatch
    print(f"Chunks: {len(chunks_df)}, Embeddings: {len(embeddings)}")

    if len(chunks_df) != len(embeddings):
        print("Length mismatch")

        if len(embeddings) < len(chunks_df):
            # Pad with zeros
            missing_count = len(chunks_df) - len(embeddings)
            print(f"Padding with {missing_count} zero embeddings")
            empty_embeddings = np.zeros((missing_count, embeddings.shape[1]))
            embeddings = np.vstack([embeddings, empty_embeddings])

        elif len(embeddings) > len(chunks_df):
            # Truncate embeddings
            print(f"Truncating to {len(chunks_df)} embeddings")
            embeddings = embeddings[:len(chunks_df)]

    # Now proceed with saving
    chunks_with_embeddings = chunks_df.copy()

    # Save as pickle for fast loading (preserves numpy arrays)
    with open(EMBEDDINGS_DIR / f"{filename_base}.pkl", "wb") as f:
        pickle.dump({
            'chunks_df': chunks_df,
            'embeddings': embeddings
        }, f)

    # Also save as CSV (without embeddings - too large)
    chunks_df.to_csv(EMBEDDINGS_DIR / f"{filename_base}_metadata.csv", index=False)

    # Save embeddings separately as numpy array
    np.save(EMBEDDINGS_DIR / f"{filename_base}_vectors.npy", embeddings)

    print(f"Saved embeddings and metadata to {EMBEDDINGS_DIR}:")
    print(f"   - {filename_base}.pkl (complete dataset)")
    print(f"   - {filename_base}_metadata.csv (chunk metadata)")
    print(f"   - {filename_base}_vectors.npy (embedding vectors)")

    return embeddings


def load_embeddings(filename_base="buffy_embeddings"):
    """Load previously saved embeddings"""
    try:
        with open(EMBEDDINGS_DIR / f"{filename_base}.pkl", "rb") as f:
            data = pickle.load(f)

        print(f"Loaded {len(data['chunks_df'])} chunks with embeddings")
        return data['chunks_df'], data['embeddings']

    except FileNotFoundError:
        print(f"No saved embeddings found at {filename_base}.pkl")
        return None, None


def test_embeddings(chunks_df, embeddings, sample_queries=None):
    """Test the embeddings with some sample similarity searches"""
    if sample_queries is None:
        sample_queries = [
            "What does Buffy say about vampires?",
            "Giles advises on vampires",
            "Willow talks about vampires",
            "Xander makes jokes about vampires"
        ]

    print("\nTesting embedding similarity...")

    for query in sample_queries:
        print(f"\nQuery: '{query}'")

        # Create embedding for query
        query_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array(query_response.data[0].embedding)

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 most similar

        print("Top matches:")
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            chunk = chunks_df.iloc[idx]
            print(f"   {i + 1}. Score: {similarity:.3f} | Episode {chunk['episode_num']}")
            print(f"      Characters: {chunk['char_combo']}")
            print(f"      Text: {chunk['text'][:200]}...")


def re_embed_failed_chunks_safely(chunks_df, embeddings, failed_indices, max_tokens=10000):
    """Re-embed failed chunks, skipping the monsters (no pun intended)"""

    failed_chunks = chunks_df.iloc[failed_indices]
    skipped_count = 0

    for i, (idx, chunk) in enumerate(failed_chunks.iterrows()):
        text = chunk['text']
        estimated_tokens = len(text) / 4

        # Skip monster chunks (parsing failures)
        if estimated_tokens > max_tokens:
            print(f"🦖 Skipping monster chunk {idx}: {estimated_tokens:.0f} tokens")
            skipped_count += 1
            continue

        # Re-embed normally sized chunks
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            embeddings[idx] = response.data[0].embedding
            print(f"✅ Fixed chunk {idx}")
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Still failed: {idx}")

    print(f"📊 Skipped {skipped_count} monster chunks, they'll stay as zero embeddings")
    return embeddings


# Main execution
if __name__ == "__main__":
    print("Loading dialogue chunks...")
    chunks_df = pd.read_csv(DATA_DIR /'buffy_chunks.csv')
    print(f"Found {len(chunks_df)} chunks to embed")

    # Check if embeddings already exist
    existing_chunks, existing_embeddings = load_embeddings()

    if existing_embeddings is not None:
        print("Using existing embeddings")
        chunks_df, embeddings = existing_chunks, existing_embeddings
    else:
        print("Creating new embeddings...")

        # Create embeddings
        embeddings, total_cost = create_embeddings_batch(
            texts=chunks_df['text'].tolist(),
            batch_size=50  # Smaller batches to be safe
        )

        # Save for future use
        save_embeddings(chunks_df, embeddings)

    # Test the embeddings
    test_embeddings(chunks_df, existing_embeddings)
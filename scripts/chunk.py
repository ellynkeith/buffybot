import pandas as pd
from pathlib import Path
import re
from typing import List, Dict


def create_character_exchanges(dialogue_df):
    """
    Group consecutive dialogue into character exchanges/conversations.
    Creates chunks that represent natural conversation flows (hopefully)
    """
    chunks = []

    # Process each episode separately
    for episode_num in dialogue_df['episode_num'].unique():
        episode_data = dialogue_df[dialogue_df['episode_num'] == episode_num].copy()
        episode_title = episode_data['episode_title'].iloc[0] if len(episode_data) > 0 else "Unknown"

        # Sort by index to maintain dialogue order
        episode_data = episode_data.sort_index()

        current_chunk = []
        current_characters = set()
        chunk_start_idx = 0

        for idx, (_, line) in enumerate(episode_data.iterrows()):
            character = line['character']
            dialogue = line['dialogue']



            if not isinstance(dialogue, str) or len(dialogue.strip()) < 3: # short dialogue likely incomplete parsing
                continue

            # during parsing, we missed instances where an action is after a line, such as
            # BUFFY: Can they do that? (goes to the counter)
            # so remove any dialogue that's between parentheses
            dialogue = re.sub(r'\s*\([^)]+\)\s*', '', dialogue).strip()

            # Decision point: should we continue this chunk or start a new one?
            should_start_new_chunk = False

            # Start new chunk if:
            # 1. We have too many characters (conversation getting unwieldy)
            if len(current_characters) >= 4:
                should_start_new_chunk = True

            # 2. Current chunk is getting too long
            elif len(current_chunk) >= 12:
                should_start_new_chunk = True

            # 3. There's a significant character change (new person dominating)
            elif len(current_chunk) >= 3:
                recent_characters = [line.split(':')[0] for line in current_chunk[-3:]]
                if character not in recent_characters and character not in current_characters:
                    should_start_new_chunk = True

            if should_start_new_chunk and len(current_chunk) >= 2:
                # Save the current chunk
                chunks.append({
                    'chunk_id': f"{episode_num}_{len(chunks)}",
                    'episode_num': episode_num,
                    'episode_title': episode_title,
                    'characters': sorted(list(current_characters)),
                    'main_characters': get_main_characters(current_chunk),
                    'text': '\n'.join(current_chunk),
                    'line_count': len(current_chunk),
                    'word_count': sum(len(line.split()) for line in current_chunk),
                    'starts_at_line': chunk_start_idx
                })

                # Start new chunk
                current_chunk = []
                current_characters = set()
                chunk_start_idx = idx

            # Add current line to chunk
            current_chunk.append(f"{character}: {dialogue}")
            current_characters.add(character)

        # last chunk of the episode
        if len(current_chunk) >= 2:
            chunks.append({
                'chunk_id': f"{episode_num}_{len(chunks)}",
                'episode_num': episode_num,
                'episode_title': episode_title,
                'characters': sorted(list(current_characters)),
                'main_characters': get_main_characters(current_chunk),
                'text': '\n'.join(current_chunk),
                'line_count': len(current_chunk),
                'word_count': sum(len(line.split()) for line in current_chunk),
                'starts_at_line': chunk_start_idx
            })

    return pd.DataFrame(chunks)


def get_main_characters(chunk_lines):
    """Identify the main characters in a chunk (those with most dialogue)"""
    character_counts = {}
    for line in chunk_lines:
        if ':' in line:
            character = line.split(':')[0].strip()
            character_counts[character] = character_counts.get(character, 0) + 1

    # Return top 2-3 characters by dialogue count
    sorted_chars = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
    return [char[0] for char in sorted_chars[:3]]


def analyze_chunks(chunks_df):
    """Analyze the created chunks to understand the data better"""
    print("=== CHUNK ANALYSIS ===")
    print(f"Total chunks created: {len(chunks_df)}")
    print(f"Average lines per chunk: {chunks_df['line_count'].mean():.1f}")
    print(f"Average words per chunk: {chunks_df['word_count'].mean():.1f}")

    print("\n=== CHUNK SIZE DISTRIBUTION ===")
    print(chunks_df['line_count'].describe())

    print("\n=== CHARACTERS PER CHUNK ===")
    chunks_df['num_characters'] = chunks_df['characters'].apply(len)
    print(chunks_df['num_characters'].value_counts().sort_index())

    print("\n=== TOP CHARACTER COMBINATIONS ===")
    chunks_df['char_combo'] = chunks_df['characters'].apply(lambda x: ', '.join(x[:3]))
    print(chunks_df['char_combo'].value_counts().head(10))

    print("\n=== SAMPLE CHUNKS ===")
    sample_chunks = chunks_df.sample(5) if len(chunks_df) >= 5 else chunks_df
    for _, chunk in sample_chunks.iterrows():
        print(f"\n--- {chunk['chunk_id']} ({chunk['episode_title']}) ---")
        print(f"Characters: {', '.join(chunk['characters'])}")
        print(f"Lines: {chunk['line_count']}, Words: {chunk['word_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")


def save_chunks(chunks_df, filename='buffy_chunks.csv'):
    chunks_df.to_csv(filename, index=False)
    print(f"Chunks saved to {filename}")


# Main execution
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data'
    # Load your cleaned dialogue data
    dialogue_df = pd.read_csv(data_path / 'all_dialogue.csv')

    print("Creating character exchange chunks...")
    chunks_df = create_character_exchanges(dialogue_df)

    # Analyze the results
    analyze_chunks(chunks_df)

    # Save for next phase
    save_chunks(chunks_df)

    print("Character exchange chunking complete!")
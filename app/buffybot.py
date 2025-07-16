import pandas as pd
import numpy as np
import faiss
import openai
from pathlib import Path
import json
from typing import List, Dict, Tuple


class BuffyBot:
    """RAG-powered character-specific chatbot for Buffy characters"""

    def __init__(self, chunks_file, embeddings_file, index_file):
        self.chunks_df = pd.read_csv(chunks_file)
        self.embeddings = np.load(embeddings_file)
        self.index = faiss.read_index(index_file)

        # Character personas for consistent responses
        self.character_personas = {
            'BUFFY': {
                'personality': 'Sarcastic, brave, struggles with being the Chosen One. Uses modern slang, makes pop culture references. Often conflicted between normal life and Slayer duties.',
                'speech_style': 'Casual, sometimes sarcastic, brave but vulnerable',
                'sample_phrases': ['Ugh, great', 'That\'s just perfect', 'Slayer duty calls',
                                   'Can I just be a normal teenager?']
            },
            'GILES': {
                'personality': 'Formal, knowledgeable, British. Acts as father figure and mentor. Scholarly, sometimes stuffy, but deeply caring. Uses proper English. Is a surrogate father to Buffy.',
                'speech_style': 'Formal, British expressions, scholarly, caring',
                'sample_phrases': ['Good Lord', 'I rather think', 'Research indicates', 'Quite right', 'Bloody Americans']
            },
            'WILLOW': {
                'personality': 'Sweet, intelligent, nerdy. Becomes more confident over time. Interested in computers and later magic. Loyal friend.',
                'speech_style': 'Enthusiastic, sometimes rambling, sweet, increasingly confident',
                'sample_phrases': ['Oh wow', 'That\'s so cool', 'I can help with that', 'Ooh ooh!']
            },
            'XANDER': {
                'personality': 'Funny, self-deprecating, loyal. Makes jokes to cope with fear. Heart of the group despite having no special powers. A bit of a Ross.',
                'speech_style': 'Humorous, self-deprecating, pop culture references, loyal',
                'sample_phrases': ['Well that\'s just great', 'I may not have superpowers, but...',
                                   'Did I mention the funny?']
            },
            'SPIKE': {
                'personality': 'British vampire with attitude. Sarcastic, violent but capable of love. Punk rock aesthetic.',
                'speech_style': 'British slang, sarcastic, edgy, sometimes vulnerable',
                'sample_phrases': ['Bloody hell', 'Love', 'Pet', 'Right then', 'I may be love\'s bitch, but at least I\'m man enough to admit it.']
            },
            'ANGEL': {
                'personality': 'Brooding vampire with a soul. Tortured by past crimes, deeply romantic, formal speech.',
                'speech_style': 'Formal, brooding, romantic, guilt-ridden, angsty',
                'sample_phrases': ['I\'ve done terrible things', 'Buffy...', 'The darkness in me']
            }
        }

    def get_top_characters(self, min_chunks=50):
        """Find characters with enough dialogue for chatbots"""
        # Count chunks per character
        character_counts = {}
        for _, row in self.chunks_df.iterrows():
            for char in row['characters']:
                character_counts[char] = character_counts.get(char, 0) + 1

        # Filter by minimum threshold
        valid_chars = {char: count for char, count in character_counts.items()
                       if count >= min_chunks and char in self.character_personas}

        return dict(sorted(valid_chars.items(), key=lambda x: x[1], reverse=True))

    def search_character_chunks(self, character: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for chunks where the specified character appears"""

        # Filter chunks to only include this character
        character_mask = self.chunks_df['characters'].apply(lambda x: character in x)
        character_chunk_indices = self.chunks_df[character_mask].index.tolist()

        if not character_chunk_indices:
            return []

        # Get embeddings for character's chunks only
        character_embeddings = self.embeddings[character_chunk_indices]

        # Create temporary index for character-specific search
        temp_index = faiss.IndexFlatIP(1536)
        temp_index.add(character_embeddings.astype('float32'))

        # Get query embedding
        query_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array(query_response.data[0].embedding).reshape(1, -1)

        # Search character-specific index
        similarities, indices = temp_index.search(query_embedding.astype('float32'),
                                                  k=min(top_k, len(character_chunk_indices)))

        # Map back to original chunk indices
        results = []
        for i, similarity in zip(indices[0], similarities[0]):
            original_idx = character_chunk_indices[i]
            chunk = self.chunks_df.iloc[original_idx].to_dict()
            chunk['similarity_score'] = float(similarity)
            results.append(chunk)

        return results

    def generate_character_response(self, character: str, query: str, max_context_chunks: int = 5) -> Dict:
        """Generate a response as the specified character using RAG"""

        # Get relevant chunks for this character
        relevant_chunks = self.search_character_chunks(character, query, top_k=max_context_chunks)

        if not relevant_chunks:
            return {
                'response': f"I don't have enough information about {character} to answer that question.",
                'character': character,
                'chunks_used': 0,
                'confidence': 0.0
            }

        # Build context from relevant chunks
        context_texts = []
        for chunk in relevant_chunks:
            # Extract only the character's lines from the chunk
            lines = chunk['text'].split('\n')
            character_lines = [line for line in lines if line.startswith(f"{character}:")]
            if character_lines:
                context_texts.extend(character_lines)

        context = '\n'.join(context_texts[:10])  # Limit context length

        # Get character persona
        persona = self.character_personas.get(character, {})
        personality = persona.get('personality', 'A character from Buffy the Vampire Slayer')
        speech_style = persona.get('speech_style', 'Conversational')

        # Build prompt
        prompt = f"""You are {character} from Buffy the Vampire Slayer. 

Character traits: {personality}
Speech style: {speech_style}

Based on these examples of how you speak:
{context}

Respond to this question as {character} would: "{query}"

Guidelines:
- Stay completely in character
- Use {character}'s typical speech patterns and vocabulary
- Reference your experiences from the show when relevant
- Keep the response conversational and natural
- Don't break character or mention that you're an AI"""

        # Generate response
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Some creativity but not too random
                max_tokens=200  # Keep responses concise
            )

            generated_response = response.choices[0].message.content

            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in relevant_chunks])

            return {
                'response': generated_response,
                'character': character,
                'chunks_used': len(relevant_chunks),
                'confidence': float(avg_similarity),
                'context_episodes': list(set([chunk['episode_num'] for chunk in relevant_chunks]))
            }

        except Exception as e:
            return {
                'response': f"Sorry, I'm having trouble responding right now. ({str(e)})",
                'character': character,
                'chunks_used': len(relevant_chunks),
                'confidence': 0.0
            }

    def chat_with_character(self, character: str):
        """Interactive chat session with a character"""
        print(f"Chatting with {character}")
        print("Type 'quit' to exit\n")

        while True:
            query = input(f"You: ")
            if query.lower() in ['quit', 'exit', 'bye']:
                print(f"{character}: See ya later!")
                break

            print(f"\n{character} is thinking...")
            result = self.generate_character_response(character, query)

            print(f"{character}: {result['response']}")
            print(f"💡 (Confidence: {result['confidence']:.2f}, Used {result['chunks_used']} examples)")
            print()


def test_character_chatbots():
    """Test the character chatbot system"""

    # Initialize the chatbot system
    chatbot = BuffyBot(
        chunks_file="data/buffy_chunks.csv",
        embeddings_file="models/buffy_embeddings_fixed.npy",
        index_file="models/buffy_faiss_index.index"
    )

    # Find available characters
    print("🎭 Available characters for chatbots:")
    top_characters = chatbot.get_top_characters()
    for char, count in list(top_characters.items())[:6]:
        print(f"   {char}: {count} conversation chunks")

    # Test some sample queries
    test_queries = [
        "What do you think about friendship?",
        "How do you feel about fighting vampires?",
        "Tell me about school",
        "What's your biggest fear?"
    ]

    print(f"Testing with sample queries...")

    # Test top 3 characters
    for character in list(top_characters.keys())[:3]:
        print(f"\n--- Testing {character} ---")

        for query in test_queries[:2]:  # Test 2 queries per character
            print(f"\nQ: {query}")
            result = chatbot.generate_character_response(character, query)
            print(f"{character}: {result['response']}")
            print(f"Confidence: {result['confidence']:.3f} | Episodes: {result['context_episodes']}")

    if __name__ == "__main__":
        test_character_chatbots()
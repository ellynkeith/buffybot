import pandas as pd
import numpy as np
import ast
import faiss
import openai
import uuid
import time
from pathlib import Path
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


class BuffyBot:
    """RAG-powered character-specific chatbot for Buffy characters"""

    def __init__(self,
                     chunks_file=DATA_DIR / "buffy_chunks.csv",
                     embeddings_file = MODEL_DIR / "buffy_embeddings_vectors.npy",
                     index_file = MODEL_DIR / "buffy_faiss_index.index",
                    max_history_length = 5
                 ):
        self.chunks_df = pd.read_csv(chunks_file)
        self.chunks_df['characters'] = self.chunks_df['characters'].apply(ast.literal_eval)
        self.embeddings = np.load(embeddings_file)
        self.index = faiss.read_index(str(index_file))
        self.max_history_length = max_history_length
        self.conversations = {}

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

    def create_session(self, character: str) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())

        self.conversations[session_id] = {
            'character': character,
            'history': [],
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get conversation session, return None if not found"""
        return self.conversations.get(session_id)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        expired_sessions = [
            sid for sid, session in self.conversations.items()
            if session['last_activity'] < cutoff
        ]

        for sid in expired_sessions:
            del self.conversations[sid]

        return len(expired_sessions)

    def add_to_history(self, session_id: str, user_message: str, assistant_response: str):
        """Add exchange to conversation history"""
        if session_id not in self.conversations:
            return False

        history = self.conversations[session_id]['history']

        # Add new exchange
        history.append({
            'user': user_message,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })

        # Trim history if too long
        if len(history) > self.max_history_length:
            self.conversations[session_id]['history'] = history[-self.max_history_length:]

        # Update last activity
        self.conversations[session_id]['last_activity'] = datetime.now()

        return True

    def build_conversational_context(self, current_query: str, history: List[Dict]) -> str:
        """Build search context from current query and conversation history"""
        if not history:
            return current_query

        # Include recent user messages for context
        context_parts = [current_query]

        # Add last 2-3 user questions for broader context
        for exchange in history[-3:]:
            context_parts.append(exchange['user'])

        return ' '.join(context_parts)


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
        character_mask = self.chunks_df['characters'].apply(lambda x: character.upper() in x)
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

    def build_conversational_prompt(self, character: str, current_query: str,
                                    relevant_chunks: List[Dict], history: List[Dict]) -> str:
        """Build prompt that includes conversation history"""

        persona = self.character_personas.get(character, {})
        personality = persona.get('personality', f'A character named {character} from Buffy the Vampire Slayer')
        speech_style = persona.get('speech_style', 'Conversational')

        # Build conversation history context
        conversation_context = ""
        if history:
            conversation_context = "\nPrevious conversation:\n"
            for exchange in history[-3:]:  # Last 3 exchanges for context
                conversation_context += f"Human: {exchange['user']}\n{character}: {exchange['assistant']}\n"
            conversation_context += "\n"

        # Build examples from retrieved chunks
        examples_context = ""
        if relevant_chunks:
            context_texts = []
            for chunk in relevant_chunks:
                lines = chunk['text'].split('\n')
                character_lines = [line for line in lines if line.startswith(f"{character}:")]
                context_texts.extend(character_lines[:3])  # Limit lines per chunk

            if context_texts:
                examples_context = f"\nExamples of how {character} speaks:\n" + '\n'.join(context_texts[:8])

        # Build the full prompt
        prompt = f"""You are {character} from Buffy the Vampire Slayer.

Character traits: {personality}
Speech style: {speech_style}
{examples_context}
{conversation_context}
Now respond to: "{current_query}"

Guidelines:
- Stay completely in character as {character}
- Remember and reference previous parts of our conversation when relevant
- Use {character}'s typical speech patterns and vocabulary
- Keep responses conversational and natural (2-3 sentences usually)
- Don't break character or mention that you're an AI
- If this is continuing a conversation topic, acknowledge that naturally
- Don't preface with the character's name (ie, instead of 'BUFFY: Nice to talk to you!' return just 'Nice to talk to you!')
"""

        return prompt

    def chat(self, session_id: str, user_message: str) -> Dict:
        """Main chat method with conversation memory"""

        # Get session
        session = self.get_session(session_id)
        if not session:
            return {
                'error': 'Session not found',
                'session_id': session_id
            }

        character = session['character']
        history = session['history']

        try:
            # Build search context from current query + history
            search_context = self.build_conversational_context(user_message, history)

            # Get relevant chunks
            relevant_chunks = self.search_character_chunks(character, search_context, top_k=5)

            # Build conversational prompt
            prompt = self.build_conversational_prompt(character, user_message, relevant_chunks, history)

            # Generate response
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )

            assistant_response = response.choices[0].message.content

            # Add to conversation history
            self.add_to_history(session_id, user_message, assistant_response)

            # Calculate confidence
            avg_similarity = np.mean(
                [chunk['similarity_score'] for chunk in relevant_chunks]) if relevant_chunks else 0.0

            return {
                'response': assistant_response,
                'character': character,
                'session_id': session_id,
                'confidence': float(avg_similarity),
                'chunks_used': len(relevant_chunks),
                'context_episodes': list(
                    set([chunk['episode_title'] for chunk in relevant_chunks])) if relevant_chunks else [],
                'conversation_length': len(session['history'])
            }

        except Exception as e:
            return {
                'error': f"Error generating response: {str(e)}",
                'character': character,
                'session_id': session_id
            }

    def start_conversation(self, character: str) -> Dict:
        """Start a new conversation with a character"""
        character = character.upper()
        if character not in self.character_personas:
            return {'error': f'Character {character} not available'}

        session_id = self.create_session(character)
        greeting = self.character_personas[character].get('greeting', f"Hello! I'm {character.capitalize()}.")

        # Add greeting to history
        self.add_to_history(session_id, "[CONVERSATION_START]", greeting)

        return {
            'session_id': session_id,
            'character': character,
            'greeting': greeting
        }

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []

        return session['history']

    def reset_conversation(self, session_id: str) -> bool:
        """Clear conversation history but keep session"""
        if session_id in self.conversations:
            self.conversations[session_id]['history'] = []
            self.conversations[session_id]['last_activity'] = datetime.now()
            return True
        return False

    def get_available_characters(self) -> List[str]:
        """Get list of available characters"""
        return list(self.character_personas.keys())

    def get_session_info(self, session_id: str) -> Dict:
        """Get session metadata"""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            'character': session['character'],
            'conversation_length': len(session['history']),
            'created_at': session['created_at'].isoformat(),
            'last_activity': session['last_activity'].isoformat()
        }

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
                max_tokens=100,
                stop=["\n\n", "Human:", "You:"]
            )

            generated_response = response.choices[0].message.content

            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in relevant_chunks])

            return {
                'response': generated_response,
                'character': character,
                'chunks_used': len(relevant_chunks),
                'confidence': float(avg_similarity),
                'context_episodes': list(set([chunk['episode_title'] for chunk in relevant_chunks]))
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
    chatbot = BuffyBot()

    # Find available characters
    print("Available characters for chatbots:")
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


def format_output(query, character, result):
    # print(f"You: {query}\n")
    print(f"{character}: {result['response']}")
    print(f"\nConfidence: {result['confidence']:.3f} | Episodes: {result['context_episodes']}")


if __name__ == "__main__":
    buffybot = BuffyBot()
    character = "Buffy"
    conversation = buffybot.start_conversation(character)
    session_id = conversation['session_id']

    test_messages = [
        "Hi Buffy! How are you doing?",
        "That sounds tough. What's the hardest part about being the Slayer?",
        "Do you ever wish you could just be a normal teenager?",
        "What keeps you going when things get really bad?"
    ]

    for message in test_messages:
        print(f"\nYou: {message}")
        result = buffybot.chat(session_id, message)

        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            format_output(message, character, result)


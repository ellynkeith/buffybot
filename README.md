# BuffyBot 🧛‍♀️⚔️ #
An AI-powered chatbot system that lets you have conversations with characters from Buffy the Vampire Slayer. Using Retrieval-Augmented Generation (RAG), each character responds based on their actual dialogue from the show, maintaining their unique personality and speech patterns.

## Character-Specific Chatbots: Chat with Buffy, Giles, Willow, Xander, and other main characters ##
- [x] RAG-Powered Responses: Uses actual show dialogue as context for generating authentic responses
- [x] Personality Preservation: Each character maintains their unique voice and mannerisms
- [x] Semantic Search: Advanced vector search to find relevant dialogue examples
- [x] Episode Context: See which episodes inform each response
- [ ] Conversational Context: Track conversation history
- [ ] Interactive Web Interface: Clean Flask-based UI for easy conversations


## Architecture ##
User Query → Character-Specific Retrieval → Context + Persona → LLM Generation → Character Response

## Tech Stack ##
### Data Processing 
- BeautifulSoup
- Pandas
- NumPy

### Embeddings
- OpenAI text-embedding-3-small
  
### Vector Search
- FAISS for fast similarity search
  
### LLM 
- OpenAI GPT-4o-mini for response generation
  
### Storage
- CSV
- NumPy arrays
- pickle

### Web Interface (not yet developed)
- Flask

## Sample Usage -- in `app/buffybot.py`
```
if __name__ == "__main__":
    buffybot = BuffyBot()
    character = "Buffy"
    query = "What do you think about AI?"
    format_output(query, character)
```

## Sample output (often hallucinates!)
```
You: What do you think about AI?

Buffy: Oh wow, AI, huh? That’s a whole thing. I mean, it’s kind of like the nerdy version of a vampire, right? Just lurking in the shadows, learning about humans and trying to figure us out. 

Confidence: 0.254 | Episodes: ['The I In Team - Buffy Episode 69 Transcript', 'Entropy', 'New Moon Rising - Buffy Episode 75 Transcript', 'Primeval - Buffy Episode 77 Transcript', 'Goodbye Iowa - Buffy Episode 70 Transcript']```


# BuffyBot 🧛‍♀️⚔️ #
An AI-powered chatbot system that lets you have conversations with characters from Buffy the Vampire Slayer. Using Retrieval-Augmented Generation (RAG), each character responds based on their actual dialogue from the show, maintaining their unique personality and speech patterns.

## Enter the [scooby chat](https://buffybot-chatbot.onrender.com)!

## Character-Specific Chatbots: Chat with Buffy, Giles, Willow, Xander, and Spike ##
- [x] RAG-Powered Responses: Uses actual show dialogue as context for generating authentic responses
- [x] Personality Preservation: Each character maintains their unique voice and mannerisms
- [x] Semantic Search: Advanced vector search to find relevant dialogue examples
- [x] Cross-Character Memory: Switch between characters while maintaining conversation context
- [x] Episode Context: See which episodes inform each response
- [x] Conversational Context: Track conversation history
- [x] Interactive Web Interface: Clean Flask-based UI for easy conversations
- [ ] Expose Metadata: Optionally display which episodes RAG used for dialogue generation
- [ ] User Feedback: Features to optionally collect feedback (Did AI stay in character voice, was response appropriate/inappropriate)


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

**Web Interface**
* Flask (backend/routing)
* HTML/CSS (90s gothic UI styling)
* JavaScript (real-time chat interactions)

## Enter the [scooby chat](https://buffybot-chatbot.onrender.com)!

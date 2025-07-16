# BuffyBot 🧛‍♀️⚔️ #
An AI-powered chatbot system that lets you have conversations with characters from Buffy the Vampire Slayer. Using Retrieval-Augmented Generation (RAG), each character responds based on their actual dialogue from the show, maintaining their unique personality and speech patterns.
Features

## Character-Specific Chatbots: Chat with Buffy, Giles, Willow, Xander, and other main characters ##
- [x] RAG-Powered Responses: Uses actual show dialogue as context for generating authentic responses
- [ ] Personality Preservation: Each character maintains their unique voice and mannerisms
- [ ] Semantic Search: Advanced vector search to find relevant dialogue examples
- [ ] Interactive Web Interface: Clean Flask-based UI for easy conversations
- [ ] Episode Context: See which episodes inform each response

## Architecture ##
User Query → Character-Specific Retrieval → Context + Persona → LLM Generation → Character Response
Tech Stack
##
Data Processing 
- BeautifulSoup
- Pandas
- NumPy

Embeddings
- OpenAI text-embedding-3-small
  
Vector Search
- FAISS for fast similarity search
  
LLM 
- OpenAI GPT-4o-mini for response generation
  
Storage
- CSV
- NumPy arrays
- pickle



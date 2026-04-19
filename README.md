# Multi-Video RAG Chatbot (YouTube)

An advanced **Retrieval-Augmented Generation (RAG)** system that allows you to:

- Ask questions from **YouTube videos**
- Query **multiple videos simultaneously**
- Compare insights across videos
- Get **accurate, context-grounded answers**

Built using **Hybrid Retrieval (FAISS + BM25)**, **Reranking**, and **LLMs**.

---

## Features

### Single Video Q&A
- Paste a YouTube link  
- Ask questions about the video  
- Get precise answers based on transcript  

---

### Multi-Video Comparison
- Load multiple videos  
- Ask:
  - “Compare these videos”
  - “What are the differences?”  
- Get structured comparisons  

---

### Hybrid Retrieval (Advanced RAG)
- Dense retrieval → FAISS (embeddings)  
- Sparse retrieval → BM25 (keyword search)  
- Combined scoring for better relevance  

---

### Reranking (CrossEncoder)
- Improves retrieval accuracy  
- Selects most relevant chunks before LLM  

---

### Graph-based Context Expansion
- Expands related content using semantic similarity  
- Improves context coverage  

---

### Conversational Memory
- Maintains chat history  
- Context-aware responses  

---

## Architecture

YouTube Video
    ↓ 
Transcript Extraction
    ↓ 
Semantic Chunking
    ↓ 
Node Creation + Graph 
    ↓ 
Hybrid Retrieval (FAISS + BM25) 
    ↓ 
Reranking (CrossEncoder) 
    ↓ 
Graph Expansion 
    ↓ 
Context Builder 
    ↓ 
LLM (OpenRouter) 
    ↓ 
Final Answer

---

## Tech Stack

- Frontend: Streamlit
- LLM API: OpenRouter (GPT-4o-mini or similar)
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- Vector DB: FAISS
- Sparse Retrieval: BM25
- Reranker: CrossEncoder (ms-marco-MiniLM)
- Transcript API: youtube-transcript-api

---

## Installation

### 1 Create virtual environment
python -m venv venv

### 2 Install dependencies
pip install -r requirements.txt

### 3 Setup environment variables

Create a .env file:

OPENROUTER_API_KEY=your_api_key_here
MODEL_NAME=openai/gpt-4o-mini

### 4 Run the App
streamlit run app.py

## Usage
Step 1: Add a Video
Paste a YouTube link
Wait for processing
Step 2: Ask Questions

Examples:

"What is attention mechanism?"
"Summarize the video"
"Explain transformers simply"
Step 3: Multi-video Comparison
Add multiple videos

Ask:
"Compare these videos"
"What are the key differences?"

## Example Queries

What is gradient descent?
Explain attention mechanism
Summarize this video
Compare these two videos
What are the differences between them?

## Limitations

- Requires videos with available transcripts
- Processing time depends on video length
- API-dependent (OpenRouter)

## What This Project Demonstrates

- Advanced RAG pipeline design
- Hybrid retrieval systems
- Semantic + keyword search combination
- LLM prompt engineering
- Multi-document reasoning
- End-to-end AI system building

## Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

## Acknowledgements

- Hugging Face
- LangChain
- Streamlit
- OpenRouter
- Sentence Transformers

## Support

If you found this project useful, consider giving it a ⭐ on GitHub!

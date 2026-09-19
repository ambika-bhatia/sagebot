# 📚 sagebot

A fully local, privacy-first RAG (Retrieval-Augmented Generation) chatbot that lets you upload your study notes (PDF) and ask questions — answered using semantic search and a locally running LLM. No API costs, no data leaves your machine.

## Features
- Upload any PDF (notes, textbook chapters, etc.)
- Semantic chunking and embedding using `sentence-transformers`
- Vector search powered by ChromaDB
- Answer generation using Llama 3.2 (via Ollama) — runs entirely on your local machine
- Clean chat interface built with Streamlit

## Tech Stack
| Component | Tool |
|---|---|
| PDF Parsing | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM (Generation) | Llama 3.2 (via Ollama) |
| UI | Streamlit |

## How It Works
1. User uploads a PDF
2. Text is extracted and split into overlapping chunks
3. Each chunk is converted into an embedding vector
4. Embeddings are stored in a local ChromaDB vector store
5. On a user query, the most relevant chunks are retrieved via semantic similarity
6. Retrieved chunks + query are passed to a local Llama 3.2 model for answer generation

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

**1. Clone the repository**
```bash
git clone https://github.com/ambika-bhatia/sagebot.git
cd sagebot
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Pull the LLM model via Ollama**
```bash
ollama pull llama3.2:3b
```

**5. Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Why Local?
This project intentionally uses a local LLM instead of a paid API — demonstrating a cost-free, privacy-preserving RAG architecture that runs entirely offline after setup.

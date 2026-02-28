# 🔬 ArXiv RAG Bot

An AI-powered question-answering bot for ArXiv research papers, built with RAG (Retrieval-Augmented Generation), FAISS vector search, and local LLMs via Ollama.

## 🚀 Demo

Load any ArXiv paper by ID and ask questions about it in natural language. The bot retrieves the most relevant sections and generates detailed answers using a local language model.

## 🛠️ Tech Stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — Vector similarity search
- **Ollama** — Local LLM inference (Mistral, LLaMA, Aya)
- **Streamlit** — Web interface
- **nomic-embed-text** — Text embeddings

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/arxiv-rag-bot.git
cd arxiv-rag-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pull required models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Run
```bash
streamlit run app.py
```

## 📖 How It Works

1. **Load** — Enter an ArXiv paper ID to download and index the PDF
2. **Chunk** — The paper is split into overlapping chunks of text
3. **Embed** — Each chunk is converted to a vector using nomic-embed-text
4. **Retrieve** — When you ask a question, FAISS finds the most similar chunks
5. **Generate** — Mistral generates a detailed answer based on the retrieved chunks

## 💡 Example Usage

Load paper `1706.03762` (Attention Is All You Need) and ask:
- *What is the purpose of the attention mechanism?*
- *How does the Transformer architecture work?*
- *What are the main experimental results?*

## 📁 Project Structure
```
arxiv-rag-bot/
├── app.py          # Streamlit web interface
├── ingest.py       # PDF download and vector store creation
├── rag.py          # RAG pipeline and question answering
├── requirements.txt
└── README.md
```
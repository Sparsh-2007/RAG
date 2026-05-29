# 🧠 Custom RAG Chat

A lightweight Retrieval-Augmented Generation (RAG) app with a ChatGPT-style web interface. Upload `.txt` or `.md` documents, then ask questions — answers are grounded in your documents via semantic search.

---

## Stack

| Layer | Technology |
|---|---|
| Web UI | Vanilla HTML/CSS/JS (ChatGPT-style) |
| API server | Flask |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | Ollama (`mxbai-embed-large`) |
| LLM | Groq (`llama-3.1-8b-instant`) |

---

## Project Structure

```
.
├── app.py          # Flask API server (upload + query routes)
├── rag.py          # RAG pipeline (ingestion, retrieval, generation)
├── static/
│   └── index.html  # Chat UI
├── uploads/        # Temporary file storage (auto-created)
├── my_chroma_db/   # ChromaDB persistent store (auto-created)
├── .env            # API keys (see setup below)
└── requirements.txt
```

---

## Prerequisites

1. **Python 3.10+**

2. **Ollama** running locally with the embedding model pulled:
   ```bash
   ollama pull mxbai-embed-large
   ollama serve
   ```

3. **Groq API key** — free at [console.groq.com](https://console.groq.com)

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here

# Optional overrides
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=mxbai-embed-large
GROQ_API_KEY="YOUR API KEY"
```

### 3. Run the server

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

### Via the Web UI

1. Click **📎** or drag-and-drop a `.txt` / `.md` / `.markdown` file into the sidebar upload zone
2. Wait for ingestion — the sidebar shows the filename and chunk count when done
3. Type a question in the input bar and press **Enter**

### Via the API directly

**Upload a document:**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@my_document.txt"
```

Response:
```json
{
  "filename": "my_document.txt",
  "ingested_chunks": 14,
  "stored_path": "uploads/my_document_<uuid>.txt"
}
```

**Query the RAG pipeline:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the document?"}'
```

Response:
```json
{
  "answer": "Based on the provided context, the key points are..."
}
```

**CLI query (no server needed):**
```bash
python rag.py --query "What does the document say about X?"
```

---

## Configuration

All tuneable constants live at the top of `rag.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `800` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `UPSERT_BATCH_SIZE` | `64` | Chunks per ChromaDB upsert batch |

And in `app.py`:

| Config | Default | Description |
|---|---|---|
| `UPLOAD_FOLDER` | `uploads` | Directory for uploaded files (env override) |
| `MAX_CONTENT_LENGTH` | `10 MB` | Max upload file size |

---

## How It Works

```
User question
     │
     ▼
ChromaDB semantic search  ──►  Top-K relevant chunks
     │
     ▼
Groq LLM (llama-3.1-8b-instant)
     │  System prompt: "Answer using ONLY the context below"
     │  + retrieved chunks as context
     ▼
Grounded answer
```

1. **Ingestion** — documents are split into overlapping chunks, embedded via Ollama, and stored in a local ChromaDB collection. Chunk IDs are content-hash based, so re-uploading the same file is safe and idempotent.
2. **Retrieval** — at query time, the question is embedded and the top-K most similar chunks are fetched from ChromaDB.
3. **Generation** — the retrieved chunks are injected into the system prompt, and the LLM is instructed to answer only from that context.

---

## Supported File Types

| Extension | Supported |
|---|---|
| `.txt` | ✅ |
| `.md` | ✅ |
| `.markdown` | ✅ |
| `.pdf`, `.docx`, etc. | ❌ Not yet |

---

## Troubleshooting

**`RuntimeError: Cannot reach Ollama`**
Make sure Ollama is running: `ollama serve`. Check `OLLAMA_URL` in your `.env` if using a non-default port.

**`RuntimeError: GROQ_API_KEY is not set`**
Add `GROQ_API_KEY=...` to your `.env` file and restart the server.

**Empty answers / "No relevant context found"**
The question may not semantically match the ingested chunks. Try rephrasing, or lower `TOP_K_RESULTS` to include more chunks.

**Re-uploading the same file adds duplicate chunks**
It won't — chunk IDs are SHA-1 hashes of content, so upsert is idempotent.

---

## Requirements

```
flask
werkzeug
chromadb
langchain-groq
python-dotenv
```

Install with:
```bash
pip install flask werkzeug chromadb langchain-groq python-dotenv
```
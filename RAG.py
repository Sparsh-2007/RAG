import argparse
import getpass
import hashlib
import os , sys
from urllib.parse import unquote, urlparse

import chromadb
import dotenv
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq

dotenv.load_dotenv()

_llm: ChatGroq | None = None
_collection = None

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 64
STREAM_READ_SIZE = 4096
TOP_K_RESULTS = 5         
MAX_TOKENS = 1024         

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")   
OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


def _ensure_groq_key() -> None:
    """Resolve GROQ_API_KEY from env; block interactive prompts entirely in server mode."""
    # 1. Check if the key already exists and isn't empty
    if os.environ.get("GROQ_API_KEY", "").strip():
        return

    # 2. Hard check: Are we running via Flask/Werkzeug local server?
    is_flask_server = (
        os.environ.get("FLASK_RUN_FROM_CLI") == "true" or 
        os.environ.get("WERKZEUG_RUN_MAIN") == "true" or
        "flask" in sys.argv[0].lower()
    )

    # 3. If it's a web server, DO NOT try to read from the terminal. Fail immediately.
    if is_flask_server or not (sys.stdin and sys.stdin.isatty()):
        raise RuntimeError(
            "\n[ERROR] GROQ_API_KEY is missing!\n"
            "Because you are running a Flask server, you cannot type the key interactively.\n"
            "Please create a '.env' file in your project root or export the variable:\n"
        )

    # 4. Pure local CLI script execution path (Safe to prompt here)
    raw_key = getpass.getpass("Enter your Groq API key: ").strip()
    if not raw_key:
        raise ValueError("API key cannot be empty.")
        
    os.environ["GROQ_API_KEY"] = raw_key

def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _ensure_groq_key()
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=MAX_TOKENS,   # FIX 4: was None
            timeout=30,
            max_retries=2,
        )
    return _llm


def get_collection():
    global _collection
    if _collection is None:
        _check_ollama()
        client = chromadb.PersistentClient(path="./my_chroma_db")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=OLLAMA_URL,
            model_name=OLLAMA_MODEL,
        )
        _collection = client.get_or_create_collection(
            name="rag_docs",
            embedding_function=ollama_ef,
        )
    return _collection


def _check_ollama() -> None:
    """Raise a clear error if the Ollama server is not reachable."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=3)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL!r}. "
            "Make sure Ollama is running, or set OLLAMA_URL in your environment."
        ) from exc


# ---------- Document ingestion helpers ----------

def normalize_path(path_or_url: str) -> str:
    if not path_or_url.startswith("file://"):
        return path_or_url
    parsed = urlparse(path_or_url)
    path = unquote(parsed.path)
    if os.name == "nt" and path[:1] == "/" and path[2:3] == ":":
        return path[1:]
    return path


def _chunk_id(prefix: str, content: str) -> str:
    """FIX 5: Deterministic, content-hash-based chunk ID to avoid duplicates on re-ingest."""
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _iter_chunks_from_stream(
    handle,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    read_size: int = STREAM_READ_SIZE,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    buffer = ""
    started = False

    while True:
        data = handle.read(read_size)

        if not data:
            # EOF
            if not started:
                return
            buffer = buffer.rstrip()
            # FIX 2: guard against infinite loop when remaining buffer < chunk_size
            while len(buffer) > 0:
                yield buffer[:chunk_size]
                next_buffer = buffer[chunk_size - overlap:]
                # If overlap == 0 and buffer was shorter than chunk_size, next_buffer
                # would be identical to buffer — break to avoid infinite loop.
                if next_buffer == buffer or len(next_buffer) == 0:
                    break
                buffer = next_buffer
            return

        if not started:
            if not data.strip():
                continue
            started = True
            data = data.lstrip()

        buffer += data
        while len(buffer) >= chunk_size:
            yield buffer[:chunk_size]
            buffer = buffer[chunk_size - overlap:]


def run_rag_pipeline(user_question: str) -> str:
    """Retrieve relevant context from ChromaDB and generate a grounded answer via Groq."""
    question = user_question.strip()
    if not question:
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    llm = get_llm()

    doc_count = collection.count()
    if doc_count == 0:
        return "No documents available. Please upload a document before querying."
    n = min(TOP_K_RESULTS, doc_count)
    search_results = collection.query(
        query_texts=[question],
        n_results=n,
    )

    retrieved_chunks = (search_results.get("documents") or [[]])[0]
    if not retrieved_chunks:
        return "No relevant context found in the vector store."

    context = "\n\n".join([f"[Chunk {i+1}]\n{doc}" for i, doc in enumerate(retrieved_chunks)])
    messages = [
        (
            "system",
            "You are a helpful, conversational technical assistant. Answer the user's question using ONLY the "
            "context provided below. You may expand on it in plain, human-friendly language. If the answer cannot be determined from the context, say so.\n\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---",
        ),
        ("human", question),
    ]

    # Step C: Generate
    ai_response = llm.invoke(messages)
    return ai_response.content


def ingest_document(file_path: str) -> int:
    """Chunk and upsert a document into ChromaDB. Returns number of chunks ingested."""
    collection = get_collection()
    normalized_path = normalize_path(file_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Document not found: {normalized_path}")

    _, ext = os.path.splitext(normalized_path)
    if ext.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError("Only .txt, .md, and .markdown files are supported for ingestion.")

    prefix = os.path.splitext(os.path.basename(normalized_path))[0]
    base_metadata = {"source": os.path.basename(normalized_path)}

    ids, documents, metadatas = [], [], []
    chunk_index = 0

    with open(normalized_path, "r", encoding="utf-8") as handle:
        for chunk in _iter_chunks_from_stream(handle):
            if not chunk.strip():
                continue
            # FIX 5: content-hash ID prevents duplicate chunks on re-ingest
            chunk_id = _chunk_id(prefix, chunk)
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk": chunk_index})
            chunk_index += 1

            if len(ids) >= UPSERT_BATCH_SIZE:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                ids.clear()
                documents.clear()
                metadatas.clear()

    if chunk_index == 0:
        raise ValueError("Document appears to be empty and cannot be ingested.")

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return chunk_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline query runner.")
    parser.add_argument("--query", required=True, help="Question to run through the RAG pipeline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ensure_groq_key()
    print(run_rag_pipeline(args.query))
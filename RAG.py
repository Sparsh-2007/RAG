import argparse
import getpass
import os
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


def _ensure_groq_key(prompt: bool) -> None:
    if "GROQ_API_KEY" not in os.environ:
        if prompt:
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
        else:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _ensure_groq_key(prompt=False)
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    return _llm


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./my_chroma_db")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name="mxbai-embed-large",
        )
        _collection = client.get_or_create_collection(
            name="rag_docs",
            embedding_function=ollama_ef,
        )
    return _collection

# ---------- Document ingestion helpers ----------
def normalize_path(path_or_url: str) -> str:
    if not path_or_url.startswith("file://"):
        return path_or_url

    parsed = urlparse(path_or_url)
    path = unquote(parsed.path)
    if os.name == "nt" and path[:1] == "/" and path[2:3] == ":":
        return path[1:]
    return path


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= text_length:
            break
    return chunks


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
            # EOF: if we never saw content, nothing to yield
            if not started:
                return
            # flush remaining buffer (strip trailing whitespace)
            buffer = buffer.rstrip()
            while buffer:
                yield buffer[:chunk_size]
                buffer = buffer[chunk_size - overlap:]
            return

        # skip leading whitespace until first non-empty content
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
    """Retrieves relevant context from ChromaDB and passes it to Groq LLM for generation."""
    question = user_question.strip()
    if not question:
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    llm = get_llm()

    doc_count = collection.count()
    if doc_count == 0:
        return "No documents available. Ingest a document before querying."

    # Step A: Retrieve context from DB
    # We fetch the top 2 matches to give the LLM sufficient context
    search_results = collection.query(
        query_texts=[question],
        n_results=min(2, doc_count)
    )

    # Flatten the retrieved documents array into a single context string
    retrieved_chunks = (search_results.get("documents") or [[]])[0]
    if not retrieved_chunks:
        return "No relevant context found in the vector store."
    context = "\n".join([f"- {doc}" for doc in retrieved_chunks])

    # Step B: Construct Prompt Structure
    # Use system instructions to strictly constrain the LLM to the retrieved data
    messages = [
        (
            "system",
            "You are a precise technical assistant. Answer the user's question using ONLY the provided "
            "Context below. If the answer cannot be determined from the context, state that you do not know.\n\n"
            f"--- CONTEXT ---\n{context}"
        ),
        ("human", question),
    ]

    # Step C: Generate Grounded Output
    ai_response = llm.invoke(messages)
    return ai_response.content


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="RAG pipeline with optional document ingestion.")
    parser.add_argument("--ingest", help="Path or file:// URL to a .txt/.md document to ingest")
    parser.add_argument("--doc-id-prefix", help="Prefix for generated document chunk IDs")
    parser.add_argument("--domain", help="Optional domain metadata (e.g., AI/ML)")
    parser.add_argument("--complexity", help="Optional complexity metadata (e.g., beginner)")
    parser.add_argument("--query", help="Question to run through the RAG pipeline")
    args = parser.parse_args()
    if not args.ingest and not args.query:
        parser.error("Provide --ingest and/or --query.")
    return args

def ingest_document(file_path: str, doc_id_prefix: str | None = None, metadata: dict | None = None) -> int:
    collection = get_collection()
    normalized_path = normalize_path(file_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Document not found: {normalized_path}")

    _, ext = os.path.splitext(normalized_path)
    ext = ext.lower()
    if ext not in {".txt", ".md", ".markdown"}:
        raise ValueError("Only .txt, .md, and .markdown files are supported for ingestion.")

    prefix = doc_id_prefix or os.path.splitext(os.path.basename(normalized_path))[0]
    base_metadata = {"source": os.path.basename(normalized_path)}
    if metadata:
        base_metadata.update(metadata)
    ids = []
    documents = []
    metadatas = []
    chunk_index = 0

    with open(normalized_path, "r", encoding="utf-8") as handle:
        for chunk in _iter_chunks_from_stream(handle):
            if not chunk:
                continue
            ids.append(f"{prefix}_{chunk_index:04d}")
            documents.append(chunk)
            metadatas.append({**base_metadata, "chunk": chunk_index})
            chunk_index += 1

            if len(ids) >= UPSERT_BATCH_SIZE:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                ids.clear()
                documents.clear()
                metadatas.clear()

    if chunk_index == 0:
        raise ValueError("Document is empty and cannot be ingested.")

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return chunk_index


if __name__ == "__main__":
    args = parse_args()
    if args.ingest:
        metadata = {}
        if args.domain:
            metadata["domain"] = args.domain
        if args.complexity:
            metadata["complexity"] = args.complexity
        ingested_count = ingest_document(
            args.ingest,
            doc_id_prefix=args.doc_id_prefix,
            metadata=metadata or None,
        )
        print(f"Ingested {ingested_count} chunks from {args.ingest}")

    if args.query:
        _ensure_groq_key(prompt=True)
        print(run_rag_pipeline(args.query))

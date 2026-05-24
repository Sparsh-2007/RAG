import getpass
import os
import dotenv
from langchain_groq import ChatGroq
import chromadb

dotenv.load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Initialize ChromaDB Local Storage Client
client = chromadb.PersistentClient(path="./my_chroma_db")

# FIX: Safely fetch or initialize without throwing exceptions
collection = client.get_or_create_collection(name="engineering_docs")

# Upsert documents into your persistent database
collection.upsert(
    ids=["doc_001", "doc_002", "doc_003"],
    documents=[
        "PyTorch handles tensors using automatic differentiation via the autograd engine.",
        "React uses a virtual DOM to optimize UI updates by minimizing direct browser paint operations.",
        "Flask is a micro-framework for Python that handles HTTP routing using Werkzeug."
    ],
    metadatas=[
        {"domain": "AI/ML", "complexity": "advanced"},
        {"domain": "Frontend", "complexity": "intermediate"},
        {"domain": "Backend", "complexity": "beginner"}
    ]
)

# Execute Semantic Search Query
results = collection.query(
    query_texts=["How does deep learning framework compute gradients?"],
    n_results=1
)

print("Retrieved Document:", results["documents"])
print("Similarity Distance:", results["distances"])
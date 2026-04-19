import os
from functools import lru_cache

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Paths
DATA_PATH = "data/maintenance_guidelines.txt"
CHROMA_DB_DIR = "data/chroma_db"

DEFAULT_GUIDELINES = [
    "Brake pads marked Worn Out should be replaced within 24 to 48 hours. Avoid highway use until service is completed.",
    "Tires marked Worn Out should be replaced immediately. Perform alignment and balancing after replacement.",
    "Weak battery status requires voltage and alternator testing. Replace battery if it fails load tests.",
    "Vehicles with risk score above 0.7 should receive a full inspection within 7 days.",
    "If days since last service exceed 180, schedule preventive maintenance within 30 days.",
    "When warranty is expiring within 60 days, prioritize warranty inspection and covered repairs.",
]


class KeywordRetriever:
    """Simple fallback retriever that scores documents by keyword overlap."""

    def __init__(self, texts: list[str]):
        self._docs = [Document(page_content=text) for text in texts]

    def invoke(self, query: str):
        query_lower = (query or "").lower()
        scored_docs = []

        for doc in self._docs:
            text_lower = doc.page_content.lower()
            score = sum(1 for token in query_lower.split() if token and token in text_lower)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        matches = [doc for score, doc in scored_docs if score > 0][:2]

        return matches if matches else self._docs[:2]

def initialize_vector_store():
    """
    Loads the maintenance guidelines, chunks the text,
    and initializes the local ChromaDB vector store.
    """
    if os.path.exists(DATA_PATH):
        print("Loading guidelines from file...")
        loader = TextLoader(DATA_PATH)
        documents = loader.load()
    else:
        print("Guidelines file missing; using built-in fallback guidance.")
        documents = [Document(page_content=text) for text in DEFAULT_GUIDELINES]

    print("Chunking documents...")
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator="\n\n")
    docs = text_splitter.split_documents(documents)

    print("Initializing embedding function (local)...")
    # Using local embeddings to avoid API charges
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating vector store...")
    db = Chroma.from_documents(docs, embedding_function, persist_directory=CHROMA_DB_DIR)
    # Chroma natively persists in recent versions without needing .persist() explicitly for this setup, 
    # but we can explicitly use it if required by older lang-chain configs. To be safe, we just return the db.
    
    print(f"Vector store initialized with {len(docs)} chunks.")
    return db

def get_retriever():
    """
    Returns the retriever interface for the vector store.
    Attempts to load from disk first, if not found, initializes it.
    """
    if os.path.exists(CHROMA_DB_DIR):
        try:
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
            return db.as_retriever(search_kwargs={"k": 2})
        except Exception:
            pass

    if os.path.exists(DATA_PATH):
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = initialize_vector_store()
        return db.as_retriever(search_kwargs={"k": 2})

    return KeywordRetriever(DEFAULT_GUIDELINES)

if __name__ == "__main__":
    # Test script for Member 2
    print("Testing Vector Store Initialization & Retrieval")
    retriever = get_retriever()
    
    query = "The ML model says the risk is high and the tires are in poor condition. What should I do?"
    print(f"\nQuerying for: '{query}'")
    
    results = retriever.invoke(query)
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(res.page_content)
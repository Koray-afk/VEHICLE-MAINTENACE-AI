import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DATA_PATH = "data/maintenance_guidelines.txt"
CHROMA_DB_DIR = "data/chroma_db"

def initialize_vector_store():
    """
    Loads the maintenance guidelines, chunks the text,
    and initializes the local ChromaDB vector store.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Guidelines file not found at {DATA_PATH}")

    print("Loading guidelines...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

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
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(CHROMA_DB_DIR):
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    else:
        db = initialize_vector_store()
    
    # Return a retriever that fetches the top 2 most relevant chunks
    return db.as_retriever(search_kwargs={"k": 2})

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
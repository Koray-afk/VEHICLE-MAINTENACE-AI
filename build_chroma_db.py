"""Bootstrap the local Chroma vector store for the project."""

from __future__ import annotations

import os

from retriever import CHROMA_DB_DIR, initialize_vector_store


def main() -> None:
    """Create the persisted Chroma DB used by the agent retriever."""
    os.makedirs(os.path.dirname(CHROMA_DB_DIR), exist_ok=True)
    db = initialize_vector_store()
    print(f"Chroma DB ready at: {CHROMA_DB_DIR}")
    print(f"Retriever type: {type(db).__name__}")


if __name__ == "__main__":
    main()
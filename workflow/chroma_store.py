from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

class ChromaStoreError(Exception):
    """Raised when there is an error storing chunks in ChromaDB."""

def store_chunks(
    chunks: List[Document],
    persist_directory: str = "chroma_db",
    collection_name: str = "pdf_chunks"
) -> Chroma:
    """
    Store Document chunks in a ChromaDB collection with local disk persistence.

    Args:
        chunks: List of Document objects to store.
        persist_directory: Directory path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.

    Raises:
        ChromaStoreError: If storing or persisting fails.
    """

    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    try:
        vector_store = Chroma.from_documents(
            chunks, 
            embedding_model,
            collection_name="paper_collection",
            persist_directory="chroma_store"
            )
        
        return vector_store
    
    except Exception as e:
        raise ChromaStoreError(f"Failed to store chunks in ChromaDB: {e}") from e
from typing import List
from langchain_core.documents import Document

def clean_chunks(chunks: List[Document]) -> List[Document]:
    """
    Clean each Document chunk by removing newline characters.

    Args:
        chunks: List of Document chunks to clean.

    Returns:
        A new list of Document objects with cleaned page_content.
    """
    cleaned = []
    for chunk in chunks:
        # Replace all newline characters with a space
        cleaned_content = chunk.page_content.replace("\n", " ")
        cleaned.append(Document(page_content=cleaned_content, metadata=chunk.metadata))
    return cleaned
from typing import List
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

# TODO: Error handling for non-parseable text files.


class PDFParsingError(Exception):
    """Raised when PDF parsing fails."""

def parse_pdf(file_path: str) -> List[Document]:
    """
    Load and parse a PDF into LangChain Document objects using PyMuPDFLoader.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of Document objects, each containing page_content and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        PDFParsingError: If parsing fails for any other reason.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise PDFParsingError(f"Failed to parse PDF '{file_path}': {e}") from e
    


result = parse_pdf("../database/non-text-searchable.pdf")

print(f"{result}")
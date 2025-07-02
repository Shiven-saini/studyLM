from typing import Dict, Any
import fitz 

class PDFMetadataError(Exception):
    """Raised when metadata extraction fails."""

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A dictionary of metadata fields (e.g., title, author, creationDate, etc.).

    Raises:
        FileNotFoundError: If the file does not exist.
        PDFMetadataError: If extraction fails.
    """
    try:
        doc = fitz.open(file_path)
    except RuntimeError as e:
        raise FileNotFoundError(f"Cannot open PDF for metadata extraction: {file_path}") from e
    except Exception as e:
        raise PDFMetadataError(f"Failed to open PDF '{file_path}': {e}") from e

    try:
        meta = doc.metadata or {}
        # Optionally normalize keys, e.g. strip leading '/'
        normalized = {k.lstrip("/"): v for k, v in meta.items()}
        return normalized
    except Exception as e:
        raise PDFMetadataError(f"Failed to extract metadata from '{file_path}': {e}") from e
    finally:
        doc.close()

metadata = extract_metadata("../database/attention-paper.pdf")
print(metadata)
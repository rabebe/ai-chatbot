from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

# --- Configuration ---
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


def process_document(document_text: str) -> List[Document]:
    """
    Splits a large document into manageable chunks for processing.

    This uses the RecursiveCharacterTextSplitter for robust splitting
    based on various delimiters, ensuring semantic coherence.

    Args:
        document_text: The full text of the document to be summarized.

    Returns:
        A list of Document objects, each containing a chunk of the text.
    """
    print(
        "--- Document Processor: Splitting document (Size: {len(document_text)} chars) ---"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.create_documents([document_text])

    print(f"--- Document Processor: Created {len(documents)} chunks ---")
    return documents

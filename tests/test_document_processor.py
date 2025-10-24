from src.core.document_processor import process_document

SAMPLE_TEXT_BASE = "This is the start of the document." + ("a" * 3000)
SAMPLE_TEXT = (
    SAMPLE_TEXT_BASE + " " + SAMPLE_TEXT_BASE + " " + SAMPLE_TEXT_BASE + " The end."
)


def test_process_document_splits_correctly():
    """
    Tests that the process_document function correctly splits a large string into the expected number of chunks based on the defined CHUNK_SIZE.
    """

    long_document = SAMPLE_TEXT

    chunks = process_document(long_document)

    assert len(chunks) == 3, "Expected 3 chunks, but got " + (len(chunks))

    for chunk in chunks:
        assert isinstance(chunk.page_content, str), "Chunk content must be a string."
        assert len(chunk.page_content) > 100, "Chunk content should not be empty."

    # total_length = sum(len(chunk.page_content) for chunk in chunks)

    print("Test passed: Document was split into", len(chunks), "chunks.")


def test_process_document_handles_small_input():
    """
    Tests that a small document (smaller than CHUNK_SIZE) is returned as a single chunk.
    """
    small_document = "This is a brief summary text under the chunk size limit"

    chunks = process_document(small_document)

    assert len(chunks) == 1, "Expected 1 chunk for small document, but got " + (
        len(chunks)
    )
    assert chunks[0].page_content == small_document

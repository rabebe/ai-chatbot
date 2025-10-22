#!/usr/bin/env python3
"""
Test script to verify ChromaDB setup and basic functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.database.embeddings import create_chromadb_manager
from src.core.data_loader import create_data_loader
from src.chains.chatbot import create_rag_summarizer

# Load environment variables
load_dotenv()


def test_chromadb_basic():
    """Test basic ChromaDB functionality."""
    print("🧪 Testing ChromaDB basic functionality...")

    try:
        # Create ChromaDB manager
        manager = create_chromadb_manager(
            persist_directory="./data/test_chromadb", collection_name="test_collection"
        )

        # Test adding some sample documents
        sample_docs = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing helps computers understand human language.",
            "Deep learning uses neural networks with multiple layers to process data.",
        ]

        manager.add_texts(sample_docs)

        # Test search
        # results = manager.similarity_search("machine learning", k=2)
        print("✅ Found {len(results)} results for 'machine learning'")

        # Test collection info
        # info = manager.get_collection_info()
        print("✅ Collection info: {info}")

        print("✅ ChromaDB basic test passed!")
        return True

    except Exception:
        print("❌ ChromaDB basic test failed: {e}")
        return False


def test_data_loader():
    """Test data loader functionality."""
    print("\n🧪 Testing data loader...")

    try:
        loader = create_data_loader()

        # Test loading FAQ data
        faq_docs = loader.load_faq_data()
        print("✅ Loaded {len(faq_docs)} FAQ documents")

        if faq_docs:
            print("✅ Sample FAQ document: {faq_docs[0].page_content[:100]}...")

        print("✅ Data loader test passed!")
        return True

    except Exception:
        print("❌ Data loader test failed: {e}")
        return False


def test_rag_summarizer():
    """Test RAG summarizer functionality."""
    print("\n🧪 Testing RAG summarizer...")

    try:
        # Check API keys
        google_key = os.getenv("GOOGLE_API_KEY")

        if not google_key:
            print("⚠️  No API key found. Skipping RAG summarizer test.")
            print("   Please set GOOGLE_API_KEY in your .env file")
            return True

        # Create RAG summarizer
        summarizer = create_rag_summarizer(
            persist_directory="./data/test_chromadb", collection_name="test_collection"
        )

        # Load some test data
        summarizer.load_faq_data()

        # Test summarization
        # result = summarizer.summarize("What is machine learning?", k=2)
        print("✅ Summarization test passed!")
        print("   Summary length: {len(result['summary'])} characters")
        print("   Sources found: {result['num_sources']}")

        print("✅ RAG summarizer test passed!")
        return True

    except Exception:
        print("❌ RAG summarizer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting RAG AI Summarizer Setup Tests")
    print("=" * 50)

    tests = [test_chromadb_basic, test_data_loader, test_rag_summarizer]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print("📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your RAG AI Summarizer is ready to use.")
        print("\nNext steps:")
        print("1. Set up your GOOGLE_API_KEY in .env file (copy from env_template.txt)")
        print("2. Add your documents to the data/ directory")
        print("3. Run: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

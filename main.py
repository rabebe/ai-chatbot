#!/usr/bin/env python3
"""
Main script for RAG-based AI Summarizer using ChromaDB.
Provides CLI interface for document summarization and question answering.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.chains.chatbot import create_rag_summarizer
from check_google_api import check_google_api_key

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment and check for required API keys."""
    # Create logs directory
    Path("./logs").mkdir(exist_ok=True)

    # Check for API keys using the centralized function
    if not check_google_api_key():
        logger.error("API key validation failed")
        return False

    logger.info("Environment setup complete")
    return True


def load_documents(rag_summarizer, file_paths: Optional[List[str]] = None):
    """Load documents into the vector store."""
    try:
        if file_paths:
            logger.info(f"Loading documents from specified files: {file_paths}")
            rag_summarizer.load_documents(file_paths)
        else:
            logger.info("Loading all documents from data directory")
            rag_summarizer.load_documents()

        # Also try to load FAQ data specifically
        rag_summarizer.load_faq_data()

        # Get collection info
        info = rag_summarizer.get_collection_info()
        logger.info(f"Collection info: {info}")

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise


def interactive_mode(rag_summarizer):
    """Run the summarizer in interactive mode."""
    print("\n🤖 RAG AI Summarizer - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  summarize <query>  - Summarize content about a topic")
    print("  ask <question>     - Ask a question")
    print("  search <query>     - Search for relevant documents")
    print("  info               - Show collection information")
    print("  quit               - Exit the program")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break

            if user_input.lower() == "info":
                # info = rag_summarizer.get_collection_info()
                print("\n📊 Collection Information:")
                print("  Documents: {info.get('document_count', 'Unknown')}")
                print("  Collection: {info.get('collection_name', 'Unknown')}")
                print("  Directory: {info.get('persist_directory', 'Unknown')}")
                continue

            # Parse command
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""

            if command == "summarize":
                if not query:
                    print(
                        "Please provide a query to summarize. Example: summarize machine learning basics"
                    )
                    continue

                print(f"\n📝 Summarizing: {query}")
                print("-" * 40)

                result = rag_summarizer.summarize(query)
                print("Summary:\n{result['summary']}")
                print("\n📚 Sources ({result['num_sources']}):")
                for i, source in enumerate(result["sources"], 1):
                    print("  {i}. {source}")

            elif command == "ask":
                if not query:
                    print(
                        "Please provide a question. Example: ask What is machine learning?"
                    )
                    continue

                print("\n❓ Question: {query}")
                print("-" * 40)

                result = rag_summarizer.answer_question(query)
                print("Answer:\n{result['answer']}")
                print("\n📚 Sources ({result['num_sources']}):")
                for i, source in enumerate(result["sources"], 1):
                    print("  {i}. {source}")

            elif command == "search":
                if not query:
                    print(
                        "Please provide a search query. Example: search machine learning"
                    )
                    continue

                print("\n🔍 Searching for: {query}")
                print("-" * 40)

                docs = rag_summarizer.search_documents(query, k=3)
                for i, doc in enumerate(docs, 1):
                    # source = doc.metadata.get('source', 'Unknown')
                    # content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print("\n{i}. Source: {source}")
                    print("   Content: {content}")

            else:
                print("Unknown command: {command}")
                print("Available commands: summarize, ask, search, info, quit")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception:
            logger.error("Error in interactive mode: {e}")
            print("Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="RAG AI Summarizer using ChromaDB")
    parser.add_argument(
        "--mode",
        choices=["interactive", "cli"],
        default="interactive",
        help="Mode to run the summarizer",
    )
    parser.add_argument("--query", type=str, help="Query to summarize (for CLI mode)")
    parser.add_argument(
        "--question", type=str, help="Question to answer (for CLI mode)"
    )
    parser.add_argument("--files", nargs="+", help="Specific files to load")
    parser.add_argument(
        "--embedding-model",
        choices=["huggingface"],
        default="huggingface",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--llm-model", choices=["google"], default="google", help="LLM model to use"
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Number of documents to retrieve"
    )

    args = parser.parse_args()

    # Setup environment
    if not setup_environment():
        sys.exit(1)

    try:
        # Create RAG summarizer
        logger.info("Initializing RAG summarizer...")
        rag_summarizer = create_rag_summarizer(
            embedding_model=args.embedding_model, llm_model=args.llm_model
        )

        # Load documents
        logger.info("Loading documents...")
        load_documents(rag_summarizer, args.files)

        if args.mode == "interactive":
            interactive_mode(rag_summarizer)
        else:
            # CLI mode
            if args.query:
                print("Summarizing: {args.query}")
                # result = rag_summarizer.summarize(args.query, k=args.k)
                print("\nSummary:\n{result['summary']}")
                print("\nSources: {', '.join(result['sources'])}")

            elif args.question:
                print("Question: {args.question}")
                # result = rag_summarizer.answer_question(args.question, k=args.k)
                print("\nAnswer:\n{result['answer']}")
                print("\nSources: {', '.join(result['sources'])}")

            else:
                print("Please provide either --query or --question for CLI mode")
                parser.print_help()

    except Exception:
        logger.error("Error in main: {e}")
        print("Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

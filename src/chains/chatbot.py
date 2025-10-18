"""
RAG-based chatbot implementation using ChromaDB and LangChain.
Handles document retrieval, summarization, and question answering.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from ..database.embeddings import ChromaDBManager
from ..core.data_loader import DataLoader
from .prompts import get_summarization_prompt, get_qa_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSummarizer:
    """RAG-based summarizer using ChromaDB for document retrieval."""
    
    def __init__(
        self,
        chromadb_manager: Optional[ChromaDBManager] = None,
        llm_model: str = "google",  # "google" only
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize RAG summarizer.
        
        Args:
            chromadb_manager: ChromaDB manager instance
            llm_model: LLM model to use ("google" only)
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM output
        """
        self.chromadb_manager = chromadb_manager or ChromaDBManager()
        self.llm = self._get_llm(llm_model, temperature, max_tokens)
        self.data_loader = DataLoader()
        
        # Initialize chains
        self.summarization_chain = self._create_summarization_chain()
        self.qa_chain = self._create_qa_chain()
    
    def _get_llm(self, model_type: str, temperature: float, max_tokens: int):
        """Get LLM based on type."""
        if model_type == "google":
            return ChatGoogleGenerativeAI(
                model="models/gemini-2.0-flash-001",
                temperature=temperature,
                max_tokens=max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                convert_system_message_to_human=True
            )
        else:
            raise ValueError(f"Unsupported LLM model: {model_type}. Only 'google' is supported.")
    
    def _create_summarization_chain(self):
        """Create the summarization chain."""
        prompt = get_summarization_prompt()
        
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents for the prompt."""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                formatted_docs.append(f"[{i}] Source: {source}\n{content}")
            return "\n\n".join(formatted_docs)
        
        chain = (
            {"context": self.chromadb_manager.get_retriever() | format_docs, "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _create_qa_chain(self):
        """Create the question-answering chain."""
        prompt = get_qa_prompt()
        
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents for the prompt."""
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                formatted_docs.append(f"[{i}] Source: {source}\n{content}")
            return "\n\n".join(formatted_docs)
        
        chain = (
            {"context": self.chromadb_manager.get_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Load documents into the vector store.
        
        Args:
            file_paths: List of file paths to load (if None, loads all from data directory)
        """
        if file_paths:
            all_documents = []
            for file_path in file_paths:
                documents = self.data_loader.load_file(file_path)
                all_documents.extend(documents)
        else:
            # Load all documents from data directory
            all_documents = self.data_loader.load_directory()
        
        if all_documents:
            self.chromadb_manager.add_documents(all_documents)
            logger.info(f"Loaded {len(all_documents)} documents into ChromaDB")
        else:
            logger.warning("No documents found to load")
    
    def load_faq_data(self, file_path: Optional[str] = None) -> None:
        """
        Load FAQ data specifically.
        
        Args:
            file_path: Path to FAQ JSON file (defaults to faq_data.json)
        """
        documents = self.data_loader.load_faq_data(file_path)
        if documents:
            self.chromadb_manager.add_documents(documents)
            logger.info(f"Loaded {len(documents)} FAQ documents into ChromaDB")
        else:
            logger.warning("No FAQ data found to load")
    
    def summarize(self, query: str, k: int = 4) -> Dict[str, Any]:
        """
        Summarize content based on a query.
        
        Args:
            query: Query to summarize about
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            # Get retriever with custom k
            retriever = self.chromadb_manager.get_retriever(search_kwargs={"k": k})
            
            # Format documents
            def format_docs(docs: List[Document]) -> str:
                formatted_docs = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content
                    formatted_docs.append(f"[{i}] Source: {source}\n{content}")
                return "\n\n".join(formatted_docs)
            
            # Create chain with custom retriever
            chain = (
                {"context": retriever | format_docs, "query": RunnablePassthrough()}
                | get_summarization_prompt()
                | self.llm
                | StrOutputParser()
            )
            
            # Get summary
            summary = chain.invoke(query)
            
            # Get source documents for metadata
            source_docs = self.chromadb_manager.similarity_search(query, k=k)
            sources = [doc.metadata.get('source', 'Unknown') for doc in source_docs]
            
            return {
                "summary": summary,
                "query": query,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "query": query,
                "sources": [],
                "num_sources": 0
            }
    
    def answer_question(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Answer a question using retrieved context.
        
        Args:
            question: Question to answer
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Get retriever with custom k
            retriever = self.chromadb_manager.get_retriever(search_kwargs={"k": k})
            
            # Format documents
            def format_docs(docs: List[Document]) -> str:
                formatted_docs = []
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content
                    formatted_docs.append(f"[{i}] Source: {source}\n{content}")
                return "\n\n".join(formatted_docs)
            
            # Create chain with custom retriever
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | get_qa_prompt()
                | self.llm
                | StrOutputParser()
            )
            
            # Get answer
            answer = chain.invoke(question)
            
            # Get source documents for metadata
            source_docs = self.chromadb_manager.similarity_search(question, k=k)
            sources = [doc.metadata.get('source', 'Unknown') for doc in source_docs]
            
            return {
                "answer": answer,
                "question": question,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "question": question,
                "sources": [],
                "num_sources": 0
            }
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.chromadb_manager.similarity_search(query, k=k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        return self.chromadb_manager.get_collection_info()


def create_rag_summarizer(
    persist_directory: str = "./data/chromadb",
    collection_name: str = "documents",
    embedding_model: str = "huggingface",
    llm_model: str = "google"
) -> RAGSummarizer:
    """
    Factory function to create a RAG summarizer.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        embedding_model: Embedding model to use (only "huggingface" supported)
        llm_model: LLM model to use (only "google" supported)
        
    Returns:
        RAGSummarizer instance
    """
    chromadb_manager = ChromaDBManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    return RAGSummarizer(
        chromadb_manager=chromadb_manager,
        llm_model=llm_model
    )

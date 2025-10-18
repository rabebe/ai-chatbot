"""
ChromaDB vector store integration for RAG summarizer.
Handles document embedding, storage, and retrieval using ChromaDB.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manages ChromaDB vector store operations for the RAG system."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        collection_name: str = "documents",
        embedding_model: str = "huggingface",  # "huggingface" only
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: Embedding model to use ("huggingface" only)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        self.embeddings = self._get_embeddings(embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = self._get_vectorstore()
        
    def _get_embeddings(self, model_type: str):
        """Get embedding model based on type."""
        if model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise ValueError(f"Unsupported embedding model: {model_type}. Only 'huggingface' is supported.")
    
    def _get_vectorstore(self) -> Chroma:
        """Get or create ChromaDB vector store."""
        try:
            # Try to load existing collection
            vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            return vectorstore
        except Exception as e:
            logger.info(f"Creating new ChromaDB collection: {self.collection_name}")
            # Create new collection
            vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            return vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
            
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Add chunks to vector store
        self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """
        Add raw texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
        """
        if not texts:
            logger.warning("No texts provided to add")
            return
            
        # Convert to documents
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        self.add_documents(documents)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        filter_dict: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
    
    def get_retriever(
        self, 
        search_type: str = "similarity",
        search_kwargs: Optional[Dict] = None
    ):
        """
        Get a retriever for the vector store.
        
        Args:
            search_type: Type of search ("similarity", "mmr", etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}


def create_chromadb_manager(
    persist_directory: str = "./data/chromadb",
    collection_name: str = "documents",
    embedding_model: str = "huggingface"
) -> ChromaDBManager:
    """
    Factory function to create a ChromaDB manager.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        embedding_model: Embedding model to use (only "huggingface" supported)
        
    Returns:
        ChromaDBManager instance
    """
    return ChromaDBManager(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

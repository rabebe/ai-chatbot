"""
Data loading utilities for different file formats.
Supports JSON, TXT, MD, and PDF files for RAG summarization.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    JSONLoader
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading documents from various file formats."""
    
    def __init__(self, data_directory: str = "./data"):
        """
        Initialize data loader.
        
        Args:
            data_directory: Directory containing data files
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
    
    def load_json_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Extract text content from common fields
                        text_content = self._extract_text_from_dict(item)
                        metadata = {
                            "source": str(file_path),
                            "index": i,
                            "type": "json"
                        }
                        # Add original item data to metadata
                        metadata.update(item)
                        documents.append(Document(page_content=text_content, metadata=metadata))
                    else:
                        # Handle list of strings
                        metadata = {
                            "source": str(file_path),
                            "index": i,
                            "type": "json"
                        }
                        documents.append(Document(page_content=str(item), metadata=metadata))
            elif isinstance(data, dict):
                # Single JSON object
                text_content = self._extract_text_from_dict(data)
                metadata = {
                    "source": str(file_path),
                    "type": "json"
                }
                metadata.update(data)
                documents.append(Document(page_content=text_content, metadata=metadata))
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def _extract_text_from_dict(self, data: Dict[str, Any]) -> str:
        """
        Extract text content from a dictionary.
        
        Args:
            data: Dictionary to extract text from
            
        Returns:
            Extracted text content
        """
        # Common text fields to look for
        text_fields = ['text', 'content', 'body', 'description', 'answer', 'question', 'summary']
        
        # Try to find text content
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        
        # If no specific text field found, concatenate all string values
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(f"{key}: {value}")
        
        return "\n".join(text_parts) if text_parts else str(data)
    
    def load_text_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "type": "text"
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_pdf_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "type": "pdf"
                })
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def load_markdown_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of Document objects
        """
        # Markdown files are treated as text files
        return self.load_text_file(file_path)
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load documents from a file based on its extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return self.load_json_file(file_path)
        elif extension in ['.txt', '.md']:
            return self.load_text_file(file_path)
        elif extension == '.pdf':
            return self.load_pdf_file(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []
    
    def load_directory(self, directory_path: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Directory to load from (defaults to data_directory)
            
        Returns:
            List of Document objects
        """
        if directory_path is None:
            directory_path = self.data_directory
        else:
            directory_path = Path(directory_path)
        
        all_documents = []
        supported_extensions = ['.json', '.txt', '.md', '.pdf']
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents = self.load_file(file_path)
                all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def load_faq_data(self, file_path: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        Load FAQ data specifically, handling the expected structure.
        
        Args:
            file_path: Path to FAQ JSON file (defaults to faq_data.json)
            
        Returns:
            List of Document objects
        """
        if file_path is None:
            file_path = self.data_directory / "faq_data.json"
        else:
            file_path = Path(file_path)
        
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle FAQ structure (list of Q&A pairs)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Create a combined text for Q&A
                        question = item.get('question', '')
                        answer = item.get('answer', '')
                        combined_text = f"Question: {question}\nAnswer: {answer}"
                        
                        metadata = {
                            "source": str(file_path),
                            "index": i,
                            "type": "faq",
                            "question": question,
                            "answer": answer
                        }
                        documents.append(Document(page_content=combined_text, metadata=metadata))
            
            logger.info(f"Loaded {len(documents)} FAQ documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading FAQ data from {file_path}: {e}")
            return []


def create_data_loader(data_directory: str = "./data") -> DataLoader:
    """
    Factory function to create a data loader.
    
    Args:
        data_directory: Directory containing data files
        
    Returns:
        DataLoader instance
    """
    return DataLoader(data_directory=data_directory)

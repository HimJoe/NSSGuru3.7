"""
Vector Store Memory Implementation

This module implements a persistent memory system using vector embeddings
for semantic storage and retrieval.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
import time
import numpy as np
from pathlib import Path

from ..utils.logger import get_logger

class VectorStore:
    """
    A vector store implementation for persistent agent memory.
    
    This class provides semantic storage and retrieval of information using
    vector embeddings, allowing agents to access relevant past interactions
    and knowledge.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        storage_path: Optional[str] = None,
        max_entries: int = 1000,
        similarity_threshold: float = 0.7,
        namespace: str = "default",
    ):
        """
        Initialize a vector store memory.
        
        Args:
            embedding_model: Model to use for text embeddings
            storage_path: Path to store the vector database
            max_entries: Maximum number of entries to store
            similarity_threshold: Minimum cosine similarity for relevance
            namespace: Namespace for memory isolation
        """
        self.embedding_model = embedding_model
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.namespace = namespace
        self.logger = get_logger(f"memory.vector_store.{namespace}")
        
        # Set up storage path
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to user's home directory
            self.storage_path = Path.home() / ".agentic_ai" / "vector_store"
        
        # Create directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize memory store
        self.entries = []
        self.embeddings = []
        
        # Load existing data if available
        self._load_from_disk()
        
        self.logger.info(f"Initialized vector store with {len(self.entries)} entries")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text string.
        
        In a real implementation, this would call an embedding API.
        For simplicity, this returns a random vector in this example.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # This is a placeholder for a real embedding API call
        # In production, you would use OpenAI, HuggingFace, or similar
        
        # For demonstration, return a random unit vector
        random_vector = np.random.randn(1536)  # Common embedding dimension
        return random_vector / np.linalg.norm(random_vector)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _load_from_disk(self) -> None:
        """Load vector store data from disk."""
        store_file = self.storage_path / f"{self.namespace}_store.json"
        embedding_file = self.storage_path / f"{self.namespace}_embeddings.npy"
        
        if store_file.exists() and embedding_file.exists():
            try:
                # Load entries
                with open(store_file, 'r') as f:
                    self.entries = json.load(f)
                
                # Load embeddings
                self.embeddings = np.load(embedding_file)
                
                self.logger.info(f"Loaded {len(self.entries)} entries from disk")
            except Exception as e:
                self.logger.error(f"Error loading vector store from disk: {str(e)}")
                # Initialize empty if loading fails
                self.entries = []
                self.embeddings = np.array([])
    
    def _save_to_disk(self) -> None:
        """Save vector store data to disk."""
        store_file = self.storage_path / f"{self.namespace}_store.json"
        embedding_file = self.storage_path / f"{self.namespace}_embeddings.npy"
        
        try:
            # Save entries
            with open(store_file, 'w') as f:
                json.dump(self.entries, f)
            
            # Save embeddings
            if len(self.embeddings) > 0:
                np.save(embedding_file, np.array(self.embeddings))
            
            self.logger.debug(f"Saved {len(self.entries)} entries to disk")
        except Exception as e:
            self.logger.error(f"Error saving vector store to disk: {str(e)}")
    
    def add(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to the vector store.
        
        Args:
            entry: Dictionary with at least a 'content' field
        """
        if 'content' not in entry:
            self.logger.warning("Cannot add entry without 'content' field")
            return
        
        # Add timestamp if not present
        if 'timestamp' not in entry:
            entry['timestamp'] = time.time()
        
        # Get embedding for the content
        embedding = self._get_embedding(entry['content'])
        
        # Add to memory
        self.entries.append(entry)
        
        # Handle empty embeddings array
        if len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Enforce max entries limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            self.embeddings = self.embeddings[-self.max_entries:]
        
        # Save to disk
        self._save_to_disk()
    
    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant entries based on semantic similarity.
        
        Args:
            query: Query text to find relevant entries for
            top_k: Number of top results to retrieve
            
        Returns:
            Formatted string with relevant entries
        """
        if not self.entries:
            return ""
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]
        
        # Get indices of top-k similar entries above threshold
        indices = np.argsort(similarities)[::-1]
        relevant_indices = [
            i for i in indices
            if similarities[i] >= self.similarity_threshold
        ][:top_k]
        
        # Format relevant entries
        if not relevant_indices:
            return ""
        
        results = []
        for idx in relevant_indices:
            entry = self.entries[idx]
            similarity = similarities[idx]
            
            # Format based on entry type
            if 'role' in entry:
                formatted = f"{entry['role'].capitalize()}: {entry['content']}"
            else:
                formatted = entry['content']
            
            results.append(f"[Relevance: {similarity:.2f}] {formatted}")
        
        return "\n\n".join(results)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for entries and return full objects with similarity scores.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to retrieve
            
        Returns:
            List of tuples (entry, similarity score)
        """
        if not self.entries:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]
        
        # Get indices of top-k similar entries above threshold
        indices = np.argsort(similarities)[::-1]
        relevant_indices = [
            i for i in indices
            if similarities[i] >= self.similarity_threshold
        ][:top_k]
        
        # Return entries with scores
        results = []
        for idx in relevant_indices:
            results.append({
                "entry": self.entries[idx],
                "similarity": similarities[idx]
            })
        
        return results
    
    def clear(self) -> None:
        """Clear all entries from memory."""
        self.entries = []
        self.embeddings = np.array([])
        self._save_to_disk()
        self.logger.info(f"Cleared vector store memory for namespace {self.namespace}")

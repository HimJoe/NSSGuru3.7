"""
Working Memory Implementation

This module implements a short-term working memory system for agents
to maintain context within a session or conversation.
"""

from typing import Dict, List, Optional, Any, Union, Deque
from collections import deque
import time
import json

from ..utils.logger import get_logger

class WorkingMemory:
    """
    A working memory implementation for short-term agent memory.
    
    This class provides a conversation-length memory system that maintains
    a fixed-size buffer of recent interactions and important contextual information.
    """
    
    def __init__(
        self,
        max_entries: int = 100,
        buffer_type: str = "fifo",
        namespace: str = "default",
    ):
        """
        Initialize working memory.
        
        Args:
            max_entries: Maximum number of entries to keep in memory
            buffer_type: Type of buffer management ('fifo' or 'priority')
            namespace: Namespace for memory isolation
        """
        self.max_entries = max_entries
        self.buffer_type = buffer_type
        self.namespace = namespace
        self.logger = get_logger(f"memory.working.{namespace}")
        
        # Initialize memory buffer
        self.buffer: Deque = deque(maxlen=max_entries)
        
        # For priority buffer, track priorities
        self.priorities = {}
        
        self.logger.info(f"Initialized working memory with buffer type {buffer_type}")
    
    def add(self, entry: Dict[str, Any], priority: float = 0.0) -> None:
        """
        Add an entry to working memory.
        
        Args:
            entry: Dictionary with memory content
            priority: Priority value (higher = more important, only used in priority mode)
        """
        if not entry:
            return
        
        # Add timestamp if not present
        if 'timestamp' not in entry:
            entry['timestamp'] = time.time()
        
        # Add entry ID if not present
        if 'id' not in entry:
            entry['id'] = f"{len(self.buffer)}-{int(time.time())}"
        
        # Handle priority buffer
        if self.buffer_type == 'priority':
            self.priorities[entry['id']] = priority
            
            # If buffer is full, we might need to remove lowest priority item
            if len(self.buffer) >= self.max_entries:
                lowest_id = min(self.priorities, key=self.priorities.get)
                lowest_priority = self.priorities[lowest_id]
                
                # Only replace if new entry has higher priority
                if priority > lowest_priority:
                    # Find and remove the lowest priority entry
                    for i, e in enumerate(self.buffer):
                        if e['id'] == lowest_id:
                            self.buffer.remove(e)
                            del self.priorities[lowest_id]
                            break
                else:
                    # Don't add if lower priority than all existing entries
                    return
        
        # Add to buffer
        self.buffer.append(entry)
        
        self.logger.debug(f"Added entry to working memory: {entry.get('id')}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all entries in working memory.
        
        Returns:
            List of all memory entries
        """
        return list(self.buffer)
    
    def get_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get most recent entries.
        
        Args:
            count: Number of recent entries to retrieve
            
        Returns:
            List of recent memory entries
        """
        count = min(count, len(self.buffer))
        return list(self.buffer)[-count:]
    
    def get_conversation(self) -> str:
        """
        Get formatted conversation history.
        
        Returns:
            Formatted conversation string
        """
        conversation = []
        
        for entry in self.buffer:
            if 'role' in entry and 'content' in entry:
                role = entry['role'].capitalize()
                content = entry['content']
                conversation.append(f"{role}: {content}")
        
        return "\n\n".join(conversation)
    
    def retrieve(self, query: str = "", count: int = 5) -> str:
        """
        Retrieve relevant entries from working memory.
        
        For working memory, this simply returns recent entries since
        the buffer is small enough to process directly.
        
        Args:
            query: Query string (not used in basic working memory)
            count: Number of entries to retrieve
            
        Returns:
            Formatted string with relevant entries
        """
        # For basic working memory, just return recent entries
        recent = self.get_recent(count)
        
        results = []
        for entry in recent:
            # Format based on entry type
            if 'role' in entry:
                formatted = f"{entry['role'].capitalize()}: {entry['content']}"
            else:
                formatted = entry.get('content', str(entry))
            
            results.append(formatted)
        
        return "\n\n".join(results)
    
    def clear(self) -> None:
        """Clear all entries from working memory."""
        self.buffer.clear()
        self.priorities = {}
        self.logger.info(f"Cleared working memory for namespace {self.namespace}")
    
    def to_json(self) -> str:
        """
        Convert working memory to JSON string.
        
        Returns:
            JSON string of memory contents
        """
        return json.dumps(list(self.buffer), default=str)
    
    def from_json(self, json_str: str) -> None:
        """
        Load working memory from JSON string.
        
        Args:
            json_str: JSON string with memory contents
        """
        try:
            entries = json.loads(json_str)
            self.clear()
            for entry in entries:
                priority = 0.0
                if 'priority' in entry:
                    priority = entry.pop('priority')
                self.add(entry, priority=priority)
            
            self.logger.info(f"Loaded {len(entries)} entries from JSON")
        except Exception as e:
            self.logger.error(f"Error loading from JSON: {str(e)}")

"""
File Operations Tool

This module implements tools for reading and writing files, enabling agents
to interact with the file system in a controlled manner.
"""

import os
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TextIO, BinaryIO

from .base_tool import BaseTool, tool
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_filename

class FileOperationsTool(BaseTool):
    """
    Tool for performing file operations like reading and writing files.
    
    This tool provides a safe interface for agents to interact with the
    file system, with appropriate restrictions and validation.
    """
    
    def __init__(
        self,
        allowed_directories: Optional[List[str]] = None,
        default_directory: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        name: str = "file_operations",
    ):
        """
        Initialize the file operations tool.
        
        Args:
            allowed_directories: List of directories the tool is allowed to access
            default_directory: Default directory for operations
            max_file_size: Maximum file size in bytes
            name: Tool name
        """
        # Define parameter schema
        params_schema = {
            "operation": {
                "type": "string",
                "description": "File operation to perform (read, write, list, delete)",
                "enum": ["read", "write", "list", "delete"]
            },
            "file_path": {
                "type": "string",
                "description": "Path to the file"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file (for write operation)"
            },
            "format": {
                "type": "string",
                "description": "Format for read/write operations (text, json, csv, yaml)",
                "enum": ["text", "json", "csv", "yaml", "binary"]
            }
        }
        
        super().__init__(
            name=name,
            description="Perform file operations like reading and writing files",
            usage=f'{name}(operation="read", file_path="data/example.txt", format="text")',
            params_schema=params_schema,
            required_params=["operation", "file_path"],
            returns_description="Result of the file operation"
        )
        
        # Set up allowed directories
        self.allowed_directories = allowed_directories or []
        
        # Add current directory if no directories specified
        if not self.allowed_directories:
            self.allowed_directories.append(os.getcwd())
        
        # Add default directory if specified
        self.default_directory = default_directory or os.getcwd()
        if self.default_directory not in self.allowed_directories:
            self.allowed_directories.append(self.default_directory)
        
        # Convert all paths to absolute paths
        self.allowed_directories = [os.path.abspath(d) for d in self.allowed_directories]
        self.default_directory = os.path.abspath(self.default_directory)
        
        self.max_file_size = max_file_size
        
        # Set up logger
        self.logger = get_logger(f"tool.{name}")
        
        self.logger.info(f"Initialized file operations tool with {len(self.allowed_directories)} allowed directories")
    
    def _validate_path(self, file_path: str) -> str:
        """
        Validate that the file path is allowed and return the absolute path.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Absolute file path
            
        Raises:
            ValueError: If the path is not allowed
        """
        # If path is not absolute, make it relative to default directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.default_directory, file_path)
        
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check if path is in allowed directories
        allowed = any(
            os.path.commonpath([abs_path, allowed_dir]) == allowed_dir
            for allowed_dir in self.allowed_directories
        )
        
        if not allowed:
            raise ValueError(f"File path '{file_path}' is not in allowed directories")
        
        return abs_path
    
    def _read_file(self, file_path: str, format: str) -> Any:
        """
        Read a file with the specified format.
        
        Args:
            file_path: Path to the file
            format: Format to read (text, json, csv, yaml, binary)
            
        Returns:
            File contents in the specified format
        """
        # Validate file path
        abs_path = self._validate_path(file_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        
        # Check file size
        if os.path.getsize(abs_path) > self.max_file_size:
            raise ValueError(f"File '{file_path}' exceeds maximum size of {self.max_file_size} bytes")
        
        # Read file based on format
        try:
            if format == "text":
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif format == "json":
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif format == "csv":
                with open(abs_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            
            elif format == "yaml":
                with open(abs_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            
            elif format == "binary":
                with open(abs_path, 'rb') as f:
                    return f.read()
            
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error reading file '{file_path}': {str(e)}")
            raise
    
    def _write_file(self, file_path: str, content: Any, format: str) -> Dict[str, Any]:
        """
        Write content to a file with the specified format.
        
        Args:
            file_path: Path to the file
            content: Content to write
            format: Format to write (text, json, csv, yaml, binary)
            
        Returns:
            Dictionary with operation result
        """
        # Validate file path
        abs_path = self._validate_path(file_path)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(abs_path)
        os.makedirs(directory, exist_ok=True)
        
        # Write file based on format
        try:
            if format == "text":
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif format == "json":
                with open(abs_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
            
            elif format == "csv":
                if not isinstance(content, list) or not content:
                    raise ValueError("CSV content must be a non-empty list of dictionaries")
                
                with open(abs_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=content[0].keys())
                    writer.writeheader()
                    writer.writerows(content)
            
            elif format == "yaml":
                with open(abs_path, 'w', encoding='utf-8') as f:
                    yaml.dump(content, f)
            
            elif format == "binary":
                with open(abs_path, 'wb') as f:
                    if isinstance(content, str):
                        f.write(content.encode('utf-8'))
                    else:
                        f.write(content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Get file size
            file_size = os.path.getsize(abs_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "abs_path": abs_path,
                "size": file_size,
                "format": format
            }
        
        except Exception as e:
            self.logger.error(f"Error writing file '{file_path}': {str(e)}")
            raise
    
    def _list_files(self, directory: str) -> List[Dict[str, Any]]:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list
            
        Returns:
            List of file information dictionaries
        """
        # Validate directory path
        abs_path = self._validate_path(directory)
        
        # Check if directory exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Directory '{directory}' not found")
        
        # Check if path is a directory
        if not os.path.isdir(abs_path):
            raise NotADirectoryError(f"Path '{directory}' is not a directory")
        
        # List files and directories
        entries = []
        for entry in os.scandir(abs_path):
            entry_info = {
                "name": entry.name,
                "path": os.path.join(directory, entry.name),
                "is_file": entry.is_file(),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if entry.is_file() else None,
                "modified": entry.stat().st_mtime
            }
            entries.append(entry_info)
        
        return entries
    
    def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with operation result
        """
        # Validate file path
        abs_path = self._validate_path(file_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        
        # Check if path is a file
        if not os.path.isfile(abs_path):
            raise IsADirectoryError(f"Path '{file_path}' is not a file")
        
        # Delete the file
        os.remove(abs_path)
        
        return {
            "success": True,
            "file_path": file_path,
            "abs_path": abs_path
        }
    
    def execute(
        self,
        operation: str,
        file_path: str,
        content: Optional[Any] = None,
        format: str = "text"
    ) -> Any:
        """
        Execute a file operation.
        
        Args:
            operation: Operation to perform (read, write, list, delete)
            file_path: Path to the file
            content: Content to write (for write operation)
            format: Format for read/write operations
            
        Returns:
            Result of the file operation
        """
        self.logger.info(f"Executing file operation: {operation} on {file_path}")
        
        if operation == "read":
            return self._read_file(file_path, format)
        
        elif operation == "write":
            if content is None:
                raise ValueError("Content is required for write operation")
            return self._write_file(file_path, content, format)
        
        elif operation == "list":
            return self._list_files(file_path)
        
        elif operation == "delete":
            return self._delete_file(file_path)
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")


@tool(
    name="read_file",
    description="Read a file with the specified format",
    params_schema={
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        },
        "format": {
            "type": "string",
            "description": "Format to read (text, json, csv, yaml, binary)",
            "enum": ["text", "json", "csv", "yaml", "binary"]
        }
    },
    required_params=["file_path"]
)
def read_file(file_path: str, format: str = "text") -> Any:
    """
    Read a file with the specified format.
    
    Args:
        file_path: Path to the file
        format: Format to read (text, json, csv, yaml, binary)
        
    Returns:
        File contents in the specified format
    """
    tool = FileOperationsTool()
    return tool.execute(operation="read", file_path=file_path, format=format)


@tool(
    name="write_file",
    description="Write content to a file with the specified format",
    params_schema={
        "file_path": {
            "type": "string",
            "description": "Path to the file to write"
        },
        "content": {
            "type": "object",
            "description": "Content to write to the file"
        },
        "format": {
            "type": "string",
            "description": "Format to write (text, json, csv, yaml, binary)",
            "enum": ["text", "json", "csv", "yaml", "binary"]
        }
    },
    required_params=["file_path", "content"]
)
def write_file(file_path: str, content: Any, format: str = "text") -> Dict[str, Any]:
    """
    Write content to a file with the specified format.
    
    Args:
        file_path: Path to the file
        content: Content to write
        format: Format to write (text, json, csv, yaml, binary)
        
    Returns:
        Dictionary with operation result
    """
    tool = FileOperationsTool()
    return tool.execute(operation="write", file_path=file_path, content=content, format=format)

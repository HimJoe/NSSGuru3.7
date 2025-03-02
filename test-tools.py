"""
Tests for the tool implementations in the Agentic AI framework.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import json
from typing import Dict, List, Any

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the framework
from src.agentic_ai.tools import BaseTool, tool
from src.agentic_ai.tools import WebSearchTool, CodeExecutionTool, FileOperationsTool
from src.agentic_ai.tools import read_file, write_file

class TestBaseTool:
    """Tests for the BaseTool class."""
    
    def test_initialization(self):
        """Test that a base tool can be initialized with required values."""
        tool = BaseTool(
            name="test_tool",
            description="A test tool",
            usage="test_tool(param1='value')",
            params_schema={"param1": {"type": "string", "description": "A parameter"}},
            required_params=["param1"],
            returns_description="Test result"
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.usage == "test_tool(param1='value')"
        assert tool.params_schema == {"param1": {"type": "string", "description": "A parameter"}}
        assert tool.required_params == ["param1"]
        assert tool.returns_description == "Test result"
    
    def test_validate_params_method(self):
        """Test the validate_params method of BaseTool."""
        tool = BaseTool(
            name="test_tool",
            description="A test tool",
            usage="test_tool(param1='value', param2=123)",
            params_schema={
                "param1": {"type": "string", "description": "A string parameter"},
                "param2": {"type": "integer", "description": "An integer parameter", "minimum": 0, "maximum": 100}
            },
            required_params=["param1"],
            returns_description="Test result"
        )
        
        # Test with valid parameters
        errors = tool.validate_params({"param1": "value", "param2": 50})
        assert errors == {}
        
        # Test with missing required parameter
        errors = tool.validate_params({"param2": 50})
        assert "param1" in errors
        
        # Test with invalid type
        errors = tool.validate_params({"param1": "value", "param2": "not_an_integer"})
        assert "param2" in errors
        
        # Test with value out of range
        errors = tool.validate_params({"param1": "value", "param2": 200})
        assert "param2" in errors
    
    def test_call_method(self):
        """Test the __call__ method of BaseTool."""
        # Create a subclass to implement execute
        class TestTool(BaseTool):
            def execute(self, param1, param2=None):
                return {"param1": param1, "param2": param2}
        
        tool = TestTool(
            name="test_tool",
            description="A test tool",
            usage="test_tool(param1='value')",
            params_schema={"param1": {"type": "string", "description": "A parameter"}},
            required_params=["param1"],
            returns_description="Test result"
        )
        
        # Test calling with valid parameters
        result = tool(param1="value", param2=123)
        assert result == {"param1": "value", "param2": 123}
        
        # Test calling with missing required parameter
        with pytest.raises(ValueError, match="Parameter validation failed"):
            tool(param2=123)
    
    def test_execute_method(self):
        """Test that the abstract execute method raises NotImplementedError."""
        tool = BaseTool(
            name="test_tool",
            description="A test tool",
            usage="test_tool()",
            params_schema={},
            required_params=[],
            returns_description="Test result"
        )
        
        with pytest.raises(NotImplementedError):
            tool.execute()


class TestToolDecorator:
    """Tests for the tool decorator."""
    
    def test_basic_decoration(self):
        """Test that the tool decorator creates a proper tool instance."""
        # Define a function with the tool decorator
        @tool(
            name="test_decorator",
            description="A test decorator",
            params_schema={"param1": {"type": "string", "description": "A parameter"}}
        )
        def test_function(param1):
            """Test function docstring."""
            return {"param1": param1}
        
        # Verify that the decorator returned a BaseTool instance
        assert isinstance(test_function, BaseTool)
        assert test_function.name == "test_decorator"
        assert test_function.description == "A test decorator"
        assert "param1" in test_function.params_schema
        
        # Test calling the tool
        result = test_function(param1="test_value")
        assert result == {"param1": "test_value"}
    
    def test_auto_schema_generation(self):
        """Test that the tool decorator automatically generates schema from function signature."""
        # Define a function with the tool decorator but no explicit schema
        @tool(
            name="auto_schema",
            description="A tool with auto schema"
        )
        def auto_schema_function(param1: str, param2: int = 0):
            """Function with type hints."""
            return {"param1": param1, "param2": param2}
        
        # Verify that the schema was generated from the function signature
        assert "param1" in auto_schema_function.params_schema
        assert "param2" in auto_schema_function.params_schema
        assert auto_schema_function.params_schema["param1"]["type"] == "string"
        assert auto_schema_function.params_schema["param2"]["type"] == "integer"
        
        # Verify that required parameters were identified correctly
        assert "param1" in auto_schema_function.required_params
        assert "param2" not in auto_schema_function.required_params
        
        # Test calling the tool
        result = auto_schema_function(param1="test")
        assert result == {"param1": "test", "param2": 0}


class TestWebSearchTool:
    """Tests for the WebSearchTool class."""
    
    def test_initialization(self):
        """Test that a web search tool can be initialized with default values."""
        tool = WebSearchTool()
        
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert tool.max_results == 5
    
    def test_initialization_with_custom_values(self):
        """Test that a web search tool can be initialized with custom values."""
        tool = WebSearchTool(
            api_key="test_key",
            search_engine="bing",
            max_results=10,
            name="custom_search"
        )
        
        assert tool.name == "custom_search"
        assert tool.api_key == "test_key"
        assert tool.search_engine == "bing"
        assert tool.max_results == 10
    
    def test_execute_method(self):
        """Test the execute method of WebSearchTool."""
        tool = WebSearchTool()
        
        # Test with basic parameters
        results = tool.execute(query="test query")
        
        # Verify the results (using mock data in the implementation)
        assert len(results) > 0
        assert "title" in results[0]
        assert "url" in results[0]
        assert "snippet" in results[0]
        assert "test query" in results[0]["title"]
        
        # Test with more parameters
        results = tool.execute(
            query="test query",
            num_results=3
        )
        
        # Verify the correct number of results
        assert len(results) == 3


class TestCodeExecutionTool:
    """Tests for the CodeExecutionTool class."""
    
    def test_initialization(self):
        """Test that a code execution tool can be initialized with default values."""
        tool = CodeExecutionTool()
        
        assert tool.name == "execute_python"
        assert "python" in tool.description.lower()
        assert tool.timeout == 5.0
        assert tool.max_output_length == 8192
    
    def test_initialization_with_custom_values(self):
        """Test that a code execution tool can be initialized with custom values."""
        tool = CodeExecutionTool(
            timeout=10.0,
            max_output_length=4096,
            allow_plots=False,
            name="custom_python"
        )
        
        assert tool.name == "custom_python"
        assert tool.timeout == 10.0
        assert tool.max_output_length == 4096
        assert tool.allow_plots is False
    
    def test_validate_code_method(self):
        """Test the _validate_code method of CodeExecutionTool."""
        tool = CodeExecutionTool()
        
        # Test with valid code
        valid_code = """
        import math
        result = math.sqrt(16)
        print(result)
        """
        assert tool._validate_code(valid_code) is None
        
        # Test with invalid imports
        invalid_imports = """
        import os
        os.system('echo "This is not allowed"')
        """
        error = tool._validate_code(invalid_imports)
        assert error is not None
        assert "os.system" in error
        
        # Test with disallowed builtins
        invalid_builtins = """
        eval("__import__('os').system('echo not allowed')")
        """
        error = tool._validate_code(invalid_builtins)
        assert error is not None
        assert "eval" in error
    
    def test_execute_method(self):
        """Test the execute method of CodeExecutionTool."""
        tool = CodeExecutionTool()
        
        # Test with simple code
        code = """
        result = 2 + 2
        print(f"The result is {result}")
        result
        """
        
        result = tool.execute(code=code)
        
        # Verify the result structure
        assert result["success"] is True
        assert "The result is 4" in result["stdout"]
        assert result["result"] == 4
        assert result["error"] is None
        
        # Test with syntax error
        invalid_code = """
        if True
            print("Missing colon")
        """
        
        result = tool.execute(code=invalid_code)
        
        # Verify the error handling
        assert result["success"] is False
        assert result["error"] is not None
        assert "SyntaxError" in str(result["error"])


class TestFileOperationsTool:
    """Tests for the FileOperationsTool class."""
    
    @pytest.fixture(scope="function")
    def file_tool(self, temp_directory):
        """Create a file operations tool with a temporary directory."""
        return FileOperationsTool(
            allowed_directories=[temp_directory],
            default_directory=temp_directory
        )
    
    def test_initialization(self):
        """Test that a file operations tool can be initialized with default values."""
        tool = FileOperationsTool()
        
        assert tool.name == "file_operations"
        assert "file" in tool.description.lower()
        assert len(tool.allowed_directories) > 0
        assert tool.default_directory is not None
    
    def test_initialization_with_custom_values(self):
        """Test that a file operations tool can be initialized with custom values."""
        tool = FileOperationsTool(
            allowed_directories=["/tmp"],
            default_directory="/tmp",
            max_file_size=1024,
            name="custom_file_tool"
        )
        
        assert tool.name == "custom_file_tool"
        assert "/tmp" in tool.allowed_directories
        assert tool.default_directory == "/tmp"
        assert tool.max_file_size == 1024
    
    def test_validate_path_method(self, file_tool, temp_directory):
        """Test the _validate_path method of FileOperationsTool."""
        # Test with valid path
        valid_path = os.path.join(temp_directory, "test.txt")
        abs_path = file_tool._validate_path(valid_path)
        assert abs_path == os.path.abspath(valid_path)
        
        # Test with path outside allowed directories
        invalid_path = "/etc/passwd"
        with pytest.raises(ValueError, match="not in allowed directories"):
            file_tool._validate_path(invalid_path)
    
    def test_read_write_operations(self, file_tool, temp_directory):
        """Test the read and write operations of FileOperationsTool."""
        # Create a test file path
        test_file = os.path.join(temp_directory, "test.txt")
        
        # Test writing a file
        content = "This is a test file content"
        write_result = file_tool.execute(
            operation="write",
            file_path=test_file,
            content=content,
            format="text"
        )
        
        # Verify write result
        assert write_result["success"] is True
        assert write_result["file_path"] == test_file
        
        # Verify the file was actually written
        assert os.path.exists(test_file)
        
        # Test reading the file
        read_result = file_tool.execute(
            operation="read",
            file_path=test_file,
            format="text"
        )
        
        # Verify read result
        assert read_result == content
    
    def test_json_operations(self, file_tool, temp_directory):
        """Test the JSON operations of FileOperationsTool."""
        # Create a test file path
        test_file = os.path.join(temp_directory, "test.json")
        
        # Test writing JSON
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        file_tool.execute(
            operation="write",
            file_path=test_file,
            content=data,
            format="json"
        )
        
        # Test reading JSON
        read_result = file_tool.execute(
            operation="read",
            file_path=test_file,
            format="json"
        )
        
        # Verify the JSON was preserved
        assert read_result["key"] == "value"
        assert read_result["number"] == 123
        assert read_result["list"] == [1, 2, 3]
    
    def test_list_operation(self, file_tool, temp_directory):
        """Test the list operation of FileOperationsTool."""
        # Create some test files
        os.makedirs(os.path.join(temp_directory, "subdir"))
        with open(os.path.join(temp_directory, "file1.txt"), "w") as f:
            f.write("content")
        
        # Test listing directory
        list_result = file_tool.execute(
            operation="list",
            file_path=temp_directory
        )
        
        # Verify list result
        assert isinstance(list_result, list)
        assert len(list_result) == 2  # subdir and file1.txt
        
        # Verify file info
        file_info = next(item for item in list_result if item["name"] == "file1.txt")
        assert file_info["is_file"] is True
        assert file_info["size"] > 0
        
        # Verify directory info
        dir_info = next(item for item in list_result if item["name"] == "subdir")
        assert dir_info["is_dir"] is True
    
    def test_delete_operation(self, file_tool, temp_directory):
        """Test the delete operation of FileOperationsTool."""
        # Create a test file
        test_file = os.path.join(temp_directory, "to_delete.txt")
        with open(test_file, "w") as f:
            f.write("content to delete")
        
        # Verify file exists
        assert os.path.exists(test_file)
        
        # Test deleting the file
        delete_result = file_tool.execute(
            operation="delete",
            file_path=test_file
        )
        
        # Verify delete result
        assert delete_result["success"] is True
        assert delete_result["file_path"] == test_file
        
        # Verify file was actually deleted
        assert not os.path.exists(test_file)
    
    def test_helper_functions(self, temp_directory):
        """Test the helper functions (read_file, write_file)."""
        # Create a test file path
        test_file = os.path.join(temp_directory, "helper_test.txt")
        
        # Mock the FileOperationsTool.execute method
        with patch("src.agentic_ai.tools.file_operations.FileOperationsTool.execute") as mock_execute:
            # Set up the mock
            mock_execute.return_value = "mock result"
            
            # Test read_file helper
            result = read_file(file_path=test_file, format="text")
            assert result == "mock result"
            mock_execute.assert_called_with(operation="read", file_path=test_file, format="text")
            
            # Test write_file helper
            result = write_file(file_path=test_file, content="test content", format="text")
            assert result == "mock result"
            mock_execute.assert_called_with(operation="write", file_path=test_file, content="test content", format="text")

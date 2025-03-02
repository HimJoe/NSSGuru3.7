"""
Code Execution Tool Implementation

This module implements a sandboxed code execution tool that allows agents to
run Python code in a controlled environment.
"""

import sys
import os
import io
import traceback
import ast
import time
from typing import Dict, Any, Optional, Union, List
import contextlib

from .base_tool import BaseTool
from ..utils.logger import get_logger

class CodeExecutionTool(BaseTool):
    """
    A tool for executing Python code in a sandboxed environment.
    
    This tool allows agents to run Python code for data processing, analysis,
    and other computational tasks in a controlled environment with security
    restrictions.
    """
    
    # Set of modules that are allowed to be imported
    ALLOWED_MODULES = {
        "math", "random", "datetime", "json", "re", "string",
        "collections", "itertools", "functools", "operator",
        "statistics", "numpy", "pandas", "matplotlib", "seaborn",
        "sklearn", "scipy", "nltk", "requests", "bs4", "urllib",
    }
    
    # Set of built-in functions that are disallowed
    DISALLOWED_BUILTINS = {
        "eval", "exec", "compile", "__import__", "open", "input", "exit", "quit",
        "help", "globals", "locals", "vars", "dir"
    }
    
    def __init__(
        self,
        timeout: float = 5.0,
        max_output_length: int = 8192,
        allow_plots: bool = True,
        additional_allowed_modules: Optional[List[str]] = None,
        name: str = "execute_python",
    ):
        """
        Initialize the code execution tool.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of output text
            allow_plots: Whether to allow matplotlib plots
            additional_allowed_modules: Additional modules to allow
            name: Tool name
        """
        # Define parameter schema
        params_schema = {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            },
            "inputs": {
                "type": "object",
                "description": "Dictionary of input variables for the code"
            },
            "timeout": {
                "type": "number",
                "description": "Maximum execution time in seconds",
                "minimum": 0.1,
                "maximum": 60.0
            },
            "description": {
                "type": "string",
                "description": "Description of what the code does (for logging)"
            }
        }
        
        super().__init__(
            name=name,
            description="Execute Python code in a sandboxed environment",
            usage=f'{name}(code="import math\\nresult = math.sqrt(16)\\nprint(result)", inputs={"x": 5})',
            params_schema=params_schema,
            required_params=["code"],
            returns_description="Dictionary with execution results, including stdout, stderr, and return value"
        )
        
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.allow_plots = allow_plots
        
        # Update allowed modules
        if additional_allowed_modules:
            self.ALLOWED_MODULES.update(additional_allowed_modules)
        
        self.logger = get_logger(f"tool.{name}")
        
        self.logger.info(f"Initialized code execution tool with timeout {timeout}s")
    
    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            Error message if validation fails, None if it passes
        """
        try:
            # Parse the AST
            tree = ast.parse(code)
            
            # Check for disallowed constructs
            for node in ast.walk(tree):
                # Check for imports of disallowed modules
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split('.')[0] not in self.ALLOWED_MODULES:
                            return f"Import of module '{name.name}' is not allowed"
                
                # Check for import from disallowed modules
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                        return f"Import from module '{node.module}' is not allowed"
                
                # Check for os.system calls and similar
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            # Check for os.system, os.popen, etc.
                            if node.func.value.id == 'os' and node.func.attr in ['system', 'popen', 'spawn', 'exec']:
                                return f"Use of os.{node.func.attr} is not allowed"
                            
                            # Check for subprocess calls
                            if node.func.value.id == 'subprocess':
                                return "Use of subprocess module is not allowed"
                    
                    # Check for built-in functions that are disallowed
                    if isinstance(node.func, ast.Name) and node.func.id in self.DISALLOWED_BUILTINS:
                        return f"Use of built-in function '{node.func.id}' is not allowed"
            
            return None
        
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def execute(
        self,
        code: str,
        inputs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            inputs: Dictionary of input variables for the code
            timeout: Maximum execution time in seconds
            description: Description of what the code does (for logging)
            
        Returns:
            Dictionary with execution results:
            - success: Whether execution was successful
            - stdout: Standard output
            - stderr: Standard error output
            - result: Return value of the code (if any)
            - error: Error message (if any)
            - execution_time: Time taken to execute the code
        """
        inputs = inputs or {}
        timeout = timeout or self.timeout
        
        # Validate the code first
        validation_error = self._validate_code(code)
        if validation_error:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": validation_error,
                "execution_time": 0
            }
        
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Prepare result container
        result_container = {"value": None}
        
        # Get start time
        start_time = time.time()
        
        # Execute the code
        try:
            # Set up globals with safe builtins
            safe_globals = {
                "__builtins__": {
                    name: getattr(__builtins__, name)
                    for name in dir(__builtins__)
                    if name not in self.DISALLOWED_BUILTINS
                }
            }
            
            # Add inputs to globals
            safe_globals.update(inputs)
            
            # Set up a function to capture the return value
            def execute_with_return():
                # Append a return statement to capture the last expression
                lines = code.strip().split('\n')
                if lines and not lines[-1].strip().startswith(('return', 'print', 'import', 'from', 'def', 'class', '#')):
                    last_line = lines[-1]
                    # Count leading whitespace to maintain indentation
                    leading_space = len(last_line) - len(last_line.lstrip())
                    whitespace = last_line[:leading_space]
                    
                    # Replace the last line with a return statement
                    lines[-1] = f"{whitespace}result_container['value'] = {last_line.strip()}"
                
                # Join the lines back together
                modified_code = '\n'.join(lines)
                
                # Execute the modified code
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    exec(modified_code, safe_globals)
            
            # Set a timeout for execution
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution timed out after {timeout} seconds")
            
            # Register timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            # Execute the code
            execute_with_return()
            
            # Cancel the alarm
            signal.alarm(0)
            
            # Get execution time
            execution_time = time.time() - start_time
            
            # Get stdout and stderr
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # Truncate output if too long
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "... [output truncated]"
            
            if len(stderr) > self.max_output_length:
                stderr = stderr[:self.max_output_length] + "... [output truncated]"
            
            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "result": result_container["value"],
                "error": None,
                "execution_time": execution_time
            }
        
        except TimeoutError as e:
            return {
                "success": False,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "result": None,
                "error": str(e),
                "execution_time": timeout
            }
        
        except Exception as e:
            # Get execution time
            execution_time = time.time() - start_time
            
            # Get traceback
            tb = traceback.format_exc()
            
            return {
                "success": False,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "result": None,
                "error": str(e),
                "traceback": tb,
                "execution_time": execution_time
            }
        
        finally:
            # Clean up
            stdout_buffer.close()
            stderr_buffer.close()

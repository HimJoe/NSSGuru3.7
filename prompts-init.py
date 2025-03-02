"""
Prompts Module

This module provides system prompts and templates for agent initialization and operation.
"""

from .system_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    TOOL_USING_SYSTEM_PROMPT,
    RESEARCH_AGENT_PROMPT,
    CREATIVE_AGENT_PROMPT,
    CODING_AGENT_PROMPT
)

from .templates import (
    get_template,
    render_template,
    register_template
)

__all__ = [
    # System prompts
    "DEFAULT_SYSTEM_PROMPT",
    "REASONING_SYSTEM_PROMPT",
    "TOOL_USING_SYSTEM_PROMPT",
    "RESEARCH_AGENT_PROMPT",
    "CREATIVE_AGENT_PROMPT",
    "CODING_AGENT_PROMPT",
    
    # Template functions
    "get_template",
    "render_template",
    "register_template"
]

# TradingAgents/graph/mcp_tool_wrapper.py

from typing import Any
from langchain_core.tools import StructuredTool


def create_mcp_tool_wrapper(mcp_tool: StructuredTool) -> StructuredTool:
    """
    Wrap MCP tool to ensure string output while preserving StructuredTool schema.

    MCP tools from langchain_mcp_adapters sometimes return lists like:
    [{"type": "text", "text": "actual content"}]

    This wrapper converts such results to plain strings.
    """

    # Get the original function
    original_func = mcp_tool.info

    def wrapped_func(*args, **kwargs) -> str:
        """Wrapped function that ensures proper string output."""
        try:
            # Call the original MCP tool function
            result = original_func(*args, **kwargs)

            # Handle different result types
            if isinstance(result, str):
                return result
            elif isinstance(result, list) and len(result) > 0:
                # Handle MCP format: [{"type": "text", "text": "..."}]
                if isinstance(result[0], dict) and "text" in result[0]:
                    return result[0]["text"]
                else:
                    return str(result)
            else:
                return str(result)

        except Exception as e:
            return f"Error executing {mcp_tool.name}: {str(e)}"

    # Create a new StructuredTool with the same schema but wrapped function
    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        func=wrapped_func,
        args_schema=mcp_tool.args_schema,  # Keep the original schema for multi-param support
    )
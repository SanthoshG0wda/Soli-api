"""
Tool Interface & Registry
=========================
Defines the Tool dataclass, the global tool registry, and the @tool decorator.
Each tool is a typed, async-callable unit that the AgentHarness can dispatch.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    name: str
    type: str           # "string" | "integer" | "number" | "boolean" | "array"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    """A single callable tool available to the agent."""
    name: str
    description: str
    parameters: list[ToolParameter]
    fn: Callable[..., Awaitable[Any]]

    async def execute(self, **kwargs: Any) -> Any:
        try:
            result = await self.fn(**kwargs)
            logger.debug("Tool %s executed: %s", self.name, str(result)[:200])
            return result
        except Exception as exc:
            logger.warning("Tool %s failed: %s", self.name, exc)
            return {"error": str(exc)}

    def to_nim_schema(self) -> dict:
        """Render this tool as an OpenAI-compatible function schema for NIM."""
        props: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            schema: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                schema["enum"] = p.enum
            props[p.name] = schema
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Central registry of all tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def nim_schemas(self) -> list[dict]:
        return [t.to_nim_schema() for t in self._tools.values()]

    async def dispatch(self, name: str, arguments: str | dict) -> Any:
        """Parse JSON arguments and execute the named tool."""
        tool = self.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        if isinstance(arguments, str):
            try:
                kwargs = json.loads(arguments)
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON arguments for tool {name}: {arguments}"}
        else:
            kwargs = arguments

        return await tool.execute(**kwargs)


# Global registry — connectors register into this at import time
registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None,
) -> Callable:
    """
    Decorator to register an async function as a tool.

    Usage:
        @tool(
            name="search_indian_kanoon",
            description="Search Supreme Court and High Court judgments",
            parameters=[ToolParameter("query", "string", "Search query", required=True)],
        )
        async def search_indian_kanoon(query: str) -> list[dict]:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        t = Tool(
            name=name,
            description=description,
            parameters=parameters or [],
            fn=fn,
        )
        registry.register(t)
        return fn
    return decorator

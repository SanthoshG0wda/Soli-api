"""
Agent Trace
===========
Records every think/act/observe step of an agent run for inspection and debugging.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    THINK = "think"       # LLM reasoning / plan text
    ACT = "act"           # Tool call dispatched
    OBSERVE = "observe"   # Tool call result received
    DONE = "done"         # Final answer produced
    ERROR = "error"       # Something went wrong


@dataclass
class TraceStep:
    kind: StepKind
    content: Any          # str for THINK/DONE/ERROR, dict for ACT/OBSERVE
    tool_name: str | None = None
    tool_args: dict | None = None
    elapsed_ms: float = 0.0
    iteration: int = 0


@dataclass
class AgentTrace:
    """Full execution trace for one agent run."""
    query: str
    steps: list[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    total_elapsed_ms: float = 0.0
    iterations_used: int = 0
    llm_used: bool = False

    def add(self, step: TraceStep) -> None:
        self.steps.append(step)

    def think(self, content: str, iteration: int = 0) -> None:
        self.add(TraceStep(StepKind.THINK, content, iteration=iteration))

    def act(self, tool_name: str, tool_args: dict, iteration: int = 0) -> None:
        self.add(TraceStep(StepKind.ACT, f"Calling {tool_name}", tool_name=tool_name, tool_args=tool_args, iteration=iteration))

    def observe(self, tool_name: str, result: Any, elapsed_ms: float = 0.0, iteration: int = 0) -> None:
        self.add(TraceStep(StepKind.OBSERVE, result, tool_name=tool_name, elapsed_ms=elapsed_ms, iteration=iteration))

    def done(self, content: Any) -> None:
        self.total_elapsed_ms = (time.monotonic() - self.start_time) * 1000
        self.add(TraceStep(StepKind.DONE, content))

    def error(self, msg: str) -> None:
        self.add(TraceStep(StepKind.ERROR, msg))

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "iterations_used": self.iterations_used,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "llm_used": self.llm_used,
            "steps": [
                {
                    "kind": s.kind.value,
                    "content": s.content if isinstance(s.content, (str, int, float, bool, type(None))) else repr(s.content)[:400],
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "elapsed_ms": round(s.elapsed_ms, 1),
                    "iteration": s.iteration,
                }
                for s in self.steps
            ],
        }

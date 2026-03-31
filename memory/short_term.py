"""
Short-Term Memory — in-context working memory for a single agent during one task.

Provides a structured, windowed message buffer that:
  - Stores the recent conversation / observation history (token-budget-aware)
  - Holds ephemeral task state: current goal, intermediate results, tool outputs
  - Offers a formatted context string agents can prepend to any LLM prompt
  - Is discarded at the end of a task (intentionally not persisted)

Inspired by:
  - NousResearch/hermes-agent (session memory, windowed chat history)
  - plastic-labs/honcho (short-session vs long-session memory split)
  - OpenAI chat completions message format
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool", "observation"]


@dataclass
class Message:
    """A single entry in the short-term memory buffer."""

    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0   # estimated token count (filled lazily)

    def estimate_tokens(self) -> int:
        """Rough token estimate: 1 token ≈ 4 characters."""
        if not self.tokens:
            self.tokens = max(1, len(self.content) // 4)
        return self.tokens


class ShortTermMemory:
    """
    Token-budget-aware windowed message buffer for one agent/task.

    Usage:
        mem = ShortTermMemory(token_budget=4096)
        mem.add("system", "You are a hypothesis generator.")
        mem.add("user", "Generate 3 hypotheses about attention efficiency.")
        mem.add("assistant", '["hypothesis 1", ...]')
        mem.add("observation", "Experiment ran. metric=0.87")

        # Get formatted context for next LLM call
        context = mem.as_prompt_context()

        # Access structured task state
        mem.set_state("current_hypothesis", "Sparse attention reduces FLOPs")
        hyp = mem.get_state("current_hypothesis")
    """

    def __init__(
        self,
        token_budget: int = 6000,
        max_messages: int = 50,
        agent_id: str = "",
        task_id: str = "",
    ) -> None:
        self._budget = token_budget
        self._max_messages = max_messages
        self.agent_id = agent_id
        self.task_id = task_id
        self._messages: deque[Message] = deque()
        self._used_tokens: int = 0
        self._task_state: dict[str, Any] = {}   # structured ephemeral state
        self._tool_results: list[dict[str, Any]] = []  # recent tool outputs

    # ------------------------------------------------------------------ #
    #  Message buffer                                                       #
    # ------------------------------------------------------------------ #

    def add(
        self,
        role: Role,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """
        Add a message to the buffer. Automatically evicts oldest non-system
        messages when the token budget is exceeded.
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        tokens = msg.estimate_tokens()

        # Evict oldest non-system messages until budget is satisfied
        while (
            (self._used_tokens + tokens > self._budget or len(self._messages) >= self._max_messages)
            and self._messages
        ):
            oldest = self._messages[0]
            if oldest.role == "system":
                # Never evict system messages — skip past them
                break
            evicted = self._messages.popleft()
            self._used_tokens -= evicted.estimate_tokens()

        self._messages.append(msg)
        self._used_tokens += tokens
        return msg

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
    ) -> None:
        """
        Store a tool output both as a buffer message and in the structured
        tool_results list for easy programmatic access.
        """
        summary = str(result)[:500]
        status = "✓" if success else "✗"
        self.add("tool", f"[{tool_name}] {status} {summary}")
        self._tool_results.append({
            "tool": tool_name,
            "result": result,
            "success": success,
            "timestamp": time.time(),
        })

    def get_messages(
        self,
        roles: list[Role] | None = None,
        last_n: int | None = None,
    ) -> list[Message]:
        """Return messages, optionally filtered by role and/or capped to last N."""
        msgs = list(self._messages)
        if roles:
            msgs = [m for m in msgs if m.role in roles]
        if last_n is not None:
            msgs = msgs[-last_n:]
        return msgs

    def last(self, role: Role | None = None) -> Message | None:
        """Return the most recent message, optionally filtered by role."""
        for msg in reversed(list(self._messages)):
            if role is None or msg.role == role:
                return msg
        return None

    def clear(self, keep_system: bool = True) -> None:
        """Clear the buffer. Optionally keep system messages."""
        if keep_system:
            system_msgs = [m for m in self._messages if m.role == "system"]
            self._messages = deque(system_msgs)
            self._used_tokens = sum(m.estimate_tokens() for m in self._messages)
        else:
            self._messages.clear()
            self._used_tokens = 0
        self._tool_results.clear()

    # ------------------------------------------------------------------ #
    #  Structured task state                                               #
    # ------------------------------------------------------------------ #

    def set_state(self, key: str, value: Any) -> None:
        """Store an arbitrary key-value in the ephemeral task state."""
        self._task_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the task state."""
        return self._task_state.get(key, default)

    def update_state(self, updates: dict[str, Any]) -> None:
        """Batch update the task state."""
        self._task_state.update(updates)

    def clear_state(self) -> None:
        self._task_state.clear()

    @property
    def task_state(self) -> dict[str, Any]:
        return dict(self._task_state)

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        return list(self._tool_results)

    # ------------------------------------------------------------------ #
    #  Context generation for LLM prompts                                  #
    # ------------------------------------------------------------------ #

    def as_prompt_context(
        self,
        include_roles: list[Role] | None = None,
        last_n: int | None = None,
        include_state: bool = True,
    ) -> str:
        """
        Format the memory buffer as a context block to prepend to an LLM prompt.

        Example output:
            [Working Memory]
            [system] You are a hypothesis generator.
            [observation] Search returned 8 papers on sparse attention.
            [assistant] Hypothesis 1: ...

            [Task State]
            current_goal: improve attention efficiency
            best_metric_so_far: 0.87
        """
        lines: list[str] = []

        msgs = self.get_messages(roles=include_roles, last_n=last_n)
        if msgs:
            lines.append("[Working Memory]")
            for msg in msgs:
                content_preview = msg.content[:300].replace("\n", " ")
                lines.append(f"[{msg.role}] {content_preview}")

        if include_state and self._task_state:
            lines.append("\n[Task State]")
            for k, v in self._task_state.items():
                lines.append(f"{k}: {str(v)[:150]}")

        return "\n".join(lines)

    def as_openai_messages(
        self,
        last_n: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Return messages in OpenAI chat completions format.
        Roles "tool" and "observation" are mapped to "user".
        """
        role_map: dict[str, str] = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool": "user",
            "observation": "user",
        }
        msgs = self.get_messages(last_n=last_n)
        return [
            {"role": role_map.get(m.role, "user"), "content": m.content}
            for m in msgs
        ]

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self._budget - self._used_tokens)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(agent={self.agent_id!r}, "
            f"messages={self.message_count}, "
            f"tokens={self.used_tokens}/{self._budget})"
        )

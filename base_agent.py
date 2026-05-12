"""
Base agent class. All agents inherit from this.
Rule-based agents override `analyze()` directly.
LLM-based agents use `_call_llm()` for structured completions.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all trading agents."""

    name: str = "BaseAgent"
    description: str = ""

    @abstractmethod
    def analyze(self, **kwargs) -> dict:
        """Run analysis and return a structured dict."""
        ...

    def __repr__(self):
        return f"<{self.name}>"

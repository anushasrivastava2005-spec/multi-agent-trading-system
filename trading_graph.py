"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.

Supports multiple providers:
  - "groq"    → Free Llama models via Groq (default)
  - "openai"  → OpenAI GPT-4o (paid)
"""

import os
from typing import Dict

from langchain_core.language_models import BaseChatModel

from agents.graph_setup import SetGraph
from agents.graph_util import TechnicalTools

# Default configuration — uses FREE Groq Llama models
DEFAULT_CONFIG = {
    "agent_llm_model": "llama-3.3-70b-versatile",       # tool-calling agent
    "graph_llm_model": "meta-llama/llama-4-scout-17b-16e-instruct",   # vision agent (pattern + trend)
    "agent_llm_provider": "groq",
    "graph_llm_provider": "groq",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "",         # OpenAI key (if using openai provider)
    "groq_api_key": "",    # Groq key (if using groq provider)
}


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()

        # Initialize LLMs
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "groq"),
            model=self.config.get("agent_llm_model", "llama-3.3-70b-versatile"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
            max_tokens=1024,
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "groq"),
            model=self.config.get("graph_llm_model", "meta-llama/llama-4-scout-17b-16e-instruct"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
            max_tokens=1024,
        )
        self.toolkit = TechnicalTools()

        # Build the LangGraph
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
        )
        self.graph = self.graph_setup.set_graph()

    def _get_api_key(self, provider: str = "groq") -> str:
        """Get API key from config or environment."""
        if provider == "groq":
            api_key = (
                self.config.get("groq_api_key")
                or os.environ.get("GROQ_API_KEY", "")
            )
            if not api_key:
                raise ValueError(
                    "Groq API key not found. Please set GROQ_API_KEY in your .env file. "
                    "Get a free key at https://console.groq.com/keys"
                )
            return api_key
        elif provider == "openai":
            api_key = (
                self.config.get("api_key")
                or os.environ.get("OPENAI_API_KEY", "")
            )
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
                )
            return api_key
        elif provider == "anthropic":
            api_key = (
                self.config.get("anthropic_api_key")
                or os.environ.get("ANTHROPIC_API_KEY", "")
            )
            if not api_key:
                raise ValueError("Anthropic API key not found.")
            return api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_llm(
        self, provider: str, model: str, temperature: float, max_tokens: int = 1024
    ) -> BaseChatModel:
        """Create an LLM instance based on the provider."""
        api_key = self._get_api_key(provider)

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def refresh_llms(self):
        """Refresh the LLM objects with the current config values."""
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "groq"),
            model=self.config.get("agent_llm_model", "llama-3.3-70b-versatile"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
            max_tokens=1024,
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "groq"),
            model=self.config.get("graph_llm_model", "meta-llama/llama-4-scout-17b-16e-instruct"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
            max_tokens=1024,
        )

        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
        )
        self.graph = self.graph_setup.set_graph()

    def update_api_key(self, api_key: str, provider: str = "groq"):
        """Update the API key and refresh LLMs."""
        if provider == "groq":
            self.config["groq_api_key"] = api_key
            os.environ["GROQ_API_KEY"] = api_key
        elif provider == "openai":
            self.config["api_key"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            self.config["anthropic_api_key"] = api_key
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.refresh_llms()

"""
Agent for technical indicator analysis using LLM + tool calls.
Uses LangChain tools to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node.
    The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        from langchain_core.tools import tool

        @tool
        def compute_macd(fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> dict:
            """Compute the Moving Average Convergence Divergence (MACD)."""
            return toolkit.compute_macd.invoke({"kline_data": state["kline_data"], "fastperiod": fastperiod, "slowperiod": slowperiod, "signalperiod": signalperiod})

        @tool
        def compute_rsi(period: int = 14) -> dict:
            """Compute the Relative Strength Index (RSI)."""
            return toolkit.compute_rsi.invoke({"kline_data": state["kline_data"], "period": period})

        @tool
        def compute_roc(period: int = 10) -> dict:
            """Compute the Rate of Change (ROC) indicator."""
            return toolkit.compute_roc.invoke({"kline_data": state["kline_data"], "period": period})

        @tool
        def compute_stoch() -> dict:
            """Compute the Stochastic Oscillator %K and %D."""
            return toolkit.compute_stoch.invoke({"kline_data": state["kline_data"]})

        @tool
        def compute_willr(period: int = 14) -> dict:
            """Compute the Williams %R indicator."""
            return toolkit.compute_willr.invoke({"kline_data": state["kline_data"], "period": period})

        # --- Tool definitions ---
        tools = [compute_macd, compute_rsi, compute_roc, compute_stoch, compute_willr]
        time_frame = state["time_frame"]
        
        # We need to slice the latest 5 candles to send to the prompt to save context length without affecting tool accuracy
        # The tools will still use the full 45 candles because they use state["kline_data"] directly
        prompt_kline_slice = {}
        for k, v in state["kline_data"].items():
            prompt_kline_slice[k] = v[-5:]

        # --- System prompt for LLM ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                    "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                    "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                    "Call them with the requested periods context. Do NOT send or make up the data yourself, it's already provided.\n\n"
                    f"⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. "
                    "You must interpret this data quickly and accurately.\n\n"
                    "Here is the most recent OHLC data (for context):\n{kline_data}.\n\n"
                    "Call necessary tools, and analyze the results.\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(prompt_kline_slice, indent=2))

        chain = prompt | llm.bind_tools(tools)
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]

        # --- Step 1: Ask for tool calls ---
        ai_response = chain.invoke(messages)
        messages.append(ai_response)

        # --- Step 2: Collect tool results ---
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                # Lookup tool by name
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                # Append result as ToolMessage
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # --- Step 3: Re-run the chain with tool results ---
        max_iterations = 5
        iteration = 0
        final_response = None

        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke(messages)
            messages.append(final_response)

            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break

            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # Extract content
        if final_response:
            report_content = final_response.content
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                for msg in reversed(messages):
                    if (hasattr(msg, 'content') and msg.content and
                        isinstance(msg.content, str) and msg.content.strip() and
                        not hasattr(msg, 'tool_calls')):
                        report_content = msg.content
                        break
        else:
            report_content = "Indicator analysis completed, but no detailed report was generated."

        return {
            "messages": messages,
            "indicator_report": report_content if report_content else "Indicator analysis completed.",
        }

    return indicator_agent_node

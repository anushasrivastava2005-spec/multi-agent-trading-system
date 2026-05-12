"""
Agent for trend analysis using vision LLM to interpret trendline charts.
"""

import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def invoke_with_retry(call_fn, *args, retries=3, wait_sec=4):
    """Retry a function call with backoff for rate limits or errors."""
    for attempt in range(retries):
        try:
            result = call_fn(*args)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in str(e):
                print(
                    f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                )
            else:
                print(
                    f"Error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                )
            if attempt < retries - 1:
                time.sleep(wait_sec)
    raise RuntimeError("Max retries exceeded")


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node.
    Uses precomputed images from state or falls back to tool generation.
    """

    def trend_agent_node(state):
        tools = [toolkit.generate_trend_image]
        time_frame = state["time_frame"]

        # --- Check for precomputed image in state ---
        trend_image_b64 = state.get("trend_image")

        messages = []

        # --- If no precomputed image, fall back to tool generation ---
        if not trend_image_b64:
            print("No precomputed trend image found in state, generating with tool...")

            system_prompt = (
                "You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                "You must first call the tool `generate_trend_image` using the provided `kline_data`. "
                "Once the chart is generated, analyze the image for support/resistance trendlines and known candlestick patterns. "
                "Only then should you proceed to make a prediction about the short-term trend (upward, downward, or sideways). "
                "Do not make any predictions before generating and analyzing the image."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Here is the recent kline data:\n{json.dumps(state['kline_data'], indent=2)}"
                ),
            ]

            chain = tool_llm.bind_tools(tools)

            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = tool_fn.invoke(tool_args)
                    trend_image_b64 = tool_result.get("trend_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("Using precomputed trend image from state")

        # --- Vision analysis with image ---
        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This candlestick ({time_frame} K-line) chart includes automated trendlines: the **blue line** is support, and the **red line** is resistance, both derived from recent closing prices.\n\n"
                        "Analyze how price interacts with these lines — are candles bouncing off, breaking through, or compressing between them?\n\n"
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend: **upward**, **downward**, or **sideways**. "
                        "Support your prediction with respect to prediction, reasoning, signals."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)

            messages = [
                SystemMessage(
                    content="You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                    "Your task is to analyze candlestick charts annotated with support and resistance trendlines."
                ),
                human_msg,
            ]

            try:
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                if "at least one message" in error_str.lower():
                    print("Retrying with HumanMessage only due to message conversion issue...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
                else:
                    raise
        else:
            final_response = invoke_with_retry(chain.invoke, messages)

        return {
            "messages": messages + [final_response],
            "trend_report": final_response.content,
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "Trend-enhanced candlestick chart with support/resistance lines"
                if trend_image_b64
                else None
            ),
        }

    return trend_agent_node

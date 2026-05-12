"""
Graph assembly: wires together Indicator → Pattern → Trend → Decision agents
using LangGraph's StateGraph.
"""

from langgraph.graph import END, START, StateGraph

from agents.agent_state import IndicatorAgentState
from agents.decision_agent import create_final_trade_decider
from agents.graph_util import TechnicalTools
from agents.indicator_agent import create_indicator_agent
from agents.pattern_agent import create_pattern_agent
from agents.trend_agent import create_trend_agent


class SetGraph:
    def __init__(self, agent_llm, graph_llm, toolkit):
        self.agent_llm = agent_llm
        self.graph_llm = graph_llm
        self.toolkit = toolkit

    def set_graph(self):
        """Build and compile the LangGraph state-graph."""
        agent_nodes = {}
        all_agents = ["indicator", "pattern", "trend"]

        # Create nodes for each agent
        agent_nodes["indicator"] = create_indicator_agent(self.agent_llm, self.toolkit)
        agent_nodes["pattern"] = create_pattern_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )
        agent_nodes["trend"] = create_trend_agent(
            self.agent_llm, self.graph_llm, self.toolkit
        )

        # Create decision node
        decision_agent_node = create_final_trade_decider(self.graph_llm)

        # Build the graph
        graph = StateGraph(IndicatorAgentState)

        # Add agent nodes
        for agent_type, cur_node in agent_nodes.items():
            graph.add_node(f"{agent_type.capitalize()} Agent", cur_node)

        # Add decision node
        graph.add_node("Decision Maker", decision_agent_node)

        # Set start edge
        graph.add_edge(START, "Indicator Agent")

        # Chain agents sequentially: Indicator → Pattern → Trend → Decision
        for i, agent_type in enumerate(all_agents):
            current_agent = f"{agent_type.capitalize()} Agent"

            if i == len(all_agents) - 1:
                graph.add_edge(current_agent, "Decision Maker")
            else:
                next_agent = f"{all_agents[i + 1].capitalize()} Agent"
                graph.add_edge(current_agent, next_agent)

        # Decision Maker → END
        graph.add_edge("Decision Maker", END)

        return graph.compile()

# TradingAgents/graph/conditional_logic.py

from tauric_mcp.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1, max_tool_rounds=3):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
        self.max_tool_rounds = max_tool_rounds

    def _count_tool_rounds(self, messages, tool_name: str) -> int:
        """Count how many times a specific tool has been called."""
        count = 0
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get('name') == tool_name:
                        count += 1
        return count

    def _should_continue_analyst(self, state: AgentState, tool_node: str, clear_node: str, tool_name: str):
        """Generic analyst routing with tool-call cap."""
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if self._count_tool_rounds(messages, tool_name) >= self.max_tool_rounds:
                return clear_node
            return tool_node
        return clear_node

    def should_continue_market(self, state: AgentState):
        return self._should_continue_analyst(state, "tools_market", "Msg Clear Market", "get_stock_market_data")

    def should_continue_social(self, state: AgentState):
        return self._should_continue_analyst(state, "tools_social", "Msg Clear Social", "get_stock_news_sentiment")

    def should_continue_news(self, state: AgentState):
        return self._should_continue_analyst(state, "tools_news", "Msg Clear News", "get_realtime_stock_news")

    def should_continue_fundamentals(self, state: AgentState):
        return self._should_continue_analyst(state, "tools_fundamentals", "Msg Clear Fundamentals", "get_stock_fundamentals_data")

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"

# TradingAgents/graph/trading_graph.py

import os
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from tauric_mcp.default_config import DEFAULT_CONFIG
from tauric_mcp.agents import FinancialSituationMemory
from common.consts import Agents
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor
from tauric_mcp.llm_adapters import ChatDashScopeOpenAI
from ..config.config import set_config
from ..llm_adapters.deepseek_adapter import ChatDeepSeek
from common.utils import output_results


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        # selected_analysts=["socail", "news", "market", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        mcp_tools: list = None
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)
        self.selected_analysts = selected_analysts


        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        # elif self.config["llm_provider"].lower() == "anthropic":
        #     self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
        #     self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        # elif self.config["llm_provider"].lower() == "google":
        #     google_api_key = os.getenv('GOOGLE_API_KEY')
        #     self.deep_thinking_llm = ChatGoogleGenerativeAI(
        #         model=self.config["deep_think_llm"],
        #         google_api_key=google_api_key,
        #         temperature=0.1,
        #         max_tokens=2000
        #     )
        #     self.quick_thinking_llm = ChatGoogleGenerativeAI(
        #         model=self.config["quick_think_llm"],
        #         google_api_key=google_api_key,
        #         temperature=0.1,
        #         max_tokens=2000
        #     )
        elif (self.config["llm_provider"].lower() == "dashscope" or
              self.config["llm_provider"].lower() == "alibaba" or
              "dashscope" in self.config["llm_provider"].lower() or
              "阿里百炼" in self.config["llm_provider"]):
            # 使用 OpenAI 兼容适配器，支持原生 Function Calling
            print("🔧 使用阿里百炼 OpenAI 兼容适配器 (支持原生工具调用)")
            self.deep_thinking_llm = ChatDashScopeOpenAI(
                model=self.config["deep_think_llm"],
                temperature=0.1,
                max_tokens=2000
            )
            self.quick_thinking_llm = ChatDashScopeOpenAI(
                model=self.config["quick_think_llm"],
                temperature=0.1,
                max_tokens=2000
            )
        elif (self.config["llm_provider"].lower() == "deepseek" or
              "deepseek" in self.config["llm_provider"].lower()):
            # DeepSeek V3配置 - 使用支持token统计的适配器
            # from trading_agents.llm_adapters.deepseek_adapter import ChatDeepSeek

            deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
            if not deepseek_api_key:
                raise ValueError("使用DeepSeek需要设置DEEPSEEK_API_KEY环境变量")

            deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

            # 使用支持token统计的DeepSeek适配器
            self.deep_thinking_llm = ChatDeepSeek(
                model=self.config["deep_think_llm"],
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                temperature=1.0,
                max_tokens=4096
            )
            self.quick_thinking_llm = ChatDeepSeek(
                model=self.config["quick_think_llm"],
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                temperature=1.0,
                max_tokens=4096
                )

            print(f"✅ [DeepSeek] 已启用token统计功能")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        # 使用 MCP tools
        if not mcp_tools:
            raise ValueError("mcp_tools is required. Please provide MCP tools.")
        self.mcp_tools = mcp_tools
        # 将 MCP tools 转换为字典以便快速查找
        self.mcp_tools_dict = {tool.name: tool for tool in mcp_tools}

        # Initialize memories (如果启用)
        memory_enabled = self.config.get("memory_enabled", True)
        if memory_enabled:
            # 使用单例ChromaDB管理器，避免并发创建冲突
            self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
            self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
            self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
            self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
            self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)
        else:
            # 创建空的内存对象
            self.bull_memory = None
            self.bear_memory = None
            self.trader_memory = None
            self.invest_judge_memory = None
            self.risk_manager_memory = None

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            self.config,
            react_llm=getattr(self, 'react_llm', None),
            mcp_tools=mcp_tools,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)
        # try:
        #     # display(Image(graph.get_graph().draw_mermaid_png()))
        #     with open("./graph_main.png", "wb") as f:
        #         png_data = self.graph.get_graph().draw_mermaid_png()
        #         f.write(png_data)  # 手动查看文件
        # except Exception as e:
        #     print(e)


    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using MCP tools."""
        return {
            "market": ToolNode(
                [self.mcp_tools_dict.get('get_stock_market_data')]
            ),
            "social": ToolNode(
                [self.mcp_tools_dict.get('get_stock_news_sentiment')]
            ),
            "news": ToolNode(
                [self.mcp_tools_dict.get('get_realtime_stock_news')]
            ),
            "fundamentals": ToolNode(
                [self.mcp_tools_dict.get('get_stock_fundamentals_data')]
            ),
        }

    async def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        # 添加详细的接收日志
        print(f"🔍 [GRAPH DEBUG] ===== TradingAgentsGraph.propagate 接收参数 =====")
        print(f"🔍 [GRAPH DEBUG] 接收到的company_name: '{company_name}' (类型: {type(company_name)})")
        print(f"🔍 [GRAPH DEBUG] 接收到的trade_date: '{trade_date}' (类型: {type(trade_date)})")

        self.ticker = company_name
        print(f"🔍 [GRAPH DEBUG] 设置self.ticker: '{self.ticker}'")

        # Initialize state
        print(f"🔍 [GRAPH DEBUG] 创建初始状态，传递参数: company_name='{company_name}', trade_date='{trade_date}'")
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        print(f"🔍 [GRAPH DEBUG] 初始状态中的company_of_interest: '{init_agent_state.get('company_of_interest', 'NOT_FOUND')}'")
        print(f"🔍 [GRAPH DEBUG] 初始状态中的trade_date: '{init_agent_state.get('trade_date', 'NOT_FOUND')}')")
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            async for chunk in self.graph.astream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = await self.graph.ainvoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        # self._log_state(trade_date, final_state)

        decision = self.process_signal(final_state["final_trade_decision"], company_name)
        output_path = self.config.get('project_dir') + f'/reports/{datetime.now().strftime("%Y-%m-%d")}/'
        final_report = final_state
        final_report.pop('messages')
        final_report['final_decision'] = decision
        output_results(final_report, company_name, output_path, Agents.tauric)

        # Return decision and processed signal
        return decision

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal, stock_symbol=None):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal, stock_symbol)

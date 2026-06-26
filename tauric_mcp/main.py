import sys
import os

# Add project root to Python path - MUST be before any project imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from datetime import datetime
from tauric_mcp.graph import TradingAgentsGraph
from common.utils import Action
import asyncio
from tauric_mcp.default_config import DEFAULT_CONFIG
from tauric_mcp.hkai_tools import get_all_tools
from dotenv import load_dotenv
load_dotenv()

config = DEFAULT_CONFIG.copy()
# Use standard OpenAI-compatible env vars instead of hardcoded DeepSeek
config["llm_provider"] = "openai"
config["deep_think_llm"] = os.getenv("LLM_MODEL", "gpt-4o-mini")
config["quick_think_llm"] = os.getenv("LLM_MODEL", "gpt-4o-mini")
config["backend_url"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
config["memory_enabled"] = False  # ChromaDB not needed for HK trading


async def tauric_main(stock_code: str) -> Action:
    """
    Trading agent main function for making buy/hold/sell decisions.

    Input:
        stock_code (str): Stock symbol (e.g., '00700.HK')

    Output:
        Action: Trading action enum (Action.BUY / Action.HOLD / Action.SELL)
    """
    ta = None
    try:
        mcp_tools = get_all_tools()
        print(f"已加载 {len(mcp_tools)} 个工具 (HK.AI MCP + DuckDuckGo)")

        ta = TradingAgentsGraph(debug=True, config=config, mcp_tools=mcp_tools)
        analyze_date = datetime.now().strftime('%Y-%m-%d')
        decision = await ta.propagate(stock_code, analyze_date)
        print(decision)

        return decision
    except KeyboardInterrupt:
        print("分析被用户中断", "Ctrl+C")
        return Action.HOLD
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return Action.HOLD
    finally:
        if ta:
            try:
                import gc
                gc.collect()
                await asyncio.sleep(0.1)
            except:
                pass


if __name__ == "__main__":
    stock_code = '00700.HK'
    result = asyncio.run(tauric_main(stock_code))
    print(result)

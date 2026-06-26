"""LangChain tools wrapping HK.AI MCP + DuckDuckGo for the trading agent graph.

Exports 4 tools matching the names TradingAgentsGraph expects:
- get_stock_market_data      → HK.AI K-line + real-time quotes
- get_stock_news_sentiment   → DuckDuckGo sentiment/news search
- get_realtime_stock_news    → DuckDuckGo latest news search
- get_stock_fundamentals_data → DuckDuckGo fundamentals/financials search
"""
import asyncio
import json
import os
import sys

from langchain_core.tools import tool

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _hk_kline_and_quote(stock_code: str) -> str:
    """Call HK.AI MCP to get daily K-line, minute K-line, and real-time quote."""
    from skills.hk_ai.trading_api import get_stock_kline, get_quote_by_symbols

    parts = []

    # Daily K-line (60 bars)
    r = get_stock_kline(stock_code, "1d", 60)
    if r.get("success"):
        data = r["data"]
        kline = data.get("data", data).get("kline", [])
        parts.append(f"=== 日K线 (最近{len(kline)}根) ===")
        for k in kline[-20:]:  # last 20 bars for readability
            parts.append(f"{k.get('date', k.get('time', '?'))} O:{k.get('open')} H:{k.get('high')} L:{k.get('low')} C:{k.get('close')}")
    else:
        parts.append(f"日K线获取失败: {r.get('error', 'unknown')}")

    # Minute K-line (60 bars)
    r = get_stock_kline(stock_code, "1m", 60)
    if r.get("success"):
        data = r["data"]
        kline = data.get("data", data).get("kline", [])
        parts.append(f"\n=== 分钟K线 (最近{len(kline)}根) ===")
        for k in kline[-20:]:
            parts.append(f"{k.get('time', '?')} O:{k.get('open')} H:{k.get('high')} L:{k.get('low')} C:{k.get('close')}")
    else:
        parts.append(f"分钟K线获取失败: {r.get('error', 'unknown')}")

    # Real-time quote
    r = get_quote_by_symbols([stock_code])
    if r.get("success"):
        data = r["data"]
        quotes = data.get("data", data)
        if isinstance(quotes, list) and len(quotes) > 0:
            q = quotes[0]
            quote_data = q.get("quote", q)
            parts.append(f"\n=== 实时行情 ===")
            parts.append(f"价格: {quote_data.get('price', '?')}")
    else:
        parts.append(f"实时行情获取失败: {r.get('error', 'unknown')}")

    return "\n".join(parts)


@tool
async def get_stock_market_data(stock_code: str) -> str:
    """获取指定港股的技术面数据，包含日K线(最近60根)、分钟K线(最近60根)、实时报价。
    输入参数 stock_code: 港股代码，格式如 '00700.HK'"""
    return await asyncio.to_thread(_hk_kline_and_quote, stock_code)


def _ddg_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search."""
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return f"搜索 '{query}' 无结果"
        lines = [f"搜索: {query}", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '?')}")
            lines.append(f"   {r.get('body', '')[:300]}")
            lines.append(f"   {r.get('href', '')}")
        return "\n".join(lines)
    except ImportError:
        return "搜索不可用: duckduckgo_search 未安装"
    except Exception as e:
        return f"搜索失败: {type(e).__name__}"


@tool
async def get_stock_news_sentiment(query: str) -> str:
    """搜索港股的新闻舆情和社交媒体情绪。输入 query: 搜索词，如 '00700.HK 腾讯 舆情'"""
    return await asyncio.to_thread(_ddg_search, query, 5)


@tool
async def get_realtime_stock_news(query: str) -> str:
    """搜索港股的最新实时新闻。输入 query: 搜索词，如 '00700.HK 最新消息'"""
    return await asyncio.to_thread(_ddg_search, f"{query} 港股 最新", 5)


@tool
async def get_stock_fundamentals_data(query: str) -> str:
    """搜索港股的基本面数据，包括财报、估值、财务指标。输入 query: 搜索词，如 '00700.HK 财报 估值'"""
    return await asyncio.to_thread(_ddg_search, f"{query} 财报 估值 基本面", 5)


def get_all_tools() -> list:
    """Return all 4 tools as a list for TradingAgentsGraph."""
    return [
        get_stock_market_data,
        get_stock_news_sentiment,
        get_realtime_stock_news,
        get_stock_fundamentals_data,
    ]

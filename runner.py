"""LLM tool-calling runner for HK.AI stock screening.

Tool registration reads function signatures + docstrings from skills/hk_ai.
Adapted from hk-ai-agent template — run() returns the LLM's final text output.
"""
import inspect
import json
import os

from openai import OpenAI

from skills import hk_ai

_TOOL_NAMES = [
    "list_selectable_stocks", "get_quote_by_symbols", "get_market_status",
    "get_stock_kline", "get_account_snapshot", "get_positions", "get_holdings",
    "buy_stock", "sell_stock",
    "get_orders_history", "get_buy_list", "get_sell_list",
    "get_settlement_list", "get_balance_log", "get_fee_log",
    "get_competition_rules",
]

_PY_TYPE_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def _build_tool_spec(name: str) -> dict:
    func = getattr(hk_ai, name)
    sig = inspect.signature(func)
    properties, required = {}, []
    for pname, param in sig.parameters.items():
        ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
        properties[pname] = {"type": _PY_TYPE_TO_JSON.get(ann, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    desc = (func.__doc__ or name).strip().split("\n")[0]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS = [_build_tool_spec(n) for n in _TOOL_NAMES]
_TOOL_FUNCS = {n: getattr(hk_ai, n) for n in _TOOL_NAMES}

MAX_TURNS = 15


def _execute_tool(name: str, args: dict) -> dict:
    func = _TOOL_FUNCS.get(name)
    if not func:
        return {"success": False, "error": f"未知工具: {name}"}
    try:
        return func(**args)
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


def run_screening(prompt: str, system_prompt: str = "") -> str:
    """Run LLM screening loop and return the final text output.

    Args:
        prompt: User prompt describing the screening task.
        system_prompt: Optional system prompt override.

    Returns:
        The LLM's final text content (should contain JSON trading plan).
    """
    if not system_prompt:
        system_prompt = (
            "你是港股交易筛选助手。通过调用工具完成用户任务。"
            "每次给买卖建议前必须先查账户和持仓。"
            "买卖数量10的整数倍，单笔不超HK$500,000。"
            "最终必须以JSON格式输出交易计划。回答用中文，简洁清晰。"
        )

    client = OpenAI()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        api_params = json.loads(os.getenv("LLM_API_PARAMS") or "{}")
    except json.JSONDecodeError:
        print(f"[runner] LLM_API_PARAMS parse failed, using empty dict")
        api_params = {}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    print(f"[runner] Starting screening with {len(TOOLS)} tools, model={model}")

    for turn in range(1, MAX_TURNS + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, **api_params,
            )
        except Exception as e:
            print(f"\n[runner] LLM API call failed (turn {turn}): {type(e).__name__}")
            return ""

        msg = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            if not content and reasoning:
                content = reasoning

            print(f"\n[runner] Final answer (finish_reason={finish_reason})")
            if content:
                print(content[:500] + ("..." if len(content) > 500 else ""))
            return content

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"[runner] turn {turn}: {name}({json.dumps(args, ensure_ascii=False)[:120]})")
            result = _execute_tool(name, args)
            preview = json.dumps(result, ensure_ascii=False)
            print(f"[runner]   → {preview[:200]}{'…' if len(preview) > 200 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    print("[runner] Max turns reached without final answer")
    return ""

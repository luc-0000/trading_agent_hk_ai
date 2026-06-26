"""Pipeline orchestrator: screening → deep analysis → execution."""
import asyncio
import json
import os
import re

from runner import run_screening
from skills.hk_ai import buy_stock, sell_stock
from common.utils import Action


def _parse_plan(text: str) -> dict:
    if not text:
        return {"buy": [], "sell": []}
    m = re.search(r'\{[^{}]*"buy"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            plan = json.loads(m.group())
            if "buy" in plan or "sell" in plan:
                return plan
        except json.JSONDecodeError:
            pass
    return {"buy": [], "sell": []}


def _clean_plan(plan: dict) -> dict:
    def _clean(orders):
        out = []
        for item in orders:
            if not isinstance(item, dict):
                continue
            sc = item.get("stock_code", "").strip()
            if not sc:
                continue
            qty = max(10, (int(item.get("quantity", 0)) // 10) * 10)
            out.append({"stock_code": sc, "quantity": qty, "reason": item.get("reason", "")})
        return out
    return {"buy": _clean(plan.get("buy", [])), "sell": _clean(plan.get("sell", []))}


async def _deep_analyze(stock_code: str):
    from tauric_mcp.main import tauric_main
    return await tauric_main(stock_code)


async def run():
    # ── Phase 1: Screening ──
    prompt = """你是港股交易筛选助手。严格按以下步骤操作：

1. get_account_snapshot → 查看可用资金
2. get_positions → 查看当前持仓
3. get_market_status → 确认市场状态
4. list_selectable_stocks → 获取可选股票列表
5. 挑出3-5只走势最强的，调用 get_stock_kline(code, "1d", 60) 看日K线
6. 对重点候选调用 get_stock_kline(code, "1m", 60) 看分钟K线
7. get_quote_by_symbols → 确认实时价格

然后以纯 JSON 格式输出交易计划（不要 markdown 代码块）：

{"buy": [{"stock_code": "00700.HK", "quantity": 100, "reason": "多头排列放量突破"}], "sell": [{"stock_code": "00388.HK", "quantity": 50, "reason": "跌破支撑止损"}]}

规则：买入最多2只，数量10的整数倍，单笔≤HK$500k。卖出填全部持仓。"""

    print("=" * 60)
    print("Phase 1: Screening")
    print("=" * 60)

    screen_text = run_screening(prompt)
    if not screen_text:
        print("[pipeline] Screening returned empty")
        return

    plan = _clean_plan(_parse_plan(screen_text))
    print(f"[pipeline] Plan: buy={json.dumps(plan['buy'], ensure_ascii=False)}")
    print(f"           sell={json.dumps(plan['sell'], ensure_ascii=False)}")

    if not plan["buy"] and not plan["sell"]:
        print("[pipeline] Empty plan — done")
        return

    # ── Phase 2+3: Deep Analysis + Execute ──
    executions = []

    for item in plan["buy"]:
        sc, qty = item["stock_code"], item["quantity"]
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Deep Analysis → {sc}")
        print(f"  Plan: BUY x{qty} — {item['reason']}")
        print("=" * 60)

        try:
            signal = await _deep_analyze(sc)
        except Exception as e:
            print(f"  Analysis failed: {e}")
            executions.append({"stock_code": sc, "action": "buy", "executed": False, "error": str(e)})
            continue

        if signal == Action.BUY:
            print(f"  Phase 3: BUY {sc} x{qty}")
            r = buy_stock(sc, qty)
            print(f"  → {json.dumps(r, ensure_ascii=False)[:200]}")
            executions.append({"stock_code": sc, "action": "buy", "quantity": qty, "executed": r.get("success", False), "result": r})
        elif signal == Action.SELL:
            print(f"  Signal SELL ≠ plan BUY — skipping")
            executions.append({"stock_code": sc, "action": "buy", "executed": False, "reason": "analysis returned SELL"})
        else:
            print(f"  Signal HOLD — skipping")
            executions.append({"stock_code": sc, "action": "buy", "executed": False, "reason": "analysis returned HOLD"})

    for item in plan["sell"]:
        sc, qty = item["stock_code"], item["quantity"]
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Deep Analysis → {sc}")
        print(f"  Plan: SELL x{qty} — {item['reason']}")
        print("=" * 60)

        try:
            signal = await _deep_analyze(sc)
        except Exception as e:
            print(f"  Analysis failed: {e}")
            executions.append({"stock_code": sc, "action": "sell", "executed": False, "error": str(e)})
            continue

        if signal == Action.SELL:
            print(f"  Phase 3: SELL {sc} x{qty}")
            r = sell_stock(sc, qty)
            print(f"  → {json.dumps(r, ensure_ascii=False)[:200]}")
            executions.append({"stock_code": sc, "action": "sell", "quantity": qty, "executed": r.get("success", False), "result": r})
        elif signal == Action.BUY:
            print(f"  Signal BUY ≠ plan SELL — skipping")
            executions.append({"stock_code": sc, "action": "sell", "executed": False, "reason": "analysis returned BUY"})
        else:
            print(f"  Signal HOLD — executing sell per plan")
            r = sell_stock(sc, qty)
            print(f"  → {json.dumps(r, ensure_ascii=False)[:200]}")
            executions.append({"stock_code": sc, "action": "sell", "quantity": qty, "executed": r.get("success", False), "hold_override": True, "result": r})

    # ── Report ──
    report = {"plan": plan, "executions": executions}
    report_path = os.path.join(os.getcwd(), "reports.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    for ex in executions:
        s = "✓" if ex.get("executed") else "✗"
        qty = ex.get('quantity', '?')
        print(f"  {s} {ex['action'].upper():4s} {ex['stock_code']:12s} x{str(qty):>6s}")
    print(f"\nReport → {report_path}")

"""Minimal StockUtils — HK-only market info (original mcp_servers/tools/stock_utils.py was deleted)."""


class StockUtils:
    @staticmethod
    def get_market_info(stock_code: str) -> dict:
        if not stock_code:
            return {
                "is_china": False, "is_hk": True,
                "market_name": "港股", "currency_name": "港元", "currency_symbol": "HK$",
            }
        code = stock_code.upper().strip()
        if ".HK" in code:
            return {
                "is_china": False, "is_hk": True, "is_us": False,
                "market_name": "港股", "currency_name": "港元", "currency_symbol": "HK$",
            }
        if ".SH" in code or ".SZ" in code:
            return {
                "is_china": True, "is_hk": False, "is_us": False,
                "market_name": "A股", "currency_name": "人民币", "currency_symbol": "¥",
            }
        if ".US" in code or ".NYSE" in code or ".NASDAQ" in code:
            return {
                "is_china": False, "is_hk": False, "is_us": True,
                "market_name": "美股", "currency_name": "美元", "currency_symbol": "$",
            }
        # Default to HK
        return {
            "is_china": False, "is_hk": True, "is_us": False,
            "market_name": "港股", "currency_name": "港元", "currency_symbol": "HK$",
        }

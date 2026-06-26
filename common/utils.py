import json
from typing import Any, Dict, Optional
from datetime import datetime
import os
from enum import Enum


def output_results(results: Dict[str, Any], stock_code :str, output_path :Any, agent_name :str, format :str ='json'):
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results.update({'report_date': current_date})
    """Display or save research results."""
    output_file = str(output_path) + '/' + agent_name + '_' + stock_code + '_' + datetime.now().strftime \
        ("%H:%M") + '.' + format
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Handle JSON output
    if format == "json":
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # For text output, results are already beautifully displayed during analysis
    # Just log completion
    if not output_file:
        return
    # 示例调用
    # Save to file if requested
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Stock Analysis Results for {results.get('stock_code', 'Unknown')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Results saved to {output_file}")


class Action(Enum):
    """Execution actions for order placement."""
    BUY = "buy"        # 买入
    SELL = "sell"      # 卖出
    HOLD = "hold"      # 持有


    def __str__(self):
        """Return the action name."""
        return self.value.upper()

    def __repr__(self):
        """Return the action representation."""
        return f"Action.{self.name}"

    def is_buy(self) -> bool:
        """Check if this is a buy action."""
        return self == self.BUY

    def is_sell(self) -> bool:
        """Check if this is a sell action."""
        return self == self.SELL

    def is_hold(self) -> bool:
        """Check if this is a hold action."""
        return self == self.HOLD

    @classmethod
    def from_string(cls, action_str: str, default: Optional['Action'] = None) -> 'Action':
        """
        Create an Action from a string.

        Args:
            action_str: String representation (e.g., "buy", "BUY", etc.)
            default: Default action if string is not recognized

        Returns:
            Action instance
        """
        if not action_str:
            return default or cls.HOLD

        # Normalize the string
        action_str = action_str.strip().lower()

        # Map various string representations to Action
        action_map = {
            'buy': cls.BUY,
            '买入': cls.BUY,
            '购买': cls.BUY,
            '买入/卖出': cls.BUY,  # Handle special case
            'sell': cls.SELL,
            '卖出': cls.SELL,
            '出售': cls.SELL,

            # Neutral
            'hold': cls.HOLD,
            '持有': cls.HOLD,
            '保持': cls.HOLD,
        }

        return action_map.get(action_str, default or cls.HOLD)


class PositionSignal(Enum):
    """Directional position signal for strategy-level decisions."""
    LONG = "long"      # 做多/看涨
    SHORT = "short"    # 做空/看跌
    HOLD = "hold"      # 观望/中性

    def __str__(self):
        """Return the signal name."""
        return self.value.upper()

    def __repr__(self):
        """Return the signal representation."""
        return f"PositionSignal.{self.name}"

    def is_long(self) -> bool:
        """Check if this is a long signal."""
        return self == self.LONG

    def is_short(self) -> bool:
        """Check if this is a short signal."""
        return self == self.SHORT

    def is_hold(self) -> bool:
        """Check if this is a hold signal."""
        return self == self.HOLD

    def to_action(self) -> Action:
        """Convert a directional signal into a compatible execution action."""
        if self == self.LONG:
            return Action.BUY
        if self == self.SHORT:
            return Action.SELL
        return Action.HOLD

    @classmethod
    def from_string(
        cls, signal_str: str, default: Optional['PositionSignal'] = None
    ) -> 'PositionSignal':
        """
        Create a PositionSignal from a string.

        Args:
            signal_str: String representation (e.g., "long", "LONG", etc.)
            default: Default signal if string is not recognized

        Returns:
            PositionSignal instance
        """
        if not signal_str:
            return default or cls.HOLD

        signal_str = signal_str.strip().lower()

        signal_map = {
            'long': cls.LONG,
            '做多': cls.LONG,
            '看涨': cls.LONG,
            'short': cls.SHORT,
            '做空': cls.SHORT,
            '看跌': cls.SHORT,
            'hold': cls.HOLD,
            '持有': cls.HOLD,
            '保持': cls.HOLD,
        }

        return signal_map.get(signal_str, default or cls.HOLD)

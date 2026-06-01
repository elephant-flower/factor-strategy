"""
随波逐流策略 — 离线回测版（重写：基于YTM趋势）

策略逻辑（简化版 - YTM趋势跟踪）：
1. 使用单一活跃国债的YTM（到期收益率）数据
2. 快慢均线交叉判断收益率趋势方向
3. 成交量放大作为信号确认
4. 收益率BP止盈止损

数据来源：本地CSV文件 (data/bond_day.csv + 宏观数据)
输出：回测结果图表 (output/ 目录)

用法：
  python strategies/sui_boliu.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest_base import (
    TradeRecord, ActionRecord,
    load_bond_day, load_macro_data,
    plot_backtest_results,
)


class SuiBoLiuBacktest:
    """随波逐流策略 - YTM趋势跟踪"""

    # ─── 参数 ─────────────────────────────────────────────
    start_date = "20210101"
    end_date = "20260519"

    # 交易标的
    target_bond = "210210"  # 21国开10(10年期活跃国开债, 覆盖2022-2026)

    # 趋势参数
    trend_fast_ma = 5        # YTM快线
    trend_slow_ma = 20       # YTM慢线

    # 成交量确认
    volume_lookback = 20
    volume_min_ratio = 0.8   # 当前量不低于均量的80%

    # 止盈止损(YTM BP)
    take_profit_yield_bp = 5
    stop_loss_yield_bp = 3

    # 信号参数
    min_ytm_change_bp = 1.0  # 最小YTM变动BP才入场

    # 仓位
    trade_size = 1

    def __init__(self, start_date: str = None, end_date: str = None):
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        # 状态
        self.position = 0
        self.entry_yield: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.entry_date: Optional[str] = None

        # YTM数据序列
        self._ytm_series: pd.Series = pd.Series(dtype=float)
        self._close_series: pd.Series = pd.Series(dtype=float)
        self._vol_series: pd.Series = pd.Series(dtype=float)
        self._trade_dates: List[str] = []

        # 方向
        self.trend_dir = 0   # 1=收益率上行(做空), -1=收益率下行(做多)
        self.signal_dir = 0

        # 统计
        self.signal_count = 0

        # 记录
        self.trades: List[TradeRecord] = []
        self.actions: List[ActionRecord] = []
        self.equity_curve: List[float] = [0.0]
        self.current_trade: Optional[dict] = None

    # ─── 数据加载 ──────────────────────────────────────────

    def _load_data(self):
        """加载债券日线数据"""
        print("正在加载债券数据...")
        bond_df = load_bond_day(self.start_date, self.end_date)

        # 提取目标债券数据
        bond = bond_df[bond_df["symbol"] == self.target_bond].copy()
        if bond.empty:
            # 尝试其他活跃券
            candidates = ["210210", "200004", "230023", "220025", "190013", "200005"]
            for c in candidates:
                bond = bond_df[bond_df["symbol"] == c].copy()
                if not bond.empty:
                    self.target_bond = c
                    print(f"  [INFO] 改用债券: {c}")
                    break

        if bond.empty:
            raise ValueError(f"无法找到目标债券 {self.target_bond}")

        bond = bond.sort_values("trade_date")
        # 过滤有效YTM
        bond = bond[bond["ytm"].between(0.5, 5.0)].copy()
        bond = bond[bond["close"] > 80].copy()

        print(f"  债券 {self.target_bond}: {len(bond)} 个交易日")
        print(f"  日期范围: {bond['trade_date'].min()} ~ {bond['trade_date'].max()}")
        print(f"  YTM范围: {bond['ytm'].min():.2f} ~ {bond['ytm'].max():.2f}")

        self._trade_dates = bond["trade_date"].tolist()
        self._ytm_series = bond.set_index("trade_date")["ytm"]
        self._close_series = bond.set_index("trade_date")["close"]
        self._vol_series = bond.set_index("trade_date")["volume"]

    # ─── 信号计算 ──────────────────────────────────────────

    def _calc_trend_dir(self, idx: int) -> int:
        """基于YTM均线交叉判断趋势方向"""
        if idx < self.trend_slow_ma:
            return 0

        ytm = self._ytm_series.iloc[:idx + 1]
        fast_ma = ytm.iloc[-self.trend_fast_ma:].mean()
        slow_ma = ytm.iloc[-self.trend_slow_ma:].mean()
        prev_fast = ytm.iloc[-self.trend_fast_ma - 1:-1].mean() if len(ytm) > self.trend_fast_ma + 1 else fast_ma
        prev_slow = ytm.iloc[-self.trend_slow_ma - 1:-1].mean() if len(ytm) > self.trend_slow_ma + 1 else slow_ma

        # YTM上行趋势 (fast > slow 且 fast在上升) → 做空
        if fast_ma > slow_ma and fast_ma >= prev_fast:
            return 1
        # YTM下行趋势 (fast < slow 且 fast在下降) → 做多
        if fast_ma < slow_ma and fast_ma <= prev_fast:
            return -1
        return 0

    def _check_volume_confirm(self, idx: int) -> bool:
        """成交量确认"""
        if idx < self.volume_lookback:
            return True
        vol = self._vol_series.iloc[:idx + 1]
        current = vol.iloc[-1]
        avg = vol.iloc[-self.volume_lookback:-1].mean()
        if avg <= 0:
            return True
        return current >= avg * self.volume_min_ratio

    def _calc_ytm_change_bp(self, idx: int) -> float:
        """计算最近N日YTM变化BP"""
        if idx < 2:
            return 0.0
        return (self._ytm_series.iloc[idx] - self._ytm_series.iloc[idx - 1]) * 100

    # ─── 交易逻辑 ──────────────────────────────────────────

    def _open_position(self, idx: int, direction: int):
        """开仓"""
        date = self._trade_dates[idx]
        ytm = self._ytm_series.iloc[idx]
        close_px = self._close_series.iloc[idx]
        vol = self.trade_size

        self.position = vol if direction < 0 else -vol  # 做多=负收益率方向
        self.entry_yield = ytm
        self.entry_price = close_px
        self.entry_date = date

        side = "LONG" if direction < 0 else "SHORT"
        action = "OPEN_LONG" if direction < 0 else "OPEN_SHORT"

        self.current_trade = {
            "symbol": self.target_bond,
            "side": side,
            "entry_time": datetime.strptime(date, "%Y%m%d"),
            "entry_yield": ytm,
            "entry_price": close_px,
            "volume": vol,
        }

        self.actions.append(ActionRecord(
            datetime=datetime.strptime(date, "%Y%m%d"),
            action=action, symbol=self.target_bond,
            price=close_px, volume=vol,
            reason=f"TREND_{'DOWN' if direction<0 else 'UP'}_CONFIRM",
            trend_dir=direction,
        ))
        self.signal_count += 1

    def _close_position(self, idx: int, reason: str):
        """平仓"""
        if self.position == 0 or self.current_trade is None:
            return

        date = self._trade_dates[idx]
        current_ytm = self._ytm_series.iloc[idx]
        current_price = self._close_series.iloc[idx]
        side = self.current_trade["side"]

        # 计算盈亏(BP)
        entry_ytm = self.current_trade["entry_yield"]
        if side == "LONG":  # 做多=收益率下行
            pnl_bp = (entry_ytm - current_ytm) * 100
        else:  # 做空=收益率上行
            pnl_bp = (current_ytm - entry_ytm) * 100

        exit_dt = datetime.strptime(date, "%Y%m%d")
        self.trades.append(TradeRecord(
            symbol=self.target_bond, side=side,
            entry_time=self.current_trade["entry_time"],
            entry_price=self.current_trade["entry_price"],
            exit_time=exit_dt,
            exit_price=current_price,
            volume=self.current_trade["volume"],
            reason=reason,
            pnl=pnl_bp,
        ))

        action = "CLOSE_LONG" if self.position > 0 else "CLOSE_SHORT"
        self.actions.append(ActionRecord(
            datetime=exit_dt, action=action, symbol=self.target_bond,
            price=current_price, volume=abs(self.position), reason=reason,
        ))

        prev = self.equity_curve[-1]
        self.equity_curve.append(prev + pnl_bp)

        self.position = 0
        self.entry_yield = None
        self.current_trade = None

    # ─── 主回测循环 ────────────────────────────────────────

    def run(self) -> Tuple[List[TradeRecord], List[ActionRecord], List[float]]:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"随波逐流策略 - YTM趋势跟踪")
        print(f"{'='*60}")
        print(f"回测区间: {self.start_date} ~ {self.end_date}")
        print(f"债券: {self.target_bond}")
        print(f"参数: YTM MA快={self.trend_fast_ma} MA慢={self.trend_slow_ma}")
        print(f"      止盈={self.take_profit_yield_bp}BP 止损={self.stop_loss_yield_bp}BP")

        self._load_data()
        n = len(self._trade_dates)
        print(f"\n开始回测 ({n} 个交易日)...")

        for i in range(n):
            date = self._trade_dates[i]

            # 计算趋势方向
            self.trend_dir = self._calc_trend_dir(i)

            # 交易逻辑
            if self.position == 0:
                # 开仓
                if self.trend_dir != 0 and self._check_volume_confirm(i):
                    change_bp = self._calc_ytm_change_bp(i)
                    if abs(change_bp) >= self.min_ytm_change_bp:
                        self._open_position(i, self.trend_dir)
            else:
                # 持仓中：检查平仓
                current_ytm = self._ytm_series.iloc[i]
                if self.entry_yield is not None:
                    if self.position > 0:  # 多头
                        move_bp = (self.entry_yield - current_ytm) * 100
                    else:  # 空头
                        move_bp = (current_ytm - self.entry_yield) * 100

                    if move_bp >= self.take_profit_yield_bp:
                        self._close_position(i, "TAKE_PROFIT")
                    elif move_bp <= -self.stop_loss_yield_bp:
                        self._close_position(i, "STOP_LOSS")
                    # 趋势反转平仓
                    elif self.trend_dir != 0 and self.trend_dir != (1 if self.position < 0 else -1):
                        self._close_position(i, "TREND_REVERSE")

            if (i + 1) % 200 == 0:
                print(f"  进度: {i+1}/{n} 天...")

        # 最终平仓
        if self.position != 0:
            self._close_position(n - 1, "END_OF_BACKTEST")

        # 计算累计盈亏（含合约乘数BP→元）
        pnl_yuan = [x * 10000 for x in self.equity_curve]  # 1BP ≈ 10000元

        print(f"\n回测完成!")
        print(f"  交易日数: {n}")
        print(f"  信号次数: {self.signal_count}")

        return self.trades, self.actions, pnl_yuan


def main():
    bt = SuiBoLiuBacktest(
        start_date="20210101",
        end_date="20260519",
    )
    trades, actions, equity = bt.run()
    plot_backtest_results(trades, actions, equity, prefix="sui_boliu")
    print(f"\n所有输出文件在 output/ 目录下")


if __name__ == "__main__":
    main()

"""
随波逐流策略 — 离线回测版

策略逻辑（三因子驱动）：
1. 宏观方向（DR007/MLF/PMI/CPI/工业增加值）
2. 资金流向（成交量/净买卖方向）
3. 趋势方向（MA快慢线交叉）
4. 三因子取两层同向确认开仓信号
5. 收益率BP止盈止损

数据来源：本地CSV文件 (data/ 目录)
输出：回测结果图表 (output/ 目录)

用法：
  python strategies/sui_boliu.py
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest_base import (
    BarData, TradeRecord, ActionRecord,
    bars_from_dataframe, load_t_futures_day, load_bond_day, load_macro_data,
    load_csv, plot_backtest_results,
)


class SuiBoLiuBacktest:
    """随波逐流策略 - 离线回测引擎"""

    # ─── 参数 ─────────────────────────────────────────────
    start_date = "20240101"
    end_date = "20260519"

    # 交易参数
    trade_size = 1
    take_profit_yield_bp = 8    # 止盈BP
    stop_loss_yield_bp = 5      # 止损BP

    # 趋势参数
    trend_fast_ma = 5
    trend_slow_ma = 10

    # 资金流向参数
    flow_lookback = 3
    volume_amplify_lookback = 5
    order_ratio_threshold = 0.55

    # 信号参数
    min_entry_layers = 2  # 最少同向层数

    def __init__(self, start_date: str = None, end_date: str = None):
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        # 状态
        self.position = 0
        self.entry_yield: Optional[float] = None
        self.entry_symbol: Optional[str] = None
        self.cur_trade_date = ""

        # 方向
        self.macro_dir = 0
        self.flow_dir = 0
        self.trend_dir = 0
        self.signal_dir = 0

        # 宏数据缓存
        self._macro_data: Dict[str, pd.DataFrame] = {}
        self._bond_cache: Dict[str, List[dict]] = defaultdict(list)

        # 统计
        self.signal_count = 0
        self.trade_count = 0

        # 记录
        self.trades: List[TradeRecord] = []
        self.actions: List[ActionRecord] = []
        self.equity_curve: List[float] = [10000000.0]
        self.current_trade: Optional[dict] = None

        # 回测日期列表
        self._trade_dates: List[str] = []

    # ─── 数据加载 ──────────────────────────────────────────

    def _load_data(self):
        """加载所有数据"""
        print("正在加载数据...")

        # 加载宏观数据
        self._macro_data = load_macro_data()
        for name, df in self._macro_data.items():
            print(f"  {name}: {len(df)} 行")

        # 加载债券日线数据
        bond_df = load_bond_day(self.start_date, self.end_date)
        print(f"  bond_day: {len(bond_df)} 行, {bond_df['symbol'].nunique()} 个券")

        # 按symbol和trade_date组织
        self._bond_data: Dict[str, Dict[str, dict]] = defaultdict(dict)
        for _, row in bond_df.iterrows():
            sym = str(row["symbol"])
            day = str(row["trade_date"])
            self._bond_data[sym][day] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "ytm": float(row["ytm"]) if not pd.isna(row.get("ytm", None)) else None,
                "dirty": float(row["dirty"]) if not pd.isna(row.get("dirty", None)) else None,
                "quotation_type": str(row.get("quotation_type", "")),
            }

        # 获取所有交易日
        all_days = set()
        for sym_data in self._bond_data.values():
            all_days.update(sym_data.keys())
        self._trade_dates = sorted(all_days)
        print(f"  交易日数量: {len(self._trade_dates)}")

    def _get_bond_symbols(self) -> List[str]:
        """获取有数据的债券代码（国债）"""
        # 优先使用国债 (代码通常以数字开头且在k_cmds_day中有ytm数据的)
        candidates = []
        for sym, days in self._bond_data.items():
            # 取最近一天有ytm的数据
            valid_days = [(d, info) for d, info in days.items() if info["ytm"] is not None]
            if valid_days:
                candidates.append(sym)
        return sorted(candidates)

    def _get_closes(self, symbol: str, end_day: str, n: int) -> List[float]:
        """获取某债券最近N天的收盘价"""
        sym_data = self._bond_data.get(symbol, {})
        available = sorted([d for d in sym_data.keys() if d <= end_day])
        closes = [sym_data[d]["close"] for d in available[-n:]] if available else []
        return closes

    def _get_ytm(self, symbol: str, day: str) -> Optional[float]:
        """获取某债券某天的收益率"""
        days = self._bond_data.get(symbol, {})
        info = days.get(day)
        if info and info["ytm"] is not None:
            return info["ytm"]
        # 找最近的
        available = sorted([d for d in days.keys() if d <= day])
        for d in reversed(available):
            if days[d]["ytm"] is not None:
                return days[d]["ytm"]
        return None

    # ─── 宏观方向 ──────────────────────────────────────────

    def _calc_macro_direction(self, day: str) -> int:
        """计算宏观方向：从CSV读取数据"""
        dr_df = self._macro_data.get("macro_dr007", None)
        mlf_df = self._macro_data.get("macro_mlf", None)
        pmi_df = self._macro_data.get("macro_pmi", None)
        cpi_df = self._macro_data.get("macro_cpi", None)
        iva_df = self._macro_data.get("macro_industrial_va", None)

        # 确保日期为字符串
        day_str = str(day)

        # DR007趋势
        dr_vals = []
        if dr_df is not None and not dr_df.empty:
            dr_df["quote_date"] = dr_df["quote_date"].astype(str)
            dr_filtered = dr_df[dr_df["quote_date"] <= day_str]
            dr_vals = dr_filtered["dr007"].dropna().astype(float).tail(20).tolist()

        if len(dr_vals) < 10:
            return 0

        dr_down = self._is_sustained_move(dr_vals, direction=-1)
        dr_up = self._is_sustained_move(dr_vals, direction=1)

        # MLF趋势
        mlf_vals = []
        if mlf_df is not None and not mlf_df.empty:
            mlf_df["quote_date"] = mlf_df["quote_date"].astype(str)
            mlf_filtered = mlf_df[(mlf_df["quote_date"] <= day_str) & (mlf_df["tenor_raw"] == "1Y")]
            mlf_vals = mlf_filtered["mlf_rate"].dropna().astype(float).tail(12).tolist()

        mlf_down = len(mlf_vals) >= 2 and mlf_vals[-1] < mlf_vals[0]
        mlf_up = len(mlf_vals) >= 2 and mlf_vals[-1] > mlf_vals[0]

        # PMI
        pmi_latest = None
        if pmi_df is not None and not pmi_df.empty:
            pmi_df["quote_date"] = pmi_df["quote_date"].astype(str)
            pmi_filtered = pmi_df[pmi_df["quote_date"] <= day_str]
            pmi_vals = pmi_filtered["pmi_mfg_index"].dropna().astype(float).tail(3).tolist()
            pmi_latest = pmi_vals[-1] if pmi_vals else None

        # CPI
        cpi_latest = None
        if cpi_df is not None and not cpi_df.empty:
            cpi_df["quote_date"] = cpi_df["quote_date"].astype(str)
            cpi_filtered = cpi_df[cpi_df["quote_date"] <= day_str]
            cpi_vals = cpi_filtered["cpi_yoy_pct"].dropna().astype(float).tail(3).tolist()
            cpi_latest = cpi_vals[-1] if cpi_vals else None

        # 工业增加值
        iva_trend = 0
        if iva_df is not None and not iva_df.empty:
            iva_df["quote_date"] = iva_df["quote_date"].astype(str)
            iva_filtered = iva_df[iva_df["quote_date"] <= day_str]
            iva_vals = iva_filtered["iva_acc_yoy_pct"].dropna().astype(float).tail(3).tolist()
            if len(iva_vals) >= 2:
                iva_trend = iva_vals[-1] - iva_vals[0]

        econ_weak = (pmi_latest is not None and pmi_latest < 50
                     and cpi_latest is not None and cpi_latest <= 2.5
                     and iva_trend <= 0)
        econ_strong = (pmi_latest is not None and pmi_latest > 50
                       and cpi_latest is not None and cpi_latest >= 1.5
                       and iva_trend >= 0)

        # 综合判断
        if dr_up and mlf_up and econ_strong:
            return 1  # 紧缩预期 -> 利率上行 -> 做空债券
        if dr_down and mlf_down and econ_weak:
            return -1  # 宽松预期 -> 利率下行 -> 做多债券
        # 仅依据DR007趋势
        if dr_down:
            return -1
        if dr_up:
            return 1
        return 0

    # ─── 资金流向方向 ──────────────────────────────────────

    def _calc_flow_direction(self, symbol: str, day: str) -> int:
        """计算资金流向方向（简化版：基于成交量和价格变化）"""
        closes = self._get_closes(symbol, day, self.flow_lookback + 2)
        if len(closes) < self.flow_lookback:
            return 0

        win = closes[-self.flow_lookback:]
        vol_data = []
        for d in sorted([d for d in self._bond_data.get(symbol, {}).keys() if d <= day])[-self.flow_lookback:]:
            info = self._bond_data[symbol][d]
            vol_data.append(info["volume"])

        if len(vol_data) < self.flow_lookback:
            return 0

        recent_vol = np.mean(vol_data[-2:]) if len(vol_data) >= 2 else 0
        prev_vol = np.mean(vol_data[:-2]) if len(vol_data) > 2 else 0
        volume_up = recent_vol > prev_vol if prev_vol > 0 else False

        if volume_up and self._is_sustained_move(win, direction=1, min_ratio=0.55):
            return 1
        if volume_up and self._is_sustained_move(win, direction=-1, min_ratio=0.55):
            return -1
        return 0

    # ─── 趋势方向 ──────────────────────────────────────────

    def _calc_trend_direction(self, symbol: str, day: str) -> int:
        """计算趋势方向（MA快慢线）"""
        closes = self._get_closes(symbol, day, self.trend_slow_ma + 2)
        if len(closes) < self.trend_slow_ma + 1:
            return 0

        cur = closes[-1]
        ma_fast = np.mean(closes[-self.trend_fast_ma:])
        ma_slow = np.mean(closes[-self.trend_slow_ma:])
        ma_fast_prev = np.mean(closes[-self.trend_fast_ma - 1:-1])
        ma_slow_prev = np.mean(closes[-self.trend_slow_ma - 1:-1])

        if cur > ma_fast and ma_fast >= ma_fast_prev and ma_fast >= ma_slow * 0.999:
            return 1  # 上涨趋势
        if cur < ma_fast and ma_fast <= ma_fast_prev and ma_fast <= ma_slow * 1.001:
            return -1  # 下跌趋势
        return 0

    # ─── 信号合成 ──────────────────────────────────────────

    def _calc_entry_signal(self, dirs: List[int]) -> int:
        """合成开仓信号：最少 min_entry_layers 层同向"""
        need = max(1, self.min_entry_layers)
        if sum(1 for x in dirs if x > 0) >= need and not any(x < 0 for x in dirs):
            return 1
        if sum(1 for x in dirs if x < 0) >= need and not any(x > 0 for x in dirs):
            return -1
        return 0

    # ─── 收益率BP计算 ──────────────────────────────────────

    def _yield_move_bp(self, entry_yield: float, current_yield: float, long_position: bool) -> float:
        """计算收益率变动BP"""
        raw_move = (entry_yield - current_yield) if long_position else (current_yield - entry_yield)
        if abs(entry_yield) <= 20 and abs(current_yield) <= 20:
            return raw_move * 100.0
        return raw_move

    # ─── 工具 ──────────────────────────────────────────────

    @staticmethod
    def _is_sustained_move(values: List[float], direction: int, min_ratio: float = 0.7) -> bool:
        """判断是否持续朝某方向移动"""
        if len(values) < 3 or direction not in (-1, 1):
            return False
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        if not diffs:
            return False
        hit = sum(1 for d in diffs if d * direction > 0)
        if hit / len(diffs) < min_ratio:
            return False
        return (values[-1] - values[0]) * direction > 0

    # ─── 交易逻辑 ──────────────────────────────────────────

    def _open_position(self, day: str, direction: int, reason: str):
        """开仓"""
        sym = self._get_trading_symbol(day)
        if not sym:
            return

        ytm = self._get_ytm(sym, day)
        if ytm is None:
            return

        self.entry_yield = ytm
        self.entry_symbol = sym
        vol = self.trade_size

        if direction > 0:
            self.position = vol
            action = "OPEN_LONG"
        else:
            self.position = -vol
            action = "OPEN_SHORT"

        self.current_trade = {
            "symbol": sym,
            "side": "LONG" if direction > 0 else "SHORT",
            "entry_time": datetime.strptime(day, "%Y%m%d"),
            "entry_yield": ytm,
            "entry_price": self._get_closes(sym, day, 1)[-1] if self._get_closes(sym, day, 1) else 0,
            "volume": vol,
        }

        self.actions.append(ActionRecord(
            datetime=datetime.strptime(day, "%Y%m%d"),
            action=action, symbol=sym,
            price=self.current_trade["entry_price"],
            volume=vol, reason=reason,
            macro_dir=self.macro_dir,
            flow_dir=self.flow_dir,
            trend_dir=self.trend_dir,
            signal_dir=self.signal_dir,
        ))
        self.trade_count += 1

    def _close_position(self, day: str, reason: str):
        """平仓"""
        if self.position == 0 or self.current_trade is None:
            return

        sym = self.current_trade["symbol"]
        current_yield = self._get_ytm(sym, day)
        if current_yield is None:
            return

        closes = self._get_closes(sym, day, 1)
        current_price = closes[-1] if closes else 0

        side = self.current_trade["side"]
        if side == "LONG":
            pnl = self._yield_move_bp(self.current_trade["entry_yield"], current_yield, long_position=True)
        else:
            pnl = self._yield_move_bp(self.current_trade["entry_yield"], current_yield, long_position=False)

        exit_dt = datetime.strptime(day, "%Y%m%d")
        self.trades.append(TradeRecord(
            symbol=sym, side=side,
            entry_time=self.current_trade["entry_time"],
            entry_price=self.current_trade["entry_price"],
            exit_time=exit_dt,
            exit_price=current_price,
            volume=self.current_trade["volume"],
            reason=reason,
            pnl=pnl,
        ))

        action = "CLOSE_LONG" if self.position > 0 else "CLOSE_SHORT"
        self.actions.append(ActionRecord(
            datetime=exit_dt, action=action, symbol=sym,
            price=current_price, volume=abs(self.position), reason=reason,
            macro_dir=self.macro_dir, flow_dir=self.flow_dir,
            trend_dir=self.trend_dir, signal_dir=self.signal_dir,
        ))

        prev_equity = self.equity_curve[-1] if self.equity_curve else 0
        self.equity_curve.append(prev_equity + pnl)

        self.position = 0
        self.entry_yield = None
        self.current_trade = None

    def _get_trading_symbol(self, day: str) -> Optional[str]:
        """获取当日可交易债券（活跃券）"""
        candidates = self._get_bond_symbols()
        for sym in candidates:
            sym_data = self._bond_data.get(sym, {})
            if day in sym_data and sym_data[day]["ytm"] is not None:
                return sym
        # 找最近的
        for sym in candidates:
            available = sorted([d for d in self._bond_data[sym].keys() if d <= day])
            if available:
                return sym
        return candidates[0] if candidates else None

    # ─── 主回测循环 ────────────────────────────────────────

    def run(self) -> Tuple[List[TradeRecord], List[ActionRecord], List[float]]:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"随波逐流策略 - 离线回测")
        print(f"{'='*60}")
        print(f"回测区间: {self.start_date} ~ {self.end_date}")
        print(f"参数: 止盈={self.take_profit_yield_bp}BP, 止损={self.stop_loss_yield_bp}BP")
        print(f"      MA快={self.trend_fast_ma}, MA慢={self.trend_slow_ma}")

        self._load_data()

        # 获取交易日列表
        all_days = self._trade_dates
        start_idx = 0
        end_idx = len(all_days)
        for i, d in enumerate(all_days):
            if d >= self.start_date:
                start_idx = i
                break
        for i, d in enumerate(all_days):
            if d > self.end_date:
                end_idx = i
                break

        trade_days = all_days[start_idx:end_idx]
        print(f"\n回测交易日数: {len(trade_days)}")

        print(f"\n开始回测...")
        for i, day in enumerate(trade_days):
            sym = self._get_trading_symbol(day)
            if not sym:
                continue

            # 计算三层方向
            self.macro_dir = self._calc_macro_direction(day)
            self.flow_dir = self._calc_flow_direction(sym, day)
            self.trend_dir = self._calc_trend_direction(sym, day)

            dirs = [self.macro_dir, self.flow_dir, self.trend_dir]
            self.signal_dir = self._calc_entry_signal(dirs)

            # 记录信号
            self.signal_count += 1

            # 交易逻辑
            if self.position != 0:
                # 检查平仓
                current_yield = self._get_ytm(sym, day)
                if current_yield is not None and self.entry_yield is not None:
                    pos = self.position
                    if pos > 0:
                        move = self._yield_move_bp(self.entry_yield, current_yield, long_position=True)
                        if move >= self.take_profit_yield_bp:
                            self._close_position(day, "TAKE_PROFIT")
                        elif move <= -self.stop_loss_yield_bp:
                            self._close_position(day, "STOP_LOSS")
                    else:
                        move = self._yield_move_bp(self.entry_yield, current_yield, long_position=False)
                        if move >= self.take_profit_yield_bp:
                            self._close_position(day, "TAKE_PROFIT")
                        elif move <= -self.stop_loss_yield_bp:
                            self._close_position(day, "STOP_LOSS")

                # 反转信号平仓
                if self.position > 0 and self._count_reverse_layers(self.position) >= 2:
                    self._close_position(day, "REVERSE_SIGNAL")
                elif self.position < 0 and self._count_reverse_layers(self.position) >= 2:
                    self._close_position(day, "REVERSE_SIGNAL")

            else:
                # 开仓
                if self.signal_dir != 0:
                    self._open_position(day, self.signal_dir, "LAYER_CONFIRM")

            # 进度
            if (i + 1) % 500 == 0:
                print(f"  处理进度: {i+1}/{len(trade_days)} 天...")

        # 最终平仓
        if self.position != 0:
            last_day = trade_days[-1] if trade_days else self.end_date
            self._close_position(last_day, "END_OF_BACKTEST")

        print(f"\n回测完成!")
        print(f"  信号次数: {self.signal_count}")
        print(f"  交易次数: {self.trade_count}")

        return self.trades, self.actions, self.equity_curve

    def _count_reverse_layers(self, pos: int) -> int:
        """计算反向层数"""
        return sum(1 for d in [self.macro_dir, self.flow_dir, self.trend_dir]
                   if (pos > 0 and d < 0) or (pos < 0 and d > 0))


def main():
    bt = SuiBoLiuBacktest(
        start_date="20240101",
        end_date="20260519",
    )
    trades, actions, equity = bt.run()
    plot_backtest_results(trades, actions, equity, prefix="sui_boliu")
    print(f"\n所有输出文件在 output/ 目录下")


if __name__ == "__main__":
    main()

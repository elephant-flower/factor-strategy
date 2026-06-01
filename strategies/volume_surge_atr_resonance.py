"""
量涌波动率因子策略 — 离线回测版

策略逻辑（匹配POC single1strategy_success.py）：
1. 计算每根5分钟K线是否触发量涌+波动率跃升共振
2. 用近N根K线突破方向确认（向上突破开多，向下突破开空）
3. 按固定止盈止损和收盘前强平规则离场
4. 每日最多1次开仓尝试，开仓冷却30根K线

数据来源：本地CSV文件 (data/t_futures_5min.csv)
输出：回测结果图表 (output/ 目录)

用法：
  python strategies/volume_surge_atr_resonance.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict, deque
from datetime import datetime, time, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest_base import (
    BarData, TradeRecord, ActionRecord,
    load_t_futures_5min,
    plot_backtest_results,
)


class VolumeSurgeATRResonanceBacktest:
    """量涌波动率共振策略 - 离线回测引擎（匹配POC逻辑）"""

    # ─── 参数 ─────────────────────────────────────────────
    lookback_days = 20              # 历史回溯天数
    min_history_days = 20           # 最少历史天数
    volume_ratio_threshold = 3.0    # 量涌倍数阈值
    volume_quantile = 0.95          # 成交量分位数阈值
    tr_ratio_threshold = 2.5        # TR倍数阈值
    tr_quantile = 0.90              # TR分位数阈值

    breakout_window = 10            # 突破回溯K线数
    take_profit_multiple = 2.0      # 止盈倍数（优化值）
    stop_loss_multiple = 0.3        # 止损倍数（优化值）

    force_flat_minutes_before_close = 15
    session_close_hour = 15
    session_close_minute = 15

    initial_capital = 10000000.0
    risk_per_trade = 0.001
    capital_utilization = 0.15
    contract_multiplier = 10000.0
    min_order_volume = 1
    max_order_volume = 50
    fallback_pricetick = 0.005

    # 开仓节流（匹配POC）
    open_cooldown_bars = 30
    max_open_attempts_per_day = 2
    aggressive_ticks = 2

    start_date = "20210101"
    end_date = "20260519"

    def __init__(self, start_date: str = None, end_date: str = None):
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        # 状态变量
        self.cur_trade_date = ""
        self.main_contract = ""
        self.day_symbol_volume: Dict[str, float] = {}
        self.main_chosen_day: Optional[str] = None
        self.day_for_volume: Optional[str] = None

        # 因子计算状态 - 使用time slot作为key，匹配POC的time().replace(second=0, microsecond=0)
        self.recent_bars: Deque[BarData] = deque(maxlen=self.breakout_window + 1)
        self._slot_volume_hist: Dict[time, Deque[float]] = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self._slot_tr_hist: Dict[time, Deque[float]] = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self._prev_close: Optional[float] = None

        # 持仓状态
        self.position = 0
        self.entry_price: Optional[float] = None
        self.entry_range: Optional[float] = None
        self.entry_symbol: Optional[str] = None
        self.entry_time: Optional[datetime] = None

        # 因子值
        self.latest_resonance = False
        self.volume_surge_flag = False
        self.atr_jump_flag = False
        self.last_volume_ratio: Optional[float] = None
        self.last_tr_ratio: Optional[float] = None

        # 开仓节流（匹配POC）
        self._bar_count = 0
        self._open_attempt_day: Optional[str] = None
        self._open_attempts_today = 0
        self._next_open_bar_count = 0

        # 统计
        self.factor_calc_count = 0
        self.resonance_true_count = 0
        self.breakout_up_count = 0
        self.breakout_down_count = 0
        self.open_attempt_count = 0
        self.open_success_count = 0

        # 记录（累计PnL从0开始）
        self.trades: List[TradeRecord] = []
        self.actions: List[ActionRecord] = []
        self.equity_curve: List[float] = [0.0]
        self.current_trade: Optional[dict] = None

    # ─── 数据准备 ──────────────────────────────────────────

    def _load_all_bars(self) -> List[BarData]:
        """加载并预处理所有5分钟K线数据"""
        print(f"正在加载T期货5分钟K线数据...")
        df = load_t_futures_5min(self.start_date, self.end_date)
        if df.empty:
            raise ValueError(f"数据为空: {self.start_date} - {self.end_date}")

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        # 过滤5分钟整数倍
        df = df[df["datetime"].dt.minute % 5 == 0].copy()
        df = df.sort_values(["instrument_id", "datetime"])

        print(f"  共加载 {len(df)} 条K线, {df['instrument_id'].nunique()} 个合约")
        print(f"  日期范围: {df['trading_day'].min()} ~ {df['trading_day'].max()}")

        all_bars = []
        for _, row in df.iterrows():
            bar = BarData(
                datetime=row["datetime"].to_pydatetime(),
                symbol=str(row["instrument_id"]),
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
                volume=float(row["volume"]),
                turnover=float(row.get("turnover", 0)),
                open_interest=float(row.get("open_interest", 0)),
            )
            all_bars.append(bar)
        return all_bars

    def _group_bars_by_time(self, all_bars: List[BarData]) -> Dict[datetime, List[BarData]]:
        """按时间戳分组"""
        groups: Dict[datetime, List[BarData]] = defaultdict(list)
        for bar in all_bars:
            groups[bar.datetime].append(bar)
        return dict(sorted(groups.items()))

    # ─── 主力合约选择（匹配POC按日成交量）────────────────────

    def _select_main_contract(self, bars_at_time: List[BarData], current_date: str) -> str:
        """按当日累计成交量选择主力合约"""
        if self.day_for_volume != current_date:
            self.day_for_volume = current_date
            self.day_symbol_volume.clear()
            self.main_chosen_day = None

        for bar in bars_at_time:
            self.day_symbol_volume[bar.symbol] = (
                self.day_symbol_volume.get(bar.symbol, 0) + bar.volume
            )

        if self.main_chosen_day is None and self.day_symbol_volume:
            main_sym = max(self.day_symbol_volume, key=self.day_symbol_volume.get)
            if main_sym != self.main_contract:
                print(f"  主力切换: {current_date} -> {main_sym} (量:{self.day_symbol_volume[main_sym]:.0f})")
            self.main_contract = main_sym
            self.main_chosen_day = current_date

        return self.main_contract

    # ─── 因子计算（匹配POC）─────────────────────────────────

    def _calc_true_range(self, bar: BarData) -> float:
        hl = bar.high_price - bar.low_price
        if self._prev_close is None:
            return max(hl, 0.0)
        hc = abs(bar.high_price - self._prev_close)
        lc = abs(bar.low_price - self._prev_close)
        return max(hl, hc, lc, 0.0)

    def _slot_key(self, dt: datetime) -> time:
        """匹配POC: time().replace(second=0, microsecond=0) 作为slot key"""
        return dt.time().replace(second=0, microsecond=0)

    def _compute_factor(self, bar: BarData):
        """计算共振因子（匹配POC _compute_local_factor_for_bar）"""
        self.factor_calc_count += 1
        slot = self._slot_key(bar.datetime)

        tr_value = self._calc_true_range(bar)
        vol_value = bar.volume

        vol_hist = self._slot_volume_hist[slot]
        tr_hist = self._slot_tr_hist[slot]

        enough_history = len(vol_hist) >= self.min_history_days and len(tr_hist) >= self.min_history_days

        self.latest_resonance = False
        self.volume_surge_flag = False
        self.atr_jump_flag = False
        self.last_volume_ratio = None
        self.last_tr_ratio = None

        if enough_history:
            vol_series = pd.Series(list(vol_hist), dtype="float64")
            tr_series = pd.Series(list(tr_hist), dtype="float64")

            vol_mean = float(vol_series.mean()) if len(vol_series) else 0.0
            tr_mean = float(tr_series.mean()) if len(tr_series) else 0.0
            vol_q = float(vol_series.quantile(self.volume_quantile)) if len(vol_series) else 0.0
            tr_q = float(tr_series.quantile(self.tr_quantile)) if len(tr_series) else 0.0

            volume_ratio = (vol_value / vol_mean) if vol_mean > 0 else None
            tr_ratio = (tr_value / tr_mean) if tr_mean > 0 else None

            if volume_ratio is not None:
                self.last_volume_ratio = volume_ratio
                self.volume_surge_flag = bool(
                    volume_ratio >= self.volume_ratio_threshold
                    and vol_value >= vol_q
                )
            if tr_ratio is not None:
                self.last_tr_ratio = tr_ratio
                self.atr_jump_flag = bool(
                    tr_ratio >= self.tr_ratio_threshold
                    and tr_value >= tr_q
                )

            self.latest_resonance = self.volume_surge_flag and self.atr_jump_flag
            if self.latest_resonance:
                self.resonance_true_count += 1

        # 每根bar均入历史（匹配POC关键修复）
        vol_hist.append(vol_value)
        tr_hist.append(tr_value)

        self._prev_close = bar.close_price

    # ─── 开仓节流（匹配POC）─────────────────────────────────

    def _can_try_open(self, dt: datetime) -> bool:
        """检查是否可以开仓（匹配POC _can_try_open_now）"""
        day_key = dt.strftime("%Y%m%d")
        if self._open_attempt_day != day_key:
            self._open_attempt_day = day_key
            self._open_attempts_today = 0

        if self._open_attempts_today >= self.max_open_attempts_per_day:
            return False

        return self._bar_count >= self._next_open_bar_count

    def _mark_open_attempt(self, dt: datetime):
        """记录开仓尝试（匹配POC _mark_open_attempt）"""
        day_key = dt.strftime("%Y%m%d")
        if self._open_attempt_day != day_key:
            self._open_attempt_day = day_key
            self._open_attempts_today = 0
        self._open_attempts_today += 1
        self._next_open_bar_count = self._bar_count + self.open_cooldown_bars

    # ─── 交易逻辑（匹配POC）─────────────────────────────────

    def _should_force_flat(self, dt: datetime) -> bool:
        close_dt = datetime.combine(dt.date(), time(self.session_close_hour, self.session_close_minute))
        deadline = close_dt - timedelta(minutes=self.force_flat_minutes_before_close)
        return dt >= deadline

    def _check_open(self, bar: BarData):
        """检查开仓（匹配POC _check_open）"""
        history = list(self.recent_bars)[:-1]
        if len(history) < self.breakout_window:
            return

        if not self._can_try_open(bar.datetime):
            return

        highest = max(b.high_price for b in history)
        lowest = min(b.low_price for b in history)

        break_up = bar.close_price > highest
        break_down = bar.close_price < lowest

        if break_up:
            self.breakout_up_count += 1
        if break_down:
            self.breakout_down_count += 1

        if not (break_up or break_down):
            return

        self._mark_open_attempt(bar.datetime)

        if break_up and self.position <= 0:
            if self.position < 0:
                self._close_position(bar, reason="reverse_to_long")
            self._open_long(bar, reason="breakout_up")
        elif break_down and self.position >= 0:
            if self.position > 0:
                self._close_position(bar, reason="reverse_to_short")
            self._open_short(bar, reason="breakout_down")

    def _check_exit(self, bar: BarData) -> bool:
        """检查平仓（匹配POC _check_exit，先止损后止盈）"""
        if self.position == 0 or self.entry_price is None or self.entry_range is None:
            return False

        tp = self.take_profit_multiple * self.entry_range
        sl = self.stop_loss_multiple * self.entry_range

        if self.position > 0:
            adverse = self.entry_price - bar.close_price
            favorable = bar.close_price - self.entry_price
            if adverse >= sl:
                self._close_position(bar, reason="stop_loss")
                return True
            if favorable >= tp:
                self._close_position(bar, reason="take_profit")
                return True
        elif self.position < 0:
            adverse = bar.close_price - self.entry_price
            favorable = self.entry_price - bar.close_price
            if adverse >= sl:
                self._close_position(bar, reason="stop_loss")
                return True
            if favorable >= tp:
                self._close_position(bar, reason="take_profit")
                return True
        return False

    def _force_flat(self, bar: BarData):
        """收盘强平"""
        if self.position != 0:
            self._close_position(bar, reason="force_flat")

    def _open_long(self, bar: BarData, reason: str = "breakout_up"):
        """开多（匹配POC）"""
        self.open_attempt_count += 1
        self.entry_price = bar.close_price
        self.entry_range = max(bar.high_price - bar.low_price, 0.0)
        self.entry_symbol = bar.symbol
        self.entry_time = bar.datetime

        vol = self._calc_order_volume()
        if vol <= 0:
            return

        self.position = vol
        self.open_success_count += 1

        self.current_trade = {
            "symbol": bar.symbol,
            "side": "LONG",
            "entry_time": bar.datetime,
            "entry_price": self.entry_price,
            "volume": vol,
        }

        self.actions.append(ActionRecord(
            datetime=bar.datetime, action="OPEN_LONG", symbol=bar.symbol,
            price=self.entry_price, volume=vol, reason=reason,
            resonance_flag=self.latest_resonance,
            volume_surge_flag=self.volume_surge_flag,
            atr_jump_flag=self.atr_jump_flag,
            volume_ratio=self.last_volume_ratio,
            tr_ratio=self.last_tr_ratio,
        ))

    def _open_short(self, bar: BarData, reason: str = "breakout_down"):
        """开空（匹配POC）"""
        self.open_attempt_count += 1
        self.entry_price = bar.close_price
        self.entry_range = max(bar.high_price - bar.low_price, 0.0)
        self.entry_symbol = bar.symbol
        self.entry_time = bar.datetime

        vol = self._calc_order_volume()
        if vol <= 0:
            return

        self.position = -vol
        self.open_success_count += 1

        self.current_trade = {
            "symbol": bar.symbol,
            "side": "SHORT",
            "entry_time": bar.datetime,
            "entry_price": self.entry_price,
            "volume": vol,
        }

        self.actions.append(ActionRecord(
            datetime=bar.datetime, action="OPEN_SHORT", symbol=bar.symbol,
            price=self.entry_price, volume=vol, reason=reason,
            resonance_flag=self.latest_resonance,
            volume_surge_flag=self.volume_surge_flag,
            atr_jump_flag=self.atr_jump_flag,
            volume_ratio=self.last_volume_ratio,
            tr_ratio=self.last_tr_ratio,
        ))

    def _close_position(self, bar: BarData, reason: str = "exit"):
        """平仓并记录成交"""
        if self.position == 0 or self.current_trade is None:
            return

        close_price = bar.close_price
        side = self.current_trade["side"]
        entry_px = self.current_trade["entry_price"]
        vol = self.current_trade["volume"]

        if side == "LONG":
            pnl = (close_price - entry_px) * vol
        else:
            pnl = (entry_px - close_price) * vol

        self.trades.append(TradeRecord(
            symbol=self.current_trade["symbol"],
            side=side,
            entry_time=self.current_trade["entry_time"],
            entry_price=entry_px,
            exit_time=bar.datetime,
            exit_price=close_price,
            volume=vol,
            reason=reason,
            pnl=pnl,
        ))

        action = "CLOSE_LONG" if self.position > 0 else "CLOSE_SHORT"
        self.actions.append(ActionRecord(
            datetime=bar.datetime, action=action, symbol=bar.symbol,
            price=close_price, volume=abs(self.position), reason=reason,
        ))

        # 累计PnL（不乘合约乘数，与POC trades.csv格式一致）
        prev = self.equity_curve[-1]
        self.equity_curve.append(prev + pnl)

        self.position = 0
        self.entry_price = None
        self.entry_range = None
        self.current_trade = None

    def _calc_order_volume(self) -> int:
        """计算下单手数（匹配POC）"""
        if self.entry_range is None or self.entry_range <= 0:
            return self.max_order_volume

        risk_lot = int((self.initial_capital * self.risk_per_trade)
                       / (self.entry_range * self.contract_multiplier))

        capital_lot = int(self.max_order_volume)
        if self.entry_price and self.entry_price > 0:
            capital_lot = int((self.initial_capital * self.capital_utilization)
                              / (self.entry_price * self.contract_multiplier))

        vol = min(risk_lot, capital_lot, self.max_order_volume)
        return max(vol, self.min_order_volume)

    # ─── 主回测循环 ────────────────────────────────────────

    def run(self) -> Tuple[List[TradeRecord], List[ActionRecord], List[float]]:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"量涌波动率共振策略 - 离线回测 (匹配POC)")
        print(f"{'='*60}")
        print(f"回测区间: {self.start_date} ~ {self.end_date}")
        print(f"参数: 量涌>{self.volume_ratio_threshold}x, TR>{self.tr_ratio_threshold}x")
        print(f"      突破={self.breakout_window}, 止盈={self.take_profit_multiple}x, 止损={self.stop_loss_multiple}x")
        print(f"      每日开仓≤{self.max_open_attempts_per_day}, 冷却={self.open_cooldown_bars}bar")

        all_bars = self._load_all_bars()
        grouped = self._group_bars_by_time(all_bars)

        print(f"\n开始回测...")
        bar_count = 0

        for dt, bars_list in grouped.items():
            self._bar_count += 1
            bar_count = self._bar_count
            trade_date = dt.strftime("%Y%m%d")

            main_sym = self._select_main_contract(bars_list, trade_date)

            target_bar = None
            for bar in bars_list:
                if bar.symbol == main_sym:
                    target_bar = bar
                    break
            if target_bar is None:
                continue

            self._compute_factor(target_bar)
            self.recent_bars.append(target_bar)

            if len(self.recent_bars) <= self.breakout_window:
                continue

            # 强平
            if self._should_force_flat(target_bar.datetime):
                self._force_flat(target_bar)
                self.actions.append(ActionRecord(
                    datetime=target_bar.datetime, action="FORCE_FLAT",
                    symbol=target_bar.symbol, price=target_bar.close_price,
                    volume=0, reason="end_of_session",
                ))
                continue

            # 先检查平仓，再检查开仓（匹配POC顺序）
            if self._check_exit(target_bar):
                continue

            if self.latest_resonance:
                self._check_open(target_bar)

            if bar_count % 20000 == 0:
                print(f"  进度: {bar_count} K线 (主力:{main_sym})...")

        # 回测结束平仓
        if self.position != 0 and self.current_trade:
            last_bar = all_bars[-1]
            self._close_position(last_bar, reason="end_of_backtest")

        # 计算累计PnL（含合约乘数，用于图表）
        pnl_with_multiplier = [x * self.contract_multiplier for x in self.equity_curve]

        print(f"\n回测完成! 共处理 {bar_count} 根K线")
        print(f"  因子计算: {self.factor_calc_count}")
        print(f"  共振触发: {self.resonance_true_count}")
        print(f"  突破向上: {self.breakout_up_count}")
        print(f"  突破向下: {self.breakout_down_count}")
        print(f"  开仓尝试: {self.open_attempt_count}")
        print(f"  开仓成功: {self.open_success_count}")

        return self.trades, self.actions, pnl_with_multiplier


def main():
    bt = VolumeSurgeATRResonanceBacktest(
        start_date="20210101",
        end_date="20260519",
    )
    trades, actions, equity = bt.run()
    # 按POC格式输出
    plot_backtest_results(trades, actions, equity, prefix="volume_surge_atr")
    print(f"\n所有输出文件在 output/ 目录下")


if __name__ == "__main__":
    main()

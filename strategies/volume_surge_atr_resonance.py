"""
量涌波动率因子策略 — 离线回测版

策略逻辑：
1. 计算每根5分钟K线是否触发量涌+波动率跃升共振
2. 用近N根K线突破方向确认（向上突破开多，向下突破开空）
3. 按固定止盈止损和收盘前强平规则离场

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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest_base import (
    BarData, TradeRecord, ActionRecord,
    bars_from_dataframe, load_t_futures_5min,
    plot_backtest_results,
)


class VolumeSurgeATRResonanceBacktest:
    """量涌波动率共振策略 - 离线回测引擎"""

    # ─── 参数 ─────────────────────────────────────────────
    # 共振参数
    lookback_days = 20          # 历史回溯天数
    min_history_days = 20       # 最少历史天数
    volume_ratio_threshold = 3.0   # 量涌倍数阈值
    volume_quantile = 0.95      # 成交量分位数阈值
    tr_ratio_threshold = 2.5    # TR倍数阈值
    tr_quantile = 0.90          # TR分位数阈值

    # 突破参数
    breakout_window = 10        # 突破回溯K线数

    # 止盈止损
    take_profit_multiple = 1.5  # 止盈倍数(入场K线波幅)
    stop_loss_multiple = 0.5    # 止损倍数(入场K线波幅)

    # 强平
    force_flat_minutes_before_close = 15  # 收盘前强平分钟数
    session_close_hour = 15
    session_close_minute = 15

    # 资金管理
    initial_capital = 10000000.0
    risk_per_trade = 0.001
    capital_utilization = 0.15
    contract_multiplier = 10000.0
    min_order_volume = 1
    max_order_volume = 50
    fallback_pricetick = 0.005

    # 数据范围
    start_date = "20240101"
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

        # 因子计算状态
        self.recent_bars: Deque[BarData] = deque(maxlen=self.breakout_window + 1)
        self._slot_volume_hist = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self._slot_tr_hist = defaultdict(lambda: deque(maxlen=self.lookback_days))
        self._prev_close: Optional[float] = None

        # 持仓状态
        self.position = 0
        self.entry_price: Optional[float] = None
        self.entry_range: Optional[float] = None
        self.entry_symbol: Optional[str] = None

        # 因子值
        self.latest_resonance = False
        self.volume_surge_flag = False
        self.atr_jump_flag = False
        self.last_volume_ratio: Optional[float] = None
        self.last_tr_ratio: Optional[float] = None

        # 统计
        self.factor_calc_count = 0
        self.resonance_true_count = 0
        self.breakout_up_count = 0
        self.breakout_down_count = 0
        self.open_attempt_count = 0
        self.open_success_count = 0
        self.entry_candidate_count = 0

        # 记录
        self.trades: List[TradeRecord] = []
        self.actions: List[ActionRecord] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.current_trade: Optional[dict] = None

    # ─── 数据准备 ──────────────────────────────────────────

    def _load_all_bars(self) -> List[BarData]:
        """加载并预处理所有5分钟K线数据"""
        print(f"正在加载T期货5分钟K线数据...")
        df = load_t_futures_5min(self.start_date, self.end_date)
        if df.empty:
            raise ValueError(f"数据为空: {self.start_date} - {self.end_date}")

        # 转换datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # 过滤5分钟整数倍
        df = df[df["datetime"].dt.minute % 5 == 0].copy()

        # 按合约+时间排序
        df = df.sort_values(["instrument_id", "datetime"])

        print(f"  共加载 {len(df)} 条K线, {df['instrument_id'].nunique()} 个合约")
        print(f"  日期范围: {df['trading_day'].min()} ~ {df['trading_day'].max()}")

        # 按交易日分组，返回每个合约的BarData列表
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
        """按时间戳分组（同一时刻可能有多个合约的K线）"""
        groups: Dict[datetime, List[BarData]] = defaultdict(list)
        for bar in all_bars:
            groups[bar.datetime].append(bar)
        return dict(sorted(groups.items()))

    # ─── 主力合约选择 ──────────────────────────────────────

    def _select_main_contract(self, bars_at_time: List[BarData], current_date: str) -> str:
        """按当日累计成交量选择主力合约"""
        if self.day_for_volume != current_date:
            self.day_for_volume = current_date
            self.day_symbol_volume.clear()
            self.main_chosen_day = None

        # 累计成交量
        for bar in bars_at_time:
            self.day_symbol_volume[bar.symbol] = (
                self.day_symbol_volume.get(bar.symbol, 0) + bar.volume
            )

        # 每天第一根K线确定主力
        if self.main_chosen_day is None:
            if self.day_symbol_volume:
                main_sym = max(self.day_symbol_volume, key=self.day_symbol_volume.get)
                if main_sym != self.main_contract:
                    print(f"  主力切换: {current_date} -> {main_sym} (量:{self.day_symbol_volume[main_sym]:.0f})")
                self.main_contract = main_sym
                self.main_chosen_day = current_date
            else:
                # 沿用上一个主力
                pass

        return self.main_contract

    # ─── 因子计算 ──────────────────────────────────────────

    def _calc_true_range(self, bar: BarData) -> float:
        high = bar.high_price
        low = bar.low_price
        if self._prev_close is None:
            return high - low
        return max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))

    def _compute_factor(self, bar: BarData):
        """计算量涌波动率共振因子"""
        tr = self._calc_true_range(bar)
        vol = bar.volume
        t = bar.datetime.time()

        vh = self._slot_volume_hist[t]
        th = self._slot_tr_hist[t]

        self.factor_calc_count += 1
        self.latest_resonance = False
        self.volume_surge_flag = False
        self.atr_jump_flag = False
        self.last_volume_ratio = None
        self.last_tr_ratio = None

        if len(vh) >= self.min_history_days and len(th) >= self.min_history_days:
            vol_arr = np.asarray(list(vh), dtype=float)
            tr_arr = np.asarray(list(th), dtype=float)

            vol_mean = float(np.mean(vol_arr)) if vol_arr.size else 0.0
            tr_mean = float(np.mean(tr_arr)) if tr_arr.size else 0.0
            vol_q = float(np.quantile(vol_arr, self.volume_quantile)) if vol_arr.size else 0.0
            tr_q = float(np.quantile(tr_arr, self.tr_quantile)) if tr_arr.size else 0.0

            if vol_mean > 0:
                self.last_volume_ratio = vol / vol_mean
                self.volume_surge_flag = (
                    self.last_volume_ratio >= self.volume_ratio_threshold
                    and vol >= vol_q
                )
            if tr_mean > 0:
                self.last_tr_ratio = tr / tr_mean
                self.atr_jump_flag = (
                    self.last_tr_ratio >= self.tr_ratio_threshold
                    and tr >= tr_q
                )

            self.latest_resonance = self.volume_surge_flag and self.atr_jump_flag
            if self.latest_resonance:
                self.resonance_true_count += 1

        vh.append(vol)
        th.append(tr)
        self._prev_close = bar.close_price

    # ─── 交易逻辑 ──────────────────────────────────────────

    def _should_force_flat(self, dt: datetime) -> bool:
        close_dt = datetime.combine(dt.date(), time(self.session_close_hour, self.session_close_minute))
        deadline = close_dt - timedelta(minutes=self.force_flat_minutes_before_close)
        return dt >= deadline

    def _check_open(self, bar: BarData):
        """检查开仓条件"""
        valid_bars = list(self.recent_bars)[:-1]
        if len(valid_bars) < self.breakout_window:
            return

        highest = max(b.high_price for b in valid_bars)
        lowest = min(b.low_price for b in valid_bars)
        close_price = bar.close_price

        break_up = close_price > highest
        break_down = close_price < lowest

        if break_up:
            self.breakout_up_count += 1
        if break_down:
            self.breakout_down_count += 1

        if not break_up and not break_down:
            return

        self.entry_candidate_count += 1

        if break_up and self.position <= 0:
            if self.position < 0:
                self._close_position(bar, reason="reverse_to_long")
            self._open_long(bar, reason="breakout_up")
        elif break_down and self.position >= 0:
            if self.position > 0:
                self._close_position(bar, reason="reverse_to_short")
            self._open_short(bar, reason="breakout_down")

    def _check_exit(self, bar: BarData) -> bool:
        """检查平仓条件"""
        if self.position == 0 or self.entry_price is None or self.entry_range is None:
            return False

        close_price = bar.close_price
        tp = self.take_profit_multiple * self.entry_range
        sl = self.stop_loss_multiple * self.entry_range

        if self.position > 0:
            if close_price - self.entry_price >= tp:
                self._close_position(bar, reason="take_profit")
                return True
            if self.entry_price - close_price >= sl:
                self._close_position(bar, reason="stop_loss")
                return True
        elif self.position < 0:
            if self.entry_price - close_price >= tp:
                self._close_position(bar, reason="take_profit")
                return True
            if close_price - self.entry_price >= sl:
                self._close_position(bar, reason="stop_loss")
                return True
        return False

    def _open_long(self, bar: BarData, reason: str = "breakout_up"):
        """开多"""
        self.open_attempt_count += 1
        self.entry_price = bar.close_price
        self.entry_range = bar.high_price - bar.low_price
        self.entry_symbol = bar.symbol

        vol = self._calc_order_volume(self.entry_price, self.entry_range)
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
        """开空"""
        self.open_attempt_count += 1
        self.entry_price = bar.close_price
        self.entry_range = bar.high_price - bar.low_price
        self.entry_symbol = bar.symbol

        vol = self._calc_order_volume(self.entry_price, self.entry_range)
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
        """平仓"""
        if self.position == 0 or self.current_trade is None:
            return

        close_price = bar.close_price
        side = self.current_trade["side"]

        if side == "LONG":
            pnl = (close_price - self.current_trade["entry_price"]) * self.current_trade["volume"] * self.contract_multiplier
        else:
            pnl = (self.current_trade["entry_price"] - close_price) * self.current_trade["volume"] * self.contract_multiplier

        self.trades.append(TradeRecord(
            symbol=self.current_trade["symbol"],
            side=side,
            entry_time=self.current_trade["entry_time"],
            entry_price=self.current_trade["entry_price"],
            exit_time=bar.datetime,
            exit_price=close_price,
            volume=self.current_trade["volume"],
            reason=reason,
            pnl=pnl,
        ))

        action = "CLOSE_LONG" if self.position > 0 else "CLOSE_SHORT"
        self.actions.append(ActionRecord(
            datetime=bar.datetime, action=action, symbol=bar.symbol,
            price=close_price, volume=abs(self.position), reason=reason,
        ))

        # 更新净值
        prev_equity = self.equity_curve[-1] if self.equity_curve else 0
        self.equity_curve.append(prev_equity + pnl)

        self.position = 0
        self.entry_price = None
        self.entry_range = None
        self.current_trade = None

    def _force_flat(self, bar: BarData):
        """强平"""
        if self.position != 0:
            self._close_position(bar, reason="force_flat")

    def _calc_order_volume(self, price: float, risk_range: float) -> int:
        """计算下单手数"""
        capital = self.initial_capital
        risk_lot = self.max_order_volume
        if risk_range > 0:
            risk_lot = int((capital * self.risk_per_trade) / (risk_range * self.contract_multiplier))

        capital_lot = self.max_order_volume
        if price > 0:
            capital_lot = int((capital * self.capital_utilization) / (price * self.contract_multiplier))

        vol = min(risk_lot, capital_lot, self.max_order_volume)
        return max(vol, self.min_order_volume)

    # ─── 主回测循环 ────────────────────────────────────────

    def run(self) -> Tuple[List[TradeRecord], List[ActionRecord], List[float]]:
        """执行回测"""
        print(f"\n{'='*60}")
        print(f"量涌波动率共振策略 - 离线回测")
        print(f"{'='*60}")
        print(f"回测区间: {self.start_date} ~ {self.end_date}")
        print(f"参数: 量涌倍数>{self.volume_ratio_threshold}x, TR倍数>{self.tr_ratio_threshold}x")
        print(f"      突破窗口={self.breakout_window}, 止盈={self.take_profit_multiple}x, 止损={self.stop_loss_multiple}x")

        # 加载数据
        all_bars = self._load_all_bars()
        grouped = self._group_bars_by_time(all_bars)

        print(f"\n开始回测...")
        bar_count = 0

        for dt, bars_list in grouped.items():
            bar_count += 1
            trade_date = dt.strftime("%Y%m%d")

            # 选择主力合约
            main_sym = self._select_main_contract(bars_list, trade_date)

            # 找到主力合约的K线
            target_bar = None
            for bar in bars_list:
                if bar.symbol == main_sym:
                    target_bar = bar
                    break

            if target_bar is None:
                continue

            # 计算因子
            self._compute_factor(target_bar)
            self.recent_bars.append(target_bar)

            if len(self.recent_bars) <= self.breakout_window:
                continue

            # 强平检查
            if self._should_force_flat(target_bar.datetime):
                self._force_flat(target_bar)
                # 记录HOLD
                self.actions.append(ActionRecord(
                    datetime=target_bar.datetime, action="FORCE_FLAT",
                    symbol=target_bar.symbol, price=target_bar.close_price,
                    volume=0, reason="force_flat_end",
                ))
                continue

            # 平仓检查
            if self._check_exit(target_bar):
                continue

            # 开仓检查
            if self.latest_resonance:
                self._check_open(target_bar)
            elif self.position == 0:
                self.actions.append(ActionRecord(
                    datetime=target_bar.datetime, action="HOLD",
                    symbol=target_bar.symbol, price=target_bar.close_price,
                    volume=0,
                    resonance_flag=False,
                    volume_surge_flag=self.volume_surge_flag,
                    atr_jump_flag=self.atr_jump_flag,
                    volume_ratio=self.last_volume_ratio,
                    tr_ratio=self.last_tr_ratio,
                ))

            # 进度显示
            if bar_count % 10000 == 0:
                print(f"  处理进度: {bar_count} K线...")

        # 回测结束，强制平仓
        if self.position != 0 and self.current_trade:
            last_bar = all_bars[-1]
            self._close_position(last_bar, reason="end_of_backtest")

        print(f"\n回测完成! 共处理 {bar_count} 根K线")
        print(f"  因子计算: {self.factor_calc_count}")
        print(f"  共振触发: {self.resonance_true_count}")
        print(f"  突破向上: {self.breakout_up_count}")
        print(f"  突破向下: {self.breakout_down_count}")
        print(f"  开仓尝试: {self.open_attempt_count}")
        print(f"  开仓成功: {self.open_success_count}")

        return self.trades, self.actions, self.equity_curve


# ─── 主入口 ────────────────────────────────────────────────

def main():
    """运行量涌波动率策略回测"""
    # 可使用更短的区间测试，例如最近3个月
    # bt = VolumeSurgeATRResonanceBacktest(start_date="20260101", end_date="20260519")

    bt = VolumeSurgeATRResonanceBacktest(
        start_date="20240101",
        end_date="20260519",
    )
    trades, actions, equity = bt.run()

    # 输出结果
    prefix = "volume_surge_atr"
    plot_backtest_results(trades, actions, equity, prefix=prefix)

    print(f"\n所有输出文件在 output/ 目录下")


if __name__ == "__main__":
    main()

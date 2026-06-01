"""
离线回测基础模块：数据加载、K线对象、回测结果记录与图表输出。
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# ─── 数据对象 ─────────────────────────────────────────────

@dataclass
class BarData:
    """K线数据"""
    datetime: datetime
    symbol: str = ""
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0

    @property
    def open(self) -> float:
        return self.open_price

    @property
    def high(self) -> float:
        return self.high_price

    @property
    def low(self) -> float:
        return self.low_price

    @property
    def close(self) -> float:
        return self.close_price


@dataclass
class TradeRecord:
    """成交记录"""
    symbol: str
    side: str  # LONG / SHORT
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    volume: float
    reason: str
    pnl: float = 0.0


@dataclass
class ActionRecord:
    """操作记录"""
    datetime: datetime
    action: str  # HOLD / OPEN_LONG / OPEN_SHORT / CLOSE_LONG / CLOSE_SHORT / FORCE_FLAT
    symbol: str
    price: float
    volume: float
    reason: str = ""
    resonance_flag: bool = False
    volume_surge_flag: bool = False
    atr_jump_flag: bool = False
    volume_ratio: Optional[float] = None
    tr_ratio: Optional[float] = None
    # 随波逐流专用
    macro_dir: int = 0
    flow_dir: int = 0
    trend_dir: int = 0
    signal_dir: int = 0


# ─── CSV数据加载 ────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_csv(filename: str) -> pd.DataFrame:
    """加载CSV数据文件"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_csv(path)


def load_t_futures_5min(start_date: str, end_date: str) -> pd.DataFrame:
    """加载T期货5分钟K线"""
    df = load_csv("t_futures_5min.csv")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    # 确保trading_day为字符串
    df["trading_day"] = df["trading_day"].astype(str)
    # 过滤日期范围
    mask = (df["trading_day"] >= str(start_date)) & (df["trading_day"] <= str(end_date))
    df = df[mask].copy()
    return df


def load_t_futures_day(start_date: str, end_date: str) -> pd.DataFrame:
    """加载T期货日K线"""
    df = load_csv("t_futures_day.csv")
    df["trading_day"] = df["trading_day"].astype(str)
    mask = (df["trading_day"] >= str(start_date)) & (df["trading_day"] <= str(end_date))
    return df[mask].copy()


def load_bond_day(start_date: str, end_date: str) -> pd.DataFrame:
    """加载债券日线"""
    df = load_csv("bond_day.csv")
    df["trade_date"] = df["trade_date"].astype(str)
    # 重命名security_id为symbol
    df = df.rename(columns={"security_id": "symbol"})
    mask = (df["trade_date"] >= str(start_date)) & (df["trade_date"] <= str(end_date))
    return df[mask].copy()


def load_macro_data() -> Dict[str, pd.DataFrame]:
    """加载所有宏观数据"""
    data = {}
    for name in ["macro_dr007", "macro_mlf", "macro_pmi", "macro_cpi", "macro_industrial_va"]:
        try:
            df = load_csv(f"{name}.csv")
            data[name] = df
        except FileNotFoundError:
            print(f"  [警告] 宏观数据文件不存在: {name}.csv")
            data[name] = pd.DataFrame()
    return data


def bars_from_dataframe(df: pd.DataFrame, symbol_col: str = "symbol") -> List[BarData]:
    """将DataFrame转换为BarData列表"""
    bars = []
    for _, row in df.iterrows():
        dt = row.get("datetime")
        if pd.isna(dt):
            continue
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)

        bar = BarData(
            datetime=dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
            symbol=str(row.get(symbol_col, "")),
            open_price=float(row.get("open_price", row.get("open", 0))),
            high_price=float(row.get("high_price", row.get("high", 0))),
            low_price=float(row.get("low_price", row.get("low", 0))),
            close_price=float(row.get("close_price", row.get("close", 0))),
            volume=float(row.get("volume", 0)),
            turnover=float(row.get("turnover", 0)),
            open_interest=float(row.get("open_interest", 0)),
        )
        bars.append(bar)
    return bars


# ─── 回测结果图表输出 ──────────────────────────────────────

def plot_backtest_results(
    trades: List[TradeRecord],
    actions: List[ActionRecord],
    equity_curve: List[float],
    prefix: str = "",
    output_dir: str = None,
):
    """绘制回测结果图表，参考poc_output的格式

    生成:
      1. {prefix}_cumulative_pnl.png - 累计盈亏曲线
      2. {prefix}_daily_pnl_distribution.png - 每日盈亏分布
      3. {prefix}_drawdown_curve.png - 回撤曲线
      4. {prefix}_trades.csv - 成交明细
      5. {prefix}_actions.csv - 操作明细
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 保存成交明细
    if trades:
        trades_path = os.path.join(output_dir, f"{prefix}_trades.csv")
        with open(trades_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "side", "entry_time", "entry_price", "exit_time",
                             "exit_price", "volume", "reason", "pnl"])
            for t in trades:
                writer.writerow([
                    t.symbol, t.side,
                    t.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{t.entry_price:.4f}",
                    t.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{t.exit_price:.4f}",
                    f"{t.volume:.0f}", t.reason, f"{t.pnl:.4f}"
                ])
        print(f"  成交明细 -> {trades_path}")

    # 保存操作明细
    if actions:
        actions_path = os.path.join(output_dir, f"{prefix}_actions.csv")
        with open(actions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 动态列头
            sample = actions[0]
            extra_keys = [k for k in vars(sample).keys()
                          if k not in ("datetime", "action", "symbol", "price", "volume", "reason")]
            headers = ["datetime", "action", "symbol", "price", "volume", "reason"] + extra_keys
            writer.writerow(headers)
            for a in actions:
                row = [
                    a.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    a.action, a.symbol,
                    f"{a.price:.4f}", f"{a.volume:.0f}", a.reason,
                ]
                for k in extra_keys:
                    val = getattr(a, k, "")
                    row.append(str(val) if val is not None else "")
                writer.writerow(row)
        print(f"  操作明细 -> {actions_path}")

    if not equity_curve or len(equity_curve) < 2:
        print("  [WARNING] No equity data, skip charts")
        return

    # 去重连续的相同净值点，避免matplotlib生成巨大mesh
    cleaned_equity = [equity_curve[0]]
    for v in equity_curve[1:]:
        if abs(v - cleaned_equity[-1]) > 1e-8:
            cleaned_equity.append(v)
    if len(cleaned_equity) < 2:
        cleaned_equity = equity_curve

    # 计算回撤
    equity = np.array(cleaned_equity)
    nav = equity + 10000000.0
    peak = np.maximum.accumulate(nav)
    drawdown = np.where(peak > 0, (nav - peak) / peak * 100, np.zeros_like(nav))

    # 按日分组计算盈亏
    daily_pnl = []
    if trades:
        by_day = defaultdict(float)
        for t in trades:
            day_key = t.exit_time.strftime("%Y-%m-%d")
            by_day[day_key] += t.pnl
        daily_pnl = list(by_day.values())

    # 1. 累计盈亏曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(equity)), equity, label="Equity", color="#2196F3", linewidth=1.5)
    ax.fill_between(range(len(equity)), equity, alpha=0.15, color="#2196F3")
    ax.axhline(y=equity[0], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Portfolio Value", fontsize=11)
    ax.set_title(f"Cumulative PnL ({prefix})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    total_pnl = equity[-1] - equity[0] if len(equity) > 1 else 0
    win_trades = sum(1 for t in trades if t.pnl > 0) if trades else 0
    loss_trades = sum(1 for t in trades if t.pnl < 0) if trades else 0
    trade_count = len(trades)
    win_rate = win_trades / trade_count * 100 if trade_count else 0
    avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if win_trades else 0
    avg_loss = abs(np.mean([t.pnl for t in trades if t.pnl < 0])) if loss_trades else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 夏普比率 (使用日收益率)
    daily_returns = []
    if trades:
        by_day_return = defaultdict(float)
        for t in trades:
            day_key = t.exit_time.strftime("%Y-%m-%d")
            by_day_return[day_key] += t.pnl
        daily_returns = list(by_day_return.values())
    sharpe = 0.0
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    max_dd = 0.0
    if len(drawdown) > 0:
        dd_finite = drawdown[np.isfinite(drawdown)]
        if len(dd_finite) > 0:
            max_dd = abs(np.min(dd_finite))

    stats_text = (
        f"Total PnL: {total_pnl:+.2f}\n"
        f"Trades: {trade_count}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"P/L Ratio: {profit_loss_ratio:.2f}\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"Max DD: {max_dd:.2f}%"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    cum_pnl_path = os.path.join(output_dir, f"{prefix}_cumulative_pnl.png")
    plt.savefig(cum_pnl_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Cumulative PnL chart -> {cum_pnl_path}")

    # 2. 每日盈亏分布
    fig, ax = plt.subplots(figsize=(12, 5))
    if daily_pnl:
        colors = ["#4CAF50" if x >= 0 else "#F44336" for x in daily_pnl]
        ax.bar(range(len(daily_pnl)), daily_pnl, color=colors, alpha=0.8, width=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Trading Day", fontsize=11)
        ax.set_ylabel("Daily PnL", fontsize=11)
        ax.set_title(f"Daily PnL Distribution ({prefix})", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        win_days = sum(1 for x in daily_pnl if x > 0)
        loss_days = sum(1 for x in daily_pnl if x < 0)
        ax.text(0.98, 0.95, f"Win Days: {win_days}  Loss Days: {loss_days}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No trade data", ha="center", va="center", fontsize=14)

    dist_path = os.path.join(output_dir, f"{prefix}_daily_pnl_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PnL Distribution chart -> {dist_path}")

    # 3. 回撤曲线
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color="#F44336")
    ax.plot(range(len(drawdown)), drawdown, color="#D32F2F", linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title(f"Drawdown Curve ({prefix})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(drawdown) > 0:
        min_idx = np.argmin(drawdown)
        if np.isfinite(drawdown[min_idx]):
            ax.annotate(f"Max DD: {drawdown[min_idx]:.1f}%",
                        xy=(min_idx, drawdown[min_idx]),
                        xytext=(min_idx + len(drawdown) * 0.05, drawdown[min_idx] + 5),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=10, color="red", fontweight="bold")

    dd_path = os.path.join(output_dir, f"{prefix}_drawdown_curve.png")
    plt.savefig(dd_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Drawdown curve -> {dd_path}")

    # 打印回测摘要
    print(f"\n{'='*50}")
    print(f"Backtest Summary ({prefix})")
    print(f"{'='*50}")
    print(f"  Total PnL:      {total_pnl:>+12.2f}")
    print(f"  Trades:         {trade_count:>12d}")
    print(f"  Win Rate:       {win_rate:>11.1f}%")
    print(f"  P/L Ratio:      {profit_loss_ratio:>11.2f}")
    print(f"  Sharpe:         {sharpe:>11.2f}")
    print(f"  Max DD:         {max_dd:>11.2f}%")
    if trades:
        print(f"  Avg Win:        {avg_win:>+12.4f}")
        print(f"  Avg Loss:       {avg_loss:>12.4f}")
    print(f"{'='*50}")

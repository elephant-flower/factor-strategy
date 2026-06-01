# Factor-Strategy

**量化因子与策略回测框架** — 离线本地运行版

本项目包含两个核心策略的完整离线回测实现：
1. **量涌波动率共振策略** (Volume Surge ATR Resonance) — T期货5分钟K线
2. **随波逐流策略** (Sui Bo Liu) — 债券日线三因子驱动

所有数据均导出为CSV文件，策略脚本可直接读取本地数据进行回测并输出图表结果。

---

## 项目结构

```
factor-stategy/
├── README.md              # 本文件
├── requirements.txt       # Python依赖
├── .gitignore
│
├── data/                  # CSV数据文件（从数据库导出）
│   ├── t_futures_5min.csv      # T期货5分钟K线（18个合约, 29万行）
│   ├── t_futures_day.csv       # T期货日K线
│   ├── bond_day.csv            # 债券日线数据（8113个券）
│   ├── macro_dr007.csv         # DR007利率
│   ├── macro_mlf.csv           # MLF利率
│   ├── macro_pmi.csv           # PMI制造业指数
│   ├── macro_cpi.csv           # CPI同比
│   └── macro_industrial_va.csv # 工业增加值
│
├── scripts/
│   ├── export_data.py     # 数据库→CSV导出脚本
│   └── run_all.py         # 一键运行两个策略回测
│
├── strategies/
│   ├── __init__.py
│   ├── backtest_base.py         # 回测基础模块（数据加载/图表输出）
│   ├── volume_surge_atr_resonance.py  # 量涌波动率策略
│   └── sui_boliu.py             # 随波逐流策略
│
└── output/                # 回测结果输出目录
    ├── volume_surge_atr_cumulative_pnl.png
    ├── volume_surge_atr_daily_pnl_distribution.png
    ├── volume_surge_atr_drawdown_curve.png
    ├── volume_surge_atr_trades.csv
    ├── volume_surge_atr_actions.csv
    ├── sui_boliu_cumulative_pnl.png
    ├── sui_boliu_daily_pnl_distribution.png
    ├── sui_boliu_drawdown_curve.png
    ├── sui_boliu_trades.csv
    └── sui_boliu_actions.csv
```

---

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 直接运行回测

```bash
# 量涌波动率策略
python strategies/volume_surge_atr_resonance.py

# 随波逐流策略
python strategies/sui_boliu.py

# 一键运行所有策略
python scripts/run_all.py
```

### 3. 查看结果

回测结果在 `output/` 目录下：
- `*_cumulative_pnl.png` — 累计盈亏曲线
- `*_daily_pnl_distribution.png` — 每日盈亏分布
- `*_drawdown_curve.png` — 回撤曲线
- `*_trades.csv` — 成交明细
- `*_actions.csv` — 操作明细

---

## 策略说明

### 量涌波动率共振策略

**交易标的：** 10年期国债期货（T合约）5分钟K线

**核心逻辑：**
1. **量涌条件：** 当前5分钟成交量 >= 过去20日同时段成交量均值×3倍，且 >= 95分位
2. **波动率跳升：** 当前TR >= 过去20日同时段TR均值×2.5倍，且 >= 90分位
3. **共振开仓：** 量涌+波动率同时触发，且价格突破过去10根K线最高/最低
4. **离场规则：** 固定止盈（入场波幅×1.5）/ 止损（入场波幅×0.5）/ 收盘前15分钟强平

**自动主力选择：** 按当日累计成交量在 Txxxx 合约中选择主力，自动拼接连续序列。

### 随波逐流策略

**交易标的：** 银行间活跃国债（日线）

**核心逻辑：**
1. **宏观方向：** 基于DR007、MLF、PMI、CPI、工业增加值判断货币政策周期
2. **资金流向：** 基于成交量和价格变化判断资金方向
3. **趋势方向：** MA快慢线（5日/10日）交叉
4. **三因子合成：** 最少两层同向确认开仓信号
5. **离场规则：** 收益率BP止盈（8BP）/ 止损（5BP）/ 反转信号平仓

---

## 数据说明

`data/` 目录下的CSV文件是从内网数据库 `quotation_test6` 导出的回测数据，覆盖范围 `20240101 ~ 20260519`。

如需重新导出数据（需要有内网数据库访问权限）：
```bash
python scripts/export_data.py
```

---

所有数据均从内网数据库导出为CSV文件，策略脚本可直接读取本地数据进行回测并输出图表结果。

---

## 回测结果

### 量涌波动率策略

| 指标 | 数值 |
|------|------|
| 回测区间 | 2021-01 ~ 2026-05 |
| 处理K线 | 37,270 根 |
| 交易次数 | 205 |
| 胜率 | 22.9% |
| 盈亏比 | 4.23 |
| 总盈亏 | **+10,300元** |
| 夏普比率 | 0.86 |
| 最大回撤 | 0.07% |

### 随波逐流策略

| 指标 | 数值 |
|------|------|
| 回测区间 | 2024-01 ~ 2026-05 |
| 交易日数 | 630 |
| 交易次数 | 9 |
| 胜率 | 11.1% |
| 总盈亏 | -57元 |

---

## 依赖

- Python ≥ 3.10
- pandas
- numpy
- matplotlib
- pymysql（仅数据导出需要）

---

## GitHub

项目地址：https://github.com/elephant-flower/factor-stategy

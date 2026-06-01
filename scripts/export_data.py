"""
数据导出脚本：从内网数据库导出回测所需数据到CSV文件。
支持量涌波动率策略（T期货5分钟K线）和随波逐流策略（债券日线+宏观数据）。

用法：
  python export_data.py

输出：
  data/t_futures_5min.csv   - T期货5分钟K线数据（所有合约）
  data/t_futures_day.csv    - T期货日K线数据（所有合约）
  data/bond_day.csv         - 债券日线数据（k_cmds_day）
  data/macro_dr007.csv      - DR007利率数据
  data/macro_mlf.csv        - MLF利率数据
  data/macro_pmi.csv        - PMI数据
  data/macro_cpi.csv        - CPI数据
  data/macro_industrial_va.csv - 工业增加值数据
"""

import csv
import os

import pymysql

# 数据库配置
DB_CONFIG = {
    "host": "10.253.48.56",
    "port": 3306,
    "user": "kingstar",
    "password": "kingstar",
    "database": "quotation_test6",
    "charset": "utf8mb4",
    "connect_timeout": 10,
}

# 数据范围
START_DATE = "20210101"
END_DATE = "20260519"

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def get_connection():
    return pymysql.connect(**DB_CONFIG)


def write_csv(out_path, rows, field_names):
    """将查询结果写入CSV文件"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for k, v in row.items():
                if v is None:
                    cleaned[k] = ""
                else:
                    cleaned[k] = v
            writer.writerow(cleaned)
    print(f"  已导出 {len(rows)} 行 -> {out_path}")


def query_all(cur, sql, params):
    """执行查询并返回字典列表"""
    cur.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        return []
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, r)) for r in rows]


def export_cffe_futures_5min(conn):
    """导出T期货5分钟K线（分批查询避免超时）"""
    print("正在导出T期货5分钟K线数据...")
    cur = conn.cursor()
    cur.execute("""
    SELECT DISTINCT instrument_id FROM k_cffe_quotation_kline_minute
    WHERE instrument_id LIKE 'T%'
      AND instrument_id NOT LIKE 'TS%'
      AND instrument_id NOT LIKE 'TL%'
      AND instrument_id NOT LIKE 'TF%'
      AND instrument_id NOT LIKE 'T_index'
      AND instrument_id NOT LIKE 'T_main'
    """)
    symbols = [r[0] for r in cur.fetchall()]
    print(f"  发现 {len(symbols)} 个T合约")

    all_rows = []
    for sym in symbols:
        cur.execute("""
        SELECT instrument_id, trading_day, `timestamp`, `open`, `high`, `low`, `close`, volume, turnover, open_interest
        FROM k_cffe_quotation_kline_minute
        WHERE instrument_id=%s AND trading_day BETWEEN %s AND %s
        ORDER BY `timestamp` ASC
        """, (sym, START_DATE, END_DATE))
        rows = cur.fetchall()
        if rows:
            columns = [desc[0] for desc in cur.description]
            all_rows.extend([dict(zip(columns, r)) for r in rows])
        print(f"    {sym}: {len(rows)} 行")

    cur.close()
    if not all_rows:
        print("  WARNING: 无T期货5分钟K线数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "t_futures_5min.csv")
    columns = list(all_rows[0].keys())
    write_csv(out_path, all_rows, columns)


def export_cffe_futures_day(conn):
    """导出T期货日K线"""
    print("正在导出T期货日K线数据...")
    cur = conn.cursor()
    sql = """
    SELECT instrument_id, trading_day, `open`, `high`, `low`, `close`, volume, turnover, open_interest
    FROM k_cffe_quotation_kline_day
    WHERE instrument_id LIKE 'T%%'
      AND instrument_id NOT LIKE 'TS%%'
      AND instrument_id NOT LIKE 'TL%%'
      AND instrument_id NOT LIKE 'TF%%'
      AND instrument_id NOT LIKE 'T_index'
      AND instrument_id NOT LIKE 'T_main'
      AND trading_day BETWEEN %s AND %s
    ORDER BY instrument_id, trading_day ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无T期货日K线数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "t_futures_day.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_bond_day(conn):
    """导出债券日线数据(k_cmds_day)"""
    print("正在导出债券日线数据...")
    cur = conn.cursor()
    sql = """
    SELECT security_id, date_day AS trade_date, `open`, `high`, `low`, `close`,
           volume, ytm, dirty, quotation_type
    FROM k_cmds_day
    WHERE date_day BETWEEN %s AND %s
    ORDER BY security_id, date_day ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无债券日线数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "bond_day.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_macro_dr007(conn):
    """导出DR007数据"""
    print("正在导出DR007数据...")
    cur = conn.cursor()
    sql = """
    SELECT quote_date, dr007
    FROM k_dr007_python
    WHERE quote_date BETWEEN %s AND %s
    ORDER BY quote_date ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无DR007数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "macro_dr007.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_macro_mlf(conn):
    """导出MLF数据"""
    print("正在导出MLF数据...")
    cur = conn.cursor()
    sql = """
    SELECT quote_date, security_id, tenor_raw, mlf_rate, operation_amount_yi
    FROM k_mlf_python
    WHERE quote_date BETWEEN %s AND %s
    ORDER BY quote_date ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无MLF数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "macro_mlf.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_macro_pmi(conn):
    """导出PMI数据"""
    print("正在导出PMI数据...")
    cur = conn.cursor()
    sql = """
    SELECT quote_date, pmi_mfg_index
    FROM k_pmi_python
    WHERE quote_date BETWEEN %s AND %s
    ORDER BY quote_date ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无PMI数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "macro_pmi.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_macro_cpi(conn):
    """导出CPI数据"""
    print("正在导出CPI数据...")
    cur = conn.cursor()
    sql = """
    SELECT quote_date, country, cpi_yoy_pct
    FROM k_cpi_python
    WHERE quote_date BETWEEN %s AND %s AND country='CN'
    ORDER BY quote_date ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无CPI数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "macro_cpi.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def export_macro_industrial_va(conn):
    """导出工业增加值数据"""
    print("正在导出工业增加值数据...")
    cur = conn.cursor()
    sql = """
    SELECT quote_date, iva_yoy_pct, iva_acc_yoy_pct
    FROM k_industrial_va_python
    WHERE quote_date BETWEEN %s AND %s
    ORDER BY quote_date ASC
    """
    rows = query_all(cur, sql, (START_DATE, END_DATE))
    cur.close()
    if not rows:
        print("  WARNING: 无工业增加值数据")
        return
    out_path = os.path.join(OUTPUT_DIR, "macro_industrial_va.csv")
    columns = list(rows[0].keys())
    write_csv(out_path, rows, columns)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"数据导出范围: {START_DATE} - {END_DATE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    conn = get_connection()
    try:
        export_cffe_futures_5min(conn)
        export_cffe_futures_day(conn)
        export_bond_day(conn)
        export_macro_dr007(conn)
        export_macro_mlf(conn)
        export_macro_pmi(conn)
        export_macro_cpi(conn)
        export_macro_industrial_va(conn)
    finally:
        conn.close()

    print("=" * 60)
    print("数据导出完成！")


if __name__ == "__main__":
    main()

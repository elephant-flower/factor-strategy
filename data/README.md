本目录存放从数据库导出的CSV数据文件。

因文件过大（最大69MB），未纳入Git版本管理。

## 生成数据

运行以下命令从内网数据库导出数据：
```bash
python scripts/export_data.py
```

## 所需数据文件

| 文件 | 说明 | 大小 |
|------|------|------|
| t_futures_5min.csv | T期货5分钟K线（18个合约） | ~70MB |
| t_futures_day.csv | T期货日K线 | ~200KB |
| bond_day.csv | 债券日线数据 | ~20MB |
| macro_dr007.csv | DR007利率 | ~50KB |
| macro_mlf.csv | MLF利率 | ~5KB |
| macro_pmi.csv | PMI制造业指数 | ~5KB |
| macro_cpi.csv | CPI同比 | ~5KB |
| macro_industrial_va.csv | 工业增加值 | ~5KB |

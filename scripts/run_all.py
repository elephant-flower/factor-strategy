"""
一键运行所有策略回测。

用法：
  python scripts/run_all.py
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_strategy(name: str, script_path: str):
    """运行单个策略回测"""
    print(f"\n{'='*60}")
    print(f"正在运行: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT,
        capture_output=False,
    )
    return result.returncode


def main():
    strategies = [
        ("量涌波动率共振策略", os.path.join(ROOT, "strategies", "volume_surge_atr_resonance.py")),
        ("随波逐流策略", os.path.join(ROOT, "strategies", "sui_boliu.py")),
    ]

    for name, path in strategies:
        if not os.path.exists(path):
            print(f"  [跳过] 文件不存在: {path}")
            continue
        ret = run_strategy(name, path)
        if ret != 0:
            print(f"  [警告] {name} 返回非零状态码: {ret}")

    print(f"\n{'='*60}")
    print(f"所有策略运行完成！")
    print(f"回测结果在 output/ 目录下")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report all cold-start trajectory controls and their score completeness."""

import argparse
import json
from pathlib import Path

import numpy as np
import regex as re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite analysis")
    text = ["# 冷首轮轨迹打分统计\n"]
    rng = np.random.default_rng(17)
    for path in args.inputs:
        report = json.loads(path.read_text())
        assert report["status"] == "complete" and report["runs"] >= 10
        text.extend(
            [
                f"## {path.name}\n",
                f"真实权重：{report['environment']['model']}；"
                f"batch-invariant={report['environment']['batch_invariant']}；"
                "每条轨迹计入第一轮冷请求，每轮重复前清空缓存。\n",
                "| 轨迹 | 模式 | 次数 | 总耗时中位数 ms | P10–P90 ms | "
                "相对 full 配对加速比 | 95% bootstrap 区间 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for index, trace in enumerate(report["traces"]):
            groups = {}
            for mode in report["config"]["modes"]:
                rows = sorted(
                    [
                        r
                        for r in report["measurements"]
                        if r["trace"] == index and r["mode"] == mode
                    ],
                    key=lambda r: r["repetition"],
                )
                if not rows:
                    continue
                assert [r["repetition"] for r in rows] == list(range(report["runs"]))
                values = np.array([r["seconds"] for r in rows])
                assert np.isfinite(values).all() and (values > 0).all()
                groups[mode] = values
            for mode, values in groups.items():
                ratios = groups["full"] / values
                sample = rng.integers(0, len(values), size=(10000, len(values)))
                ci = np.quantile(np.median(ratios[sample], axis=1), [0.025, 0.975])
                p10, median, p90 = np.quantile(values * 1000, [0.1, 0.5, 0.9])
                text.append(
                    f"| {index + 1} | {mode} | {len(values)} | {median:.2f} | "
                    f"{p10:.2f}–{p90:.2f} | {np.median(ratios):.3f} | "
                    f"{ci[0]:.3f}–{ci[1]:.3f} |"
                )
            text.append("")
        text.extend(
            [
                "\n| 轨迹 | 模式 | 缓存场景 | 各轮命中 Token | "
                "缺少动作分数 | 最大绝对误差 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in report["diagnostics"]:
            text.append(
                f"| {row['trace'] + 1} | {row['mode']} | {row['scenario']} | "
                f"{row['cached_tokens']} | {len(row['missing'])} | "
                f"{row['max_abs_error']} |"
            )
        text.append("")
    text.append(
        "unsafe 表示上游显式允许缓存并在调用端按命中数恢复偏移的强对照，"
        "只在当次完整返回动作分数时计时；它在重复请求上的缺分另列。"
        "区间只描述本次固定回放的运行内波动，不代表真实训练或跨硬件泛化。\n"
    )
    args.output.write_text(re.sub(r"\n{3,}", "\n\n", "\n".join(text)))


if __name__ == "__main__":
    main()

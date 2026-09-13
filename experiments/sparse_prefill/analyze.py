# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize complete paired runs; preserve weight provenance and uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(report):
    if report["status"] != "complete":
        raise ValueError("Incomplete runs cannot enter the performance summary")
    config = report["config"]
    expected = config["runs"] * len(config["lengths"]) * len(config["modes"])
    assert len(report["measurements"]) == expected
    rng = np.random.default_rng(17)
    summaries = []
    for length in config["lengths"]:
        grouped = {
            mode: sorted(
                [
                    row
                    for row in report["measurements"]
                    if row["length"] == length and row["mode"] == mode
                ],
                key=lambda row: row["repetition"],
            )
            for mode in config["modes"]
        }
        for rows in grouped.values():
            assert [row["repetition"] for row in rows] == list(range(config["runs"]))
        baseline = np.array([row["seconds"] for row in grouped["full"]])
        assert np.isfinite(baseline).all() and (baseline > 0).all()
        for mode, rows in grouped.items():
            seconds = np.array([row["seconds"] for row in rows])
            assert np.isfinite(seconds).all() and (seconds > 0).all()
            speedups = baseline / seconds
            median, p10, p90 = np.quantile(seconds, [0.5, 0.1, 0.9])
            interval = None
            if len(rows) >= 10:
                indices = rng.integers(0, len(rows), size=(10000, len(rows)))
                samples = np.median(speedups[indices], axis=1)
                interval = np.quantile(samples, [0.025, 0.975]).tolist()
            summaries.append(
                {
                    "length": length,
                    "mode": mode,
                    "pairs": len(rows),
                    "median_seconds": float(median),
                    "p10_seconds": float(p10),
                    "p90_seconds": float(p90),
                    "median_paired_speedup": float(np.median(speedups)),
                    "bootstrap_95_percent_interval": interval,
                }
            )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing report")
    sections = ["# 配对性能测量\n"]
    for path in args.inputs:
        report = json.loads(path.read_text())
        rows = summarize(report)
        config, environment = report["config"], report["environment"]
        sections.extend(
            [
                f"## {path.name}\n",
                f"模型配置：{config['model']}；权重：{environment['weights_kind']}；"
                f"GPU：{environment['gpu']}；eager={config['eager']}。\n",
                "| 长度 | 模式 | 配对数 | 中位数 ms | P10–P90 ms |"
                " 配对加速比 | Bootstrap 95% 区间 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            interval = row["bootstrap_95_percent_interval"]
            interval_text = (
                f"{interval[0]:.3f}–{interval[1]:.3f}"
                if interval is not None
                else "不足 10 对，不估计"
            )
            sections.append(
                f"| {row['length']} | {row['mode']} | {row['pairs']} | "
                f"{row['median_seconds'] * 1000:.2f} | "
                f"{row['p10_seconds'] * 1000:.2f}–{row['p90_seconds'] * 1000:.2f} | "
                f"{row['median_paired_speedup']:.3f} | {interval_text} |"
            )
        sections.extend(
            [
                "\n| 长度 | 模式 | 实际投影行 | 最大动作 logprob 误差 |"
                " 结果 JSON 字节 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for diagnostic in report["diagnostics"]:
            sections.append(
                f"| {diagnostic['length']} | {diagnostic['mode']} | "
                f"{diagnostic['projected_prompt_rows']} | "
                f"{diagnostic['max_abs_logprob_error']:.8g} | "
                f"{diagnostic['score_json_bytes']} |"
            )
        sections.append("")
    sections.append(
        "区间使用固定种子、10,000 次配对重采样和配对加速比中位数。"
        "区间仅描述这次运行内的波动，不覆盖跨机器、跨时段或真实工作负载变化。"
        "随机权重数据是结构成本代理；合成轨迹没有执行真实工具或训练。\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections))


if __name__ == "__main__":
    main()

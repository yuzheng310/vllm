# 冷首轮轨迹打分统计

## cache-score-invariant-main.json

真实权重：Qwen3-0.6B；batch-invariant=1；每条轨迹计入第一轮冷请求，每轮重复前清空缓存。

| 轨迹 | 模式 | 次数 | 总耗时中位数 ms | P10–P90 ms | 相对 full 配对加速比 | 95% bootstrap 区间 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | full | 10 | 672.97 | 666.91–674.36 | 1.000 | 1.000–1.000 |
| 1 | sparse | 10 | 598.33 | 589.27–599.05 | 1.126 | 1.125–1.128 |
| 1 | cache | 10 | 269.59 | 267.48–270.26 | 2.497 | 2.492–2.505 |
| 1 | unsafe | 10 | 294.92 | 293.47–325.10 | 2.281 | 2.267–2.288 |

| 2 | full | 10 | 220.55 | 219.93–221.43 | 1.000 | 1.000–1.000 |
| 2 | sparse | 10 | 196.34 | 194.89–197.58 | 1.124 | 1.117–1.132 |
| 2 | cache | 10 | 103.46 | 101.69–104.58 | 2.127 | 2.112–2.154 |
| 2 | unsafe | 10 | 111.90 | 111.36–112.80 | 1.971 | 1.960–1.979 |

| 轨迹 | 模式 | 缓存场景 | 各轮命中 Token | 缺少动作分数 | 最大绝对误差 |
| --- | --- | --- | --- | --- | --- |
| 1 | full | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 1 | full | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |
| 1 | sparse | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 1 | sparse | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |
| 1 | cache | cold-first-turn | [0, 192, 5424, 6480] | 0 | 0.0 |
| 1 | cache | repeat-after-complete | [96, 5344, 6416, 7184] | 0 | 0.0 |
| 1 | unsafe | cold-first-turn | [0, 192, 5424, 6480] | 0 | 0.0 |
| 1 | unsafe | repeat-after-complete | [192, 5424, 6480, 7216] | 246 | 0.0 |
| 2 | full | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | full | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | sparse | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | sparse | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | cache | cold-first-turn | [0, 192, 1568, 2272] | 0 | 0.0 |
| 2 | cache | repeat-after-complete | [96, 1504, 2192, 2960] | 0 | 0.0 |
| 2 | unsafe | cold-first-turn | [0, 192, 1568, 2272] | 0 | 0.0 |
| 2 | unsafe | repeat-after-complete | [192, 1568, 2272, 3008] | 256 | 0.0 |

## cache-score-original-baseline.json

真实权重：Qwen3-0.6B；batch-invariant=1；每条轨迹计入第一轮冷请求，每轮重复前清空缓存。

| 轨迹 | 模式 | 次数 | 总耗时中位数 ms | P10–P90 ms | 相对 full 配对加速比 | 95% bootstrap 区间 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | full | 10 | 672.88 | 671.89–674.12 | 1.000 | 1.000–1.000 |

| 2 | full | 10 | 220.48 | 219.40–249.88 | 1.000 | 1.000–1.000 |

| 轨迹 | 模式 | 缓存场景 | 各轮命中 Token | 缺少动作分数 | 最大绝对误差 |
| --- | --- | --- | --- | --- | --- |
| 1 | full | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 1 | full | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | full | cold-first-turn | [0, 0, 0, 0] | 0 | 0.0 |
| 2 | full | repeat-after-complete | [0, 0, 0, 0] | 0 | 0.0 |

unsafe 表示上游显式允许缓存并在调用端按命中数恢复偏移的强对照，只在当次完整返回动作分数时计时；它在重复请求上的缺分另列。区间只描述本次固定回放的运行内波动，不代表真实训练或跨硬件泛化。

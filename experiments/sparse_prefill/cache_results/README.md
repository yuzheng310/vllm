# 第二阶段原始证据

主结论见 [CACHE_RESULTS.md](../CACHE_RESULTS.md)。这里的数值文件未经筛选，
同时保留没有通过正确性门槛的首次尝试和已有开关的强对照。

| 文件 | 用途 |
| --- | --- |
| `recorded-code-workload.json` | 实验前固定的动作、真实工具输出、源码版本与来源 |
| `cache-score-correctness.json` | 默认数值模式未通过门槛，status=running，不进入性能结论 |
| `cache-score-invariant-correctness.json` | 统一批次不变模式，runs=0，仅正确性 |
| `cache-score-invariant-main.json` | 首次主测量，80 个完整轨迹计时 |
| `cache-score-original-baseline.json` | 五处相关源码还原冻结原版，20 个轨迹计时 |
| `cache-score-invariant-replication.json` | 独立进程复测，80 个轨迹计时 |
| `cache-score-final-analysis.md` | 全部正式结果与缺分对照的统计 |
| `cache-benchmark-main.py.txt` | 主实验执行脚本的精确字节，匹配记录 SHA256 |
| `logs/` | 对应启动、数值门槛失败和回归输出 |
| `ARTIFACTS.json` | 16 个导出文件的字节长度及 SHA256 |

已核对 16 份归档哈希、180 个完整轨迹计时、主实验/复测的五处源码与
当前实现一致，以及冻结原版的五处源码与 98dff2a 一致。
原版和复测使用增加了 --modes 兼容选项的最终 cache_benchmark.py，
主实验精确版本另存，不混淆它们的脚本哈希。

复测仍使用原有预编译扩展，没有替换模型权重或 CUDA 内核。
环境版本号可能仍显示 editable 安装时的开发版本；实验身份以逐文件
源码哈希为准，不能只看包的版本字符串。

主机、地址和专用路径已从发布日志移除，完整原始文件留在远程专用目录。
预提交格式检查删除了统计 Markdown 的多余空行；清单同时保留其导出时哈希
和规范化后的发布哈希，数值未变。

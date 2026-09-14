# RepoCompass 的 vLLM 前端分词复用

2026-09-14。当前选定的局部开发方向：在真实 vLLM renderer 调用边界，
按 Qwen/ChatML 已注册消息结束标记复用历史消息的 token IDs。
目标是补充 RepoCompass 的一条局部性能工作，未新建模型评测或多 LoRA 业务。
尚未实现新的 vLLM 执行路径，未产生新的性能成绩；当前材料是开发前依据。

## 与实际项目的联系

RepoCompass 的 OpenHands / LiteLLM 请求每轮发送完整消息，并要求精确
prompt 与生成 token IDs；SkyRL 将请求交给 vLLM ChatCompletion。
模型读取新工具结果时，旧消息的文本和分词结果通常没有变化，服务器却
再次分词整个 rendered prompt。其消费者是已经存在的 rollout 生成链路。

不是给 Agent 增加评分，也不是缓存 GPU KV。本方案不改模型看到的输入，
不减少实际 attention 上下文；目标仅是降低 API 前端的重复分词 CPU 成本。
现有 Ray 载荷裁剪发生在 Conversation 结束后，这两项不能重复计收益。

## 此次获得的数据证据

审计使用上一轮实验前就冻结的相同 48 条轨迹、233 请求。从原始 parquet
消息经固定 CodeScout tokenizer 和 chat template 恢复文本，包含所有冷首轮。
没有按结果挑选任务、截断请求或重复填充文本；没有测 GPU 或分词时间。

| 项目 | 结果 |
| --- | --- |
| 从原始消息完整分词，与旧冻结 input IDs 一致 | 233/233 |
| 按 `<\|im_end\|>` 分段重拼，与完整分词一致 | 233/233 |
| 全部输入 token | 2,661,314 |
| 属于当前轨迹内已出现段的 token | 1,839,537，约 69.12% |
| 段内容 UTF-8 字节 | 11,185,665 |
| 已出现段对应 UTF-8 字节 | 7,726,648，约 69.08% |
| 单轨迹理想缓存内容最大值 | 292,942 bytes |

最后一项仅为 UTF-8 内容 + 每 ID 四字节的理想载荷核算，没有 Python 对象、
锁、索引等成本，不是 RSS 实测。审计为每轨迹独立字典、无淘汰，允许同请求
重复段命中；实际有界并发 LRU 必须另测。69% 是结构性复用比例，不是提速。

检查使用 transformers 4.57.6 / tokenizers 0.22.2 的 CPU 环境。
GPU 回放曾采用 Qwen3-4B 基础权重，而此处精确文本契约取固定的
OpenHands/CodeScout-4B tokenizer；它含 NFC normalizer，不能假设没有归一化。
项目训练仍固定 vLLM 0.11.0；接入 v0.29.0 fork 不等于训练依赖已完成升级。

## 已有工作与取舍

核对时下列四个 PR 均 OPEN；固定 head 和原始正文保存在 sources/。

| 先例 | 采用或排除依据 |
| --- | --- |
| [PR 45514](https://github.com/vllm-project/vllm/pull/45514)，消息分段分词缓存 | 借鉴注册特殊边界的思路；当前只重写 encode，而已核对 v0.29.0 BaseRenderer 调用 __call__，需适配实际接入位置。其通用 delimiter 警告不能代替拒绝不安全参数。 |
| [PR 47583](https://github.com/vllm-project/vllm/pull/47583)，前缀增量编码 | 借鉴 renderer 接入、有限缓存和回退契约；暂不照搬通用预分词窗口拼接，先限定项目实际 ChatML 语义。 |
| [PR 51281](https://github.com/vllm-project/vllm/pull/51281)，解码初始化裁尾 | 维护者质疑任意 token 边界的安全性，且原生 Rust 前端可能已处理；本轮次选。 |
| [PR 55771](https://github.com/vllm-project/vllm/pull/55771)，预分词 chat 输入 | 原生已有 KV 参数别名；本项目没有路由端重复分词的现成消费者。搬移分词位置本身不能消除重复。 |

不将作者的 CPU/TTFT 数据当作我们的结果，不开重复上游 PR。
本轮未证明任意 tokenizer 都支持分段；需限定 marker 的注册属性、无空白
吞并和独立边界，核对重叠 added tokens、normalizer、特殊 token 注入及
截断/offset 参数。异常条件应回退完整分词。

## 下一步的唯一实现范围

[协议](PROTOCOL.md) 已在审计前以提交 4f5345c 固定。实现只触及 renderer
分词接入、一个受预算限制的缓存模块和必要配置；不会改 scheduler 或 CUDA。
缓存以固定 tokenizer 和 cache_salt 为边界；接口仍返回完整原生 token IDs。

CPU 验证先比较本批完整分词、无缓存分段、缓存分段，包含首轮和交错请求；
还须核查原生 fastokens 后端的替代性。暂定累计分词耗时下降至少 30% 才
进入真实服务接入检查，任何 token 不一致停止。该门槛针对局部阶段，
不把它扩写成训练吞吐、准确率或 GPU 加速。

当前还不能直接填简历数字。通过后可写的句式是：针对多轮代码定位 rollout
的长历史重复分词，扩展 vLLM 前端消息段复用，使固定真实工作负载的分词
阶段耗时下降 X%，且 token 序列与完整分词一致。X 只取后续实测。

## 复核入口

[inspect_reuse.py](inspect_reuse.py) 只审计数据与一致性，不含计时。
[reuse_audit.json](reuse_audit.json) 保存逐请求计数、文本/来源/脚本指纹。
输入仍是同一冻结 workload 及 qwen3-runtime/workloads/code_localization，
原始工具返回文本留在公共 parquet，不另行提交大文本或权重。

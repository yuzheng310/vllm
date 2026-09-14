# 评分优化的已有实践与副作用审查

审查日期：2026-09-14。回应“别人是否做过、是否有详细论证、会否有副作用”。
本轮只做文献与源码核查，没有新增 GPU 实验，也没有改变运行时代码。

## 更新后的判断

**两项核心机制都有直接先例，尤其 SGLang 已有评分起点限制缓存命中的机制。**
本项目应定位为 vLLM 面向非连续动作位置评分的工程扩展，不能声称发明了
稀疏评分、评分感知缓存边界或 APC。先前 NEXT_GAIN_RESEARCH 的查重集中于
vLLM；“没有发现同等功能”不能延伸成整个推理生态没有先例。

目前没有找到替我们的完整组合与支持范围背书的独立论文。已有实现证明路线
不是凭空设想；数学依赖关系解释何时等价；本项目实验只验证已记录的配置。
这三种证据不能互相替代，也不能推导出训练收敛已获验证。

## 1. 最直接的同类实现：SGLang

本轮用 GitHub API 核对 main，并固定快照
`6388b6cfb1d93c253714a408f1d66a093302acd7`。

- [schedule_batch.py 的缓存边界](https://github.com/sgl-project/sglang/blob/6388b6cfb1d93c253714a408f1d66a093302acd7/python/sglang/srt/managers/schedule_batch.py#L1576)：
  `_compute_max_prefix_len` 在请求输入概率时将可命中长度限制到
  `min(input_len - 1, logprob_start_len)`。
- [同文件目标 token 对齐](https://github.com/sgl-project/sglang/blob/6388b6cfb1d93c253714a408f1d66a093302acd7/python/sglang/srt/managers/schedule_batch.py#L2722)：
  评分目标从 `global_start_idx + 1` 开始，说明其起点指预测用的源行。
  本项目 API 指目标位置 j，因此对应的源行是 j−1；不能直接比较参数名下的数字。
- [logits_processor.py 的裁行](https://github.com/sgl-project/sglang/blob/6388b6cfb1d93c253714a408f1d66a093302acd7/python/sglang/srt/layers/logits_processor.py#L627)：
  在 LM head 之前按每请求起点裁剪隐藏状态，有带位置数组的注释示例。

这不仅是“也使用 KV Cache”，而是与我们的因果边界直接对应的设计。
核对的路径是连续后缀评分；本次没有全面审计 SGLang 所有 API，不能由此声称
它不可能支持其他稀疏形式。其同一文件也包含 multi-item delimiter 位置选行。
未在相同机器上比较 SGLang 与我们的性能，不作跨框架领先声明。

## 2. vLLM 内部的相邻开发

[PR #54335: Add fixed-token prefill scoring](https://github.com/vllm-project/vllm/pull/54335)
本轮核对仍为 OPEN。建议读 Design、Scoring reuses the sampler's kernel、
Relationship to SGLang、Verification。

它通过起点只处理所需后缀的 LM-head 行，返回指定候选 token 的概率。
作者说明复用原有 FP32 归约的 logprob 内核，避免 BF16 词表归约误差，
并记录因果对齐、chunking、preemption 的验证。正文明确把 prefix-cache reads
列为非目标。它支持“评分位置裁剪”的设计依据，不证明我们的缓存实现正确。
这属于尚未合并的工程提案及作者自测，不是经同行评审的论文。

## 3. 适合先读的生产技术博客

[LinkedIn: Scaling LLM-Based ranking systems with SGLang](https://www.linkedin.com/blog/engineering/ai/scaling-llm-based-ranking-systems-with-sglang-at-linkedin)

重点看 Stage three: Reusing query work with in-batch prefix caching。
作者解释公共 query 前缀只计算一次，后续候选复用其 KV；后缀仍需关注完整的
前缀与自身。文中还解释拆分 attention 后如何合并归一化结果。

这是生产排序评分、同批内复用；我们是逐轮动作概率评分、跨请求复用。
它证明“评分任务专门减少重复前缀计算”有工业先例，不是对我们的非连续
动作接口或实现的验证。不能照搬其排序效果、吞吐或时延数字。

## 4. 最值得读的副作用论证

[Horace He / Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

重点看 Batch invariance and determinism、Batch-invariant attention、
True on-policy RL。文章通过内核归约顺序解释：批形状、分块和缓存边界的变化
可能改变浮点结果；其 RL 实验还说明训练/采样概率不一致可能需要校正。
数学等价并不保证有限精度逐位相等，也不能把概率偏差自动当成无害噪声。
文章的 RL 结论限于其设置，不表示本项目必然导致同样问题。

[vLLM PR #34046](https://github.com/vllm-project/vllm/pull/34046)
仍为 OPEN，专门讨论 cache-hit/miss 改变 GEMM 行数导致数值差异，并提出
调度分块方案。它提供相关机制与作者诊断，但不能替代对我们误差根因的逐层定位。

[vLLM PR #54998](https://github.com/vllm-project/vllm/pull/54998)
已 MERGED，记录 CPU offload 忽略 skip 标记后导致 prompt logprob 缺失。
这是评分行被缓存跳过的真实故障先例；本项目当前直接拒绝相关外部 KV 路径。

## 本项目为何在数学上可以成立

以下是根据标准稠密因果 Transformer 的依赖关系所作推导，不冒充外部定理：

1. 固定权重、相同 token 前缀、位置与 attention 语义时，历史 token 的 K/V
   不依赖未来 token。复用它们不会在精确算术下改变后续计算。
2. 新的 query 仍关注全部应关注的历史 KV；没有裁掉 attention 上下文。
3. 目标 token j 的概率来自 hidden[j−1]。KV 不包含用于输出的这行 final hidden，
   因此命中长度 C 必须满足 C <= j−1，并由 coordinator 向下对齐合法块。
4. LM head 对位置逐行计算。只选需要的行可保留目标概率；每行仍保留完整词表
   的归一化，不能只在候选词上重新 softmax，否则概率语义发生变化。

## 副作用、当前处理与未验证部分

| 风险 | 依据与当前状态 |
| --- | --- |
| 缓存跳过目标所需的隐藏行，导致少分数 | SGLang 的上界设计与 vLLM 缺分修复支持此风险。本项目保留 j−1；冷/热检查未缺分。 |
| 浮点结果随形状变化 | 本项目默认模式实测最大 logprob 绝对误差 0.25782，未通过 1e-4 门槛。不能声称默认模式无精度影响，也尚未逐层归因到某个内核。 |
| 批次不变模式的性能代价与支持限制 | 所有正式对照均启用同一模式，本次误差为 0；这不代表跨模型、硬件、训练实现均一致。未测默认模式基线到新确定性配置的迁移净收益。 |
| 权重更新后旧 KV 无效 | 当前限定不可变 reference/teacher。更新 actor 权重必须清缓存或按版本严格隔离；未开发或验证权重更新协议。 |
| 内存压力与缓存淘汰 | 复用原有 KV 池与淘汰机制，命中不足会重算；高并发、短请求下可能收益变小，调度/选行开销甚至使性能倒退。未验证在线 P99。 |
| 模型/系统配置范围扩大 | 当前单卡、稠密 decoder-only 文本路径；外部 KV、LoRA、投机等按已记录限制拒绝。其他架构不能直接沿用本结论。 |
| 推断成训练效果 | 没有运行完整训练、梯度更新或收敛比较。前向评分一致不等于训练与采样栈完全一致。 |

[官方 batch invariance 文档](https://docs.vllm.ai/en/stable/features/batch_invariance/)
本次仍标记 beta。0.25782 是自然对数概率的绝对差，不能写成“0.25782%”。
正式性能与数值证据继续以 [CACHE_RESULTS.md](CACHE_RESULTS.md) 为准。

53%–60% 是同一批次不变模式下对默认正确全量评分路径的耗时下降。
上游显式开缓存且恰好不覆盖动作的强对照只落后约 7.5%–8.6%；这应与
完整性优势一起披露，不能称为相对所有已有方法提速 2.5 倍。

## 阅读建议

先读 LinkedIn 的 Stage three 理解收益来源，再读 Thinking Machines 理解
数学等价与数值一致的区别。需要源码依据时看 SGLang 的两个函数和 vLLM
PR #54335 的 Design / Verification。现有资料支持继续把它当作有明确边界的
vLLM 二开成果；不支持“原创算法、无副作用、训练已验证”的表述。

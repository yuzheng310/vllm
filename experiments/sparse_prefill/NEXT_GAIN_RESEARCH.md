# 下一步收益审查：动作打分与安全前缀复用

后续补充：SGLang 已有按评分起点限制缓存命中的直接先例。见
[已有实践与副作用审查](PRIOR_ART_AND_RISKS.md)。本文的早期查重结论主要限于
vLLM，不能作为该机制在整个生态中原创的依据。

研究日期：2026-09-14。只做源码与上游审查，未运行 GPU 实验。
本地项目检查点：`7e27ac4f13a1670091740c25e0b3b9d1f54322a3`；
上游 main 检查点：`dec0b5d63fc42dc1d1789caf572195c73dd86ea6`。

## 结论与选择

推荐下一项有界开发：**让非连续动作打分安全复用首个待评分动作之前的
本地 KV 前缀**。具体场景优先选不可变 reference/teacher 模型的逐轮动作打分，
每轮只请求新动作分数，输入保留此前全部对话与真实工具返回。
它能删掉重复的 Transformer 前缀计算，突破仅删 LM head 行所受的成本上限。
官方 APC 文档明确将多轮会话列为适用场景，但 APC 本身是既有能力，不能
描述成我们发明了前缀缓存。[官方 APC 说明](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/docs/features/automatic_prefix_caching.md)

**确定性只到“少算哪些工作”，不能预先确定某个提速百分比。**
我们能在数学和源码上确认这一扩展可删除重复前向；实际请求吞吐、轨迹打分
耗时及数值偏差必须在一张真实 GPU 上验证。尤其应与上游显式启用缓存的
强对照区分：若调用端能保证从不越过动作边界，已有 APC 已经可以省掉相同
Transformer 计算，新增性能余量可能仍主要来自稀疏 LM head。
本开发真正补齐的是**任意缓存状态下，缓存复用与完整动作分数同时成立**。
[上游参数默认行为](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/vllm/sampling_params.py)
[上游缓存命中边界](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/vllm/v1/core/kv_cache_manager.py)

## 因果边界的推导

目标位置集合为递增集合 A，首个目标为 a。位置 j 的目标分数依赖隐藏状态
H[j−1]，而 KV 缓存只保存各层 K/V，不保存本次需要输出的 H[j−1]。
如果缓存复用 C 个 Token，则本轮从位置 C 开始前向，因此必须满足：

```text
C <= a - 1
C_safe = largest valid cached prefix with length <= min(L - 1, a - 1)
```

标准全注意力、块大小 B 的路径通常向下对齐到 B 的倍数。边界处理应交给
现有 coordinator，不能自行假设所有模型只用一种块大小。对于 a=1，安全
命中为 0，自然退化成原完整前向。对于首个动作恰好在块起点，必须保留
包含 a−1 的前一块，不能把“动作起点”直接作为可跳过长度。
[已有因果行路由](../../vllm/v1/worker/gpu/sample/prompt_logprob.py)
[coordinator 的最大命中长度接口](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/vllm/v1/core/kv_cache_coordinator.py)

设第 t 轮完整长度为 L_t，实际安全命中长度为 C_t；未缓存的累计处理行数是
ΣL_t，复用后为 Σ(L_t−C_t)。缓存首轮成本已经包含在求和中。
在增长对话且缓存未被驱逐时，C_t 可以包含上一轮已评分动作与此前工具返回。
单次全轨迹打分却只能命中第一动作前缀，不能跨过早期动作直接跳到最后一轮。
这是选择“增量逐轮打分”的原因，不应靠人为扩长初始 prompt 制造收益。

Transformer 的成本并不只与新增行数成比例：新 query 仍需要读取历史 KV，
固定调度成本、块对齐重算、显存带宽与数值内核形状也在。因此
ΣL_t/Σ(L_t−C_t) 是工作量描述，**不能当成墙钟加速比**。
[官方 APC 的 prefill 限制](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/docs/features/automatic_prefix_caching.md)

## 最小实现触点与边界

1. `SamplingParams`：允许带动作位置的请求安全启用缓存；保留显式
   `skip_reading_prefix_cache=True` 的强制重算语义。初版可让调用端显式设
   False，避免悄然改变此前默认数值路径。
2. `KVCacheManager.get_computed_blocks`：在调用已有
   `find_longest_cache_hit` 前，把最大命中长度限制到首个目标的前驱位置。
   原 `Request` 已保留 sampling_params，无需新跨进程字段。
3. `InputProcessor`：首版拒绝外部 KV connector，保留现有单卡 CUDA V2、
   完整 decoder-only 文本、无 LoRA/投机/多模态等限制；优先只覆盖已验证的
   Qwen3 稠密全注意力路径，不为混合 Mamba 模型扩展工程范围。
4. GPU/CPU 稀疏分数路径原则上已按绝对位置和 computed-token offset 处理，
   但必须验证首次 schedule 的非零 computed offset、分块、重试和抢占。

以上触点依据本地现有实现和上游 request/cache/scheduler 接口。
特别是 scheduler 有独立 `connector.get_num_new_matched_tokens` 调用，
限制本地 cache-hit 上界并不能限制外部命中，故首版应拒绝该配置。
[Request 参数保留与缓存标记](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/vllm/v1/request.py)
[Scheduler 的本地与外部命中处理](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/vllm/v1/core/sched/scheduler.py)

权重更新会使缓存失效。推荐实验固定 reference/teacher 权重，不引入更新
协议，也不把同一 Token 前缀在不同权重版本下的 KV 当成可复用对象。
现有 RepoCompass 的特定 GSPO 路径把旧策略分数合并到带梯度前向，并可能
关闭 KL/ref 路径；该扩展应称为后训练打分服务能力，不能直接宣称已经加速
该项目的训练步骤。[当前任务侧逻辑](../../../repocompass/src/async_trainer.py)

## 上游查重与不能忽略的强基线

通过 agent-reach 的 GitHub/gh 路由检查上述上游快照，以及公开 PR/issue。
检索式包括 `"prompt_logprobs" "cache"`、`"prompt_logprob_start"`、
`"logprobs" "prefix caching"`。搜索只说明本次没有发现同等功能，不能证明
整个生态不存在相关设计。

| 上游材料 | 对本项目的含义 |
| --- | --- |
| [PR #54335：fixed-token prefill scoring](https://github.com/vllm-project/vllm/pull/54335) | 开放 PR，提供连续起点与固定候选词集合；正文明确把 prefix-cache reads 排除在本阶段范围外，不覆盖非连续动作集合的安全命中上界。 |
| [PR #54998：SimpleCPUOffload bypass](https://github.com/vllm-project/vllm/pull/54998) | 已合并。修复缓存导致 prompt 分数缺失，采取遵守 skip 标记的方式，不是安全复用未评分前缀。 |
| [PR #50795：外部 KV 导致空分数](https://github.com/vllm-project/vllm/pull/50795) | 外部 KV 可能使提示行完全不前向。正文提供真实风险依据，不能假设单卡就自动排除 connector。 |
| [Issue #42019 与讨论](https://github.com/vllm-project/vllm/issues/42019) | 协作者指出当前 prompt-logprob 默认跳过 cache，所见数值偏差在关闭 APC 时也存在；不能照抄旧 issue 的根因。 |
| [PR #34046：确定性前缀分块](https://github.com/vllm-project/vllm/pull/34046) | 开放 PR，处理 cache-hit/miss 改变 GEMM M 维产生的数值差异，不是本功能；同时提醒我们不能预先承诺缓存与全量 bit-identical。 |

**最强现有开关对照必须保留。** 上游允许显式
`skip_reading_prefix_cache=False`。当缓存恰好只覆盖未评分历史时，原 worker
仍可计算剩余尾部的目标分数；精心控制的调用端可根据实际输出长度推断偏移，
并在外部过滤动作。因此不应说“上游绝不可能复用缓存做任何打分”。
但上游最大命中只限制到 L−1，没有要求目标的因果前驱重新执行；若相同动作
前缀已经缓存，所需分数可能全部缺失。旧 CPU 路径还从现有列表尾部直接追加，
并没有本项目的绝对动作位置恢复。
[基线 GPU 累积](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/worker/gpu/sample/prompt_logprob.py)
[基线 CPU 输出](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/v1/engine/logprobs.py)

因此，原生正确基线是 APC 配置开启、但 full prompt-logprob 请求按上游语义
自动 bypass。额外强对照使用上述显式 False，并同时展示其可返回的完整动作
数与位置，不能用少返回结果的耗时来判它更快，也不能藏起它在无越界命中时
已有的性能优势。若调用端维护完整得分缓存能处理重复请求，那是另一个有效
方案；本项目不能把打分缓存本身能消除的工作包装成独有收益。

## 真实任务相关性与有界验证建议

RepoCompass 的 generator 目前显式使用 session_id 保持同轨迹路由，并将
随机工作目录隐藏为 `$PWD` 以使相同 prompt 前缀一致；说明增长对话和前缀
稳定性是实际工程需求。不过这些是生成路径，尚不能证明 reference 打分
服务已接入。应从真实公开代码与项目工具格式构造固定回放，注明“记录工具
返回的回放”，不伪装成真实训练 rollout。
[任务 generator](../../../repocompass/src/generator/code_search_generator.py)

最低限度的性能协议应包括：

- 固定一份自然增长的代码搜索多轮回放，保存完整 Token IDs、动作边界和工具
  返回来源；先定输入，再运行，不按结果挑长度或缓存命中比例。
- 两个主要等价任务：上游原生完整打分后过滤当前动作；新原生稀疏安全 APC
  打分。每条轨迹第一次请求也计时，不能只报已有热缓存的最后一轮。
- 一个消融：相同稀疏位置但强制 bypass，识别收益来自 LM head 还是缓存。
- 一个强对照：上游显式 False；分别报告自然首次递增请求和已缓存覆盖动作
  的重试/共享动作场景，执行结果完整性检查后才接受耗时作为可比结果。
- 同一重复内按轨迹独立 cache_salt 隔离，避免前一组或前一模式预热了后一组。
  每模式使用不同 salt，但每条轨迹内部 salt 稳定；不以不同随机 Token 改变
  任务。计入首轮即可防止免费预热。正式测量不得修改轨迹或重复利用前一轮
  完整缓存而称之为冷启动。
- 预定主指标为完整多轮打分总耗时及已评分动作 Token/s，另报每轮 P50/P95、
  完整输出比例、冷启动开销和实际命中；行数/测试数量只作机制证据，不作简历
  成果。若样本轮数不足，不把少量重复的 P95 当成稳定在线尾延迟。

不同 cache 状态改变 GEMM/attention 形状，数学等价不自动等于 BF16 位级一致。
应先比较相同 KV/相同剩余前向中的稀疏与全尾部得分，再独立比较 warm/cold
原生上游路径的数值漂移，完整记录最大误差和分布。不能把新路径宽容差而
基线严格容差。官方 batch-invariant 模式可以作为有界诊断选择，但它仍在
beta，文档未把 Qwen3-0.6B列入已验证列表，且会改变性能配置；若使用，两边
必须一致，不预先承诺它解决所有误差。
[官方 batch invariance 文档](https://github.com/vllm-project/vllm/blob/dec0b5d63fc42dc1d1789caf572195c73dd86ea6/docs/features/batch_invariance.md)

## 简历可接受的结果边界

这条路线的可写成果应是“在某个明确的多轮代码工具回放上，把**完整有效的
动作打分**总耗时从实测 X 降为实测 Y，吞吐提升 Z”，并以真实权重为主证据。
措辞应是“使选定位置打分安全利用 vLLM APC”，不是“实现 APC”。
若较强的可用上游方案已达到相同速度，必须如实记录，不能通过较弱配置换取
更大的简历百分比；届时现有功能可以保留，但不足以支撑泛化的吞吐领先声明。

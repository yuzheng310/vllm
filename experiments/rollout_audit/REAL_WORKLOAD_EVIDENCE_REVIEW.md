# 真实后训练链路与已有收益的独立复核

日期：2026-09-14。仅阅读本地代码、固定依赖源码和已有结果；未联网、未运行 GPU、未修改运行时代码。此记录回答“已有实验是否已经支持另一个必要且高收益的 vLLM 二开”，不把新假设当作已取得的成绩。

## 结论

**没有找到已经同时满足真实需求、原生功能缺口和显著任务级收益的新增候选。若继续投入，优先补已有会话调度在目标 4B 上的证据，而不是先开发另一个功能。** 0.6B 的原生 c8 对照只说明该模型/容量下可用调参替代，不能直接外推 4B；后文给出容量推导和限制。

另有一个真实但不适合立即作为 vLLM 新课题的重复数据边界：多轮 HTTP 响应中的完整 prompt token IDs。它尚无耗时收益证据，而且可在现有 SkyRL 代理层实现增量编码，不必改 vLLM 内核。

会话调度已有固定并发下的真实条件收益，可以诚实保留；它尚未证明优于原生合理并发配置，不能通过忽略该对照升级为新立项依据。训练端、Ray 传输、仓库准备和权重发布的多项已实现优化，也不能换个名称再次计为 vLLM 新功能。

## 事实来源与版本边界

- 当前简历仍位于 [肖小云-简历.html](/Users/xiaoyuzheng/面试/面试项目/肖小云-HTML简历/肖小云-简历.html:575)，写的是 RFT/GSPO、多轮代码定位与既有基础设施收益。
- 工作区指南提到的 `codeDisc/CONTEXT.md` 本次未找到，不能假定其仍存在或猜测内容。`codeDisc/repocompass/` 实现、`codeDisc/docs/adr/`、`codeDisc/research/notes/` 均在。
- RepoCompass 当前工作树含大量既存未提交修改，其 Git HEAD `34789dcb21cc9e6595c2d27ad832f69ab146006f` 不能代表当前全部实现。以下实现事实依据当前文件，而不是将这些修改伪装成该提交中的代码。
- [pyproject.toml](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/pyproject.toml:20) 固定 vLLM 0.11.0；同文件固定 SkyRL `81e5a97c7430503c0c4e6508497cc5aa01a0c624`、OpenHands SDK `85ecfd9333d2d2cc4404dd460fd38868d9b978e2`。本次直接读取了对应本地 uv Git checkout。
- 会话实验的纯发布版基线为 v0.29.0 `98dff2a81d747d1dba01a47f939f48c3526d4206`，并非上述真实训练依赖已完成升级的证明。

## 实际任务消费什么

实际路径是 `CodeSearchGenerator.generate` → 并发 Ray rollout → OpenHands Conversation → LiteLLM / SkyRL HTTP proxy → vLLM ChatCompletion → 执行工具 → 下一轮。轨迹结束后构造训练 token 序列、动作 loss mask、reward。

- [训练脚本](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/scripts/run_async_training_4b.sh:72) 选择 GRPO 优势、GSPO loss、每批一次更新、异步生成，工具为终端搜索/读取和定位结束工具。
- [请求参数](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/generator/code_search_generator.py:144) 要求 `return_token_ids=True`，没有在该实际 ChatCompletion 请求中要求 prompt logprobs、top-k teacher 分布或候选动作评分。
- [GeneratorOutput](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/generator/code_search_generator.py:639) 明确设置 `rollout_logprobs=None`。因此“RL 需要概率”不能推出“本项目需要额外通过 vLLM 对整段历史评分”。
- [trainer 条件](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/async_trainer.py:99) 检查 one-update GSPO、无 TIS/KL/entropy 等约束；[worker](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/infra/skyrl_worker.py:74) 已在训练前向中计算 current logprobs，并用 detach 提供该配方的旧策略分母。它仍需要训练端带梯度的 forward；vLLM 稀疏推理评分不能代替它。

### 会话 ID 的真实集成缺口

[generator](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/generator/code_search_generator.py:151) 已给每轨迹稳定 session ID，但固定 SkyRL 的 [InferenceEngineClient.chat_completion](/Users/xiaoyuzheng/.cache/uv/git-v0/checkouts/035e10f733380df5/81e5a97/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py:355) 会 `pop("session_id")`，仅用它选择引擎。**现有真实链路不会自动把这个字段继续传给新 vLLM 调度器。**

会话回放则通过 [AsyncLLM.generate](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/session_affinity/replay.py:180) 直接传入 session ID。移植后的端到端接入仍需处理该差异；修正字段传递本身不是另一项性能成果。

## 已有实测排除了什么

| 已有证据 | 能推出的结论 | 不能推出的结论 |
| --- | --- | --- |
| 会话回放纯原版 c16 约 60.1s、候选 c16 约 53.5s，缓存命中增加 182,368 token | 固定并发下确实减少了历史未命中计算；约 11% 墙钟收益有条件成立 | 已证明真实训练提速，或优于原版最佳配置 |
| 纯原版 c8 为 54.728s，候选 c8 为 55.266s | 原生降低活动轨迹并发已追回大部分差距 | 实际工具链必须固定 16 并发，或 1% 差异是稳定回归 |
| 上述所有主回放抢占为 0 | 当前受测收益不是减少调度抢占 | 需要开发防抢占准入策略 |
| batch invariant 模式 233/233 完整输出一致，但耗时仅下降约 1.2% | 在该受测模式没有发现输出错误 | 默认模式的 11% 与逐 token 一致性可拼为同一组“无损收益” |
| 公开真实轨迹统计 2395 请求，输入中位数 9981、输出 87 token | 长历史、短工具动作是实际负载结构 | prefill 主导耗时；短输出意味着 decode 不值得关注 |

数值来源：[会话结果](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/session_affinity/RESULTS.md:18)、[公开轨迹统计](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/rollout_audit/README.md:29)。

会话回放 [skip_tokenizer_init=True](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/session_affinity/replay.py:87)、[detokenize=False](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/session_affinity/replay.py:178)，直接给定 token IDs；工具等待为模拟值。因此它没有测到 HTTP JSON、模板/分词、SDK 事件构造和真实工具耗时。它既不能证实这些部分很慢，也不能证实这些部分可忽略。

## 有真实重复数据、但不能据此立项的 HTTP 候选

**多轮请求只返回新增 prompt token IDs，客户端精确恢复训练所需的完整序列。**

真实重复工作已由源码确定：

1. 每轮请求都要 token IDs；当前 vLLM [协议](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/entrypoints/openai/chat_completion/protocol.py:414) 的 `return_token_ids` 同时控制输出 IDs 和完整 prompt IDs。
2. [非 streaming 响应](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/entrypoints/openai/chat_completion/serving.py:1163) 实际回传 `final_res.prompt_token_ids` 全量列表；固定 SkyRL [服务路径](/Users/xiaoyuzheng/.cache/uv/git-v0/checkouts/035e10f733380df5/81e5a97/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:613) 明确只支持非 streaming 并 `model_dump` 后返回。
3. RepoCompass 已做的 [compact_intermediate_prompt_ids](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/infra/rollout_payload.py:8) 发生在 Conversation 全部结束后、Ray 返回之前，无法消除此前逐轮 HTTP 返回、反序列化和 TokenEvent 存储。
4. 默认 non-step-wise 训练只用首轮和末轮完整 prompt，见 [generator](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/generator/code_search_generator.py:419)。这是已知消费者约束，不是虚构新评分场景。

保留它的理由仅是重复数据真实存在且位于尚未优化的边界。**目前没有该边界的实际延迟、CPU 饱和、字节汇总或任务级 A/B，所以不能预告收益百分比，也不能直接承诺为下一主课题。** 旧 [10 轮合成载荷实验](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/docs/infra_step_latency_design.md:238) 的 77.9% 减少是 Ray envelope 结果，不能直接挪作新 HTTP 优化成绩。

必要的反证与工程边界：

- 关闭整个 `return_token_ids` 会丢失用于训练的精确动作 token，不能拿这个不等价配置作基线；输出文本重分词也不自动等价。
- 任意结束轮次、截断、模板变更、tokenizer 变化、上下文凝缩和边界重分词均可能打破简单拼接，必须能校验并回退完整结果。
- 单纯减少字节不等于减少轨迹墙钟，HTTP 压缩或原生 token-in/token-out 接口还可能成为更便宜的对照；本轮没有确认哪个足以替代。
- 该优化不减少 Transformer 已执行的计算。若剩余时间主要在 GPU 或真实工具，端到端上限可能很小。

### 更强的替代和接口边界

**现有代理层足以实现主要线上的字节收益，无须 fork vLLM。** 固定 SkyRL 的 [handle_openai_request](/Users/xiaoyuzheng/.cache/uv/git-v0/checkouts/035e10f733380df5/81e5a97/skyrl-train/skyrl_train/inference_engines/inference_engine_client_http_endpoint.py:144) 先获得普通 Python 字典，再交给 `JSONResponse`。在这两个步骤之间，轨迹感知的适配层可保存前一轮 IDs、核对前缀、返回 delta；OpenHands 对应适配器恢复完整字段。这样仍有工程成本和状态管理，但不要求 vLLM 新调度器、KV 结构或 CUDA kernel。

反向代理也可以解析完整 HTTP 响应后再编码 delta，代价是上游完整 JSON 的构建和传输已发生；SkyRL 字典边界比纯外部代理少一层 JSON 解析。若要同时去掉引擎 actor → SkyRL 的 Ray 传输，还可在现有 SkyRL engine wrapper 的 [model_dump 边界](/Users/xiaoyuzheng/.cache/uv/git-v0/checkouts/035e10f733380df5/81e5a97/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py:648) 适配。仍不能直接证明需要修改 vLLM 核心。

HTTP gzip 是必须保留的无新协议对照；它不消除应用端 JSON 对象和解析工作，但可能已足够消除网络瓶颈。主审完成的冻结数据离线核算表明 delta 后字段仍比逐响应 gzip 后字段更小；该核算没有测真实 HTTP CPU/网络或轨迹吞吐，且未计 base-ID/hash 等安全字段。不能据此跳过“网络是否在关键路径”的问题。

现有几个原生接口不能混为一谈：

- `return_token_ids=False` 同时去掉训练要用的 completion IDs；`usage` 只有长度等统计，不能恢复 token 身份。仅保留 completion IDs 也缺少工具观测及模板组成的完整训练序列。
- [ChatCompletion streaming 描述](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/entrypoints/openai/chat_completion/protocol.py:414) 的 delta 是**单次响应内生成 token 的增量**，每次新请求仍需返回自己的完整 prompt。这不等于跨工具轮次的 prompt ID 增量。固定 SkyRL 此路径本来就拒绝 `stream=True`。
- [AsyncLLM StreamingInput](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/v1/engine/async_llm.py:516) 是原生输入流机制；[校验](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/v1/engine/async_llm.py:604) 限制 `n>1`、FINAL_ONLY、stop strings 等。它并非当前非 streaming ChatCompletion 接口的一项 delta 开关，迁移还涉及请求生命周期和权重发布语义，不能宣称原生毫无会话支持。
- [Responses previous_response_id](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/entrypoints/openai/responses/serving.py:365) 会找存储的上一响应并构造下一请求；不能直接等同当前 ChatCompletion 所需的精确 prompt ID delta 返回。改用它属于另一层 API/SDK 迁移。
- 客户端按同版本 tokenizer/chat template 重新编码完整历史是一种潜在替代，但必须证明与服务端工具模板、特殊 token、截断完全一致；还可能多做一次完整分词。不能默认可无成本复原。

因此，这个候选的具体缺口是“当前 ChatCompletion 缺少独立的跨轮 prompt-ID delta 输出契约”，不是“vLLM 不支持增量生成”。接口切片和代理传输优化可以有产品价值，但在没有真实 CPU/网络收益前，不宜包装成深度引擎性能优化。

## 为什么 0.6B 的否决不能替代目标 4B 的判断

本地已下载的 Qwen3-4B `config.json` 给出 36 layers、8 KV heads、head_dim=128。BF16 KV 每 token 为：

`2(K,V) × 36 × 8 × 128 × 2 bytes = 147456 bytes = 144 KiB`。

三片官方权重合计 8,044,982,000 bytes，约 7.493 GiB。按 RTX 4090 D 的 nvidia-smi 24,564 MiB、`gpu_memory_utilization=0.8`，完全不计 activation/workspace/CUDA graph/非 KV 开销时，理想 KV 预算约 85.2k token。主审另从实际 vLLM 日志核对到 torch 可见总容量 23.51 GiB、desired 18.81 GiB；用该账本扣权重则约 82.4k token。两者都是约 8.2–8.5 万 token 的扣工作区前量级，不能混用总显存口径或冒充真实已分配 block 数。权重文件字节也不严格等同 GPU 权重占用，实际容量仍以目标引擎启动报告为准。

主审核对的 0.6B 引擎日志为 162,752 token 的 KV 池。因此“0.6B c8 能保住全部历史”不能推出“4B c8 仍能做到”。若要进一步降低到 c4 来保缓存，较大的 4B 权重还可能使较小 decode batch 的权重读取摊销变差。**缓存容量下降和 decode batching 的冲突具有明确物理依据，是继续验证现有策略比转投未知 HTTP 开销更合理的理由。**

但这仍不是大收益证明：更多重算不等于等待队列中存在尚存的热前缀，缓存若已在工具执行期间被淘汰，重排也救不回来；原生 full-input admission、较低并发或其他原生配置仍可能胜出。4B 较大的权重读取成本也可能稀释省 prefill 的墙钟收益。不能提前给出百分比或承诺 4B 一定通过。

建议证据顺序只有一项：将已有移植、同一冻结工作负载、纯发布版与强原生并发对照补到真正目标 4B，先证明“调参仍不能追回”的剩余任务级优势；若需要放弃完整轨迹、降低输出长度或人为拉长工具延时才能通过，则不应升级成绩。当前 0.6B 结果仍保留为容量宽裕时的无收益/可替代对照，而非删除。

这项修正只取消旧 no-go 对未测 4B 的过强外推，**不推翻 0.6B 的任何实际对照，也不把 4B 未知结果改成预计成功**。准确状态应是“0.6B 不支持主要简历立项；目标 4B 由于容量边界明显不同，仍有一项有依据且可复用既有代码的待验证问题”。

## 不应再次计为新二开的工作

- 仓库镜像/checkout、重复 Ray prompt 压缩、后台轨迹落盘、TensorBatch 单次序列化、GSPO 前向融合和训练 padding 裁剪已经在 [infra 设计及实现](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/docs/infra_step_latency_design.md:19) 中；其中多项仍标为 CPU-validated、GPU A/B pending。
- [request drain fence](/Users/xiaoyuzheng/面试/面试项目/codeDisc/repocompass/src/infra/skyrl_client.py:10) 已替换旧 SkyRL 五秒 abort/replay；原生新 vLLM 又已有 pause wait/keep/abort。不能把升级/启用现有能力说成 vLLM 首次具备。
- 新版 vLLM [NCCL packed transfer](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/vllm/distributed/weight_transfer/nccl_engine.py:193) 已有多 tensor 打包；旧训练文档中的逐 tensor RPC 问题不能直接用来为新版重复立项。
- 原生 APC、同批完整前缀共享、full-input admission、水位控制、通用 ngram/suffix 投机已在 [上轮源码核查](/Users/xiaoyuzheng/面试/面试项目/codeDisc/vllm-rollout/experiments/rollout_audit/README.md:62) 确认。缺少真实接受率的投机设想不能从“输出短”自动推出。

## 旧运行时成绩能提供的边界

[Qwen3 runtime 公开结果](/Users/xiaoyuzheng/面试/面试项目/personnalinfra/qwen3-runtime/publication/RESULTS.md:175) 保留一轮历史 GRPO：generate 513.35→408.68s，整步 1844.14→1732.44s，但 session arm 同时改了 sleep 权重处理，不能归因于单个功能。另一次 serial/batched driver 为 380.67→289.34s，来自不同 sampled trajectories，不可把两项加速相乘。

该历史卡片里的 weight sync 仅 2.23s：即使整个删去，相对 1844.14s 的整步也只有约 0.12% 的直接时间量级。它不能代表现在异步训练，但足以反对“权重同步一定是最大可挖收益”的无证据判断。旧 decode/prefill 占比是拟合估计，公开结果明确没有把它当 profiler 实测；不能拿 token 占比充当当前可减少的墙钟上限。

同文 [四臂 KV 对照](/Users/xiaoyuzheng/面试/面试项目/personnalinfra/qwen3-runtime/publication/RESULTS.md:283) 中，APC-only 和 Session+APC 的回放接近；且原始记录有 dirty、forced-length 未通过等限制。它与新会话实验共同说明：**重新实现 KV 留存不天然优于已有 APC，需要独立证明目前真正丢掉了什么。**

最终证据等级：会话调度已有受限的真实回放收益，其 4B 必要性仍待目标容量下的对照；HTTP prompt 数据重复有源码证据但无性能结论，且代理层可替代 vLLM 二开；其余所列工作或已实现、或原生已有、或没有当前负载的占比证据。本次不新增收益声明，不运行 GPU 实验。

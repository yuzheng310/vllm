# 实际 rollout 链路探测

本轮日期：2026-09-14。只做源码核查及已存在公开轨迹的离线元数据统计。
没有运行 GPU 推理、启动训练、改动运行时代码或生成新的收益指标。

后续用户要求将资源优先用于已有未合并工作的筛选。最新选择及一手证据见
[未合并工作筛选](UNMERGED_WORK_SELECTION.md)：会话续跑优先调度进入实现
审查首选。下文保留此前源码探测结论，不再以全面耗时诊断作为默认下一步。

## 已确认的实际路径

RepoCompass 的 `CodeSearchGenerator.generate` 并发启动 `code_search_loop`，
后者调用 Ray 的 `init_and_run`，创建工作目录并运行 OpenHands Conversation。
Conversation 通过 LLM/LiteLLM 发起生成，执行工具后继续下一轮。
生成参数携带 `return_token_ids=True` 和每轨迹 session_id；结束后压缩
重复 prompt IDs，再交给训练侧组织 loss mask / reward 等数据。

源码依据：`codeDisc/repocompass/src/generator/code_search_generator.py`。
当前统计只把工作目录创建、Agent 初始化、整个 conversation 分开；
conversation 内没有在这里进一步区分排队、GPU、前后处理和工具执行。
权重发布另有 `weight_publish_seconds` / `pause_drain_seconds` 计时点，
但本轮没有找到足以分析其占比的当前训练数值记录。

`pyproject.toml` 固定 vLLM 0.11.0，而 fork 基线是 0.29.0。新版提供某功能
不表示旧的 SkyRL 集成已经使用，也不表示升级的兼容成本为零。

## 原始数据复核

使用已有公开 CodeScout-4B evaluation parquet，SHA-256：
`14130218f8789f6214ade31182a5b8f4372b528f2e8112e601afeaf23428aa2f`。
494 个不同任务、2395 次请求，所有请求均能通过 response_id 找到延迟记录。
这是公开评估数据，不是 RepoCompass 当前训练日志，也不是同题八采样记录。

| 可观察内容 | 结果 |
| --- | --- |
| 每轨迹轮数中位数 | 5 |
| 每次输入 token 中位数 / P95 | 9981 / 27807 |
| 每次输出 token 中位数 / P95 | 87 / 170 |
| 记录的请求延迟中位数 / P95 | 6.877 / 16.834 秒 |
| 输入超过 16384 token 的请求 | 636 / 2395 |
| 输入超过当前训练配置 40960 上限的请求 | 17 / 2395 |
| cache_read_tokens | 2395 条均记录为 0 |

字段与结果见 [trace_observations.json](trace_observations.json)。长度来自
provider usage；冻结 tokenizer 的重建长度存在已记录的差异，不能混用。

这些数值支持“长输入、短动作”的负载判断，但不支持下列推断：

- 输入 token 多不等于 prefill 占主要墙钟时间，decode 每个 token 会逐步执行。
- cache_read_tokens 为 0 不等于引擎 APC 全部 miss。缺少引擎开关和
  vLLM→代理→LiteLLM 的字段映射证据；不得用它宣称 70% GPU 重算浪费。
- 请求延迟合计不是并发任务完成时间；数据没有到达时间线、工具持续时间或
  引擎 queue/prefill/decode 分解。
- 当前 max_len=16384 的评分实验不能直接承载完整公开轨迹。超过上限的
  请求不能静默截断、删除后冒充完整任务结果。

## 候选逐项审查

以下上游能力以原版 v0.29.0 固定快照
`98dff2a81d747d1dba01a47f939f48c3526d4206` 为依据。

| 候选 | 查到的依据 | 当前决定 |
| --- | --- | --- |
| 显存感知准入 | 原生已有 full-input admission 与 watermark | 维持上一轮不立项 |
| 同题多采样共享首次 prefill | 原生在分配阶段缓存完整块，支持同批后续请求复用 | 不把八采样假设成八次全量 prefill，不重复开发 |
| 权重更新时等待短工具调用完成，避免 abort/replay | RepoCompass 已有 request fence；新版 vLLM 还有 pause 的 wait/keep/abort | 先核对实际同步耗时，不能当新功能 |
| 多权重 tensor 合并传输 | 原生 NCCL 已有 packed broadcasting、轮转缓冲和流重叠 | 不开发通用打包传输；单卡也不能代表跨卡通信收益 |
| 通用 ngram/suffix 投机 | 原生已有 ngram、ngram_gpu、suffix 方法 | 使用现成功能属于基线调优；没有本任务接受率证据 |
| 每轮完整 token IDs 返回及前后处理 | 当前确实请求完整 prompt IDs，Ray 侧压缩发生在 HTTP 返回之后 | 是可达开销，尚无耗时占比，暂不立项 |

同批共享的关键链路是 `KVCacheManager.allocate_slots` 调用
`coordinator.cache_blocks`，之后的请求可以查到完整块；官方设计文档
`docs/design/prefix_caching.md` 的 Block Allocation 明确描述 same-batch reuse。
这不意味着部分尾块永不重复、跨引擎可自动共享、所有架构路径无条件相同。
`block_pool.py` 的“不去重”注释描述另一种已有块的重复情况，不能据此否定
新请求可以复用同批完整前缀。

## 不需要二开就能补齐的下一层观测

新版 `OpenAIServingChat` 已提供 `enable_per_request_metrics`，
`build_per_request_timing_metrics` 返回 queue、TTFT、generation interval 等。
`enable_prompt_tokens_details` 控制缓存统计输出。它们应作为诊断基线的能力，
不能再开发一套后称为简历成果。

- streaming 的 per-request metrics 随最终 usage chunk 返回，必须请求 usage；
  `n>1` 时被抑制，因此组采样需用引擎层统计或独立 n=1 请求采集，不能把
  改变请求形态后的性能无条件等同原链路。
- queue、TTFT、decode interval 是运行时时间戳口径，不是纯 GPU kernel 时间。
  如需判断 CUDA 优化空间，必须另用 profile，不能相减后冒称 GPU 净耗时。
- 工具执行、HTTP/模板/分词、权重发布与阻塞需要各自时间点，保持
  request_id / trajectory_id / policy version 的关联；不可把它们全部加和
  当作异步系统关键路径。

这轮结果仍是“尚无通过必要性审查的新功能”。下一步合理的资源投入是
在原生引擎上形成有版本、真实权重、实际请求配置的基线观测，先回答时间花
在哪里。不能从这份公共统计预先许诺具体加速比，不能默认 0.6B 或 dummy
4B 的性能就代表真实 4B 训练模型。

## 复核命令

从工作区根目录执行，输出路径必须不存在：

```bash
uv run --no-project --with pyarrow --python 3.13 python \
  codeDisc/vllm-rollout/experiments/rollout_audit/inspect_public_trace.py \
  --parquet personnalinfra/qwen3-runtime/workloads/code_localization/raw/swe_bench_verified.parquet \
  --output /tmp/rollout-trace-observations-new.json
```

程序只导出字段名、分布和源文件哈希，不导出问题文本、工具输出或凭据。

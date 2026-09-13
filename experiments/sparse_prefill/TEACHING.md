# 学习与复现路线

## 先回答这个项目解决什么问题

给定已经生成的多轮 Agent 轨迹，reference/teacher scorer 可能只需要动作
Token 的条件概率。工具返回仍属于因果上下文，但无需为这些位置返回概率。
本项目将这个需求实现为引擎能力；尚未接入 RepoCompass 的训练更新链路。

从 [DESIGN.md](DESIGN.md) 的公式开始，自己推导目标位置 j 为什么对应
hidden-state 行 j−1。然后用两个分块手算跨块目标，最后再读代码。

## 四处代码分别负责什么

| 文件 | 职责 | 阅读时应能解释的问题 |
| --- | --- | --- |
| `vllm/sampling_params.py` | 原生参数与序列化契约 | 为什么位置必须递增、唯一且不能为 0？ |
| `vllm/v1/engine/input_processor.py` | 输入长度与运行配置校验 | 为什么不支持的 runner 必须拒绝而不能忽略参数？ |
| `vllm/v1/worker/gpu/sample/prompt_logprob.py` | packed row 映射、LM head 前选行、跨块累积 | 空动作块、批次重排、抢占重算分别如何处理？ |
| `vllm/v1/engine/logprobs.py` | 恢复输入位置与原输出语义 | 为什么稀疏分数仍保留空槽？UTF-8 解码需要谁的上下文？ |

“Python 实现”不意味着只是外层包装：是否在实际 GPU 投影之前改变工作量，
以及是否正确维护引擎状态，才是区别。这里复用了已有概率 CUDA 内核，
没有编写新 kernel，也不应在简历中写成 CUDA kernel 优化。

## 实验命令

以下命令在专用远程工作目录执行。`src/vllm` 是源码，`.venv` 是独立环境，
`experiments/sparse_prefill` 是与提交版本一致的实验脚本副本。
启动器把模型、编译和包缓存限制在该目录内；不需要改系统驱动。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
./run-in-workspace.sh .venv/bin/python -B -m pytest \
  src/vllm/tests/test_sampling_params.py \
  src/vllm/tests/v1/engine/test_logprobs_processor.py -q
./run-in-workspace.sh .venv/bin/python -B -m pytest \
  src/vllm/tests/v1/sample/test_logprobs.py -q -k selected_prompt
VLLM_TEST_QWEN3_MODEL="$PWD/models/Qwen3-0.6B" \
  ./run-in-workspace.sh .venv/bin/python -B -m pytest \
  src/vllm/tests/v1/sample/test_logprobs.py -q -s \
  -k test_prompt_logprobs_with_chunking_and_preemption
./run-in-workspace.sh .venv/bin/python -B \
  experiments/sparse_prefill/benchmark.py \
  --model models/Qwen3-0.6B --output reports/my-real-run.json
```

结果文件拒绝覆盖。正式比较固定 BF16、FA2、512 Token 分块、batch=1、
关闭 APC、默认编译与 CUDA Graph、FlatLogprobs、prompt_logprobs=0，
每次请求包含必要的一步生成。预热、行数诊断和正式计时相互分离。

0.6B 真实/随机权重校准使用相同参数，分别添加 `--load-format auto`
和 `--load-format dummy`，并固定 `--lengths 8192 --max-model-len 8193`
与 `--modes full actions`。4B 随机权重测试必须标注为结构成本代理。

## 为什么保留失败和负结果

- 驱动 API 能初始化，不等于 PyTorch 和模型执行成功。环境验证分层推进。
- 依赖已安装不等于子进程能找到命令；ninja 问题实际是 PATH。
- 观测函数不应为了方便而放开 RPC pickle 限制；使用具名 worker 扩展。
- Top-K 并列值的顺序与概率正确性是不同契约；独立全词表参照更可靠。
- 连续后缀少算了一部分行，也可能因其他开销而更慢。
- 逐轮前缀和整条轨迹的数值差异应增加定位对照，不能直接归罪稀疏实现，
  也不能直接放宽误差阈值。

具体证据与处理见 [WORK_LOG.md](WORK_LOG.md)。原始结果不删除，
同一次运行内的 bootstrap 区间也不代表跨机器、跨时段的稳定性保证。

## 面试中必须能守住的边界

能说明原生参数怎样进入 worker、位置偏移如何推导、为何保留全词表
归一化，以及为何投影行减少约 97% 只换来有限的完整请求收益。
能解释输出仍有 O(L) 占位，以及多卡、LoRA、推测解码等为何暂不支持。

不要声称完成了后训练收敛实验、真实工具轨迹评测、线上部署或上游合并。
项目由 AI 辅助开发；面试前应亲自运行最小测试并逐行复核核心改动。

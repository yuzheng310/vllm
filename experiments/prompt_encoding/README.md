# RepoCompass：vLLM 多轮消息分词复用

2026-09-14 最终验收：**保留为 RepoCompass 内的一条 vLLM 前端优化工作**。
在已有 fastokens 优化的强基线上，真实 HTTP 渲染回放的分词阶段累计耗时
再下降 **37.76%**，HTTP 渲染累计耗时下降 **12.03%**。达到开发前固定的
局部分词 30% 门槛。不是训练吞吐或模型质量收益。

后续单卡端到端实验已固定[协议](E2E_PROTOCOL.md)，当前连接与执行状态见
[E2E_STATUS.md](E2E_STATUS.md)。尚未取得新的 GPU 端到端结果，不能将下表
的 HTTP 渲染指标直接当作完整生成或训练指标。

## 场景和改动，用一句话说清楚

代码定位 Agent 每查一次仓库，就把“旧对话 + 刚查到的文件”一起发给模型。
vLLM 收到文字后要重新转成 token。我们的改动让**已经转过的完整历史消息
直接复用 token IDs，只对新内容分词**，然后仍返回完整、原样的输入序列。

例如：第一轮把任务说明转成 token；第二轮查到文件后，复用任务说明的
分词结果，只处理新增文件内容；第三轮再复用前两轮未改变的消息。
模型仍然读取全部历史，GPU 的 attention、KV cache 和生成过程没有改动。

真实消费者是 RepoCompass 已有的 OpenHands → SkyRL → vLLM ChatCompletion
rollout 链路。项目 4B 训练脚本本来就启用了自动工具调用和 Hermes parser。
此处使用同源公共代码定位轨迹的原始 messages/tools 回放验证，无须重跑
多卡 RL。当前 RepoCompass 训练依赖仍固定 vLLM 0.11.0；本次是在 v0.29.0
源码上验证扩展，**没有宣称已经迁移训练依赖或接入了完整训练循环**。

## 最终实测

固定 48 条轨迹、233 个请求、2,661,314 个输入 token；最长请求 33,551 token。
所有请求完整保留。客户端在同机 loopback 顺序发送，按 8 个活跃轨迹轮询；
这是确定的交错顺序，不是假装重放真实工具等待或 8 路负载吞吐。

每个后端按关闭/开启/开启/关闭重启四轮，每轮从空缓存开始。
下表为两次同组累计耗时的中位数；第一轮请求计入，初始化不计入。

| 原生后端 | 指标 | 原版 | 加缓存 | 下降 |
| --- | --- | ---: | ---: | ---: |
| HF tokenizers | 实际 renderer 分词累计耗时 | 3,846.25 ms | 1,052.40 ms | 72.64% |
| HF tokenizers | HTTP 渲染累计耗时 | 4,690.37 ms | 1,779.39 ms | 62.06% |
| **fastokens 0.3.1** | **实际 renderer 分词累计耗时** | **292.72 ms** | **182.18 ms** | **37.76%** |
| **fastokens 0.3.1** | **HTTP 渲染累计耗时** | **1,005.23 ms** | **884.30 ms** | **12.03%** |
| fastokens 0.3.1 | 分词窗口内进程 CPU 时间累计 | 1,277.72 ms | 1,079.53 ms | 15.51% |

强基线的两个配对中，分词下降 **39.73% / 35.84%**，HTTP 下降
**12.01% / 12.05%**。两次重复证明本配置下结果接近，不代表跨机器、
所有负载或生产流量上的稳定性保证。CPU 时间包含后端线程，不能把
37.76% 的墙钟下降写成等幅 CPU 资源节省。

原生 fastokens 比 HF + 缓存更快，因此部署建议优先采用 fastokens，再
叠加本扩展。不能只展示 HF 的大百分比，隐去更强的原生替代方案。
绝对收益也应保留：强基线每 233 请求少约 **121 ms** 的 HTTP 渲染时间，
平均约 **0.52 ms/请求**；没有证明完整 RL 总耗时有显著下降。

原始逐请求数据见 [service_results](service_results/)，严格匹配源码指纹、
输入文本哈希和客户端时间窗口后的结果见 [service_analysis.json](service_analysis.json)。
计算脚本为 [analyze_service.py](analyze_service.py)。每个 HTTP 主请求都
一一匹配实际 `_tokenize_prompt` 调用，没有用独立 helper 时间冒充服务结果。
服务原始日志以 `.log.gz` 无损保存，避免格式化工具改写原始输出。
[artifact_manifest.json](artifact_manifest.json) 记录文件及解压后日志 SHA。

## 输入一致性与限制

- 最终 8 轮主回放均 **233/233** 个请求逐 token 匹配对应的原生 API 输入；
  每轮额外 8 个并发请求、左右截断也通过。这里只证明输入一致，不声称
  跑过生成结果、定位准确率、reward 或完整训练质量评测。
- HF 的 offsets 返回检查通过；fastokens 原生不支持字符 offsets，关闭和
  开启缓存均返回相同 501。需要 offsets 的调用方应使用支持它的后端。
- 基本回归 3 项、四入口 salt 回归 4 项均通过；最终既有 renderer 文件
  全部 **32 项通过**。这些是验证证据，不作为简历收益。
- 16 MiB / 2048 条目预算，4096 字符起用。含额外隔离检查的最终缓存账面
  占用约 **6.0–6.3 MiB**，没有淘汰；这是含对象开销的预算统计，**不是 RSS**。
  小容量测试另外验证了淘汰与超大单项不入缓存。
- 真正完全冷的第一个 fastokens 请求原版 2.80/2.46 ms，缓存版 2.53/2.94 ms；
  冷启动不保证改善，个别配对有回退。含 48 个任务首轮的总时长已算入主指标。
- 256 字符的非目标回退控制中，HF 差值约 -0.2 μs，视为噪声；fastokens
  平均增加约 1.2 μs。该人为截取的短输入仅检查额外成本，不参与收益计算。

## 为什么需要在 vLLM 内修改

固定源码的 `BaseRenderer._tokenize_prompt` 调用 `tokenizer(...)`，每轮完整
处理渲染后的文本。GPU prefix caching 发生在这一步之后，不能省掉分词。
fastokens 降低每次分词的成本，但仍处理重复历史，因此两种优化可以叠加。

这次实际修改了三个生产文件：

- [chatml_encoding_cache.py](../../vllm/tokenizers/chatml_encoding_cache.py)：
  按注册的 `<|im_end|>` 特殊边界拆分，批量分词缺失段，缓存紧凑 IDs，
  保留分隔符重建完整序列；LRU 同时受条目数和账面存储预算约束。
- [base.py](../../vllm/renderers/base.py)：接入真实 `__call__` 路径；在同步/
  异步 Chat 和 Completion 的分词前传递 cache_salt，保留原生前后校验。
  原生 extras 原本在分词后注入，只在缓存 key 中写 salt 并不足以实现隔离。
- [envs.py](../../vllm/envs.py)：新增 `VLLM_CHATML_ENCODING_CACHE_MB`，默认 0，
  正值显式启用。只对符合条件的 decoder-only renderer 生效。

不切任意字符窗口。只接受已验证的 BPE/ByteLevel、NFC/无 normalizer、
无自动 BOS/EOS 注入、无 dropout、无空白吞并的 non-normalized marker。
拒绝跨边界竞争的 added token；返回 ID 列表独立，不把可变缓存暴露给调用方。
词表/特殊标记属性、归一化或后处理状态变化会失效回退；不支持任意原地
替换同大小 BPE 模型，tokenizer 必须在 renderer 生命周期内保持逻辑不可变。

模板内容进入完整文本 key。需要 offsets、短输入或不支持参数时走原路径。
fastokens 的私有兼容状态依赖已核对的 0.3.1 接口，更换版本需重新核查。
启用示例为 `VLLM_USE_FASTOKENS=1 VLLM_CHATML_ENCODING_CACHE_MB=16`，
这两个环境变量要传给运行该 fork 的 API/renderer 进程。

## 两个容易被忽略的验证问题

**直接 tokenizer 输入与 HTTP 输入不是同一个冻结文件。** 原生 vLLM 的
工具 schema 经 `model_dump()` 后字段顺序变化，导致直接 HF 参考的 233 个
序列全部与 HTTP 参考不同。首次关闭缓存的 HTTP 请求就触发了断言，比较
立即停止。后续只按原生序列化重建服务期望值，原始 messages/tools 请求体
不变，并断言工具字典值相同；先完整通过原版 A1，再运行候选。
[canonicalization.json](service_results/canonicalization.json) 保存这一差异和两份指纹。
不能把两套输入的“精确一致”混在一起，也没有通过改候选期望值掩盖不一致。

**第一版确实没有收益。** HF 第一版约 2.02s→2.08s，因为词表大小校验在
旧 tokenizers 中触发了 O(词表大小) 操作。一次小范围 profile 定位后，改为
基础词表大小 + 独立 added-token 属性检查，保留失效保护。失败数据在
[cpu_hf_macos_initial.json](cpu_hf_macos_initial.json)，定位记录在
[hf_initial_profile.txt](hf_initial_profile.txt)。没有反复调缓存参数挑好结果。

## 环境与复现

服务 CPU 为 Intel Core i9-14900KF；Transformers 5.17.0 / tokenizers 0.23.2，
fastokens 0.3.1。生产源码来自干净 v0.29.0 的
`98dff2a81d747d1dba01a47f939f48c3526d4206` 加本次三文件补丁，未带之前的
session scheduler 实验。依赖复用既有预编译 wheel，包元数据为
`0.29.1.dev2+g5f7b949fa.precompiled`；源码与 wheel 标签分别记录，不能混称。
`CUDA_VISIBLE_DEVICES` 为空，使用 `vllm launch render`，没有加载模型权重
或启动 GPU 推理 worker。四个 Rayon 线程、八个 renderer worker 各组相同。

开发前协议：[PROTOCOL.md](PROTOCOL.md)；在各阶段计时前固定的细节及修订：
[IMPLEMENTATION.md](IMPLEMENTATION.md)。从消息准备、helper 对照、原生 API
规范化，到 HTTP 回放、观测、分析均有可执行脚本，逐轮原始数据保留。
源文件指纹与最终本地代码三份全部匹配；检查入口在
[run_service_matrix.py](run_service_matrix.py)。

本版原生无 GPU 服务关闭流程会挂起，脚本在回放结束后先发信号保存计时，
再终止自建进程，必要时停止该 PID；最终退出 -9 明确保留。清理不在计时内，
所有组相同，不能声称这些运行优雅退出。现有公共 renderer 回归也未掩盖此问题。

## 面试与简历口径

推荐作为 RepoCompass 下的一条：

> 针对多轮代码定位 rollout 的历史消息重复分词，扩展 vLLM 前端消息段缓存与
> 请求隔离；在 48 条真实轨迹、233 个请求的 HTTP 回放中，相对已启用
> fastokens 的基线，分词阶段耗时降低 37.8%、HTTP 渲染耗时降低 12.0%，
> 输入 token 序列与原生服务完全一致。

面试时补充：这是公共代码定位轨迹、单机 CPU、固定交错顺序的回放结果，
不是完整训练指标；当前训练依赖升级尚未完成。适合证明一项具体的 vLLM
二次开发能力，不包装成独立推理引擎项目或原创通用 tokenizer 算法。

思路参考已核对的 [PR 45514](https://github.com/vllm-project/vllm/pull/45514)
和 [PR 47583](https://github.com/vllm-project/vllm/pull/47583)：借鉴特殊边界
复用与 renderer 接入，适配实际 `__call__`、有界缓存、严格支持范围和
完整请求隔离；未复制作者性能数据，也未新开重复上游 PR。来源快照在 sources/。

全部失败、配置问题、决策与操作失误见 [WORK_LOG](../sparse_prefill/WORK_LOG.md)。
其中一次远端 Git hook 误用了专用目录外的默认 pre-commit 缓存，已停止并
向用户告知，未继续操作该目录；不能声称整个过程毫无目录边界失误。

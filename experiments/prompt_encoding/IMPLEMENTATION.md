# 实现与验证边界

2026-09-14，在性能对照前固定。

- 只接 BaseRenderer._tokenize_prompt；原生模板、预校验、后校验和 offsets
  路径保留。通过 VLLM_CHATML_ENCODING_CACHE_MB 显式启用，默认 0。
- 每 renderer 一个缓存；对象绑定 tokenizer，按 cache_salt 和完整消息段
  文本做键，不使用仅哈希命中。词表大小、normalizer/pretokenizer/postprocessor
  状态或 backend 实例变化时清空并失效。任意原地替换等不受控 tokenizer
  变更不在支持范围，和原生 cached tokenizer 一样要求重建 renderer。
- 仅 BPE、无 dropout、ByteLevel 解码、NFC/无 normalizer、无自动 BOS/EOS
  注入，且 im_end 为无空白吞并的 non-normalized 特殊 token。拒绝其他
  added token 包含 im_end 的歧义。模板变化按实际文本重新命中，不拼旧模板。
- 使用 tokenizer 公开批处理接口处理缺失段，允许原生 tokenizer 线程池
  管理 backend；缓存锁只覆盖查询/更新，编码在锁外。返回独立完整列表。
- 性能首轮固定 16 MiB、2048 条目、4096 字符起用；本工作负载中的长历史
  可在普通 CPU 内存中容纳。不因结果调整预算、并发或最短输入门槛。
  统计 accounted_bytes 是含保守对象开销的缓存预算，不冒称 RSS。
- 测试目标：保证输出 token 契约，避免边界归一化、截断、跨 salt、淘汰、
  返回列表被修改与并发访问造成训练输入变化。最低层对真实 tokenizer 做
  差分；通过后再检查真实 renderer，不以只测 helper 代替在线接入。
- 完整模型推理尚未开始。CPU 对照在相同包版本、线程设置、固定输入下
  比较原生 HF、无缓存分段、缓存分段；原生 fastokens 单独进程核对。
  先固定全部请求的轮次交错顺序，再执行 A/B/B/A；冷缓存包含在总时间内。

计时序列固定为 native/cache/segmented/segmented/cache/native，分段消融
插在 A/B/B/A 中间。每轮重新加载 tokenizer、清空缓存；仅用无关短句暖机，
不预热评测请求。8 条轨迹按冻结源顺序轮询并补位，不冒充真实工具耗时。
全部组使用 TOKENIZERS_PARALLELISM=true、RAYON_NUM_THREADS=4。
初始化、模板渲染和一致性比较在计时之外；分段、查缓存、重拼、回退在内。
每请求记录墙钟及进程 CPU 时间。缓存键允许无 salt 的公共请求共享相同段，
与此前每轨迹独立字典的结构审计不同，不能直接用 69% 推算本轮命中率。

fastokens 0.3.1 shim 的 added-token getter 和 to_str 都会解析整个词表，
不能放在热路径。其词表变更 API 为 no-op，本实现检查保留的不可变 JSON、
词表大小、postprocessor 和特殊 token 开关；依赖该已核对的兼容接口，
接口不可用时不启用缓存。原生 HF 检查 added-token 属性，覆盖同大小的标记
属性修改；任何同大小 BPE 模型的任意原地改写仍要求重建 renderer。

## 真实服务接入协议（CPU 门槛通过后、HTTP 计时之前）

选择 vLLM 原生 `launch render` 和 `/v1/chat/completions/render`，它仅需
模型配置/tokenizer，不加载权重或启动推理 worker。远端源码从干净的
v0.29.0 新建工作树，仅应用本次三文件补丁，避免夹带此前 scheduler 实验。
比较相同源码下开关关闭/开启，按 native/cache/cache/native 固定顺序，
每轮重启服务清空缓存。原生 HF 和 fastokens 各做这一组，不做参数扫描。

使用同一冻结的 233 份原始 messages/tools，客户端运行在同机回环网络；
按 CPU 协议的固定轨迹交错顺序顺序发送。主指标是实际 renderer 分词阶段
总耗时；HTTP 请求总耗时单列，包含模板、校验、序列化、HTTP 框架等成本。
包装原方法只计时并记录，不能替换实际渲染逻辑；输出在服务退出时写入。
计时主回放结束后再检查 offsets、左右截断、8 请求并发的精确输入一致性。
这一步不验证模型生成/训练 reward，不把渲染 HTTP 性能当作 RL 吞吐。

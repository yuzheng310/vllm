# 单卡端到端实验执行状态

## Material Passport

- 状态：**协议和脚本已准备；启动遇到断连，尚未取得 GPU 端到端结果**。
- 日期：2026-09-14，Asia/Shanghai。
- 事前协议：[E2E_PROTOCOL.md](E2E_PROTOCOL.md)，先提交于 `a7f9f00`。
- 实施提交：`ad740e0`，已推送个人 fork；生产三文件未更改。
- 所有远端实验文件限定于 `/home/aa205/vllm-sparse-prefill.JbyyBUjL`。

## 连接和启动记录

最初 SSH 控制连接未返回输出，独立重连与 Tailscale 探测超时。本机 VPN
Running、无健康警告，但目标 Online=false。关闭已核实的本地控制进程后，
旧命令返回了延迟的 GPU/内存输出；由于没有当前时间戳，不用它直接启动。

08:59:20 UTC 全新 SSH 返回当前时间，并确认 GPU 无计算进程。随后脚本
成功传到专用源码树，远端对 native API 冻结输入的 CPU 契约检查通过。
传输包中六个文件的 SHA 与本地提交一致。

首次提交运行时，控制连接断开，SSH 退出 255。通过 Tailscale 已确认属于
同一主机的 IPv6 地址、沿用同一 SSH 主机密钥，在 **09:04:56 UTC** 只读确认
`e2e-run-1/run.json` 和服务日志都不存在，GPU 计算进程为空。因此此前运行
没有创建实验输出，也没有观测到启动模型。

随后通过 IPv6 提交 nohup 运行，日志指定在专用目录，控制程序自身有超时
和资源检查。连接在密码提交后再次超时，未收到 controller PID；新的监控
连接也超时。**这次提交是否执行尚未确认，不能声称 GPU 实验已经运行或未运行。**
恢复连接后先检查目录、日志和进程，不重复提交、不覆盖已有日志。

最初的连接状态与第一次恢复证据见 [e2e_connectivity.json](e2e_connectivity.json)。
没有修改 VPN、网络路由、系统配置或共享软件环境。

## 已完成的检查与尚未验证的部分

本地和远端均已验证 48 轨迹、233 请求、22,046 个生成 token 的输入契约；
模拟 SSE 检查确认缺少结束标记或少生成 token 会失败，不会将不完整响应
计为提速。不同流式分块的相同文本按同一输出比较。五份脚本通过 Python
AST 解析和仓库 hooks。远端检查使用已有 native API canonical 输入。

这些是客户端和数据契约检查，**没有验证原生 GPU 服务集成或 CUDA 推理**。
本轮没有可报告的端到端提升百分比。此前每请求 0.52 ms 节省仍只是 CPU
HTTP 渲染结果，不能填入本轮 GPU HTTP 结果，更不能称训练收益。

## 恢复入口

- [serve_e2e.py](serve_e2e.py)：原生 GPU Chat Completions 服务和仅实验内的
  loopback 开关；GPU APC reset 成功后重建或关闭 CPU 缓存。
- [bench_e2e.py](bench_e2e.py)：完整请求及输出长度、4 并发轨迹、固定四轮，
  保存延迟、输入 token 摘要和输出文本；每轮结束检查全部 233 个输入。
- [run_e2e.py](run_e2e.py)：目录/资源预检、权重 SHA、单常驻模型、5 秒资源
  采样与有界清理。新目录必须不存在，禁止覆盖。
- [analyze_e2e.py](analyze_e2e.py)：固定全部四轮的配对、组内波动和输出比较。
- [check_e2e_contract.py](check_e2e_contract.py)：不完整响应与少生成的拒绝检查。

恢复后优先只读检查：

```text
experiments/prompt-encoding/e2e-controller-1.log
experiments/prompt-encoding/e2e-run-1/run.json
experiments/prompt-encoding/e2e-run-1/server.log
experiments/prompt-encoding/e2e-run-1/client.log
experiments/prompt-encoding/e2e-run-1/replay.json
```

只有确认不存在活跃的同一实验，才可启动新轮。失败结果保留，使用新输出
目录与新控制日志。在专用根目录执行入口的形式为：

```bash
bash run-in-workspace.sh .venv/bin/python \
  src/vllm-prompt-encoding/experiments/prompt_encoding/run_e2e.py \
  --root /home/aa205/vllm-sparse-prefill.JbyyBUjL \
  --output /home/aa205/vllm-sparse-prefill.JbyyBUjL/experiments/prompt-encoding/e2e-run-2
```

该命令是恢复指引，不表示第二轮已提交。取得数据后核对源码 SHA、实际
KV 池、原生阶段指标、资源曲线、输出长度与全部失败记录，再更新本文件。

# 原始实验资料索引

主结论以 `feature-0.6b-views-main.json` 为准，结构成本补充以
`feature-4b-dummy-views-main.json` 为准。`primary-views-analysis.md`
和 `proxy-calibration-analysis.md` 是 analyze.py 生成的完整统计表。

- `*-trace-*.json`：实际输入 Token IDs、动作区间与位置。
- `calibration-*-views-8k.json`：最终代码真实/随机权重校准。
- `feature-0.6b-main.json`、`calibration-0.6b-real-8k.json`：历史 gather 版本。
- `original-*.json`：源码修改前基线与启动尝试。部分失败文件保持 running，
  不进入统计；对应错误在 logs/ 中。
- `per-turn-geometry-*`：逐轮与整条轨迹的数值差异定位。
- `logs/`：包括失败的模型下载、启动、安装和测试输出。
- 模型清单、CUDA smoke 和包探测：环境与模型来源证据。
- `ARTIFACTS.json`：61 个导出文件的发布副本 SHA256。

所有最终实验的四个引擎源码 SHA256 与提交 60f5943 的代码逐一匹配。
`benchmark-measured.py.txt` 保存实际执行脚本的精确字节，匹配结果中的
benchmark_sha256；与规范化后的 benchmark.py 仅差文件末尾一个空行，
Python AST 完全一致。精确复现可将这个文件复制为 benchmark.py 执行。

发布副本去除了专用路径、主机名、SSH 地址和 ANSI 控制码。
完整原始安装终端记录、驱动探测与 4904 文件扩展清单仍保留在远程专用目录，
没有上传共享机器的其他用户信息。日志清理不修改数值结果。

# 共享机器环境修复存档

这里保存实际远程修复用到的源文件，去掉了主机地址和凭据。
Python 文件仅补充仓库版权头并调整格式，核心操作与远程脚本一致。
目录布局对应实验专用工作区；使用时将这些文件复制到自己的专用目录，
让 `run-in-workspace.sh` 位于该目录根部。先创建 `reports/`、`tmp/` 和 `cache/`。

这些是特定环境的复现材料，条件是 Ubuntu 22.04、x86_64，
已加载 NVIDIA 内核模块 580.173.02，用户态库却为另一个版本。
环境本来正常时无需执行这一修复。

## 操作顺序与理由

1. 在专用目录内准备 uv、托管 Python 和 `.venv`；缓存与临时目录也指向这里。
2. 用 `.venv/bin/python vendor/fetch_driver_libraries.py` 下载三个官方包。
   脚本从 NVIDIA `Packages.gz` 取得 SHA256，核对文件长度和完整哈希。
   此目录中的 `driver-packages.json` 保存本次实际选中的版本和校验值。
3. 使用 `dpkg-deb --extract` 将这三个包逐一解到
   `vendor/driver-580.173.02/`，不执行 `dpkg --install`。
4. 用 `bash run-in-workspace.sh nvidia-smi` 检查管理接口。
5. 用 `bash run-in-workspace.sh .venv/bin/python scripts/cuda_smoke.py`
   检查实际 CUDA 分配、PTX JIT、kernel 启动和拷回校验。

启动脚本根据自身位置确定目录并检查内核版本，然后仅对自己的子进程
覆盖库路径和缓存路径。它没有修改系统库、模块、驱动配置或其他进程。
如果内核版本已变化，脚本拒绝继续，需重新检查匹配关系。

## 原始验证范围

本次 CUDA smoke 校验 1024 个整数，GPU 缓冲仅 4096 字节，结果全部符合
`output[i] = 3*i + 7`，随后释放分配与上下文。
这可以证明实际 CUDA 计算可用，不能证明 vLLM、PyTorch 或模型已经正常运行。
这些后续层次各自需要独立记录结果。

最初 bootstrap 在创建虚拟环境前使用系统 Python 的标准库，
进入 vLLM 开发阶段后 Python 命令统一使用 uv 管理的 `.venv/bin/python`。

所有安装、解包和缓存必须限制在操作者自己的专用目录；
本实验未使用 sudo、系统包安装、重启或终止其他用户任务。

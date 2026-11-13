# 2026-11-13 上游优先移植

本轮比较自维护基线 `0a7d69a` 与官方新范围 `65760bc..106f2ae`。官方新增 16 个提交，包含一个合并提交，最新版本为 v0.1.2。交付包含自维护分支源码，以及 Flash、Qwen27 两个本地目录的引擎与原生启动器。

## 合入内容

| 官方提交 | 移植内容 | 验证 |
|---|---|---|
| [5d3966a](https://github.com/IIIIIllllIIIIIlllll/Qwenox/commit/5d3966ab3f2478f09f892f5f328f61ceb3c2fe8b) | Flash CLI 缺少参数值时返回退出码 2 和明确错误，避免越界读取 argv | 实际引擎的 26 个带值选项 |
| [0acd2ee](https://github.com/IIIIIllllIIIIIlllll/Qwenox/commit/0acd2ee24cab3eee223f1090883b230a0a504ea9) | Linux PLE 的排队与在途请求总数受 SQ 深度约束，短提交后继续提交剩余条目，容忍 EINTR | 提取实际生产方法，模拟延迟完成、短提交、中断及短读回退，原版本可复现超限 |
| [d2c620f](https://github.com/IIIIIllllIIIIIlllll/Qwenox/commit/d2c620f96e2eb154b6cddf6873af2990c5bf6dfe) | 补齐 radix-sort 存储类型的 typename 限定 | Windows HIP 完整编译 |

Linux PLE 验证使用确定性的队列模拟。Linux 内核及其内存 cgroup 下的真实冷读场景仍需单独实测。Windows IOCP 的 21 项现有回归已通过。

## 性能筛选

试移植了 [f8b3a22](https://github.com/IIIIIllllIIIIIlllll/Qwenox/commit/f8b3a223a522aaf64388a45a5e7796fde60256c9) 的 greedy 置信度门控。草稿仍按原长度计算，随后以草稿 top-1 概率缩短验证前缀，首个低置信度 token 仍进入验证。

在本机 gfx1151、ROCm 10.3.0a20261111、Flash HGN、aligned、gamma 2、双槽配置下，以三个固定提示各生成 72 个 token，得到以下解码总耗时。请求顺序执行，并另外检查两路并发。

| 门控 | MTP 解码总耗时 | ngram/MTP 混合解码总耗时 |
|---|---:|---:|
| 关闭 | 4118.9 ms | 4059.9 ms |
| 0.5 | 4117.7 ms | 4078.7 ms |
| 0.9 | 4182.9 ms | 4155.7 ms |

0.9 时 MTP 总耗时增加 1.55%，其中 Python 代码用例吞吐下降约 7.5%。本轮样本未显示采用门控的收益，因此最终提交采用上表三项修复。门控结果属于固定输入、单轮测量，不能推广为全部工作负载的结论。

门控原型在关闭、0.5、0.9 及关闭异步草稿的 0.9 模式下，均通过 14 个 Flash 请求的 token 与返回 logprob 对照。覆盖串行、MTP、ngram/MTP 混合、固定种子采样和双请求并发。GPU 候选构建通过 56 项检查，包括 55 项既有回归和一组新增置信度检查，后者包含 15 种输入。

## 其他新提交

- `2d3eecd` 的优化作用于官方 `k_ht_wmma`。当前分支使用独立的 HGN 解量化与矩阵内核，移植需要连同该内核体系做精度及性能对照。
- `f8b3a22 / 878bbcf / 99c4673` 的联合起草需要一并适配短窗 ngram 表、每槽状态、采样验证和并发长度边界，适合独立候选验证。
- `6c4c783` 的 QDV1 子词表会将选中的 HT 输出头行重新量化为 q4cp。应先验证中文覆盖率、接受率及额外重打包成本。
- `ebbd77d` 的接受率拆分涉及行协议、reqstat v1/v2 迁移、API 和页面。当前分支的嵌入式控制台与官方静态页面需要成套适配。
- 设置面板、TUI、采样方案和发布文档属于后续界面与打包整合工作。

## 复核入口

`tools/engine_cli_test.py --engine <candidate>` 验证实际 CLI。`tools/ple_uring_test.py --cxx <compiler>` 验证生产队列方法，`--source` 可指定旧版源文件进行失败复现。`tools/ple_win_iocp_test.py` 检查 Windows 读路径。`tools/engine_host_config_test.py` 覆盖原生与 shell 启动配置。

最终构建的模型验证与文件摘要见 [验证数据](benchmarks/upstream-priority-20261113.json)。

## 部署构建与兼容性

部署引擎构建使用本机已有的 ROCm 10.0.0 / AMD Clang 23。其 HIP 运行库 SHA-256 与现有两个部署目录一致。MSVC 14.52 的六个新增 constexpr 比较函数与旧 HIP 头冲突，构建时使用私有 `cmath` 副本，屏蔽这六个重复重载，由 HIP wrapper 提供设备端实现。该副本及编译命令保存在验证工作树的 `build/msvc-compat` 和 `build/build-release.command.json`。

现有 Flash 引擎第一次与第二次回放，在同一条固定种子的 MTP 采样请求上出现不同输出。第二次旧引擎回放与部署候选的全部 14 个请求 token、返回 logprob 完全一致。其余 greedy、混合采样与并发请求在这些回放中一致。本轮记录这项旧版本已存在的波动，根因尚未定位。

Qwen27 的部署候选与原部署在 8 个请求上的 token、返回 logprob 全部一致。两个目录均完成引擎和原生启动器替换，并经桌面启动器启动后验证健康接口、模型列表、算术生成和 1–20 数字序列。Flash 在 18861、Qwen27 在 18871 的数字序列请求分别接受 31/34、37/39 个草稿 token。验证后恢复原先的停止状态。

两份配置通过 SHA-256 核对。Flash 的三个性能选项均为 1，Qwen27 使用 MTP gamma 3。Flash 的 270 个保留文件、Qwen27 的 263 个保留文件通过哈希对照，包括 API、运行库和内核数据库。回滚数据位于各部署的 `backups/upstream-priority-20261113T054908Z`，保存原引擎、启动器、配置、部署记录以及本次验证结果。

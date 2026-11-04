# Qwenox 名称对齐与兼容

自维护仓库沿用官方名称 Qwenox。模型、端口、精度及自维护功能不随名称变更。

| 用途 | 当前名称 | 兼容旧名 |
| --- | --- | --- |
| 自有远端 | `jerrydong1988/Qwenox` | GitHub 旧仓库 URL 重定向 |
| Windows 引擎 | `build/qwenox-engine-win.exe` | `gdec-win.exe` |
| Windows API | `build/qwenox-win.exe` | `gdec-api-win.exe` |
| Linux 引擎 / API | `build/qwenox-engine` / `build/qwenox-api` | 启动脚本回退 `gdec` / `gdec-api` |
| GPU 源码入口 | `src/gpu/qwenox.cpp` | `gdec.cpp` 包含新入口 |
| 独立构建输出 | `QWENOX_ENGINE_OUT` / `QWENOX_API_OUT` | `GDEC_ENGINE_OUT` / `GDEC_API_OUT`，新名优先 |

运行时继续使用已验证的 `GDEC_*` 配置键，不机械替换成官方数值配置。缓存格式、
魔数、tokenizer、模型权重及全部数值内核不变，此次不加入官方其他推理路径。

## 本机目录与入口

- 源码：`C:\Users\Jerry\Projects\Qwenox`。
- Flash：`C:\Users\Jerry\llama.cpp\Qwenox`，端口 18860/18861。
- Qwen27：`C:\Users\Jerry\llama.cpp\Qwenox-qwen27`，端口 18870/18871。

三个旧目录保留为 junction，支持原脚本、聊天目录及历史回滚路径。历史实验目录、
日志和备份名称不批量改写。移动主目录后修复 linked-worktree 路径，不重写 Git 历史。

部署中新 exe 名为指向原 `gdec-*.exe` 的相对符号链接；GPU 引擎字节不变。旧备份
脚本恢复原文件名时，新入口随链接读取恢复版本。不要改成两个独立 exe 副本。
新启动器优先新名称，也可回退旧文件名。目录 canonicalization 使旧/新路径共用
一把实例锁，Flash 与 Qwen27 的锁相互独立，控制台和托盘显示 Qwenox。

## 配置与回滚

两份 `service.conf` 原样保留。Flash 原 262K/524288 页池、两路并发、aligned 精度；
Qwen27 原 8K、512 compensated Prefill、FP32 dense KV、MTP gamma 3 和可选 DFlash2
gamma 4 保持原值。正式实例迁移前后保持停止，不启动高内存模型配置。

迁移前的启动器、API、引擎、配置、deployment.json 与使用说明另存到每套部署的
`backups/qwenox-rename-<UTC时间>`，之前所有备份内容及文件时间保持不变。
停止对应实例后，恢复该备份的 `start_win.exe`、`build/gdec-api-win.exe` 和
`deployment.json` 可撤回本轮前端名称变更，新入口和旧目录兼容链接仍可用。

## 验证边界

新启动器和 API 实际编译通过；有效配置/精度环境、新旧目录等价、旧二进制回退、
两个部署实例锁、24 项构建输出、启动器 host/RoPE、Messages 与控制 token 缓存
回归通过。GPU 入口仅迁移文件名，所有 `src/gpu/parts` 不变。

本次不重编或替换 GPU 引擎，不新增真实模型性能或高内存稳定性结论。API 回归使用
隔离协议夹具，目录迁移后只运行原生 `--check`，不启动正式模型服务。

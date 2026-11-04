# 独立构建输出与取消回归

官方参考固定为 Qwenox `b34a6862c9d1dcb08cbbe95ba88dae5d4e11449a`：
[独立构建输出 #44](https://github.com/IIIIIllllIIIIIlllll/Qwenox/pull/44)、
[请求取消隔离与回归工具 #47](https://github.com/IIIIIllllIIIIIlllll/Qwenox/pull/47)。
自维护基准为 `f973566c17124de1758b55f260ee217f78e3d866`。

## 独立输出

沿用本分支 GDEC 命名，增加 `GDEC_ENGINE_OUT`、`GDEC_API_OUT`，分别对应官方
`QWENOX_ENGINE_OUT`、`QWENOX_API_OUT`。未设置或空字符串仍输出原默认文件。
编译成功后才将同路径 `.tmp` 原子替换为目标，编译失败保留现有目标文件。

```bash
GDEC_ENGINE_OUT='build/gdec candidate.exe' bash build_win.sh engine
GDEC_API_OUT='build/gdec api candidate.exe' bash build_win.sh api
```

输出目录须已存在。开关只改变引擎/API 的输出路径；运行库及 kernel db 仍按原逻辑
暂存到 `build/`，建议在该目录使用不同文件名。跨目录运行时需另行配置运行依赖。
编译参数、精度默认值、模型路径、启动器及配置文件均沿用原有逻辑。

## 取消回归工具

`tools/prefill_cancel_test.py` 原样提取生产 `prefill_batch` 方法，以 CPU 主机桩
测试其控制流。C++ 桩兼容自维护的预取、计时及 MTP 接口，也能测试官方新旧源码。
没有重写、删去或替换被测方法中的取消检查。

```text
python tools/prefill_cancel_test.py --cxx <clang++> --output <新结果目录>
python tools/prefill_cancel_test.py --cxx <clang++> --source <旧40_model.inc> --output <新结果目录>
```

`tools/prefill_cancel_live_test.py` 针对一个已经启动的、隔离的双槽 Flash-Next 实例。
合成提示为 32768/32768/1024 token，需要实例有足够上下文；不是 Qwen27 8K 的测试。
工具本身不启动、停止或重新配置任何服务。

```text
python tools/prefill_cancel_live_test.py --endpoint 127.0.0.1:18960 --tokenizer <Flash-Next-tokenizer目录> --output <新结果目录>
```

默认 `--observer progress` 使用自维护引擎现有的 `GEN ... PROGRESS` 消息。
脚本要求按接收时间观察到 A → B → A 的进度报告后才取消 B，检查 A 的输出与串行
参考一致、B 被取消、同连接恢复以及后续 MTP 活动；若没有观察到交接则判失败。
PROGRESS 是按请求发送的进度证据，不等同于全局 GPU 当前状态快照。

`--observer snap` 保留官方原有的全局 SNAP 复现模式，GEN 不附加 PROGRESS。
仅该上游模式使用 `QWENOX_PREFILL_SRPT=0`；自维护引擎未加入此开关或 SNAP 命令。
本次没有移植调度器或修改任何引擎协议。

## 本轮回归与边界

- 实际 `build_engine` / `build_api` 函数配合假编译器：16 项通过，覆盖默认与空变量、
  相对/绝对含空格路径、成功替换和失败保留目标；引擎编译参数透传不变。
- CPU 取消：自维护与官方修复版各 7/7；官方旧 `91b111a` 稳定触发 4 项预期失败，
  包括预取消、错误取消对端、取消丢失和 verifier 继承对端标志。
- 实机客户端的模型无关协议夹具：PROGRESS 与 SNAP 两种模式各通过正常、错误取消
  对端、缺少交接三个正负场景，共 6 项。负场景必须以相应原因失败。
- 既有 PLE 预取、引擎 host、Qwen27 格式主机测试通过。
- Bash 语法、Python 语法、CLI 及 32K/32K/1K 实际 tokenizer 提示生成通过。

复现构建路径与客户端夹具：

```text
python tools/build_output_test.py --bash <Git-Bash> --output <结果.json>
python tools/prefill_cancel_live_harness_test.py --output <新结果目录>
```

这些是构建路径、控制流和客户端验证。本次没有完整重编 GPU 引擎、启动模型或新增
真实推理证据；全部 `src/` 与自维护基准一致，Qwen27、并发取消和精度实现保持原样。
既有正式部署和所有回滚备份不参与本轮构建、启动或替换。

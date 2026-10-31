# 官方控制 token 原文与 API 定向融合

基线为自维护分支 `codex/qwen38-27b` 的
`e122649c143158aed0e205a1e35de51bc8bd2b1c`，独立分支为
`codex/api-literal-20261031`。来源固定为官方 Qwenox main
`91b111aeb4bf85723433a821f200398ca8d91cc2`（0.1.0）。这是选择性回移，
没有将自维护分支整体升级成官方 0.1.0。2026-10-31 完成验证，未替换正式部署。

## 变更与保留范围

- 渲染器标记客户端文本的原文区间，tokenizer 按区间区分用户输入的控制 token
  拼写与模板实际插入的控制 token。工具名称、参数、结果、系统/用户正文与历史助手正文
  都经过同一路径；真实图像占位符单独处理，避免与输入的同名字面串混淆。
- transcript token cache 比较实际 added-token 布局，避免相同字节、不同控制语义的前缀
  被错误复用。持久化版本由 1 升为 2，旧文件忽略后重建；不改变 GPU KV checkpoint
  格式。将来部署候选 API 时应预期一次 token cache 冷重建，本轮没有读取或写入正式缓存。
- 新增 `POST /v1/messages`、`POST /v1/messages/count_tokens`、
  `POST /v1/tokenize`、`POST /v1/detokenize`。Messages 使用现有生成、并发、取消、
  容量协商与严格 XML 工具校验；提供 Anthropic 内容块、具名 SSE、thinking、tool_use、
  tool_result、stop sequence、usage 和错误封装。count_tokens 不生成文本。
- tokenize 接受 text/prompt，支持 UTF-8 **字节** offsets；detokenize 接受 tokens/ids，
  支持 skip_special_tokens。这两个原始 tokenizer 工具端点保留 added-token 原始语义，
  与经过聊天模板保护的消息正文有意区分。
- 保留原模板字节、默认 xhigh、XML 工具、严格错误、Responses、WebP、32 图上限、
  晚启动恢复、Qwen27 文本模式及既有精度配置。未改变 GPU/Qwen27 源码、启动器、
  http/toolparse、Windows profiles；未引入官方的新调度、量化、推测后端或运行库。
- 为保留模板策略，未回移官方的默认 medium、自动关闭工具思考、JSON 模板模式和
  工具载荷截断等策略。Messages 对 preserve_reasoning、auto_disable_thinking_with_tools、
  max_tool_arg_chars、max_tool_response_chars、think_guard_test_cut 显式报 400，
  不静默忽略。标准 Messages tool_use 的 JSON 参数仍受支持。

## 构建与确定性回归

使用现有 AMD clang 23.0.0 / Windows SDK 和静态 WebP 1.6.0 构建 API；没有安装或更换
运行库，没有修改官方数值路径的兼容补丁。模型引擎源码没有变化，因此复用 SHA256
`50dd5e4c20ece57c68b776985397bbb2f636e9c804018ae2c15557d9e4e9b315`
的 e122649 源码构建引擎，配合本轮新编译的 API 进行全新推理。API SHA256 为
`03d7088f2c3c207bd42c1fc6fabcb15d3057ac44bc0cc6a40b1bcdc03134c373`。

| 检查 | 本轮结果 |
| --- | --- |
| 原 API 协议 | 单槽 90/90，双槽 90/90 |
| 严格工具错误 | 458/458 |
| 断连与恢复 | 单槽 49/49，双槽 49/49；取消 ACK 超时恢复套件通过 |
| 容量/晚启动恢复 | 3/3、6/6 |
| WebP/工具图像 | 30/30 |
| 原模板渲染/错误兼容 | 796 项字节/错误对照一致 |
| 控制 token 原文与图像占位符 | 2,902 个有效用例，166 个非法随机输入被拒绝，0 失败 |
| 检查点切分新旧等价 | 2,013 有效用例，0 跳过、0 失败 |
| cache 开/关与伪造历史 | 36 项通过 |
| cache v1 忽略/v2 持久化与真实复用 | 5 项通过 |
| tokenize/detokenize | 19/19 |
| Messages 确定性协议/错误/断连 | 26/26 |

## 本轮真实模型验证

隔离工作目录、配置、缓存和端口 18960/18961；未使用正式目录启动服务。
权重/tokenizer 与基线一致。Flash-Next 保留 aligned 精度、MTP gamma 2、双槽、
262144 context 与原算术开关；仅将磁盘 KV 快照关闭、将日志/缓存位置转入实验目录。
冷/暖指引擎和应用缓存，不声称清空 Windows 文件缓存。完整有效环境保存在各 run manifest。

| 检查 | 本轮结果 |
| --- | --- |
| 相同 r23-e 办公/代码工具任务与预算 | 基线 2/2、候选 2/2；均完成实际 SQL/工具执行，初始代码测试失败后恢复 |
| 成功任务耗时（办公/代码） | 基线 260.03/106.24 秒；候选 257.52/105.74 秒，样本不足以声称加速 |
| 实际 TTL/LRU 项目修复 | OpenAI 与 Messages 均通过实际文件修改、Python 子进程公测与独立私测 |
| Messages 文本/count、中文 SSE、控制串原文、真实工具往返与项目 | 5/5 |
| Anthropic smoke + 已安装 SDK 0.101.0 | 31 项通过，含 SDK 非流工具、流式 thinking/text、count_tokens |
| 长提示冷/暖 | 实际用户 token 31,713 与 138,911；答案正确且观察到前缀复用 |
| 并发/取消 | 贪心与固定种子采样一致；54K prefill 取消后 5.51 秒释放，0 输出 token，同连接、并发同伴与 API 恢复通过 |
| serial/MTP 贪心 | 按真实 EOS 停止的 4/4 用例 token 一致；不等同 BF16 真值 |
| 图像与兼容 API | PNG/JPEG/WebP、历史/双图、Responses 与 SSE 通过；真实 32 图成功、33 图拒绝后恢复、文字占位符+真图成功 |
| Qwen27 MTP | 原 13 项引擎/模型 + 7 项 API + 新 6 项 Messages 通过 |
| Qwen27 DFlash2 | 同上 26 项通过；实际统计为 dflash2，观察到 drafted/accepted 正计数 |

Qwen27 保留 context 8192、单槽、FP32 dense KV、compensated prefill/chunk 512、
MTP gamma 3；DFlash2 独立 opt-in、gamma 4、全词表。两个模式的 Messages 和 count_tokens
均保持拒绝图像。没有把 Qwen27 作为官方共同模型对照。

普通短任务仍为 **8/12**：发票计算得到 15540 而正确值为 15680；排程遗漏 B/C 冲突，
各重复两次。API 默认推测的并发测试保留原始 passed=false，因为答案仍有这些错误；
串行与并发的答案一致，不能用一致性掩盖正确性失败。两任务配对和固定有限提示不代表
所有模型、长尾延迟或整体精度；本轮没有重新进行完整官方/自维护大对比。

## 证据与复现

原始请求、响应、SSE、工具执行、失败样本、配置与构建身份位于本机隔离结果目录
`C:\Users\Jerry\Projects\gfx1151-api-validation-20261031`，总报告为 `REPORT.md`，
结构化汇总为 `integration-results.json`，保全证据为 `final-audit.json`。
`run_cpu.py`、`run_live_integration.py`、`run_vision_boundaries.py` 与复制的固定任务框架
均保留在该目录；不要在有正式服务或其他任务占用测试端口时运行实时模型复现。

新增 CPU 测试可按以下方式在自行指定的隔离输出目录复现：

```bash
TOK_DIR=/path/to/model/tokenizer THEROCK=/path/to/existing/sdk bash build_win.sh literal-test
python -B tools/messages_api_test.py --api /path/to/gdec-api-win.exe --output /new/isolated/messages
python -B tools/api_literal_tcache_test.py --api /path/to/gdec-api-win.exe --tokenizer /path/to/model/tokenizer
python -B tools/api_token_cache_version_test.py --api /path/to/gdec-api-win.exe --tokenizer /path/to/model/tokenizer --output /new/isolated/cache
python -B tools/api_tokenize_test.py --base http://127.0.0.1:18961
python -B tools/anthropic_messages_smoke.py --base http://127.0.0.1:18961
```

最初 SDK 探针依赖官方 health.tokenizer 字段，与保留的本地 health 不兼容；在开始 SDK
推理前结束该探针，测试 readiness 改为兼容 engine_info_ready，随后在新进程中通过 31 项。
原始预检日志留存，不作为推理失败或通过证据。构建期间的旧字段适配和测试链接修复也单独留档。

正式两套部署、精度配置、启动器、运行库、模型权重、全部回滚备份及原工作区未跟踪的
docs/design 均保留；测试结束没有遗留本轮服务。两个正式实例在测试前后均为停止状态。

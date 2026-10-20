# 三项高优先级上游变更融合与验证

在 `codex/upstream-priority-20261020` 上完成，基线为
`8199e51463732c0d31398fa3c1c39ad99e5a6c6d`。未替换正式部署。

## 融合范围

| 来源 | 本地落地 |
| --- | --- |
| `2e8e30e` | Flash-Next 的 prompt 预填在分块前后检查取消，并按取消语义回收当前槽位。取消指针是函数调用参数，不存入双槽共享的模型对象；投机 verify 不传该参数，保留原有回滚边界。 |
| `f18286b` 的过滤修复 | 以实际预填起点过滤 SNAPS/SNAP/SNAP2，并同步修复本地 retry checkpoint 的同类判断。全量重算起点是 0；续接和恢复仍从已缓存前缀之后选择切点。未引入该提交的分叉点缓存功能。 |
| `9b7067b` | 将 `compute_snap_cuts` 独立成编译单元，倒序找最近 8 个有效切点。保留旧实现作为等价测试参考，并接通 Linux/Windows 构建及包含 API main 的现有测试。 |

保留原 XML 模板、工具校验、WebP/32 图支持、晚启动恢复、控制台、Qwen27、MTP、全词表 DFlash2。
`service.conf`、Windows 算术 profiles、启动器、数值 kernel 和精度默认值没有修改。
未整线合并上游，也未引入新动态 gamma、合批调度、A4、DFlash2 Flash-Next 后端或运行库升级。

## 构建与 CPU 回归

使用既有 TheRock 10.0.0 gfx1151 和 VS 2022 BuildTools，候选引擎及 API 构建通过。

- 检查点切分：10,013 个有效文本/工具/思考/图像/长对话用例，新旧结果完全一致；跳过 0，失败 0。适配上游测试夹具的工具参数对象和真实 user 消息，以遵守本地模板约定。
- GPU 测试停止后的主机微基准：401 条消息，每实现 20 次，切点计算平均由 33.659 ms 降至 1.402 ms。这仅是主机端切点计算耗时。
- 90 项 API 协议、458 项工具错误、49 项断连检查、取消 ACK 超时套件、3 组容量协商、6 组晚启动恢复均通过。
- WebP/工具图片 30 项、模板渲染/错误对照 796 项、Qwen27 非法 CLI 10 项通过。
- 两个构建脚本语法检查、切分代码严格主机编译警告检查通过。

## Flash-Next 真实模型对照

旧二进制来自 source-base 对应的既有 Qwen27 部署，复制后核对部署清单 SHA-256；
它包含相同 Flash-Next 自维护路径。候选与旧版顺序启动，使用独立端口和独立工作目录。

沿用正式 Flash-Next 的有效算术配置：BF16 KV、aligned、MTP gamma 2、
`GDEC_V2_HT_DEQ_EXACT=1`，prefill chunk 8192、单请求上下文 262144、KV pool 524288、RAM checkpoint 2。
确定性缓存复现使用单槽，其余使用双槽；没有改写持久配置。

- 缓存缺陷复现：A/B 共享 4500 token，B 全量重算时请求保存 4096 切点。旧版下一请求 cached=0，候选 cached=4096。
- 恢复后的 32 个 token、logprob 和结束原因，与淘汰缓存后按相同切点冷算的结果一致；MTP 保持启用。该请求预填耗时约 1176 ms / 冷算 4803 ms。
- 中英文、Python、长上下文摘要共 4 个输入，各测 greedy 和固定 seed 采样。每版 8 个串行参考请求加 2 轮双槽并发，共 24 个请求，均与该版串行参考一致。
- 两版 8 个串行参考请求，共 512 个输出 token，其 token、logprob、结束原因全部相同。
- 32K prompt 在观察到预填进度后取消：旧版算完 32768 token，等待 20.063 s；候选在 8192 处停止，等待 2.390 s。另一槽输出与串行参考一致，原连接后续生成正常。
- 另一次独立启动复核取消与断连：取消等待 2.422 s；断连后观察到对应请求的 `prefill aborted` 清理日志，后续请求正常。该次断连与恢复检查耗时 4.797 s。
- 真实 32 图 SSE 顺序识别、33 图拒绝后工具调用、Responses SSE 通过。结束时 API in_flight=0、queued=0。

## Qwen27 自维护功能

用同一候选引擎分别启动 MTP gamma 3 和全词表 DFlash2 gamma 4，均为 8192 context、
compensated/512 prefill。每个模式的 13 项协议/真实推理检查和 7 项 API 检查通过：
包括黄金 token、固定 seed、缓存续接、超过 2051 token 的注意力与续接、取消后恢复、
中文计算、工具调用/结果往返、Chat SSE、Responses 和图像拒绝。

## 身份、复现与边界

结构化结果与源码/二进制 SHA-256 见 [priority-upstream-20261020.json](benchmarks/priority-upstream-20261020.json)。
原始日志和逐请求输出保留在本机忽略目录 `logs/priority-upstream-20261020/`。

```text
# 已准备好的 VS/TheRock 环境；TOK_DIR 指向原模型 tokenizer
bash build_win.sh engine
bash build_win.sh api
bash build_win.sh snap-cuts-test

# 对自己启动的隔离模型服务；checkpoint 需要单槽和 RCKPT_MAX=2
python tools/upstream_priority_live_test.py --port <PORT> --stage checkpoint --expect-fixed --output checkpoint.json
python tools/upstream_priority_live_test.py --port <PORT> --stage cancel --expect-fixed --engine-log <ENGINE_LOG> --output cancel.json
python tools/upstream_priority_live_test.py --port <PORT> --stage parity --tokenizer-cli <TOK_CLI> --tokenizer <TOKENIZER_DIR> --output parity.json
```

测试针对 Windows；Linux 构建脚本仅作语法检查。没有重做双路满 262144 输入压力、
通用模型能力或完整 BF16 真值语料评测；场景耗时不能推广为通用吞吐提升。
正式部署的二进制、配置和部署清单哈希保持一致，所有本轮测试服务在结束后停止。

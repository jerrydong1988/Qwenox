# Qwen3.8-27B HGN 文本后端

2026-10-14。本后端自动识别 Halogen `Qwen3.8-27B-p1-d2`、HGN1/version 2，
对应文件 `qwen3.8-27b-p1w4d-d2.hgn`。支持 Windows / gfx1151 上的文本生成、
OpenAI 兼容对话接口、流式输出、工具调用、采样、取消和连续前缀复用。

它使用独立的 27B 执行路径；Flash-Next 的 MoE、HC、PLE、专用 KV 几何和
量化类型解释没有被改写。API 与 tokenizer 沿用现有实现。原有服务可以继续使用原配置。

## 权重与计算

已校验本地文件大小 **35,865,565,184 字节**，SHA256：

```text
274c3fc767fb57faf025dd03c76376b1fc1b32448aef45aaf02d1898fa962089
```

对应发布者 Hugging Face revision `800ec932fc4a83ddd18b83d293cfd147946e77f9`。
容器有 1,352 个张量。本后端加载文本主干，跳过 `.i4l` 镜像、MTP 和 DFlash 草稿。
模型名、版本、主干张量形状、dtype、qparam 和载荷大小均在加载时检查。
本实现的完整权重文件验证不等于每次启动都会重新计算 SHA256。

| 项目 | 实现 |
|---|---|
| 主干 | 64 层，hidden 5120，dense FFN 17408 |
| GDN | 48 层；16 key heads、48 value heads、128 维；FP32 递归状态 |
| 全注意力 | 16 层；24 Q / 4 KV heads、head dim 256；64 维 partial RoPE，theta 1e7 |
| dtype 0 | BF16 原值转换成 FP32 |
| dtype 6 | E4M3FN codes，末尾逐行 **BF16** scale |
| dtype 5 / qparam 0 | NVFP4 codebook，逐行 `[nibbles][每 16 个值一个 FP8 scale]` |
| decode | 向量化加载并原位解码，FP32 FMA；HIP graph 和 dense split attention |
| prefill | 有界 FP32 解码工作区与 rocBLAS SGEMM；默认 256 token 分块，短输入走逐 token 路径 |

这里的 dtype 5 不是 Flash-Next 的 Q4CP 布局；dtype 8 `.i4l` 也不是引擎的
合成 q8g32。不能只凭 HGN version 或 dtype 数字共用旧解码器。

主干权重保持 HGN 中的数值，没有再次量化、没有使用低精度 activation MAC。
**把量化值展开为 FP32 不会恢复量化前丢失的精度。**
GDN 输出使用 SiLU 门控和普通 RMSNorm；主干/QK 归一化使用零中心权重，
全注意力输出使用 sigmoid 门控。

## 构建和启动

在 VS 2022 x64 Native Tools 环境中使用 TheRock 10.0.0：

```bat
set THEROCK=C:/path/to/therock-10.0.0
build_win.bat engine
build_win.bat api
build_win.bat launcher
build_win.bat qwen27-test
```

本机验证使用 VS 2022 的头文件；不要把另一个 VS 版本的头文件隐式混入该 HIP 工具链。
未使用 `-ffast-math`。Linux 构建未在本次验证范围内。

将 [配置样例](../tools/windows_qwen27.conf.example) 放入独立部署目录，命名为
`service.conf`，设置 `MODEL_DIR` 后使用新编译的 `start_win.exe`。二进制及依赖
仍按原部署布局放在 `build/`。`API_MODEL` 控制 `/v1/models` 和响应中的模型名称。
默认端口为引擎 18870、API 18871，默认上下文 8192、一个请求槽。

也可以直接运行：

```powershell
.\build\gdec-win.exe C:\models\halogen-qwen3.8-27b\qwen3.8-27b-p1w4d-d2.hgn --serve --host 127.0.0.1 --port 18870 --maxctx 8192
.\build\gdec-api-win.exe --tokenizer C:\models\halogen-qwen3.8-27b\tokenizer --engine 127.0.0.1:18870 --listen 127.0.0.1:18871 --model qwen3.8-27b-hgn --context 8192 --text-only
```

可用诊断参数：

- `--tokens 1,2,3,4 --gen 16` 或 `--tokens-file path`：直接生成 token ID。
- `--qwen27-no-graph`：不用 HIP graph，使用普通全注意力计算作为对照。
- `--qwen27-serial-prefill`：逐 token prefill，便于数值定位。
- `--qwen27-prefill-chunk 128`：覆盖分块长度，范围 1–512；也可设置 `GDEC_PREFILL_CHUNK`。
- `--qwen27-trace path`：导出各层 FP32 状态；该诊断模式关闭 graph，不能用其耗时代表服务性能。

## 验证基线

不需要再下载一份原始 BF16 模型。`tools/qwen27_reference.py` 用 NumPy 独立解析
**同一份 HGN** 并计算完整文本主干，用来区分引擎实现误差与模型既有量化误差。
其方程对照 [Transformers Qwen3.5 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
这是本项目构建的参考计算，不是发布者提供的 Halogen 运行结果。

```powershell
python tools/qwen27_reference.py C:\models\halogen-qwen3.8-27b\qwen3.8-27b-p1w4d-d2.hgn --tokens 1,2,3,4 --output build/qwen27-reference
.\build\gdec-win.exe C:\models\halogen-qwen3.8-27b\qwen3.8-27b-p1w4d-d2.hgn --tokens 1,2,3,4 --gen 1 --maxctx 128 --qwen27-serial-prefill --qwen27-trace build/qwen27-gpu
python tools/qwen27_compare.py build/qwen27-reference build/qwen27-gpu
python tools/qwen27_live_test.py --engine-port 18870 --api http://127.0.0.1:18871 --long
python tools/qwen27_api_smoke.py --api http://127.0.0.1:18871
```

已完成：

- CPU 格式边界用例通过；真实权重 98,304 个抽样值，C++ 与独立 NumPy 解码逐位相同。
- 四个输入 token、64 层与最终词表 logits 对比。向量化 decode 的最大层相对 L2
  误差 `1.19025e-6`，最大 logit 绝对误差 `1.04904e-5`，KL 最大约 `2.75e-12`，四个 top-1 全部一致。
- FP32 batched prefill 的同一组对照最大层相对误差 `5.20893e-6`，top-1 全部一致。
- 协议/接口回归覆盖 EOS 不泄露、非法 token、固定 greedy 序列、采样复现、缓存续写、
  跨分块续写、取消及恢复、中文算术与结构化工具调用。
- 2,056 token 提示跨过旧 QSA 的 2,051 token 边界，继续生成再续写，缓存与重新 prefill 一致。
- 相同长提示启用/关闭 HIP graph 的 16 个输出 ID 一致。
- 最终部署通过 13 项引擎/接口检查、7 项 API 交付检查；API 另通过 90 项模拟引擎回归。
  原有 Flash-Next 在其配套 n-gram 与运行参数下仍能加载并生成；HGN v2 与 overlay 格式测试通过。
  纯文本部署的 `/health` 正确声明不支持图片，Chat/Responses 图片请求返回 HTTP 400。

机器可读的结果摘要和二进制 SHA256 见 [QWEN27_VALIDATION.json](QWEN27_VALIDATION.json)。

这些检查证明上述输入上的实现一致性，不是全语料质量评测。未将原始 BF16 模型
作为质量基准，也不宣称复现 Halogen 闭源引擎的性能。

## 性能和内存

本机为 Ryzen AI Max+ 395 / gfx1151 / 128 GiB，单槽，8K 上下文；以下是本次
具体输入的测量，不保证任意长度或并发下都相同。

| 相同工具调用任务（294 prompt token） | 端到端耗时 |
|---|---:|
| 最初的逐 token 正确性实现 | 约 107.6 秒 |
| FP32 批量 prefill + 初始 GEMV | 约 20.7 秒 |
| 再加入向量化 GEMV（128-token 分块） | 约 12.2 秒 |
| 最终版本（256-token 分块） | **约 11.3 秒** |

在缓存四 token 提示后生成八 token 的测试中，实际七次 decode forward 从约
2.36 秒降为 0.729 秒，即约 **9.6 token/s**。首个输出来自 prefill logits，
不把这一步当作一次额外 decode forward。128-token 分块下，2,056 token prefill
约 53.1 秒（38.7 token/s）；最终 256-token 分块版本约 **45.8 秒（44.9 token/s）**。
HIP graph 单独的收益很小，主要 decode 收益来自 GEMV。

权重主干约 **20.16 GiB**。8K FP32 KV 约 **1 GiB**，48 层 GDN 状态约
**144 MiB**，其余是矩阵解码与分块工作区。最终版本在 8K 上下文的设备分配实测
**21.84 GiB**；进程 RSS、文件缓存和驱动额外开销不能
与设备分配混为一谈。每增加 8K 上下文，FP32 KV 大约增加 1 GiB。

## 当前边界

- 纯文本、单请求槽；连续完整前缀可复用。分叉/截断历史重新 prefill，无 SSD 状态缓存。
- MTP / DFlash2 / `.i4l` 加速镜像尚未接入；请求投机解码会被拒绝。
- 文件不含视觉塔，本后端不接收图像。它不会把 Flash-Next 视觉权重套用到 27B。
- 默认 8K；更长上下文需要额外内存与测试。代码中的容量上限不代表该长度已经验证。
- 仅此 HGN recipe 经验证；其他量化 recipe、GGUF 或原始 safetensors 不能直接传给本后端。

进一步优化应先评估无损的批量 GEMM/量化读取融合，再独立验证 MTP 和 DFlash2
的格式、状态推进与接受率。每项都应保留上述参考对照和可关闭的性能路径。

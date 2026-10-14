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
容器有 1,352 个张量。本后端加载文本主干，可选加载内置 MTP，跳过 `.i4l` 镜像和 DFlash 草稿。
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
| dtype 5 / qparam 1 | MTP NF4，逐行 `[nibbles][每 32 个值一个 FP16 scale]`；不能按 NVFP4 解释 |
| decode | 向量化加载并原位解码，FP32 FMA；HIP graph 和 dense split attention |
| prefill | 默认 512 token 分块，BF16 主项和残差两次矩阵乘、FP32 累加；保留 FP32 SGEMM 对照模式 |
| MTP | 单层草稿，最多提前猜 3 token，主模型批量验证；GDN/卷积状态可按接受长度回退 |

这里的 dtype 5 不是 Flash-Next 的 Q4CP 布局；dtype 8 `.i4l` 也不是引擎的
合成 q8g32。不能只凭 HGN version 或 dtype 数字共用旧解码器。

解码阶段使用原 HGN 权重数值和 FP32 FMA。加速预填充先将 FP8 行缩放或
NVFP4 全局缩放移到点积之后；码本不满足条件时自动使用 FP32 路径。
激活值拆成 BF16 主项和 BF16 残差，再以 FP32 累加，减少单次 BF16 舍入误差。
这仍是近似 FP32 计算，不是逐位等价，也不是官方 Halogen 的 W4A4 激活量化路径。
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
.\build\gdec-api-win.exe --tokenizer C:\models\halogen-qwen3.8-27b\tokenizer --engine 127.0.0.1:18870 --host 127.0.0.1 --port 18871 --model qwen3.8-27b-hgn --context 8192 --text-only
```

可用诊断参数：

- `--tokens 1,2,3,4 --gen 16` 或 `--tokens-file path`：直接生成 token ID。
- `--qwen27-no-graph`：不用 HIP graph，使用普通全注意力计算作为对照。
- `--qwen27-serial-prefill`：逐 token prefill，便于数值定位。
- `--qwen27-prefill-mode fp32`：使用 FP32 SGEMM；默认 `compensated`。
- `--qwen27-mtp` / `--qwen27-no-mtp`：启用/关闭服务端 MTP；默认不开启，提供的服务配置启用。
- `--qwen27-mtp-gamma 3`：草稿长度，范围 1–3。
- `--qwen27-prefill-chunk 128`：覆盖分块长度，范围 1–512；也可设置 `GDEC_PREFILL_CHUNK`。
- `--qwen27-trace path`：导出各层 FP32 状态；该诊断模式关闭 graph，不能用其耗时代表服务性能。
- `--qwen27-force-batch`：诊断时强制短输入也走预填充矩阵路径。

原生启动器可在 `service.conf` 设置 `GDEC_QWEN27_MTP="1"`、
`GDEC_QWEN27_MTP_GAMMA="3"`、`GDEC_QWEN27_PREFILL_MODE="compensated"`。
`PREFILL_CHUNK="512"` 控制分块。MTP 权重在同一个 HGN 中，`MTP_FILE` 仍留空。
CLI `--gen` 用于串行目标模型诊断；推测生成通过 `--serve` 后的引擎协议或 API 验证。

采样模式每一步都从主模型分布抽样，只有抽中的 token 与草稿一致才接受；
拒绝时恢复随机数状态和已输出 token 计数。温度、top-k/top-p/min-p、惩罚、
bias 和 logprobs 仍由目标采样器计算。草稿不会绕过主模型核验。

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

启用 MTP 的服务给 `qwen27_live_test.py` 增加 `--mtp`。独立 MTP 数值对照：

```powershell
python tools/qwen27_mtp_reference.py C:\models\halogen-qwen3.8-27b\qwen3.8-27b-p1w4d-d2.hgn --target build/qwen27-reference --output build/qwen27-mtp-reference
.\build\gdec-win.exe C:\models\halogen-qwen3.8-27b\qwen3.8-27b-p1w4d-d2.hgn --tokens 1,2,3,4 --gen 0 --maxctx 64 --qwen27-serial-prefill --qwen27-mtp-probe 5 --qwen27-trace build/qwen27-mtp-probe
```

已完成：

- CPU 格式边界用例通过；包含 MTP NF4 后，真实权重 129,024 个抽样值，C++ 与独立 NumPy 解码逐位相同。
- 四个输入 token、64 层与最终词表 logits 对比。向量化 decode 的最大层相对 L2
  误差 `1.19025e-6`，最大 logit 绝对误差 `1.04904e-5`，KL 最大约 `2.75e-12`，四个 top-1 全部一致。
- FP32 batched prefill 的同一组对照最大层相对误差 `5.20893e-6`，top-1 全部一致。
- 补偿预填充同一组对照最大层相对误差 `5.08075e-5`，logit 最大绝对误差
  `2.93017e-4`，最大 KL `2.60e-9`，四个 top-1 一致。
- MTP 单层与独立 NumPy 对照的 logit 相对 L2 `6.09e-7`、最大绝对误差
  `1.04e-5`、KL `2.49e-12`，top-1 一致。四种布局的代表矩阵中，小批量 GEMV 的 1–4 行结果与串行逐位一致。
- 协议/接口回归覆盖 EOS 不泄露、非法 token、固定 greedy 序列、采样复现、缓存续写、
  跨分块续写、取消及恢复、中文算术与结构化工具调用。
- MTP 七组 greedy 用例与串行输出一致，覆盖数列、代码、中文、英文、短输出和长输入；
  接受与拒绝分支均被实际执行，续写结果一致。代码/中文各三组采样种子也保持 token
  序列一致，包含惩罚、bias、top-k/top-p/min-p 与 logprobs，最大 logprob 差约 `1e-5`。
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
| 首个兼容版本（256-token 分块） | 约 11.3 秒 |
| 补偿预填充（256-token 分块，串行 decode） | 约 7.8 秒 |
| 正式部署（512-token 分块，MTP gamma 3） | **约 3.9 秒** |

在缓存四 token 提示后生成八 token 的测试中，实际七次 decode forward 从约
2.36 秒降为 0.729 秒，即约 **9.6 token/s**。首个输出来自 prefill logits，
不把这一步当作一次额外 decode forward。128-token 分块下，2,056 token prefill
约 53.1 秒（38.7 token/s）；首个兼容版本的 256-token 分块约 **45.8 秒（44.9 token/s）**。
HIP graph 单独的收益很小，主要 decode 收益来自 GEMV。

本轮加入补偿预填充和 MTP 后，512-token 分块下相同 2,056 token 提示的预填充
约 **13.0 秒（158 token/s）**。以下 greedy 比较均生成相同 token ID；速度按
`(输出数 - 1) / decode 时间` 计算，首个输出由 prefill logits 提供。

| 输入 / 输出数 | 串行 decode | MTP gamma 3 decode | 加速比 |
|---|---:|---:|---:|
| 数字序列 / 64 | 6.55 秒 | 3.29 秒 | 1.99× |
| Python / 96 | 9.86 秒 | 3.91 秒 | 2.52× |
| 中文 / 96 | 9.87 秒 | 4.67 秒 | 2.11× |
| 英文 / 96 | 9.81 秒 | 5.21 秒 | 1.88× |
| 2,056 token 提示 / 32 | 3.28 秒 | 1.54 秒 | 2.13× |

这些 MTP 测试约为 **18–24 token/s**，草稿接受率约 **71%**，随输入与采样参数变化。
单/双 token 输出避免无收益的草稿验证。MTP 启用、实际尝试次数与提交数通过协议统计区分。

权重主干约 **20.16 GiB**。8K FP32 KV 约 **1 GiB**，48 层 GDN 状态约
**144 MiB**，其余是矩阵解码与分块工作区。8K 上下文的设备分配初版实测
为 **21.84 GiB**；MTP 与补偿工作区启用后为 **22.95 GiB**。进程 RSS、文件缓存和驱动额外开销不能
与设备分配混为一谈。每增加 8K 上下文，FP32 KV 大约增加 1 GiB。

## 当前边界

- 纯文本、单请求槽；连续完整前缀可复用。分叉/截断历史重新 prefill，无 SSD 状态缓存。
- 已接入 MTP；DFlash2 和 `.i4l` 加速镜像未接入。
- 文件不含视觉塔，本后端不接收图像。它不会把 Flash-Next 视觉权重套用到 27B。
- 默认 8K；更长上下文需要额外内存与测试。代码中的容量上限不代表该长度已经验证。
- 仅此 HGN recipe 经验证；其他量化 recipe、GGUF 或原始 safetensors 不能直接传给本后端。

尚未建立全语料困惑度/任务质量基准；数值误差和上述输入的一致性不能等同于全模型
质量保证。DFlash2 仍需独立验证其卷积、草稿状态与候选筛选，不承诺闭源 Halogen 的性能。

# Qwen27 DFlash2 与扩展质量验证

本轮在原有文本后端、补偿预填充和 MTP 上增加独立 DFlash2 路径。基础说明见
[QWEN27.md](QWEN27.md)。HGN 文件和 tokenizer 均未修改，不需要另外下载原始 BF16 权重。

## 实现与精度边界

- 严格核对 HGN 的 `drafter.config`、浮点配置、全部需要的张量形状和量化布局。
- 主干第 5/19/33/47/61 层的输出拼接后投影，生成五层草稿各自的上下文 KV。
- 草稿使用普通 RMSNorm、128 维完整 RoPE、2,048-token 滑动上下文，以及块内非因果注意力。
- 注意力和 MLP 前后都有两抽头、每组 16 个通道的动态卷积。
- 每个位置先取 16 个候选，再用 rank-256 前驱/后继码本与隐藏状态投影选择路径。
- 草稿矩阵保留 HGN NF4 数值，使用 FP32 FMA。主模型验证批量扩展到 8，GDN/卷积状态保存至 7 个回退位置；只同步已接受 token 的草稿上下文。
- 采样仍从目标模型分布抽取，草稿只用于匹配；拒绝后恢复随机数状态和计数。草稿不直接决定最终输出。

参考方程来自 [z-lab/dflash 的 MLX 实现](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)。
本地配置也与 [公开 Qwen3.8-27B DFlash2 配置](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json) 的几何相符。
实际数值对照使用独立 NumPy 实现解析**本地同一份 HGN**；不把公开原始权重或闭源 Halogen 的结果当作已测基准。
本实现使用固定草稿长度，不使用附带的 `alpha_seed`，也不会据此跳过目标验证。

## 为什么默认使用完整词表

HGN 自带 `draft_lm_head` 的词表只有 98,304 项。实测中文用例的 96 个输出中，
47 个不在这个词表中。精简词表版本仍保持目标输出正确，但这些位置必然拒绝草稿，
中文测试甚至比串行更慢。因此 DFlash2 默认复用主模型的 248,320 项词表头，
不会重复加载该权重，也不会重新量化。`--qwen27-dflash-reduced-vocab` 仅用于复现对照。

## 运行与诊断

```powershell
.\build\gdec-win.exe C:\models\qwen3.8-27b-p1w4d-d2.hgn --serve --port 18874 --maxctx 8192 --qwen27-dflash --qwen27-dflash-gamma 4
```

`--qwen27-dflash-gamma` 接受 1–7，表示额外草稿数。DFlash2 与 MTP 选择其一；
启用 DFlash2 时优先使用它。服务配置可设置 `GDEC_QWEN27_DFLASH="1"` 和
`GDEC_QWEN27_DFLASH_GAMMA="4"`；停用设置为 `0`。原有 MTP 配置仍可使用。
协议 drafter `0` 是串行、`1` 是 MTP、`2` 是 DFlash2，未加载的草稿类型会拒绝。
API 沿用引擎默认模式；请求里的 `drafter` 不是动态切换模型的入口。

独立数值对照：

```powershell
python tools/qwen27_dflash_reference.py C:\models\qwen3.8-27b-p1w4d-d2.hgn --target build/qwen27-reference --output build/qwen27-dflash-full-cpu --full-vocab
.\build\gdec-win.exe C:\models\qwen3.8-27b-p1w4d-d2.hgn --tokens 1,2,3,4 --gen 0 --maxctx 64 --qwen27-serial-prefill --qwen27-dflash-probe 5 --qwen27-dflash-gamma 7 --qwen27-trace build/qwen27-dflash-full-gpu *> build/qwen27-dflash-full-probe.log
python tools/qwen27_dflash_compare.py build/qwen27-dflash-full-cpu build/qwen27-dflash-full-gpu --log build/qwen27-dflash-full-probe.log
```

参考输入来自主干独立 NumPy 对照，
GPU 输入来自实际主干计算，包含主干误差。五层最大相对 L2 为 `1.19e-6`，
完整词表 logits 的相对 L2 为 `1.22e-6`、最大绝对误差 `2.48e-5`、最大 KL `9.73e-12`；
8 行 top-1 和最终 7 个草稿全部一致。五种代表矩阵的 1–8 行批量 GEMV 与串行逐位相同。

## 固定语料的逐 token 评分

`tools/qwen27_quality.py` 用相同 HGN 比较串行 FP32 与补偿预填充，输入每一步均固定为
语料中的真实 token。这样不会让一次生成分歧污染后面的比较。评分文件保留每步
目标 token、top-1、logprob，以及输入 token SHA256。

| 样本 | 评分位置 | 串行 PPL | 补偿 PPL | 平均 NLL 差 | top-1 一致 |
|---|---:|---:|---:|---:|---:|
| WikiText-2 test 前缀切片 | 2,048 | 4.972987 | 4.973003 | 3.27e-6 | 2,048/2,048 |
| 原创中文说明 | 320 | 7.309710 | 7.309732 | 2.99e-6 | 320/320 |
| Python 代码 | 512 | 4.529381 | 4.529407 | 5.81e-6 | 512/512 |

最大单 token logprob 差 `0.0003443`。英文来源为
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1/test`，
取前 100 行中的前缀并按本模型 tokenizer 截断；下载内容的 SHA256 随验证记录保存。
这是 **2,880 个位置的实现一致性检查**，不是完整 WikiText 测试集的标准 PPL，
也不是 MMLU/GSM8K 全集得分或量化前 BF16 的质量对照。

## 性能选择

同一机器、同一 HGN、相同提示和 96 个输出 token；串行参考一次，MTP 与 DFlash2 各两次。
正式复测期间未并行运行 CPU 参考计算、编译或其他本任务 GPU 请求。速度按
`95 / 平均 decode 秒数` 计算，首个输出来自 prefill。全部输出 ID 相同。

| 测试 | 串行 token/s | MTP gamma 3 | DFlash2 gamma 4 |
|---|---:|---:|---:|
| 数列 | 9.58 | 21.03 | 25.06 |
| Python | 9.58 | 24.49 | 26.93 |
| 中文解释 | 9.58 | 20.39 | 15.79 |
| 英文解释 | 9.61 | 18.28 | 18.21 |

DFlash2 在数列和代码用例中更快，但中文更慢，英文接近；因此正式服务继续默认使用
**MTP gamma 3**，DFlash2 gamma 4 作为可选配置。不能把代码用例的收益推广到所有请求。
其他草稿长度和每次测量见机器可读验证记录。早期与 CPU 参考并行的探索性测量不用于本表。

完整词表 DFlash2 已通过 7 组 greedy、6 组固定种子采样、13 项引擎检查与 7 项 API 检查；
包括接受/拒绝、EOS、长于 2,051 token 的输入、缓存续写、取消恢复和工具调用。
Greedy 对照中实际提出 479 个草稿、接受 256 个；采样输出与串行一致，最大 logprob 差约 `1e-5`。

8K 上下文、完整词表 DFlash2 预热后的设备分配实测为 **24.48 GiB**，默认 MTP 约
**22.95 GiB**。这不包括系统、文件缓存和驱动的全部开销。
完整记录、每次测速、输入摘要及测试二进制 SHA256 见
[QWEN27_DFLASH_VALIDATION.json](QWEN27_DFLASH_VALIDATION.json)。

## 尚未覆盖

全语料和大型任务集评测、长于默认 8K 的上下文、并发多槽、视觉、`.i4l` W4A4 预填充、
Linux 性能，以及与闭源 Halogen 的同环境对照。单个或少数测试的输出一致不等于
所有输入逐位相同；浮点实现仍存在上述已量化的误差。

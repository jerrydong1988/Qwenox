> 历史记录：本文对应提交10223a5的四项候选验证。当前分支已撤回Q6、MoE和KV恒等页表优化；最新范围与验证见 [仅多泵I/O报告](PREFILL_IO_ONLY_20261102.md)。

# Prefill 融合与黑屏后验证报告（2026-10-31）

本机完整证据目录：`C:\Users\Jerry\Projects\gfx1151-prefill-validation-20261031`。本文件为随源码保留的报告；逐请求原始证据和失败版本保留在该独立目录。

结论：本模型的中短输入 Prefill 有小幅收益，收益主要由 Windows 多泵读取对照支持；decode 和成功任务总时间没有明确提升。原95GiB高占用场景发生过真实GPU超时，最终V2只完成64K低占用验证，不能批准原262K配置的稳定性或据此替换正式部署。

## 黑屏结论

Windows有141/117 GPU/显示超时记录，随后用户强制关机。第二次加载时CPU日志仍推进至13:22:53，不能确认整个CPU系统已死锁。就绪时HIP仍报空闲11.89GiB，没有证据把事件确定为显存耗尽。具体内核/驱动根因未定位；完整时间线、未证实因素和原始证据见本机验证目录中的INCIDENT.md。

硬件为AMD Ryzen AI MAX+395 / Radeon8060S gfx1151，128GiB安装内存，OS可见约31.65GiB（UMA划分）；驱动32.0.32015.7029。共同模型为同一Flash-Next HGN权重及tokenizer。

## 最终源码范围

- 基线GPU二进制50dd5e4c…，对应e122649/6d5dbf6相同GPU文件树；API03d7088f…来自已部署e7e4b87。最终独立分支codex/prefill-safe-20261031基于e7e4b87，不改原工作区。官方固定91b111aeb4bf85723433a821f200398ca8d91cc2（0.1.0）。
- 保留Q6合并读取、MoE BN128/BN64分组、仅大P普通Prefill的恒等页表寻址，以及Windows独立IOCP多泵。decode、prefetch与请求分别拥有读取器；覆盖整轮读取的互斥、线程join与消费事件保持所有权。
- 撤回第一版三缓冲/两chunk预取；原双缓冲单future生命周期恢复。原始失败版保留在candidate/与v1-source/，最终V2单独保存在candidate-v2/。未融合D/E中的大tile/共享专家合并或算法分桶，没有启用额外精度捷径。
- 原API、51_host_cfg、53_qwen27、55_qwen27_dflash源码不变，取消指针仍为调用栈参数。保留HT exact、FP32 index/pairs、关闭split-K/前缀Dense WMMA、固定IPROJ=-711等现有设置。
- 精度开关之外的测试差异：Flash统一maxctx65536、KVpool131072、arena slack0，约77.7GiB。正式配置的262144上下文未改。官方采用共同参数对齐配置，其内部数值路径仍不同；本轮没有新增官方默认配置测试。全部参数、命令与版本哈希见final-configurations.json及逐run manifest。
- 同一AMD clang23.0.0/TheRock SDK与运行库；主机SDK所需cmath兼容include单列在compat/与构建脚本，哈希be27854556e6b9d782a907e7f2ae403684e589c17851b958e54b071f69e30a16。没有悄悄修改官方数值源码。最终测试引擎SHA：98892410082b05ae6ac778421aab3ae4ddcded0ff2d9dd77ac438b37eda50fa9

## 真实模型 Prefill / decode

单位为tok/s，D字段按引擎一致口径计数；原始文件同时保留首token、端到端wall、去掉首token后的decode等指标。每轮3个不同提示，每个长度输出128token；同一提示在反向顺序再次运行，属于3个提示×2种顺序，不能当成6个独立任务。

|轮次|输入|基线PP|V2 PP|官方PP|V2/基线 PP|基线decode|V2 decode|官方decode|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|first|1024|721.63|754.11|947.62|+4.50%|36.33|36.49|41.39|
|first|8192|1056.04|1095.17|1183.82|+3.71%|39.27|38.90|37.69|
|first|32768|1111.60|1118.56|1127.08|+0.63%|37.43|36.94|39.01|
|reverse|1024|700.27|731.74|—|+4.49%|35.80|36.08|—|
|reverse|8192|1003.60|1080.92|—|+7.70%|39.06|38.80|—|
|reverse|32768|1075.53|1100.34|—|+2.31%|36.49|37.07|—|

首轮A→B→官方，后续B→A。OS文件缓存未清空，GPU时钟/温度没有连续遥测。首轮基线32K-r0与一次约7秒保护文件哈希有重叠，作为受干扰测量明确保留，不用它单独证明收益；reverse轮未并发哈希、编译或另一模型。总体只支持本组8K约4%–8%的PP收益、32K小收益和decode基本持平，不能泛化为所有模型或宣称追平官方。

缓存：冷请求均cached0；另有严格8192前缀复用、4个新token的暖请求与serial8K，单独见v2-benchmark-results.json，不能把8192已缓存token除以增量耗时冒充新Prefill吞吐。

输出复现性：首轮12项比较有一条32K-r1在第91个token出现“审计员/审计师”差异。原样基线在reverse也发生同一变化，且reverse基线/V2全部9组逐token一致；相同新启动与请求顺序的baseline/full/io-only三个控制组也一致。没有据此认定新增精度回归，也不隐去首轮差异或宣称全局确定性；既有基线的请求顺序/数值边界敏感机制尚未定位。

重启前数据分开：V1在原262K/95GiB下48次计时请求完成，16个基线/V1配对逐token一致，8K PP+7.11%，32K+1.50%，128K单样本-0.64%。随后开关筛选只完成24/30并出现真实GPU失败，因此不能用这些已通过请求覆盖后续失败，也不能把V1指标作为最终V2原配置验收。多泵对照支持主要收益来自读取并行；Q6/MoE、恒等寻址没有分辨出稳定独立收益，预取加深没有额外收益，不将各历史百分数相加。

## 正确率、真实工具与恢复

基线与V2均13/18组通过，其中6种普通任务各重复2次，另有文件修复、SQL恢复、API、图像、并发与取消组。账单计算与会议冲突各两次失败，SQL虽真实执行查询并恢复到2000分，但最终多了说明文字，违反仅JSON要求而失败。评分规则没有放宽，失败原始请求/响应均保留。

文件任务由模型实际读写独立cache.py并运行子进程unittest，修复前失败、修复后6项公开与3项独立隐藏检查通过。V2约62.06秒，基线65.83秒，各单次，不能推断普遍提升。成功组总wall约222.20/223.18秒，失败组耗时约8.84/8.65秒；最大组是串行与并发4次32K验证约129秒，该验证组耗时不是单请求p99。本轮没有任务端到端显著加速证据。

单图真实颜色识别、Chat/Responses/Messages计数与SSE、控制token原文、多槽同时32K与取消后同连接/同伴恢复均通过。图像仅此有限任务；没有以health、启动、旧记录替代本次推理。

## 页表、IO与Qwen27

兼容性总判定：True；正向路径证据：True。包括页读取逐字节验证、单槽预取实际enabled/finished、乱序页表、关闭快路径回退、基线/V2的4096检查点与4097非页对齐尾页COW恢复/新算一致。原始检查见compatibility-summary.json与positive-path-audit.json。

Qwen27总判定：True。原8192上下文、单槽、512 compensated Prefill与FP32 dense KV不变；MTP gamma3，DFlash2为opt-in gamma4/full vocabulary。实际引擎与API长输入、继续生成和错误后恢复检查，外加每模式3个2048输入/256输出配对，要求实际drafter代码与正的proposed/commit，见qwen27-final-summary.json与qwen27-results.json。它是自维护额外能力，没有把它冒充官方支持的共同模型对照。

|Qwen27模式|版本|PP中位数|decode中位数|n|
|---|---|---:|---:|---:|
|mtp|baseline|131.64|12.85|3|
|mtp|candidate-v2|130.96|13.02|3|
|dflash2|baseline|128.02|10.80|3|
|dflash2|candidate-v2|127.40|11.07|3|

Qwen27 API夹具修正：部署专用脚本初次因写死正式模型名qwen3.8-27b-hgn，在隔离服务声明的isolated-prefill-test处失败。原失败日志与qwen27-summary.json保留；仅把两处模型名预期参数化后，在相同MTP/DFlash2配置重新加载并完整重跑，见qwen27-api-fixture-adaptation.patch与qwen27-final-summary.json。推理、格式、工具及图像拒绝断言没有放宽。

底层差分测试的五种Q6尺寸全BF16位与guard、MoE D256/mid128及D2560/mid640的2944专家槽完整f16隐藏值/FP32 pairs/guard、页表reserve/pin/adopt/COW/多槽/乱序，以及原CPU PLE队列测试均通过。参考是同一量化权重与旧计算顺序，不是原始未量化BF16真值，也不等同模型所有题目都正确。

冷启动长尾：最后一轮V2检查点实例仅分配75.11GiB，但加载耗时281.25秒，对应基线31.45秒；随后4096/4097恢复推理均正常，无新141/117。该加载异常没有排除或重试后替换，也尚未定位原因，不能说低占用后所有资源问题都已解决。下一轮较小Qwen27基线加载约28秒。

日志说明：单槽lookahead中的single_iocp=1是继承的串行pump接口字段；最终内部PleWinReaders默认8个独立IOCP，并非只有1个句柄。positive-path-audit.json要求实际enabled与finished，而非仅出现fallback字样。

失败证据边界：原GPU失败请求没有完整D结束帧，保留的是固定输入/配置、引擎报错、连接重置与调用堆栈；不补造完整成功wire。

## 维护建议与实验边界

保留现有精度、Qwen27、取消并发和API能力；本次最明确值得保留的是有独立所有权的Windows多泵读取。Q6/MoE与恒等页表的独立收益小于本组可分辨范围，不把它们列为显著加速来源；不扩大到数值风险更高的D/E路线。

最终分支是可审查的隔离候选。保持正式部署与备份；原高占用失败仍是未解除的部署门槛。本轮没有再次盲跑95GiB，也没有宣称已修复GPU黑屏。后续如需批准原配置，需另行定位并验证高占用稳定性。本次结论限于固定Flash权重/工具链、64K诊断配置与有限任务，Qwen27单列。

## 证据与复现

v2-build-provenance.json记录源码逐文件哈希、工具链兼容项、运行库与二进制。原始每请求wire T/D、逐请求时间、API请求/响应、真实工具执行、失败及清理状态在runs/及根目录JSON。验证目录中的v2_campaign.py、reverse_benchmark.py、diagnose_parity.py、compatibility_checks.py、qwen27.py与common.py是本轮驱动；reproduce_gpu_failure.py为被中断的原95GiB失败复现，不能当作建议直接重跑命令。

本轮仅创建测试实例并在结束后清理，正式实例在任务开始时已停止，结束保持原状态。最终118项部署/配置/启动器/备份哈希与6项154.7GB模型/tokenizer哈希另见preservation-final.json与model-inputs-final.json；最终进程端口和GPU事件复核见final-resource-state.json与post-reboot-final-gpu-events.json。实际最终校验如有失败，以这些原始判定为准。

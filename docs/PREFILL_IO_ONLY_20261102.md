# 仅多泵 I/O 融合验证（2026-11-02）

结论：只保留Windows多泵I/O后，本组8K Prefill有约5.5%的中位收益，32K收益较小，decode基本持平。可以保留这个小补丁作为独立候选；原95GiB黑屏场景仍未放行，Qwen27无辅助加载也出现过内存保护停止。正式部署和回滚备份保持原状。

## 源码与配置

硬件为Ryzen AI MAX+395 / Radeon8060S（gfx1151），128GiB安装内存，OS可见约31.65GiB。当前驱动32.0.32015.7029；未连续记录温度和时钟。

- 独立分支`codex/prefill-io-only-20261102`，工作区`C:\Users\Jerry\Projects\gfx1151-io-integration-20261102`，从10223a5收窄。
- 相对e7e4b87自维护基线，生产源码仅`src/gpu/parts/10_ple_io.inc`不同。Q6合并解量化、MoE混合tile、KV恒等页表已从源码完整撤回，相关28/30/40文件与基线一致，不是只关闭开关。
- 保留独立decode读取器、默认8个文件/IOCP的Prefill读取池、请求自有预取池、整轮互斥和异常join。保留原双缓冲/单future生命周期。GDEC_PLE_RINGS=1可作回退。
- 51_host_cfg、53_qwen27、55_qwen27_dflash及API源码未改；原取消参数、Qwen27/MTP/DFlash2保留。HT exact、FP32 index/pairs、关闭split-K/前缀Dense WMMA、IPROJ=-711等精度设置不变。
- Flash隔离配置maxctx65536、KVpool131072、slack0、双槽，约77.69GiB；单槽兼容性约75.11GiB。原262K正式配置未改，也未再次盲测95GiB。
- 同一AMD clang23/TheRock运行库及单列的cmath兼容include。候选引擎SHA256为`8bfc34832081141cd250e29c48e1bfb6bab4a8412953070cd40ca7dde393fd2a`。基线引擎50dd5e4c…，相同API03d7088f…；见build-provenance.json和逐run manifest。

## Prefill / decode

单位tok/s。四次独立加载顺序A/B/B/A，每版每个长度为3个不同提示×2轮，128个输出token。配对反向降低顺序漂移影响；没有清空OS文件缓存，也没有连续时钟/温度遥测。重复提示不是6个独立任务。本轮对照是现用自维护GPU二进制，没有新增官方版本的性能测量。

|输入token|基线PP|仅I/O PP|变化|基线decode|仅I/O decode|
|---:|---:|---:|---:|---:|---:|
|1024|899.40|929.31|+3.33%|42.16|42.29|
|8192|1343.74|1417.25|+5.47%|44.65|45.36|
|32768|1442.65|1460.89|+1.26%|42.76|42.22|

18组计时配对输出逐token一致。另有serial8K和严格cached8192/new4的暖前缀复用请求，均单列，不能将缓存token计入新增Prefill吞吐。原始T/D帧、TTFT、wall、引擎decode口径和去首token口径均保留。

单泵同二进制控制的3个8K PP为1349.63/1287.14/1154.89，存在明显抖动；其行读取时间由约0.50升到1.20秒，物理盘读取却很低。控制组请求历史/文件缓存未与主实验完全匹配，不把更大的差值称作额外收益。主要结论只取平衡A/B/B/A的约5.5%。本轮未证明普遍端到端加速。

## 真实任务、工具和兼容性

主回归基线14/18、候选13/18。两版均在两次账单计算、两次会议冲突失败；候选另有SQL结果多写说明而违反“仅JSON”的失败。原始分差和严格评分保留。新启动同序SQL诊断为基线0/3、候选1/3、单泵候选0/3，均真实执行查询并恢复到正确金额。说明格式表现会随请求历史/运行变化；没有证据确认新增数值回归，也不能宣称全面质量不降或有所提升。

真实文件任务两版均完成读写cache.py、确认初始失败，再通过6项公开及3项独立隐藏检查。13个共同成功组wall为195.77/173.59秒；其中文件任务72.48/51.46秒，但生成token为3160/2037、写文件2/1次，且基线文件任务与一次约5秒CPU测试重叠。这组单样本时间差不能直接归因于I/O优化。主吞吐测量期间没有并发编译或全量文件哈希。

API Messages/count/SSE、Responses、控制token原文、单张48x48红色图像、双槽同时32K一致性、取消后的同伴和同连接恢复均通过。图像与任务覆盖有限，不泛化为所有工具任务/视觉能力。

兼容性9/9：真实页I/O逐字节验证，单槽预取实际enabled/finished且4chunks/32768rows/pending0/complete1，4096 RAM检查点及4097非页对齐COW恢复与新算一致，单槽每版4次实际部分Prefill取消后恢复，以及配对输出一致。取消循环句柄数均无增长，RSS基本稳定。继承日志的single_iocp=1指串行pump接口，内部候选仍为独立多IOCP池。

新的CPU专项从实际生产类提取编译，行149/页151共300项通过，涵盖1/8泵、边界与乱序重复行、guard、共享池/独立池/同步decode并发、反复创建销毁、worker异常join后恢复及句柄守恒。原有PLE双缓冲队列测试也通过。没有用编译或health代替上述推理证据。

## Qwen27：真实推理通过，加载需单列限制

原8192上下文、单槽、512 compensated Prefill、FP32 dense KV不变；MTP gamma3默认，DFlash2 gamma4/full vocabulary为opt-in。每模式3组2048输入/256输出的基线/候选配对全部逐token一致，并且实际drafter为1/2、proposed及accepted=commit-rounds均为正；候选两模式的live和API检查通过。它是自维护额外能力，没有当作官方共同模型对照。Qwen27不使用Flash的PLE读取，本表只用于功能/兼容性核查，速度波动不归因于本次I/O优化。

|模式|版本|PP中位数|decode中位数|n|
|---|---|---:|---:|---:|
|mtp|baseline|167.29|15.89|3|
|mtp|candidate-io|165.57|15.65|3|
|dflash2|baseline|158.82|13.53|3|
|dflash2|candidate-io|161.11|14.35|3|

重要负样本：第一次无辅助Qwen27基线加载至53/64层时，OS可用内存1.63GiB，进程RSS16.84GiB、私有约124MiB，触发外部监视器2GiB停止阈值。是监视器主动终止，未见GPU超时；候选尚未运行。原失败日志、manifest、保护快照均保留在incidents/及原run目录。

确认实例退出、GPU空闲恢复107.72GiB且无新GPU事件后，才做另一组诊断。只对本轮自建Qwen27进程，在加载阶段可用内存低于8GiB时调用Windows EmptyWorkingSet；加载后不再调用，原2GiB停止阈值未放宽。基线首次回收约10GiB，随后约28秒就绪并完成推理。四次成功Qwen27验证对两版采用相同条件，事件逐条记录在telemetry.jsonl和manifest。此辅助仅在隔离测试框架，不写入引擎或正式配置；它支持驻留页累积的诊断，不能作为无辅助加载稳定性的验收。原API夹具只参数化隔离模型别名，其余断言不变。

## 黑屏与历史慢加载

用户授权一次UAC后，已取得10月31日三个精简转储和两份WER。用本机DbgEng及匹配微软公共PDB，确认13:14:52.274/13:15:03.763为141引擎超时及重置，13:16:09.374为117/TDR硬件挂起报告；责任模块指针均落在amdkmdag.sys+0x1c1450，版本32.0.32015.7029。公共堆栈包含VidSchiCheckHwProgress、VidSchiResetHwEngine、VidSchiReportHwHang。它支持GPU无进展/恢复链路，不能确定某个推理内核、物理硬件损坏或驱动缺陷，更不能宣称整个CPU死锁。AMD私有符号/内部TDR类型不可得，!analyze扩展缺失的失败也保留。转储仅本地读取，没有上传。

上轮Flash281秒加载已定位为bulk权重上传到33.8GiB后降至约0.06GiB/s，发生在PLE初始化之前；该加载器本就用无缓冲ReadFile→pinned槽→H2D并同步。没有历史分段计时，仍不能区分存储、主机内存和GPU上传等待，也不能把本次Qwen27映射页现象直接当作它的根因。

本轮9次Flash加载为27.30–31.68秒，未复现281秒停顿；加载和推理期间外部资源采样均保留。无新超时与低占用通过都不解除原95GiB稳定性限制。详见INCIDENT_FOLLOWUP.md、windows-dumps/*-final-analysis.log、loading-summary.json。

微软代码定义：[141](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x141---video-engine-timeout-detected)、[117](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x117---video-tdr-timeout-detected)；测试辅助API：[EmptyWorkingSet](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset)。

## 交付与复现

建议保留这个I/O小补丁作为候选，继续把原高占用GPU超时、Flash加载长尾与Qwen27加载驻留管理作为独立问题。没有本轮证据支持整批恢复Q6/MoE/KV优化，也不建议据此替换原高占用正式部署。

正式配置、启动器、运行库及回滚备份118项SHA/mtime一致；6项模型/tokenizer已完整重算SHA并核对mtime。正式实例初始为停止状态，结束保持该状态。最终进程、端口、GPU事件和Git状态以postflight-summary.json为准。

复现驱动campaign.py、single_pump.py、sql_diagnostic.py、compatibility.py、qwen27.py、common.py、telemetry.py均位于隔离证据目录；全部命令/env在逐run manifest。Qwen27诊断需显式TEST_QWEN27_LOAD_TRIM=1，其余配置见model_config。先保证无其他引擎及足够资源，再在新的结果目录运行，不能复用现存run路径，也不要将64K结果改称262K/95GiB验证。负样本和逐请求原始响应保留，不自动重试覆盖。

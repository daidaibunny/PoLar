# PoLar LLaMA V2 五百题全标签报告

更新日期：2026-07-28

V2 数据与双卡运行代码提交：`ae8d2ef`

双卡一次性安全调度提交：`532dac3`

严格无损吞吐优化与恢复调度提交：`3a53035`

## 1. V2 目标与边界

V2 固定使用 `meta-llama/Llama-3.2-3B-Instruct`，从 ReDM-Public V1 的五个固定
难度带中各选择 100 个唯一问题，共 500 个问题。每个问题都运行 Predictor-compatible
多步 MCTS 并生成标签。

V2 对 train、validation、test 三个 split 全部使用 ground truth 计算 MCTS 二值奖励。
因此这里的 test 是 fully-searched oracle-labelled 数据，不再是严格未搜索测试集。V2 可以用于
分析搜索和训练 Predictor，但不能用其 test split 报告无泄漏的在线泛化结果。后续严格评估
必须另建从未参与 MCTS 的 held-out 数据。

## 2. 数据构建

构建脚本：`scripts/build_redm_public_v2.py`

数据目录：`data/redm-public-v2`

父数据：`data/redm-public`，父 manifest SHA-256：
`2b364a4df8c906e41b66a602cf70e036059164b20302f30c4177cc5f2bc57582`。

每个难度在对应 V1 难度带内使用固定种子 42 做 domain-stratified 抽样，然后重新按
62.5% / 12.5% / 25% 分割。100 题的确定整数分配为 63 train、12 validation、25 test。
记录保留 `parent_split`，便于追踪它在 V1 中的原始 split。

| 难度 | 总数 | Train | Validation | Test | 数据文件 SHA-256 |
|---|---:|---:|---:|---:|---|
| DM-1 | 100 | 63 | 12 | 25 | `1c1524f18a403ad7eb665a3beb6bb6f65a98d036410040c5197170d4f2801ae1` |
| DM-2 | 100 | 63 | 12 | 25 | `7189fcb53edfc0da169bd1818346ac6eec77e017547162e83d237ed4e57e9426` |
| DM-3 | 100 | 63 | 12 | 25 | `4f0a074e238048b19f936b2ef7adc1675e0488bda9de18c4b73a20812e274f69` |
| DM-4 | 100 | 63 | 12 | 25 | `5ee44d85465af341be09ed2f4aedbe82ba239426a670b9006f11d98373517f83` |
| DM-5 | 100 | 63 | 12 | 25 | `f65f9c5d6c2dcc58f86dadcb63ceb167671a997a6ec032b4a843999cef2319b5` |
| 合计 | 500 | 315 | 60 | 125 | 500 个 query_id 全部唯一 |

## 3. MCTS 配置

| 参数 | V2 设置 |
|---|---|
| 搜索实现 | `f35e77a` 引入的 random-unexplored 多步 MCTS |
| 模式 | Predictor-compatible |
| 每题 simulations | 200 |
| block 长度 | 1 至 4 |
| repeat | 每个 segment 额外执行一次 |
| 随机未探索概率 | 0.1 |
| UCB exploration constant | sqrt(2) |
| 路径长度惩罚 | 5.0 |
| seed | 42 |
| generation | greedy |
| max new tokens | 50 |
| reward | 官方 PoLar DART-Math batch evaluator 的二值正确性 |
| 模型 revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| 并行方式 | gyy1 两张 A800，各负责固定 question shard；每卡并发两个 difficulty worker |
| 每张卡有效 batch/search width | 50 |
| 同轮 reward 评估 | 合并本轮全部新答案后调用一次官方 evaluator |

V2 不使用 beam-MCTS。这里使用的是已经在真实 LLaMA smoke 中验证能够产生纯 skip、纯
repeat 和联合 skip+repeat segment 的当前版本。官方 Predictor 推理阶段的 beam decoding
也不属于本次标签搜索。

## 4. 全量运行验收

全量运行结束后必须满足：500 个 question_id 全部有 trace；每题完成 200 次 simulation；
每题树深大于 1 且发生 tree-policy selection；所有路径均通过官方 Predictor parser；问题内
没有重复执行路径；全局至少存在答对的纯 skip、纯 repeat 和 skip+repeat 程序。

运行目录、耗时、路径类别分布、有效标签数量和失败问题统计将在任务完成并自动校验后写入
本节。

## 5. 运行调度状态

输出目录：
`/mnt/afs/liyiwei/PoLar/outputs/v2/mcts_labels_v2_500_multistep_2gpu_b50_532dac3_20260728_041502`

一次性监控 tmux：`polar_v2_500_wait`

监控日志：`/mnt/afs/liyiwei/PoLar/logs/polar_v2_500_wait.log`

一次性状态文件：`/mnt/afs/liyiwei/PoLar/logs/polar_v2_500_wait.state.json`

2026-07-28 04:15 UTC 提交调度时，两张 A800 均有非本项目计算进程，各占约 27 GiB。
监控不会终止或抢占这些进程。只有两张卡同时连续 6 次、每次间隔 30 秒满足无计算进程、
显存不超过 64 MiB、利用率不超过 5%，才会触发一次 V2 双卡运行。触发后状态文件永久保留，
不会重复启动第二次。

监控在 2026-07-28 04:49:56 UTC 满足条件并启动原双 worker 运行。原运行每张卡只串行处理
一个 difficulty shard；多步路径分叉后，官方 reward evaluator 的实际输入平均约 2.1 条，
约 84% 的 evaluator 调用只有一个答案，两张卡利用率约 50%。

2026-07-28 05:58:39 UTC，在完成 115 项本地测试、10 项远端针对性测试，以及 16 条真实缓存
答案的逐条评估与合并评估完全一致验证后，向旧 tmux 前台进程发送一次中断信号。两个旧 CUDA
worker 在 1 秒内退出；DM-1 两个 append-only cache 分别保留 8,166 和 7,782 条，末行均可
解析，没有删除或改写既有结果。

优化续跑 tmux：`polar_v2_500_resume_opt_3a53035`

优化续跑总日志：`/mnt/afs/liyiwei/PoLar/logs/polar_v2_500_resume_opt_3a53035.log`

优化恢复器继续使用原 batch/search width 50、每题 200 simulations、seed 42、相同模型 revision、
相同样本顺序、相同 layer path 和相同官方 evaluator。变化仅有两项：每张卡同时运行两个不同
difficulty worker；同一 MCTS 轮内先完成所有模型生成，再把本轮新答案一次性交给官方 evaluator，
并在下一轮树选择前按原顺序回填完全相同的 reward。

2026-07-28 05:58:56 UTC 优化续跑从同一输出目录启动。首批映射为 GPU 0 上 DM-1 shard 0 与
DM-2 shard 0，GPU 1 上 DM-1 shard 1 与 DM-2 shard 1。模型加载后每张卡有两个本项目 CUDA
进程；实测显存约 16--25 GiB，GPU 利用率达到 96%--100%。每个 worker 使用独立 trace、cache
和日志；任务完成后恢复器会统一合并五档标签并运行 500 题最终校验。

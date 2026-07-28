# PoLar LLaMA 复现与改进汇报

更新日期：2026-07-28  
当前代码分支：`reconstruction/mcts-llama`  
当前代码提交：`8c0ab40ff7dc5e5cbb6e10ebd69b3de3655f85b4`  
对照官方仓库提交：`30d0efde953a5d139e220f61a773c0eaf92d478f`

## 1. 汇报结论

第一阶段只复现 `meta-llama/Llama-3.2-3B-Instruct`。PoLar Predictor 内部仍按论文和
官方仓库使用冻结的 `Qwen/Qwen3-Embedding-0.6B` 作为问题编码器；该编码器不是本阶段
要复现的基础大语言模型，也不会产生 Qwen 基础模型的实验结果。

目前已经完成以下工作：

1. 重建 7,500 个唯一问题的五档公开数据，每档 1,500 个问题。
2. 验证 LLaMA 标准路径执行器与逐层路径执行逻辑。
3. 在训练和验证部分的 5,625 个问题上完成 200 次路径执行搜索。
4. 合并出五个官方格式的 `merged_mcts_samples.json`。
5. 使用官方 `PolarDataset` 路径解析逻辑严格验证全部 167,627 条有效路径；解析失败为 0。
6. 构造不含测试集 MCTS 标签的 Predictor 数据目录。
7. 完成五个独立 Predictor 的双卡训练脚本和严格在线评估脚本。

当前不能把这批标签称为论文多步 MCTS 的完整复现。每个 LLaMA 问题在根节点有 212 个
单步动作，而搜索预算只有 200 次。现有 5,625 个搜索全部只到达树深 1，UCB tree-policy
selection 次数全部为 0。因此标签只包含一次 skip 或一次 repeat，没有联合 skip+repeat
程序。这批数据可以作为官方 Predictor 的单编辑监督基线，但在正式报告论文复现结果前，
需要先修复多步搜索覆盖并重新生成监督标签。

进一步核查 2025 preliminary paper 后确认了一个实现偏差：论文写的是以 0.1 概率选择
`random unexplored child`，当前重建却在已经展开的 children 中随机选择。论文同时明确说
从初始路径进行多轮 skip/repeat 编辑。因此当前树深恒为 1 不是论文预期行为。

## 2. 已核验的官方来源

- 2026 PoLar 论文：[arXiv 2606.06574](https://arxiv.org/html/2606.06574)
- 官方代码仓库：[tianyi-lab/PoLar](https://github.com/tianyi-lab/PoLar)
- 固定版本 README：
  [README at 30d0efd](https://github.com/tianyi-lab/PoLar/blob/30d0efde953a5d139e220f61a773c0eaf92d478f/README.md)
- 固定版本数据加载器：
  [polar/data.py at 30d0efd](https://github.com/tianyi-lab/PoLar/blob/30d0efde953a5d139e220f61a773c0eaf92d478f/polar/data.py)
- 固定版本 Predictor：
  [polar/model.py at 30d0efd](https://github.com/tianyi-lab/PoLar/blob/30d0efde953a5d139e220f61a773c0eaf92d478f/polar/model.py)
- 固定版本训练循环：
  [polar/train.py at 30d0efd](https://github.com/tianyi-lab/PoLar/blob/30d0efde953a5d139e220f61a773c0eaf92d478f/polar/train.py)
- 固定版本在线评估：
  [polar/eval.py at 30d0efd](https://github.com/tianyi-lab/PoLar/blob/30d0efde953a5d139e220f61a773c0eaf92d478f/polar/eval.py)
- 2025 preliminary study：
  [arXiv 2507.07996](https://arxiv.org/abs/2507.07996)

论文说明 MCTS 是离线诊断工具；公开仓库重点提供 Predictor 的训练和评估代码，没有发布
作者生成论文监督标签所用的完整 MCTS 实现。论文 Appendix B 给出搜索空间、二值奖励、
UCB 形式和算法框架，但没有公布足以逐项恢复作者实验的全部搜索实现细节。

## 3. 第一阶段边界：只复现 LLaMA

| 项目 | 第一阶段设置 | 是否属于 LLaMA 复现结果 |
|---|---|---|
| 基础大语言模型 | `meta-llama/Llama-3.2-3B-Instruct` | 是 |
| LLaMA 层数 | 28 | 是 |
| LLaMA 权重 revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | 是 |
| 问题编码器 | 冻结的 `Qwen/Qwen3-Embedding-0.6B` | 官方 Predictor 组件，不是 Qwen 结果 |
| Predictor 数量 | 5，每个 difficulty 独立训练一个 | 是 |
| 训练基础模型参数 | 完全冻结 | 是 |
| 第一阶段评估 | 五个公开难度测试集上的 LLaMA PoLar pass@1--5 | 是 |
| Qwen1.5、Qwen2.5、Qwen3 基础模型实验 | 不运行 | 否，留到后续阶段 |
| 跨数据集 OOD 评估 | 暂不运行 | 否，先完成域内基线 |

论文明确使用冻结的 Qwen3-Embedding-0.6B 产生 token-level 问题表示，再用约 2.11M 参数
的 Predictor 为 LLaMA 生成层执行程序。Qwen 编码器的存在不改变“基础模型只复现 LLaMA”
这一实验边界。

## 4. 官方实际训练的五个 Predictor

论文在每个 DART-Math 难度上独立划分训练、验证和测试集。官方 CLI 的 `--target_diff`
一次只选择一个难度，因此论文的 LLaMA 域内结果对应五个单独训练的 Predictor，而不是一个
混合五档难度的 Predictor。

### 4.1 Predictor 架构

每个 Predictor 包含以下组件：

1. 冻结的 Qwen3-Embedding-0.6B，将问题编码为 token-level 表示。
2. 线性投影，将问题表示投影到 256 维。
3. 28 个可学习的 LLaMA 层 query embedding。
4. 4-head cross-attention，使每个层 query 读取问题 token。
5. 两层 Transformer encoder，在 28 个层位置之间建模全局依赖。
6. segmentation head，预测每个位置是否开始新的连续层段。
7. operation head，在层段起点预测 `skip`、`keep` 或 `repeat`。

LLaMA Predictor 约有 2.11M 个参数，论文报告其大小约为 3.61B LLaMA 的 0.0586%。

### 4.2 官方仓库起始训练配置

| 配置 | 值 |
|---|---:|
| Epochs | 10 |
| Batch size | 128 |
| Learning rate | `5e-4` |
| Optimizer | AdamW |
| Learning-rate schedule | cosine |
| Warmup steps | 10 |
| 每问题最多有效路径 | 50 |
| 每问题权重归一化 | 开启 |
| 原始路径处理 | 有更短有效路径时保留但降权 |
| 原始路径权重 | 0.30 |
| Beam size | 5 |
| Top paths | 5 |
| Seed | 42 |
| Predictor hidden size | 256 |
| Predictor heads | 4 |
| Predictor encoder blocks | 2 |
| 自动混合精度 | 官方示例未开启 |
| AdamW weight decay | 默认 0.0 |
| AdamW epsilon | 默认 `1e-8` |

论文 Appendix D 说明作者实际在验证集上搜索 learning rate
`{1e-4, 3e-4, 5e-4, 8e-4, 1e-3, 3e-3}`、batch size `{32,128,256}` 和
epochs `{3,10}`。官方 README 只给出一组可运行的起始配置，没有公布五个 LLaMA Predictor
各自最终选中的超参数。因此当前统一使用 README 配置是可核验的代码复现起点，但不能声称
它就是作者五个难度各自的最终最优配置。

### 4.3 五个本地 Predictor 的训练输入

| Predictor | 训练问题 | 有标签训练问题 | 零有效路径 | cap=50 后训练样本 | 验证问题 | 有标签验证问题 | 验证样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DM-1 | 938 | 821 | 117 | 22,779 | 187 | 159 | 4,067 |
| DM-2 | 938 | 773 | 165 | 18,588 | 187 | 153 | 3,681 |
| DM-3 | 938 | 726 | 212 | 17,957 | 187 | 148 | 3,962 |
| DM-4 | 938 | 729 | 209 | 18,316 | 187 | 131 | 3,576 |
| DM-5 | 938 | 617 | 321 | 15,628 | 187 | 125 | 3,657 |
| 合计 | 4,690 | 3,666 | 1,024 | 93,268 | 935 | 716 | 18,943 |

训练脚本将五个 Predictor 作为五个独立任务分配到两张 A800，而不是对单个 Predictor 做
分布式数据并行：GPU 0 顺序训练 DM-1、DM-2，GPU 1 顺序训练 DM-3、DM-4、DM-5。
每个难度保存独立 checkpoint、日志和配置清单。

## 5. `PolarDataset` 与官方仓库的逐项对照

本地 `polar/data.py` 相对官方提交 `30d0efd` 只增加了 `split_filter`，用记录中的
`search_metadata.data_split` 选择公开版 train、validation、test。选出样本后的路径处理、
解析、采样和权重逻辑未修改。

| 行为 | 官方仓库 | 当前复现 | 结论 |
|---|---|---|---|
| 训练正标签来源 | 只读 `final_valid_transitions` | 相同 | 对齐 |
| `final_invalid_transitions` | 不用于训练 | 相同 | 对齐 |
| 没有有效路径 | `continue`，不产生训练样本 | 相同 | 对齐 |
| 是否强加原始路径正例 | 否 | 否 | 对齐 |
| 每题路径超过 50 | 固定 seed 随机采样 50 条 | 相同 | 对齐 |
| 路径解析 | canonical segmentation，层段最大长度 4 | 相同 | 对齐 |
| repeat 标签 | 每个层段额外执行一次 | 相同 | 对齐 |
| 无法解析的路径 | 跳过 | 相同 | 对齐；当前实际失败为 0 |
| 每个有效路径 | 构造一个训练 example | 相同 | 对齐 |
| 每题权重 | 可按有效路径数归一化 | 已开启 | 对齐 README |
| 原始路径与更短路径同时有效 | 保留原始路径并乘 0.30 | 相同 | 对齐 README |
| 随机种子 | 42 | 42 | 对齐 |
| 论文固定位置切分 | `[0:1250]`、`[1250:1500]`、`[1500:2000]` | 不适用于 1,500 条公开数据 | 有意差异 |
| 公开版切分 | 官方无此数据 | 按记录 split 筛选 `938/187/375` | 必需的适配 |

“零有效路径”不等于负样本。官方 Predictor 是从已知正确程序学习分段和操作的监督模型，
并没有为搜索失败的问题生成“所有路径都错误”的分类标签。当前复现保持了这一行为。

## 6. 数据规模与论文数据的差异

论文使用每档 2,000 个问题，共 10,000 个问题；每档切分为 1,250 train、250 validation、
500 test。公开的 DART-Math query-info 只能恢复 7,500 个唯一 MATH train query，因此当前
独立重建按 pass rate 从高到低分成五个等大的 1,500 问题区间。

| 项目 | 论文 | 当前公开重建 | 差异 |
|---|---:|---:|---:|
| 每档总问题 | 2,000 | 1,500 | -25.0% |
| 每档训练 | 1,250 | 938 | -25.0% 左右 |
| 每档验证 | 250 | 187 | -25.2% |
| 每档测试 | 500 | 375 | -25.0% |
| 五档总问题 | 10,000 | 7,500 | -25.0% |
| 五档训练 | 6,250 | 4,690 | -25.0% 左右 |
| 五档验证 | 1,250 | 935 | -25.2% |
| 五档测试 | 2,500 | 1,875 | -25.0% |

当前数据不是作者未公开固定 split，五档也不是作者原始 DM-1 至 DM-5 的逐题恢复。因此论文
表中的准确率是参考目标，不是当前公开重建的逐点验收门槛。

## 7. 当前 LLaMA 标签数据分析

分析对象：训练和验证部分共 5,625 个问题。测试集没有运行 MCTS，也没有 oracle 路径。

### 7.1 搜索成功率与零有效路径

| 难度 | 问题数 | 有至少一条有效路径 | 搜索成功率 | 零有效路径 | 零路径率 | 有效路径总数 |
|---|---:|---:|---:|---:|---:|---:|
| DM-1 | 1,125 | 980 | 87.11% | 145 | 12.89% | 44,657 |
| DM-2 | 1,125 | 926 | 82.31% | 199 | 17.69% | 33,575 |
| DM-3 | 1,125 | 874 | 77.69% | 251 | 22.31% | 31,851 |
| DM-4 | 1,125 | 860 | 76.44% | 265 | 23.56% | 30,867 |
| DM-5 | 1,125 | 742 | 65.96% | 383 | 34.04% | 26,677 |
| 合计 | 5,625 | 4,382 | 77.90% | 1,243 | 22.10% | 167,627 |

全部路径无重复，全部能被官方 canonical parser 解析。1,330 个问题拥有超过 50 条有效路径，
训练时会按官方逻辑使用固定 seed 采样到 50 条。

### 7.2 有效路径结构

| 难度 | 原始路径 | skip-only | repeat-only | skip+repeat | 平均执行层数 | 中位数 | P10--P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DM-1 | 352 | 13,851 | 30,454 | 0 | 29.10 | 30 | 26--32 |
| DM-2 | 286 | 10,311 | 22,978 | 0 | 29.10 | 30 | 26--32 |
| DM-3 | 263 | 9,953 | 21,635 | 0 | 29.07 | 29 | 26--32 |
| DM-4 | 289 | 9,060 | 21,518 | 0 | 29.17 | 30 | 26--32 |
| DM-5 | 303 | 6,125 | 20,249 | 0 | 29.47 | 30 | 27--32 |
| 合计 | 1,493 | 49,300 | 116,834 | 0 | 29.17 | 30 | 约 26--32 |

总体路径组成是原始路径 0.89%、skip-only 29.41%、repeat-only 69.70%、联合路径 0%。
repeat-only 占比很高并不能单独证明 LLaMA 更偏好 recurrence，因为当前搜索只访问根节点的
单编辑程序，路径空间覆盖本身存在偏差。

### 7.3 搜索树诊断

| 指标 | DM-1 至 DM-5 的结果 |
|---|---:|
| 每题根节点合法动作 | 212 |
| 每题 simulation | 200 |
| 完成 200 次 simulation 的问题 | 5,625 / 5,625 |
| 最大树深为 1 的问题 | 5,625 / 5,625 |
| UCB selection 次数为 0 的问题 | 5,625 / 5,625 |
| 含 skip+repeat 的有效标签 | 0 / 167,627 |

因此当前实现虽然保留了 MCTS 数据结构和回传逻辑，但在本次预算下退化为根节点的随机顺序
单编辑枚举，没有实际进入多步 tree policy。这个限制必须在正式 PoLar 训练前解决，或者把
当前实验明确命名为 `single-edit supervision baseline`。

### 7.4 与 2025 preliminary MCTS 的进一步核查

对照来源：[Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs](https://arxiv.org/pdf/2507.07996)

| 项目 | 2025 preliminary paper | 当前重建 | 判断 |
|---|---|---|---|
| 每题 simulations | 200 | 200 | 对齐 |
| 路径长度惩罚 | 5.0 | 5.0 | 对齐 |
| 随机探索概率 | 0.1 | 0.1 | 数值对齐 |
| 0.1 概率选择对象 | `random unexplored child` | 已展开 children 中随机选择 | 不对齐 |
| 搜索过程 | 从初始路径进行多轮 skip/repeat 编辑 | 只完成根节点单次编辑 | 不对齐 |
| block size | `k` 属于 1 至 4 | 1 至 4 | 对齐 |
| repeat count | diagnostic CoLa 中 `r` 属于 1 至 4 | Predictor 标签固定额外重复一次 | 有意收窄 |
| 联合搜索空间 | 同一搜索允许 skip 与 recurrence | 允许该语言，但本次没有到达第二层 | 形式允许，实际未覆盖 |
| 最终路径长度上限 | 论文说有限制，但没有给数值 | Predictor parser 语言提供结构上界 | 无法逐项对齐 |
| child 生成和限宽 | 未说明 | 枚举全部合法 start、block 和 operation | 独立选择 |

preliminary paper 的 action 描述使用“skip 下一段”或“repeat 下一段”的措辞，但没有说明
child 是否包含所有起点、是否只从当前 cursor 产生操作，也没有公布 progressive widening、
每节点 child 上限或 action prior。作者的 200 次搜索能够报告联合空间的显著收益，并明确称为
多轮编辑，所以其实际实现必然避免了“212 个根动作耗尽全部预算”的退化；具体机制没有公开。

官方 `tianyi-lab/PoLar` 仓库只有 `main` 分支、没有 tag。最初的 `Release POLAR code`
提交与后续历史均只包含 Predictor、在线层路径执行器和数学 evaluator，没有 MCTS 源文件。
README 也明确说代码发布聚焦于从已发现程序训练的 Predictor。因此无法从官方仓库恢复作者
2025 MCTS 的 child expansion 代码。

## 8. 与论文表 1 的对照

论文表 1 报告 LLaMA 的标准路径 Base accuracy 和 Skip&Loop 搜索准确率。当前的 Base 是
5,625 个训练/验证问题上标准 28 层路径的正确率；当前“搜索成功率”是至少找到一条正确的
单编辑路径的比例。因为数据 split、问题组成和搜索深度均不同，下面的差值只用于定位偏差，
不能解释为复现误差或算法提升。

| 难度 | 论文 Base | 当前 Base | 当前减论文 | 论文 Skip&Loop | 当前任一有效路径 | 当前减论文 |
|---|---:|---:|---:|---:|---:|---:|
| DM-1 | 37.9 | 31.29 | -6.61 | 84.7 | 87.11 | +2.41 |
| DM-2 | 28.1 | 25.42 | -2.68 | 72.3 | 82.31 | +10.01 |
| DM-3 | 23.2 | 23.38 | +0.18 | 65.2 | 77.69 | +12.49 |
| DM-4 | 22.8 | 25.69 | +2.89 | 57.0 | 76.44 | +19.44 |
| DM-5 | 27.1 | 26.93 | -0.17 | 59.1 | 65.96 | +6.86 |
| 五档宏平均 | 27.82 | 26.54 | -1.28 | 67.66 | 77.90 | +10.24 |

主要观察：

1. 当前 Base 宏平均比论文低 1.28 个百分点，但不同难度方向不一致，说明 difficulty 划分
   和问题组成的影响大于一个统一的模型偏移。
2. 当前搜索成功率反而普遍高于论文 Skip&Loop，不能据此声称搜索更好。当前只统计 train 和
   validation，且公开难度分段不同；现有搜索还不是多步 Skip&Loop MCTS。
3. DM-5 的 Base 高于 DM-3 和 DM-4，与论文中 LLaMA 的非单调有效难度趋势相似，但当前五档
   是由公开 DART pass rate 而不是 LLaMA correctness 定义，不能把这种相似视为复现证据。

## 9. 论文表 2：第一阶段最终要报告的指标

论文表 2 的 LLaMA PoLar pass@k 参考值如下。当前复现列需要在五个 Predictor 训练完成并对
每档 375 个未搜索测试问题严格在线执行 top-5 路径后填写。

| k | 论文 DM-1 | 论文 DM-2 | 论文 DM-3 | 论文 DM-4 | 论文 DM-5 | 当前公开复现 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 46.2 | 30.2 | 28.2 | 28.8 | 30.2 | 待训练与在线评估 |
| 2 | 56.6 | 37.4 | 34.8 | 32.8 | 36.6 | 待训练与在线评估 |
| 3 | 62.8 | 42.8 | 39.4 | 35.6 | 40.2 | 待训练与在线评估 |
| 4 | 66.8 | 45.6 | 42.6 | 38.0 | 42.8 | 待训练与在线评估 |
| 5 | 68.4 | 48.0 | 46.0 | 40.4 | 45.8 | 待训练与在线评估 |

严格评估规则：

1. 每档只评估记录为 `test` 的 375 个问题。
2. 测试记录的 `final_valid_transitions` 和 `final_invalid_transitions` 都为空。
3. Predictor 输出的 top-5 路径全部在冻结 LLaMA 上在线执行。
4. 禁止用 MCTS cache 或 ground-truth oracle 直接判定测试路径正确。
5. 只接受直接输出的 `\boxed{ANSWER}`，使用官方 `dart_math` 数学等价性判断。
6. 一次 top-5 在线运行即可从前缀计算 pass@1 至 pass@5，不重复调用模型。

论文每档测试集是 500，当前只有 375。最坏情况下二项比例的 95% 误差范围约为当前
`±5.1` 个百分点、论文 `±4.4` 个百分点，因此小于约 1 个百分点的差别不应过度解释。

## 10. 当前文件与运行位置

| 内容 | 位置 |
|---|---|
| 重建数据清单 | `data/redm-public/manifest.json` |
| 原始 MCTS 运行根目录 | `/mnt/afs/liyiwei/PoLar/outputs/full/mcts_labels_public_2gpu_b192_6b80844_20260727_0752` |
| Predictor 输入目录 | `/mnt/afs/liyiwei/PoLar/outputs/full/predictor_data_public_5625_8c0ab40_20260728` |
| Predictor 数据清单 | 上述目录中的 `predictor_data_manifest.json` |
| 双卡训练脚本 | `scripts/run_predictor_training_2gpu.py` |
| 双卡评估脚本 | `scripts/run_predictor_eval_2gpu.py` |
| Predictor 数据准备 | `scripts/prepare_predictor_data.py` |

当前远端缺少 `Qwen/Qwen3-Embedding-0.6B` cache，因此 Predictor smoke 和正式训练尚未开始。

## 11. Limitations

### 11.1 数据无法逐题恢复论文 split

公开 query-info 只有 7,500 个唯一 MATH train query，论文使用 10,000 个问题。当前五档是
独立重建，不是作者 split。该差异同时影响 Base accuracy、MCTS 成功率、训练标签数量和最终
pass@k。

### 11.2 当前监督是单编辑，不是完整多步 MCTS

这是目前最严重的限制。212 个根动作超过 200 次预算，导致搜索从未进入树的第二层，联合
skip+repeat 标签为 0。论文强调程序级组合，并在 Appendix B 将 action 定义为对当前程序
继续修改；当前标签没有覆盖这一核心能力。

### 11.3 2026 MCTS 细节未完整公开

论文没有给出所有实现选择，例如如何控制巨大 branching factor、一次 expansion 生成多少
child、是否 progressive widening、root action prior 和完整超参数。当前 200 simulations、
长度惩罚 5.0、随机探索 0.1 来自 preliminary 配置，运行清单已经明确标注，不能声称是
2026 作者实验的确定设置。2025 preliminary paper 虽然明确写了 0.1 概率选择
`random unexplored child`，但仍没有给出 unexplored child 集合的构造和限宽方式。当前实现
随机选择已展开 child，属于需要修正的确定偏差。

### 11.4 README 配置不等于作者五个最终超参数

官方 README 给出统一的 `10 epochs / batch 128 / lr 5e-4` 命令，但论文说最终配置经验证集
网格选择，未公布每档的选择结果。统一配置应报告为“official README starting config”。

### 11.5 零有效路径问题不会参与 Predictor 训练

当前 22.10% 的已搜索问题没有有效路径，DM-5 达 34.04%。这与官方 loader 一致，但意味着
Predictor 的监督分布偏向 MCTS 能成功的问题；难题和搜索失败问题被系统性排除。

### 11.6 每题最多 50 条路径会改变标签分布

有 1,330 个问题超过 50 条有效路径。固定 seed 采样保证可复现，但并不保证对 skip、repeat、
路径长度或奖励边界进行分层平衡。大量 repeat-only 路径可能支配训练目标。

### 11.7 测试规模较小

每档 375 而不是 500，使 pass@k 的统计波动更大。报告应同时给出正确题数、比例和置信区间，
不要只比较一个小数点后的差异。

### 11.8 尚无 Predictor 结果

目前只能比较数据、Base 和搜索标签。论文表 2 的 PoLar pass@1--5 必须等五个 checkpoint 和
严格在线评估结束后再填，不能用训练 MCTS 搜索成功率代替 Predictor 指标。

### 11.9 当前只做域内 LLaMA

本阶段不覆盖论文其他三个基础模型，也不覆盖 ASDiv、MAWPS、MMLU-Pro 等 OOD 结果。这样
有利于先隔离数据、搜索、Predictor 和在线路径执行的误差来源，但报告结论不能外推到论文的
全部模型和数据集。

## 12. 下一步复现决策与改进路线

### 12.1 在正式训练前先修复搜索覆盖

1. 保持 LLaMA、公开数据、直接答案 prompt、官方 evaluator 和 50-token 输出不变。
2. 设计并记录 branching-factor 控制，使 200 次预算能够进入多步 tree policy；候选方案包括
   progressive widening 或按层段结构生成有限 child。
3. 在小规模样本上验证 `maximum_tree_depth_reached > 1`、`tree_policy_selection_count > 0`，并
   确认出现 parser-compatible 的 skip+repeat 程序。
4. 比较 50、100、200 simulations 下搜索成功率、程序类型和每次 simulation 耗时。
5. 通过 smoke 后重新生成五档 train/validation 标签。

现有单编辑标签应保留，作为后续消融实验：比较“单编辑监督”与“多步 MCTS 监督”对 Predictor
pass@k 的影响。

### 12.2 官方 Predictor 基线

1. 下载并固定 Qwen3-Embedding-0.6B revision。
2. 先用 batch 128、少量样本跑训练 smoke，检查显存、吞吐、loss 和 checkpoint。
3. 分别训练 DM-1 至 DM-5 五个 Predictor。
4. 每档选择最低 validation loss checkpoint。
5. 对五个 375 问题测试集严格在线执行 top-5 路径。
6. 将 pass@1--5、执行层数、skip/keep/repeat 比例和延迟写回本报告。

### 12.3 基线完成后的改进方向

在不改变测试集和评估规则的前提下，依次研究：

1. 更均衡的 MCTS child expansion，减少根节点动作数造成的浅搜索。
2. 路径类型分层采样，避免 repeat-only 标签数量主导训练。
3. 缓存冻结问题编码器表示，减少重复 Qwen 前向但保持训练结果不变。
4. 对比一个共享五档 Predictor 与五个独立 Predictor；该项属于方法改进，不是官方复现。
5. 对零有效路径问题研究失败感知或无监督目标；该项也必须作为改进实验单独报告。

所有改进都必须在同一公开 split、同一 LLaMA revision、同一在线 evaluator 下与官方结构基线
比较，不能用 MCTS oracle 搜索率替代部署时 Predictor pass@k。

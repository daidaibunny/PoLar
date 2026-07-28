# 从 PoLar 复现到多模态检索路径学习：阶段汇报 V2

更新日期：2026-07-28

研究目标：为冻结的多模态大模型发现可部署的 skip、repeat 和组合执行路径，生成
Query 路径监督标签，并定义这些路径造成的 hidden-state drift。

## 1. 从 PoLar 出发：先复现 MCTS，但明确公开缺口

[PoLar 2026](https://arxiv.org/pdf/2606.06574) 将标准 Transformer 前向定义为根路径，
通过蒙特卡洛树搜索对当前路径的连续层段执行 skip 或 repeat。一个程序在数学题上输出正确
答案就被视为 valid program。论文给出的树策略为：

$$
\operatorname{UCB}(\pi)
=
\frac{R(\pi)}{v(\pi)}
+c\sqrt{\frac{\ln V}{v(\pi)}}
-\lambda\frac{|\pi|}{D}.
$$

其中，`R` 是累计二值奖励，`v` 是节点访问次数，`V` 是总模拟次数，`D` 是标准模型
深度。动作的 block 长度和 repeat 次数均不超过 4。PoLar Predictor 最终只允许每个连续
层段执行 skip、keep 或额外 repeat 一次。

### 1.1 论文和仓库没有完整给出什么

| 项目 | 已公开内容 | 无法从公开材料恢复的内容 |
|---|---|---|
| 2026 PoLar 论文 | 状态、skip/repeat 动作、二值奖励、UCB 形式、算法框架 | 模拟次数、`c`、`lambda` 的实验数值；child 如何生成和限宽；完整 rollout、去重与终止实现 |
| [2025 preliminary paper](https://arxiv.org/pdf/2507.07996) | 每题 200 次模拟、路径惩罚 5.0、以 0.1 概率选 random unexplored child | `c` 的数值；unexplored child 集合如何构造；最终路径长度上限；完整源码 |
| [官方 PoLar 仓库](https://github.com/tianyi-lab/PoLar) | Predictor 训练、beam 解码、路径执行和数学答案 evaluator | MCTS 标签生成源码、论文使用的 `merged_mcts_samples.json`、预训练 Predictor checkpoint |

因此，我们使用的 `200 / 5.0 / 0.1` 是 2025 preliminary defaults，不是 2026 实验的
完整官方配置；当前 `c = sqrt(2)` 是独立重建选择。论文没有公开足以逐项复刻搜索树行为的
信息，后续结果必须表述为 independent reconstruction。

截至 2026-07-28，官方仓库中请求发布
[Predictor checkpoint](https://github.com/tianyi-lab/PoLar/issues/1) 和
[Predictor 训练数据](https://github.com/tianyi-lab/PoLar/issues/2) 的两个 Issue 均保持 open，
页面没有作者团队回复。准确表述是“当前未公开且请求尚未获回复”，不能推断作者未来不会提供。

### 1.2 DART-Math 并没有公开 PoLar 的五档 10,000 题划分

[DART-Math-Hard 数据说明](https://huggingface.co/datasets/hkust-nlp/dart-math-hard/blob/main/README.md)
中的 difficulty-aware 指：根据每道题的 difficulty score 决定需要采样多少条正确 response；
Prop2Diff 会给更难的题分配更多采样次数。它没有定义 PoLar 论文中的五个离散
`DM-1` 至 `DM-5` 问题集合。

公开数据的实际结构是：

- [dart-math-pool-math](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math)
  有 1,615,233 行 response；同一问题被重复采样，不能把每一行当成独立题目。
- [dart-math-pool-math-query-info](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math-query-info)
  只有 7,500 行唯一 MATH train query，字段包括原始 MATH `level`、`domain` 和连续
  `pass_rate`。
- PoLar Appendix D 声称每档 2,000 题，共 10,000 题，每档再切成
  1,250 train、250 validation、500 test，但没有发布题目 ID、分档规则或 split。

所以，公开 DART-Math 有难度信号，但没有 PoLar 所用的五个 2,000 题分组。据现有公开材料
推断，PoLar 的五档是作者自行构建但未公开映射的实验划分，无法从 DART-Math-Hard README、
response pool 或 query-info 逐题恢复。我们把 7,500 个唯一 query 按公开 `pass_rate` 分成五个
等量区间，只能称为公开数据上的独立重建，不能称为官方 `DM-1` 至 `DM-5`。

### 1.3 论文实际是每个难度训练一个 Predictor

PoLar 的域内实验在每个难度内独立 train/test。官方代码的 `--target_diff` 一次读取一个
`dart-math-diff-k/merged_mcts_samples.json` 并产生对应 checkpoint。因此 LLaMA 的域内结果
对应五个独立 Predictor，而不是一个 Predictor 同时处理任意难度。论文把五档训练集合并只用于
跨数据集的 out-of-distribution 实验。

### 1.4 我们当前完成到哪里

- 基础模型固定为 `meta-llama/Llama-3.2-3B-Instruct`，模型 revision 固定，所有权重冻结。
- 修复后的多步 MCTS smoke 在 20 题、每题 20 次模拟中到达最大树深 5；发现 17 条答对且
  能被官方 Predictor parser 解析的 skip+repeat 路径，同时覆盖纯 skip 和纯 repeat。
- V2 数据从公开重建的五个区间各取 100 个唯一问题，共 500 题；固定种子 42，每题执行
  200 次模拟。
- 截至 2026-07-28 06:34 UTC，双 A800 优化续跑已完成 DM-1 的 100 题，DM-2 正在运行；
  任务完成后将自动合并五组标签并运行 500 题校验。
- 这 500 题仍以数学答案正确性为奖励，只用于验证 PoLar 式路径搜索和标签生成链路，不是
  最终的多模态检索标签。

## 2. 我们真正的任务：固定 Candidate 多路径，Query 选择一个路径

设冻结的多模态模型有标准路径 `pi_0`，离线选出有限路径库：

$$
\mathcal{P}=\{\pi_0,\pi_1,\ldots,\pi_{n-1}\}.
$$

对每个 Candidate，预先计算并保存全部 `n` 个路径向量；每条路径建立一个独立 Candidate
索引。在线时 Predictor 只观察 Query，输出一个路径 ID；Query 沿该路径编码，并且只查询同一
路径的 Candidate 索引：

$$
z_q^{\pi}=\operatorname{Norm}(E_{\pi}(q)),
\qquad
z_c^{\pi}=\operatorname{Norm}(E_{\pi}(c)),
$$

$$
s_{\pi}(q,c)=\left(z_q^{\pi}\right)^{\top}z_c^{\pi}.
$$

这带来一个部署约束：Candidate 只能预计算有限个索引，所以 Predictor 不能在测试时任意生成
从未见过的新路径。PoLar 的开放式程序预测在本项目中应改为“从固定路径库选择路径 ID”，除非
未来接受在线重算全部 Candidate。

## 3. 检索收益驱动的覆盖式 MCTS

### 3.1 为什么不能用相似度提升作为 reward

单独提高 Query 与某个正样本的 cosine similarity，不保证该正样本在完整 Candidate gallery
中的排名提高；错误 Candidate 可能提升得更多。路径 reward 必须来自完整排序或可靠的困难负样本
近似，而不是单个正样本对。

对 Query `q`，令 `R(q)` 是相关 Candidate 集合，`H(q)` 是当前路径下排名最高的一组错误
Candidate。定义连续的困难负样本间隔：

$$
m_{\pi}(q)
=
\frac{1}{|\mathcal{R}(q)|}
\sum_{c^{+}\in\mathcal{R}(q)}s_{\pi}(q,c^{+})
-
\tau
\log
\left(
\frac{1}{|\mathcal{H}(q)|}
\sum_{c^{-}\in\mathcal{H}(q)}
\exp\frac{s_{\pi}(q,c^{-})}{\tau}
\right).
$$

每个 Query 的路径效用以排序指标为主，连续间隔只负责降低离散指标的并列和噪声：

$$
u(q,\pi)
=
\operatorname{nDCG@K}(q,\pi)
+\epsilon_m
\sigma\left(
\frac{m_{\pi}(q)-m_{\pi_0}(q)}{\tau_m}
\right)
-\epsilon_c\frac{|\pi|}{D}.
$$

其中，`epsilon_m` 和 `epsilon_c` 必须足够小，不能推翻 nDCG 的主要排序。单正样本数据可将
nDCG 替换为 reciprocal rank；最终同时报告 Recall、mean average precision 和 nDCG。

### 3.2 先搜索共享路径库，再生成 Query 标签

直接为每个 Query 搜索任意路径不可部署，因为每条新路径都需要重新编码整个 Candidate gallery。
更合适的是两阶段算法。

第一阶段构建共享路径库。初始库只含标准路径：

$$
\mathcal{B}_0=\{\pi_0\}.
$$

给定已有路径库 `B`，新路径的 reward 是它相对于现有路径库为训练 Query 带来的边际检索增益：

$$
R_{\mathrm{cov}}(\pi\mid\mathcal{B})
=
\frac{1}{|\mathcal{Q}_{\mathrm{search}}|}
\sum_{q\in\mathcal{Q}_{\mathrm{search}}}
\left[
\max\left(
u(q,\pi),
\max_{b\in\mathcal{B}}u(q,b)
\right)
-
\max_{b\in\mathcal{B}}u(q,b)
\right].
$$

这个 reward 会优先寻找能解决现有路径失败 Query 的互补路径，而不是重复发现对同一批简单
Query 有效的相似路径。

MCTS 的节点仍是当前 layer program，动作仍是对当前程序连续层段做 skip 或 repeat。为和
Candidate 路径库以及 PoLar Predictor-compatible 语言一致，首轮设置为：block 长度 1 至 4，
每段最多额外 repeat 一次，允许多步组合 skip+repeat。

为避免 200 次预算被数百个根动作耗尽，每个节点采用 progressive widening：

$$
|\mathcal{A}_{\mathrm{open}}(v)|
\le
\left\lceil
k_{\mathrm{pw}}N(v)^{\alpha_{\mathrm{pw}}}
\right\rceil,
\qquad
0<\alpha_{\mathrm{pw}}<1.
$$

新 action 按 skip/repeat、层位置和 block 长度分层抽样；以固定概率探索未访问 action，其余
时间沿最高 UCB child 下行。树中每个新路径只计算一次 Candidate embedding 并缓存。

考虑到完整 gallery 编码昂贵，搜索时先使用固定的代表性 Candidate 子集和动态困难负样本池；
每轮 MCTS 产生的 top paths 必须在完整 gallery 上重新编码和验证。只有完整 gallery 的验证集
增益显著为正时，才把路径加入 `B`。重复此过程，直到路径数达到存储预算 `n`，或边际增益不再
显著。

第二阶段生成 Query 标签。在固定路径库上对每个训练 Query 做完整 gallery 检索，得到 oracle
路径：

$$
\pi^{*}(q)=\operatorname*{arg\,max}_{\pi\in\mathcal{B}}u(q,\pi).
$$

当多个路径接近并列时，不强制产生不稳定的单一类别，而生成 soft label：

$$
y(q,\pi)
=
\frac{
\exp\left(
[u(q,\pi)-\max_{b\in\mathcal{B}}u(q,b)]/\tau_y
\right)
}{
\sum_{b\in\mathcal{B}}
\exp\left(
[u(q,b)-\max_{b'\in\mathcal{B}}u(q,b')]/\tau_y
\right)
}.
$$

标签同时记录最优路径、次优路径、效用差和 bootstrap 置信度。MCTS 只用于 train；validation
用于路径库、reward 权重和 Predictor 选择；test 不运行 oracle 搜索，只评价 Query Predictor
选择的路径。这是防止标签泄漏的必要边界。

### 3.3 Predictor 的输入时机

若 Predictor 读取标准路径的最终 Query embedding 后才选路，就已经支付了一次完整前向成本。
若目标包含计算效率，Predictor 应读取共享前缀某一层的 Query hidden state，并且可选路径只能
修改该路由层之后的执行。若当前阶段只研究检索质量，可以先允许完整标准 embedding 作为
Predictor 输入，但必须把额外计算量单独报告。

## 4. skip/repeat 导致的偏移距离

偏移不能只用一个数描述。至少需要记录“中间 hidden state 偏移”“最终对齐空间偏移”和
“检索排序偏移”。大幅偏移可能提高正负样本间隔，因此 drift 大不等于 drift 坏。

### 4.1 中间 hidden state drift

路径不同后，不能直接比较不同语义阶段的层编号。应在 skip/repeat segment 结束、路径重新进入
相同官方层序后的 rejoin layer 比较同一 token 位置。设标准路径和路径 `pi` 在 canonical
layer `l` 的 hidden states 分别为 `h_0` 和 `h_pi`：

$$
d_{\mathrm{hidden}}^{(l)}(x,\pi)
=
\frac{1}{T}
\sum_{t=1}^{T}
\left[
1-
\cos\left(
\operatorname{LN}(h_{0,l,t}),
\operatorname{LN}(h_{\pi,l,t})
\right)
\right].
$$

同时在 rejoin layer 和最终层记录该值。定义尾部网络对偏移的放大率：

$$
A_{\pi}(x)
=
\frac{
d_{\mathrm{hidden}}^{(D)}(x,\pi)
}{
d_{\mathrm{hidden}}^{(l_{\mathrm{rejoin}})}(x,\pi)+\varepsilon
}.
$$

`A` 小于 1 表示官方尾部在收缩 skip/repeat 注入的偏移；大于 1 表示偏移被继续放大。

### 4.2 最终对齐空间的距离和方向

令 `z_0(x)` 是标准最终输出 `o28` 经官方 pooling、projection 和 L2 normalization 后的向量，
`z_pi(x)` 是修改路径的对应向量。单位球面上的角距离为：

$$
d_{\mathrm{geo}}(x,\pi)
=
\arccos\left(
\operatorname{clip}
\left[
z_0(x)^{\top}z_{\pi}(x),
-1,
1
\right]
\right).
$$

距离只有大小，没有方向。为了表达“从 `o28` 被推向哪里”，使用单位球面在 `z_0` 处的
logarithmic map：

$$
v_{\pi}(x)
=
\operatorname{Log}_{z_0(x)}z_{\pi}(x)
=
\frac{\theta}{\sin\theta}
\left[
z_{\pi}(x)-\cos\theta\,z_0(x)
\right],
\qquad
\theta=d_{\mathrm{geo}}(x,\pi).
$$

`v_pi` 是我们需要保存的 drift vector，其范数就是角距离，方向表示 skip/repeat 相对标准
`o28` 的移动方向。

### 4.3 是否需要向 `o28` 拉回

在球面上沿 drift vector 做可控插值：

$$
z_{\pi,\rho}(x)
=
\operatorname{Exp}_{z_0(x)}
\left(
\rho v_{\pi}(x)
\right)
=
\cos(\rho\theta)z_0(x)
+
\sin(\rho\theta)
\frac{v_{\pi}(x)}{\theta},
\qquad
0\le\rho\le1.
$$

`rho = 0` 完全回到标准 `o28`，`rho = 1` 保留完整 looped path，二者之间是严格保持单位范数
的 pullback。对每条路径只在 validation 上选择：

$$
\rho_{\pi}^{*}
=
\operatorname*{arg\,max}_{0\le\rho\le1}
\operatorname{nDCG@K}
\left(
z_{\pi,\rho}
\right).
$$

若稳定得到小于 1 的最优值，说明该路径提供了有益方向，但位移幅度过大，需要拉回；若最优值
接近 1，则不应仅因为 drift 大而强行回到标准输出。

### 4.4 用检索几何判断 drift 的好坏

路径对一个 Query-Candidate pair 的相似度改变为：

$$
\Delta s_{\pi}(q,c)
=
s_{\pi}(q,c)-s_{\pi_0}(q,c).
$$

真正决定检索是否改善的是正负 margin 的变化：

$$
\Delta m_{\pi}(q)
=
m_{\pi}(q)-m_{\pi_0}(q).
$$

只有 `Delta m`、nDCG、Recall 或 mean average precision 提升，才能把 drift 称为有益。
Query 与正确 Candidate 同向移动可能产生较大的单点 drift，但仍改善排序；只看
`d_geo` 会误判这种情况。

### 4.5 用 drift 限定 Candidate 路径空间

对每条路径，在 validation 上只使用“属于相关集合，且所在 Query 的检索收益不下降”的
Candidate 估计可信半径：

$$
r_{\pi}
=
\operatorname{Quantile}_{1-\alpha}
\left\{
d_{\mathrm{geo}}(c,\pi)
\;\middle|\;
q\in\mathcal{Q}_{\mathrm{val}},
\ c\in\mathcal{R}(q),
\ \Delta m_{\pi}(q)\ge0
\right\}.
$$

Candidate 仍存入对应路径索引，但对超过可信半径的 Candidate 先采用软惩罚，而不是直接删除：

$$
s_{\pi}^{\mathrm{trust}}(q,c)
=
s_{\pi}(q,c)
-\kappa
\left[
\max\left(
0,
d_{\mathrm{geo}}(c,\pi)-r_{\pi}
\right)
\right]^2.
$$

只有当 validation 和 held-out test 都证明硬过滤不降低 Recall 时，才把软约束改成删除 Candidate。
这样，路径 ID 先把检索限定到对应 latent space，drift radius 再限制该空间内不可信的过度偏移。

## 5. 最小实验路线与判定标准

1. PoLar 搜索复现：完成当前 500 题多步 MCTS，验证纯 skip、纯 repeat、skip+repeat、树深、
   parser 成功率和有效路径数量；该阶段只验证搜索机制。
2. 检索存在性实验：冻结多模态模型和 Candidate gallery，比较标准路径、单 skip、单 repeat、
   组合路径；完整 gallery 排名优于标准路径才算成功。
3. 共享路径库搜索：用覆盖式 MCTS 选择少量互补路径，并报告每增加一个 Candidate 索引带来的
   边际 nDCG、Recall、mean average precision、存储和延迟。
4. 标签质量实验：比较 hard label、soft label、置信度过滤，以及随机路径、单编辑路径和
   MCTS 路径；评价 Predictor 对 oracle 路径的逼近程度和最终在线检索结果。
5. Drift 实验：同时报告 rejoin hidden drift、最终角距离、drift vector、margin change 和最优
   pullback 系数；检验 drift 大小能否预测检索退化。
6. 严格测试：测试 Query 不参与 MCTS，不使用 ground truth 选路径；Predictor 只看 Query，
   Candidate gallery 与索引固定。

## 6. 当前结论

PoLar 提供了“用 skip/repeat 搜索样本级执行程序”的出发点，但其 MCTS 实现、监督数据、
Predictor 权重和五档数据映射尚未公开，无法做逐项完全复刻。我们的 500 题任务是在验证一个
可工作的多步 MCTS 重建。

面向多模态检索，核心改变是把二值答案奖励换成完整 gallery 的排序收益，并把 per-query
开放路径搜索改成“先发现有限共享路径库，再为每个训练 Query 生成路径标签”。hidden-state
drift 应同时保存 rejoin-layer 偏移、最终球面距离、球面切向 drift vector 和检索 margin 变化；
是否拉回 `o28` 由 validation retrieval metric 决定，而不是由 drift 大小单独决定。

## 7. 主要来源

- [PoLar 2026 paper](https://arxiv.org/pdf/2606.06574)
- [CoLa 2025 preliminary paper](https://arxiv.org/pdf/2507.07996)
- [PoLar official repository](https://github.com/tianyi-lab/PoLar)
- [PoLar Predictor checkpoint request](https://github.com/tianyi-lab/PoLar/issues/1)
- [PoLar supervision-data request](https://github.com/tianyi-lab/PoLar/issues/2)
- [DART-Math paper](https://arxiv.org/abs/2407.13690)
- [DART-Math official repository](https://github.com/hkust-nlp/dart-math)
- [DART-Math-Hard README](https://huggingface.co/datasets/hkust-nlp/dart-math-hard/blob/main/README.md)
- [DART-Math response pool](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math)
- [DART-Math query-info](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-math-query-info)

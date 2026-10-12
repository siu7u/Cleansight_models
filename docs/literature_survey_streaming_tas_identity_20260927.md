# CleanSight 文献调研：在线/流式 TAS 架构 与 段级标签同一性（segment identity）目标

> 调研日期：2026-09-27（UTC）
> 目标问题：**"整段认错"**（51.6% 真值段多数预测标签错误、78% 帧错误远离边界），而不是边界抖动。
> 约束：144 维手工特征（非 I3D / 非像素）、~3 万帧、9 类 + idle、最稀类仅 18 个真值段、
> 流式滑窗（window≈32）、CPU 侧 p95 延迟预算数毫秒级、已有 `temporal_feed_mode` 对比离线/流式一致性。

---

## 0. 验证方法与诚实声明

**怎么做的**：用 `curl` 直接抓 arXiv abs 页 / CVF OpenAccess 页 / GitHub API，逐篇核对标题、摘要、venue、代码与 license。
参考文献里的每一个 URL 都是我在本次调研中实际抓取过的页面。

**本报告的证据分级**：

| 标记 | 含义 |
|---|---|
| ✅ 已核验 | 我抓到了原始页面（arXiv abs / CVF / ICDM 页），标题与摘要为我亲眼所见 |
| ⚠️ 仅题录 | 只在检索结果里见到标题/ID/日期（arXiv 检索页），**未读到摘要**；或只见到二手页面 |
| ❌ 未核验 | 论文里声称的东西我无法确认（例如代码仓库 404、报出的数字未见原文） |

**明确没能核验的东西（不猜）**：

1. 绝大多数论文的**具体指标数字**我**没有**逐表核对——只有少数写在摘要里、或我下载 PDF 逐行读过的（ProTAS 表 1、DIR-AS 摘要）才给出数字。凡正文只写"摘要未报数字"，就是我确实没在摘要里看到。
2. `Improving Temporal Action Segmentation via Constraint-Aware Decoding`（ICPR 2026）摘要里的代码链接
   `https://github.com/LUNAProject22/CAD` 在 2026-09-27 返回 **HTTP 404**——代码是否公开、license 是什么，**❌ 未核验**。
3. 用户提到的 **"ICF"**：我在 TAS 文献里**没有**检索到以此为名的论文（普通检索命中的全是无关内容）。
   我把这一条理解并展开为"prototype / contrastive / 聚类原型"这一族（TCL、Permutation-Aware、HybridTAS 等）。
   若 "ICF" 是某篇具体论文的缩写，**请补充出处**，我不编造。
4. **HMM/CRF 平滑**在 TAS 里的"专门论文"我没有找到足够强的现代代表作（见 §B2 的说明）。
   我给出的替代是 CAD（Viterbi + 统计结构先验），并明确标注这是替代而非原意。

**本报告包含一次公开的自我纠错**：§2 的"证据反方"由独立子代理用另一套检索（含 PubMed/NCBI eutils、
Crossref、ar5iv 全文）复核，**它推翻了我初稿里的一个过度断言**——
我原本写"架构买结构、不买帧级正确率"，但同特征同划分的对照数据显示帧级准确率**确实**随架构族变化
（TCN 49.1 → Transformer 52.3 → diffusion 56.2）。
修正后的表述见 **§2.5**。此外子代理**证实**了 CleanSight "损失权重已穷尽"的结论在文献上就是预期结果（§2.7），
并发现了**指标口径通胀**这一条对项目影响最大的证据（§2.6）。

---

## Part A — 在线 / 流式架构

### A1 直接血缘：Online TAS 基线（最贴 CleanSight 的一族）

#### 1. OnlineTAS: An Online Baseline for Temporal Action Segmentation ✅
- **Venue/Year**: NeurIPS 2024 · **URL**: https://arxiv.org/abs/2411.01122
- **机制**（摘要原文核验）：核心是 **adaptive memory**（随时间容纳上下文动态变化"adapting to dynamic changes in context"）
  + **feature augmentation module**（用 memory 增强当前帧） + 一个**专门针对在线场景严重过分割的后处理**。
  三个基准上 SOTA。
- **报出的结果**：摘要未给数字。**摘要未报数字**。
- **代码 / license**: https://github.com/QingZhong1996/OnlineTAS — GitHub API 核验：**MIT**，5 stars。
- **对 CleanSight 的适配**：★★★★★ memory 里存的是"帧特征"，我们的帧特征只有 144 维，
  显存/算力成本几乎为零；**延迟上，memory 读写在 CPU 上也是 O(1)**，不会突破 p95 预算。
  真正有价值的是它把"在线场景的过分割"当成一等公民问题——这正好是段级指标（edit / F1@IoU）的主要杀手。

#### 2. Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos (ProTAS) ✅
- **Venue/Year**: CVPR 2024 · **URL**: https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf
- **机制**（我下载 PDF 逐页读过）三件事：
  (a) 把 MS-TCN / ASFormer 改造成 **causal**（causal dilated conv / causal attention），得到 CAS；
  (b) **Action Progress Prediction (APP)**：额外回归"当前动作进行到百分之几"，用进度 refine 分类 logits；
  (c) **Task Graph (TG)**：从训练视频学任务图（所有合法 transcript），推理时**从 logits 里减去违反任务图的惩罚项**
  （原文式 (12)：$y_{t,k}=e^{o_{t,k}-\eta(\alpha^p_{t,k}+\alpha^s_{t,k})}/\sum_{k'}e^{...}$）——这是**实打实的"段序列层面的受限解码"**。
- **报出的结果（表 1，MS-TCN 为底座；我逐行抄自 PDF）**：

| 配置 | 推理 | GTEA Acc / Edit | EgoProceL Acc / Edit | EgoPER Acc / Edit |
|---|---|---:|---:|---:|
| Base | Offline | 76.3 / 79.0 | 69.2 / 56.9 | 83.0 / 85.9 |
| Base | Online | 47.0 / 58.8 | **18.3 / 19.9** | 20.2 / 31.0 |
| CAS | Online | 74.0 / 64.4 | 64.5 / 42.5 | 71.8 / 48.9 |
| CAS+APP | Online | 76.0 / 67.0 | 66.3 / 47.1 | 72.7 / 55.0 |
| CAS+APP+TG | Online | 74.3 / **69.2** | 67.8 / **48.8** | 70.2 / **60.7** |

  这是本篇调研里**最有价值的一组数字**：离线模型直接上线会崩（EgoProceL Edit 56.9→19.9，Acc 69.2→18.3），
  把架构改成因果（CAS）能捞回绝大部分；而**段级结构先验（TG）在 Edit 上再加成**（GTEA 64.4→69.2，
  EgoProceL 42.5→48.8，EgoPER 48.9→60.7），代价是 Acc 几乎不涨甚至略降。
  **这正是 CleanSight 的分工假说**：容量/架构买"段怎么排"，段级先验买"段标签序列对不对"。
- **代码 / license**: https://github.com/Yuhan-Shen/ProTAS — GitHub API 核验：**license = None（未声明）**。
  ⚠️ 无 license 默认即"保留所有权利"，工业项目**不能直接拷贝代码**，只能借鉴算法自行实现。
- **适配**：★★★★★ 但注意它的 task graph 假设"同一个任务存在有限种合法流程"。
  CleanSight 是"清洁 + 抽吸"固定流程（flush → insert → withdraw → sbc），**这个假设大概率成立**，
  而且 task graph 可以从**已有标注的分段序列**直接统计出来，不需要任何新标注。
  推理成本：一次图查表 + 几个 logit 减法，微秒级。

#### 3. End-to-End Streaming Video Temporal Action Segmentation with Reinforcement Learning (SVTAS-RL) ✅
- **Venue/Year**: arXiv 2309.15683 v1 2023-09 / v2 2024-05（comment 写 "submit to TNNLS"）
  · **URL**: https://arxiv.org/abs/2309.15683
- **机制**：明确定义 **STAS（streaming TAS）**任务，并**专门分析把离线 TAS 模型搬到流式的崩坏原因**——
  归因为 **model bias + optimization dilemma**；用端到端建模消除模型偏差，用强化学习缓解优化困境。
- **报出的结果**：摘要称显著优于既有 STAS 模型，并在超长视频 EGTEA 上与 SOTA 离线 TAS 有竞争力。
  **具体数字摘要未给**（我未逐表核验）。
- **代码**：摘要只说 "Code is available at this https URL"，我未取到直链 → ❌ **未核验**。
- **适配**：★★★★☆ 它的价值不在模型，而在**问题定义和诊断框架**：它给了"离线→流式为什么崩"的可引用论述，
  可以直接用来解释 CleanSight `temporal_feed_mode` 里看到的离线/流式 gap。

#### 4. O-TALC: Steps Towards Combating Oversegmentation within Online Action Segmentation ✅
- **Venue/Year**: TAHRI 2024（short / unindexed paper）· **URL**: https://arxiv.org/abs/2404.06894
- **机制**：两个 backbone 无关的训练/推理改动：
  (a) **surround dense sampling** 让训练与推理的 clip 分布对齐（"training vs. inference clip matching"）；
  (b) **Online Temporally Aware Label Cleaning (O-TALC)**，显式抑制在线推理时的过分割。
- **报出的结果**：摘要称优于同类在线工作、并在细粒度数据集上追平许多能用全时域信息的离线模型。**摘要未报数字**。
- **代码**: 摘要未给 → ❌ 未核验。
- **适配**：★★★★☆ (a) 与 CleanSight 的 `temporal_feed_mode` 是**同一个问题的两种叫法**——训练是全序列、推理是滑窗。
  这是最小成本的"先试"项：不换架构、不换损失权重，只改采样/标签清洗。

#### 5. Online Temporal Action Localization with Memory-Augmented Transformer (MATR) ✅
- **Venue/Year**: ECCV 2024 · **URL**: https://arxiv.org/abs/2408.02957
- **机制**：**memory queue** 选择性保留过去的 segment 特征，从而在只输入定长 segment 的前提下用到长时上下文；
  定位时"用当前 segment 预测进行中动作的**结束**时间，用 memory 里的历史估计**开始**时间"。
- **报出的结果**：THUMOS14 / MUSES 上优于既有在线方法，甚至超过部分离线 TAL 方法（摘要原话）。**摘要未报数字**。
- **适配**：★★★☆☆ 任务是 On-TAL（实例定位）而非逐帧分割，但"memory 里存 segment 而不是存帧"的思路
  对 CleanSight 有直接启发：**在 memory 里维护"已确认的段序列"比维护帧特征更省，也更直接服务于 edit / F1@IoU。**

#### 6. TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks ✅
- **Venue/Year**: MICCAI 2020 · **URL**: https://arxiv.org/abs/2003.10751
- **机制**：把 MS-TCN 搬到手术阶段识别，关键点是**因果（causal）dilated 卷积**：
  "Causal, dilated convolutions allow for a large receptive field and **online inference with smooth predictions
  even during ambiguous transitions**"。在腹腔镜胆囊切除两套数据上超过多种 LSTM。
- **适配**：★★★★☆ **这是 CleanSight 现有 MS-TCN++ 的"流式可行性"最直接的理论依据**：
  只要把非因果卷积换成因果卷积，感受野不变、延迟变成 O(1)（每帧只需常数次卷积）。
  ⚠️ 但注意：TeCNO 的"平滑预测"针对的是**过渡帧**，它对"整段认错"没有承诺。

#### 7. MS-TCN / MS-TCN++（基线本身）✅
- MS-TCN: CVPR 2019 · https://arxiv.org/abs/1903.01945
- MS-TCN++: TPAMI 2020 · https://arxiv.org/abs/2006.09220
- **机制**：多阶段（stage）扩张时序卷积逐级 refine 首阶段预测；MS-TCN++ 加 dual dilated layer，
  并**解耦 first stage 与 refining stage 的设计**（"different requirements of these stages"）。
- **适配**：MS-TCN++ 已是 CleanSight 最优。**注意其 refining stage 的非因果性**——
  它是全序列 refine，天生不能在流式里跑；流式要么因果化，要么接受"每帧重跑一个滑窗"。

---

### A2 流式注意力 / 因果 Transformer

#### 8. TeSTra: Real-time Online Video Detection with Temporal Smoothing Transformers ✅
- **Venue/Year**: ECCV 2022 · **URL**: https://arxiv.org/abs/2209.09236
- **机制**：把视频 transformer 的 **cross-attention 用 kernel 视角重写**，引入 box kernel / Laplace kernel 做时间平滑，
  于是 streaming attention **复用大量跨帧计算，每帧只需常数时间更新**；
  "takes in arbitrarily long inputs with **constant caching and computing overhead**"。
- **报出的结果**：2048 帧流式场景下比等价滑窗 transformer **快 6×**；THUMOS'14 与 EPIC-Kitchens-100 上
  online action detection / anticipation SOTA；实时版本在 THUMOS'14 上仅次于一个先前方法。**均为摘要原话。**
- **代码 / license**: https://github.com/zhaoyue-zephyrus/TeSTra — GitHub API 核验：**Apache-2.0**，119 stars。
- **适配**：★★★★☆ **如果 CleanSight 要试 Transformer 家族，这是唯一"改对了地方"的版本**。
  CleanSight 现有结论是"Transformer 显著更差"，但那很可能是因为**用了非因果/全局注意力的 Transformer**，
  在 3 万帧小数据上既过拟合又无法流式。TeSTra 的平滑核在数学上等价于"因果 + 可学习的指数衰减先验"，
  对低维（144 维）输入的计算量极小。

#### 9. StreamFormer: Learning Streaming Video Representation via Multitask Training ✅
- **Venue/Year**: arXiv 2504.20041（v1 2025-04，v2 2025-07）· **URL**: https://arxiv.org/abs/2504.20041
- **机制**：在预训练 ViT 里注入**因果时间注意力（causal temporal attention）**，逐帧处理流、
  保留历史信息、做低延迟决策；配 multitask 视觉-语言对齐训练。
- **适配**：★★☆☆☆ 强依赖像素/视觉 backbone，与 CleanSight 的 144 维手工特征不兼容。
  只作为"因果注意力在视频 backbone 里可行"的旁证。

#### 10. 流式 KV cache / attention sink 一族（旁证，非 TAS）⚠️/✅
- LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval —— https://arxiv.org/abs/2505.15269 ✅
- StreamingVLM: Real-Time Understanding for Infinite Video Streams —— https://arxiv.org/abs/2510.09608 ✅
- Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression —— https://arxiv.org/abs/2512.05081 ✅
- **机制**：把"无限流"压成一个**有界 KV cache**，并用 attention sink / 压缩来保住长程信息。
- **适配**：★☆☆☆☆ **结论：这一族不要用在 CleanSight。** 它们解决的是"视频-语言大模型在无限流上显存爆炸"，
  而 CleanSight 的序列长度是几万帧 × 144 维，一个 32 帧滑窗的状态根本不存在这个问题。
  我把它们列出来是为了**明确排除**：不要被"streaming attention"这个词吸引而引入不必要的复杂度。

---

### A3 状态空间模型（S4 / Mamba / Mamba-2 / Vision-Mamba）

> **结论先行：把 Mamba/SSM 用在"逐帧 TAS"上的公开工作是稀薄的**——
> 但在 OAD（§11）、TAD（§12）、骨架 TAS（§13）上都有正式工作，所以这条路本身成立，只是**没人占过这个组合位**。

#### 11. Backtrace Mamba / MOAD: Reviving Critical Temporal Contexts via Hierarchical Memory Compression for Online Action Detection ✅
- **Venue/Year**: AAAI 2026 · **URL**: https://ojs.aaai.org/index.php/AAAI/article/view/38139
  （AAAI OJS 页面的 DC.Description 元数据；已核验）
- **机制**（摘要逐字核验）：提出 **Mamba-based OAD 框架 MOAD**。动机原话是既有方法
  "either suffer from **slow training and limited temporal receptive fields**, or face **high computational costs and
  delayed inference**, lacking the capability to tackle **extra-long video inputs**"。
  三件套：
  (a) **hierarchical memory**——按 **motion-aware similarity** 智能地**只存"高价值"的 scene/action 帧**，
  以 online 方式保留关键历史；
  (b) **memory quantization** 压缩已存历史特征，进一步降存储；
  (c) **temporal soft pruning**——基于 memory bank 动态剔除冗余特征，在降时序冗余的同时保持时序连贯。
  四个基准上显著优于既有方法。
- **报出的结果**：摘要只写 "significantly outperforms existing methods"，**未给数字**。
- **代码**：摘要未给 → ❌ 未核验。
- **适配**：★★★★☆ 这是本节**最贴 CleanSight 的一篇**，理由有三：
  (i) 它把 OAD 的痛点定义为"**延迟 + 超长输入**"，与我们的流式 p95 约束同题；
  (ii) "按语义相似度只存高价值帧"这个机制，对**段级身份**是正向的——它让 memory 里保留的是
  "上一个不同动作的帧"，而不是一堆重复的空闲帧，这恰好是判断"这一段到底是什么"所需的上下文；
  (iii) 它仍在 OAD（当前帧分类）而非逐帧分割，段级指标不是它的目标。
  ⚠️ 风险：Mamba 在 CPU 上（尤其无 CUDA）的算子支持与延迟未经验证，p95 预算下必须先做 latency spike。

#### 12. Efficient Spatial-Temporal Focal Adapter with SSM for Temporal Action Detection ✅
- **Venue/Year**: arXiv 2604.09164（2026-04）· **URL**: https://arxiv.org/abs/2604.09164
- **机制**（摘要已读）：诊断 CNN / Transformer 在长视频上的两个瓶颈是
  "**feature redundancy** and **degraded global dependency modeling**"，并把 SSM 引进来——
  摘要原话强调 SSM "offer a promising alternative with **linear long-term modeling** and robust global temporal reasoning"；
  具体是在预训练层里插入 **Efficient Spatial-Temporal Focal (ESTF) Adapter**。
- **适配**：★★★☆☆ 任务是 TAD（实例检测）而非逐帧分割，但它给出了 Mamba/SSM 在**长序列时序视频任务**上的
  正经理由：线性复杂度 + 全局时序推理，且以 **adapter** 形式接入（不需要重训整个骨干）。
  对 CleanSight 的含义：如果试 SSM，**adapter 式接入比换主干更符合小数据现实**。

#### 13. Global and Local Fusion Mamba for Skeleton-based Temporal Action Segmentation ⚠️
- **Venue/Year**: IEEE（会议/期刊待核，题录来自 IEEE Xplore）· **URL**: https://ieeexplore.ieee.org/abstract/document/11394985
- **机制**：Mamba 做骨架 TAS 的全局+局部融合。**摘要未核验**（IEEE 页面被付费墙/反爬挡住）→ ⚠️。
- **适配**：★★★☆☆ 骨架 TAS 与 CleanSight 同为"低维特征 + 长序列"，是 Mamba-TAS 最接近的先例。

#### 14. S4/S5 → TAS 的空白（诚实结论）
- 在我完成的全部检索里（"Mamba temporal action segmentation"、"state space model temporal action segmentation"、
  "Mamba action segmentation online" 等），**没有找到一篇把 Mamba/SSM 作为逐帧 TAS 主干的、可核验的代表作**。
  命中的是 Mamba 用于动作识别、点云视频、动作检测、骨架识别、事件检测等周边任务。
- **这意味着**：Mamba-TAS 目前是一个**真实但未被充分占领的方向**；对 CleanSight 而言，
  "Mamba 会不会解决段标签认错"**在文献上没有被验证过**，属于探索而非收敛选择。
  这既是机会，也是"没有前人踩坑地图"的风险。
- **但要公平补一句**：Mamba 在**同族的因果时序视频任务**上已经有正式工作，说明它不是空想——
  最接近的是 §11（AAAI 2026 的 MOAD，在线动作检测）与 §13（骨架 TAS）。
  如果 CleanSight 要押这一注，**应当以 §11 为基线复现，而不是直接从 Mamba 论文起步**。

#### 15. 工业视频的旁证：ConsensusTAS（自监督 TAS for 长时程施工视频）✅
- **URL**: https://arxiv.org/abs/2608.24043（2026-08）· **摘要已核验**（arXiv 检索页全文摘要）。
- **机制**：面向"协作式人机作业"的**施工场景**（机器人要理解工人当前/下一步动作以便递工具），
  但既有研究只做**活动类别分类**（climbing / lifting / walking），而非**长时程序列里的细粒度活动转移**；
  难点是**给长施工视频标注动作时间边界极其费时**。
  作者提出 **ConsensusTAS**：**label-free、自监督**，通过"候选分割之间的内部共识"把连续视频流切成分明的工作阶段。
- **适配**：★★★☆☆ 它是**工业流程视频 + 无标注分段**，与 CleanSight 的场景最同域；
  但它解决的是"没有标注怎么办"，而 CleanSight 的痛点是"有标注但段身份学不对"。**参考价值在领域而非方法。**

---

### A4 离线 vs 在线 gap：专门报这块的文献

| 文献 | 它到底报了什么 | URL | 强度 |
|---|---|---|---|
| **ProTAS (CVPR 2024)** ✅ | 表 1 给出了完整的"同一模型离线 vs 在线"对照；EgoProceL 上 Edit 56.9→19.9、Acc 69.2→18.3 | [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf) | **强**（我逐行读过表） |
| **SVTAS-RL** ✅ | 把 STAS 与 TAS 的差异**当作研究对象**，归因 model bias + optimization dilemma | [abs](https://arxiv.org/abs/2309.15683) | 中（论述强、数字未见） |
| **A Comprehensive Study on Temporal Modeling for Online Action Detection** ✅ | 在 OAD 上**公平对比四类时序建模**：temporal pooling / temporal convolution / RNN / temporal attention，并总结 good practices | [abs](https://arxiv.org/abs/2001.07501) | **强**（这就是 CleanSight "GRU vs TCN vs Transformer" 对照的文献版） |
| **Ego-METAS** ✅ | 首个**第一人称在线多模态能效 TAS 基准**：100+ 小时、5 种模态（RGB/音频/注视/IMU/单目），研究"选择性访问最有信息量的传感器以平衡能耗与精度" | [abs](https://arxiv.org/abs/2606.02246) | 中（基准价值 > 方法价值） |
| **VidParse (2026)** ✅ | 在线、**免训练**的图约束推理框架，直接点名标准帧级在线模型的失败模式是 "severe over-segmentation and structural collapse" | [abs](https://arxiv.org/abs/2608.27562) | 中（与 CleanSight 症状描述高度一致） |
| **Online Spatiotemporal Action Detection and Prediction via Causal Representations** ✅ | 博士论文，明确论证 **causal（online）表示可以在 TAS 等任务上接近离线 3D CNN** | [abs](https://arxiv.org/abs/2008.13759) | 中 |

---

## Part B — 面向"段标签同一性"的段级目标与解码

### B1 段级 / 集合预测型目标函数

#### 16. FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation ✅
- **Venue/Year**: CVPR 2024 · **URL**: https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html
- **机制**（摘要逐字核验）：双分支并行——(i) **frame branch** 用卷积学帧级信息，
  (ii) **action branch** 用 transformer 学 **action token** 之间的依赖，(iii) cross-attention 让两者双向传递信息、迭代 refine。
  **关键**：提出一个 **matching loss，保证每个 action token 唯一地编码一个动作段**（"ensure each action token uniquely encodes
  an action segment thus better captures its semantics"）。
- **报出的结果**：4 个数据集（2 自中心 + 2 第三人称）显著超过 SOTA，且比既有 transformer 方法**快 3×**（摘要原话）。
- **代码 / license**: https://github.com/ZijiaLewisLu/CVPR2024-FACT — GitHub API 核验：**MIT**，113 stars。
- **对 CleanSight 的适配**：★★★★★ **这是最贴"段标签同一性"的一篇。**
  CleanSight 的失败模式是"整段认错"，而 FACT 的 matching loss 恰恰是**把"一个段 ↔ 一个语义原型"做强绑定**；
  它不依赖像素（只要逐帧特征），计算量比纯 transformer 低（摘要称快 3×），
  而且 action branch 的 token 数 = 类别数，跟 144 维小输入非常匹配。

#### 17. DIR-AS: Decoupling Individual Identification and Temporal Reasoning for Action Segmentation ✅
- **Venue/Year**: arXiv 2304.02110（2023-04）· **URL**: https://arxiv.org/abs/2304.02110 · **venue 未在 abs 页标明**（摘要页无 comment/journal 字段）
- **机制**：把 TAS 显式**解耦成两个目标**：
  (1) **individual identification**（"这一帧是什么"）用逐帧监督；
  (2) **temporal reasoning**（"段怎么排"）用 **action set prediction**（集合预测，DETR 式）+ 一个 **action alignment module** 融合两种粒度的预测。
  论文的动机原话是"most of them take advantage of frame-wise supervision, **which cannot effectively tackle the evaluation metrics
  with different granularities**"——即**逐帧损失喂不出段级指标**。
- **报出的结果**（摘要）：GTEA **82.8%（+2.6%）**、Breakfast **74.7%（+1.2%）**。
- **代码**：摘要写 "The code will be made available later" → ❌ **未核验**（是否放出未知）。
- **适配**：★★★★★ **概念上最干净的一篇**。它给出的正是 CleanSight 需要的诊断：
  把"帧分类头"和"段集合预测头"分开，段级指标由后者负责。
  风险：集合预测（Hungarian 匹配）在**小数据**上不稳定，且集合预测头需要额外一轮匹配计算（CPU 上仍在微秒级）。

#### 18. Permutation-Aware Action Segmentation via Unsupervised Frame-to-Segment Alignment ✅
- **Venue/Year**: WACV 2024 · **URL**: https://arxiv.org/abs/2305.19478
- **机制**：帧级模块用 transformer encoder + **temporal optimal transport** 训练；
  再加 **segment-level prediction module**（transformer decoder 估 transcript）与 **frame-to-segment alignment module**
  （把帧特征与段特征匹配），产出一致性更强的 **permutation-aware** 分割；并用 OT 造伪标签。
- **报出的结果**：50 Salads / YouTube Instructions / Breakfast / Desktop Assembly 上与既有无监督方法相当或更好。**摘要未报数字**。
- **适配**：★★★☆☆ 无监督设定与 CleanSight 全监督不同，但 **frame-to-segment alignment** 这个模块
  可以直接搬到全监督里当辅助损失：**强制同一段内所有帧指向同一个段原型**——正是"段标签同一性"。

#### 19. Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation ✅
- **Venue/Year**: CVPR 2026 Workshop "AI-driven Skilled Activity Understanding, Assessment & Feedback Generation (SAUAFG)"
  · **URL**: https://arxiv.org/abs/2604.01859
- **机制**（摘要逐字）：轻量双损失框架，**只加一个输出通道 + 两个辅助损失**：
  (a) boundary-regression loss（单通道边界预测）；
  (b) **CDF-based segment-level regularization loss**——匹配预测段与真值段的**累积分布**，
  以"encourage coherent within-segment structure"。
  明确宣称 **architecture-agnostic**，可直接接进 MS-TCN / C2F-TCN / FACT。
- **报出的结果**：三个基准上提升 F1 与 Edit；
  **关键**："**Frame-wise accuracy remains largely unchanged**, highlighting that precise segmentation can be achieved
  through simple loss design rather than heavier architectures or inference-time refinements."
  → 这与 CleanSight "edit 涨、acc 不涨"的现象**机制一致**（见 §21.2 的结论）。
- **代码**：摘要未给 → ❌ 未核验。
- **适配**：★★★★★ **实现成本最低的段级目标**：不换架构、不加参数（只加一个输出通道），
  且它的 CDF 正则目标正是"段内一致"，可直接对着 CleanSight 的 `edit` 指标优化。

#### 20. Temporally Consistent Unbalanced Optimal Transport (TUOT) ✅
- **Venue/Year**: CVPR 2024（**Oral**）· **URL**: https://arxiv.org/abs/2404.01518
- **机制**：把 **temporal consistency 先验编码进 Gromov-Wasserstein 问题**，
  从"帧 × 动作类别"的噪声亲和矩阵里**解码出时间一致的分割**；**不需要预先知道动作顺序**；
  用 projected mirror descent 几次迭代就能在 GPU 上求解。用于无监督自训练造伪标签。
- **报出的结果**：Breakfast / 50-Salads / YouTube Instructions / Desktop Assembly 上无监督 SOTA。**摘要未报数字**。
- **适配**：★★★★☆ 这是**"从混淆的帧级分数里整体解码出一条合法段序列"的数学框架**，
  理论上正好对应"标签错是整段错"——因为 OT 是**在段/序列层面做全局最优匹配**，而不是逐帧独立 argmax。
  风险：mirror descent 迭代在 CPU 上的延迟需要实测（论文只说 GPU 上几轮迭代）。

#### 21. CLOT / D-CLOT: Closed Loop (Double) Optimal Transport for Unsupervised Action Segmentation ✅(题录)
- **URL**: https://arxiv.org/abs/2507.03539 （CLOT, 2025-07）· https://arxiv.org/abs/2608.05877 （D-CLOT, 2026-08）
- **机制/结果**：题录来自 arXiv 检索页；**摘要我未逐字核验** → ⚠️。属于 §20 的后续工作。
- **适配**：★★★☆☆ 同上，作为 OT 路线的活跃度证据。

#### 22. TCL: Unsupervised Action Segmentation by Joint Representation Learning and Online Clustering ✅
- **Venue/Year**: CVPR 2022 · **URL**: https://arxiv.org/abs/2105.13353
- **机制**：把视频帧聚类当 pretext，**同时**做表示学习与**在线聚类**；用 **temporal optimal transport**，
  在标准 OT 里加**保持动作时序的时间正则项**来算伪标签。
  **且明确以 mini-batch 在线方式处理**（"processes one mini-batch at a time in an online manner"），
  而不是先存全数据集特征再离线聚类。
- **报出的结果**：50-Salads / YouTube Instructions / Breakfast / Desktop Assembly 上持平或更好，且显存约束小得多。**摘要未报具体数字**。
- **代码**：摘要给了 research website → ❌ 未取到直链，未核验。
- **适配**：★★★★☆ "**在线聚类 + 时序正则 OT**"是**离线可用、在线可跑**的原型分配机制，
  对"每类一个原型，强制段内帧归到同一原型"这个想法是最直接的实现蓝图。

#### 23. UVAST: Unified Fully and Timestamp Supervised TAS via Sequence to Sequence Translation ✅
- **Venue/Year**: ECCV 2022（Main）· **URL**: https://arxiv.org/abs/2209.00638
  （UVAST 是代码库名；论文标题如上）
- **机制**：把 TAS 当 **seq2seq 翻译**：帧序列 → 段序列。
  为适应"长输入、短输出、视频数少"，做了一系列改动与辅助损失：
  encoder 上加逐帧辅助监督；**单独一个 alignment decoder 做隐式时长预测**；
  并把框架扩到 timestamp 监督，用 **constrained k-medoids** 造伪段。
- **报出的结果**：全监督与 timestamp 监督两种设定下在多个数据集上超过或持平 SOTA。**摘要未报数字**。
- **代码 / license**: https://github.com/boschresearch/UVAST — GitHub API 核验：**AGPL-3.0**，40 stars。
  🚨 **AGPL-3.0 是强 copyleft**：工业闭源产品直接链接/修改其代码有传染风险，**只能读算法、自己重写**。
- **适配**：★★★★☆ 它对应 CleanSight 关心的"**constrained decoding**"：
  seq2seq 解码天然带"段数有限、段有先后、段有典型时长"的约束，
  而 duration decoder 直接给"段级时长先验"。但 Transformer seq2seq 在 3 万帧小数据上有过拟合风险。

---

### B2 解码 / 后处理：直接改"段身份"

#### 24. Improving Temporal Action Segmentation via Constraint-Aware Decoding (CAD) ✅
- **Venue/Year**: **ICPR 2026** · **URL**: https://arxiv.org/abs/2605.10149
- **机制**（摘要逐字核验，这一篇的核心正是用户问的东西）：
  从标注数据里直接抽取**统计结构先验**——**transition confidence（动作转移置信度）、action boundary sets、
  per-class duration（逐类时长）**——并把它们整合进一个**修改版 Viterbi 解码**：
  "allowing **inference-time refinement without retraining or added model complexity**"。
  论文还点名动机：既有 grammar-based 方法靠复杂 parsing、不可扩展；
  而本文面向 "new or **low-resource domains**"。
- **报出的结果**：摘要称在**全监督与半监督**模型上都提升，且保持高效率。**摘要未报具体数字**。
- **代码**: 摘要给出 https://github.com/LUNAProject22/CAD ，但**我在 2026-09-27 抓取返回 HTTP 404** → ❌ **未核验**。
- **适配**：★★★★★ **与我们问题的匹配度是全部条目里最高的**：
  (a) 它是**纯推理期改动**——不重训、不加参数，直接绕开"容量/损失权重/增强都无效"的死路；
  (b) **transition confidence + per-class duration + Viterbi** 正是"整段认错"的克星：
  一个 `flush` 段被逐帧 argmax 打成 `water_injection`，只要 `flush` 的典型时长先验与转移先验支持它，Viterbi 就能整段纠正；
  (c) 它明确面向 low-resource 域。
  风险：Viterbi 是 O(T·K²)（T≈32 的滑窗，K≈10 → 完全可忽略），但**它只能在滑窗内做局部 Viterbi**，
  跨滑窗的全局一致性需要额外设计（见 §1 的排序理由）。

#### 25. ProTAS 的 task graph 受限解码（见 §A1-2）✅
- 已在 Part A 展开。此处只强调它是 **B2 意义上的"受限解码"**：
  用**任务图**惩罚"违反流程"的预测，等价于一个从数据里学出来的、比一阶转移矩阵更强的结构先验。
  **在 EgoProceL 上 Edit 42.5→48.8、EgoPER 上 48.9→60.7（同样 Acc 不涨）**——这是"段级先验直接买 edit"的实测证据。

#### 26. Viterbi / HMM / CRF 平滑 —— 诚实说明 ⚠️
- 我**没有**在本次检索中定位到一篇"现代 TAS + HMM/CRF 平滑"的强代表作（检索命中的多为无关内容或二手引用）。
- **最接近且已核验的可引用项就是 CAD（§24）**，它用的正是带结构先验的 Viterbi。
- 经典 HMM/Viterbi 在**手术阶段识别**里是标准工具（TeCNO 之前的 LSTM/HMM 路线），
  但具体引用我没能给出可核验 URL → **此处不列表、不编造**。

#### 26b. VidParse: Online Parsing of Egocentric Procedures Like a Pro ✅
- **Venue/Year**: **ECCV 2026** · **URL**: https://arxiv.org/abs/2608.27562
- **机制**（摘要逐字核验）：**在线、免训练**框架，把活动理解当成 **graph-constrained inference** 问题。
  它先用**冻结基础模型**抽 manipulation-anchored 特征（优先前景手-物交互），用时间相似度矩阵**动态识别语义转移**；
  然后一个 **beam search decoder 利用"诱导出的过程任务图"显式强制合法动作转移、剪掉不可能轨迹**
  （"explicitly enforce valid action transitions and **prune impossible trajectories**"）。
  动机原话直接点名标准帧级在线模型的失败是 "severe over-segmentation and **structural collapse**"。
- **报出的结果**：**"up to a 10x improvement in complex multi-step parsing accuracy over strong online baselines,
  all without requiring a single gradient update"**（摘要原话——注意这是"多步解析准确率"的提升倍数，
  不是 F1 的倍数，不要误读）。
- **代码**：摘要未给 → ❌ 未核验。
- **适配**：★★★★★ **这是 CAD 之外第二个"纯解码"方案，而且比 CAD 更激进**：
  CAD 是 Viterbi + 一阶转移/时长先验；VidParse 是 **beam search + 任务图 + 剪枝不可能轨迹**，
  对"整段认错"的作用更直接（一个不可能出现的段序列会被整条路径剪掉）。
  它与 ProTAS 的区别值得注意：ProTAS 是**软惩罚**（logit 减法），VidParse 是**硬剪枝**。
  在 CleanSight 上可以两者都试：软惩罚更保守、硬剪枝更激进（也更可能把稀有类误剪，需小心）。
  ⚠️ VidParse 的输入是视觉基础模型特征，CleanSight 只有 144 维检测特征——
  但**它的解码器部分与输入表示无关**，可以直接移植。

#### 27. 其他可作"段身份"约束的先验来源
- **MSBATN（Multi-Stage Boundary-Aware Transformer with hierarchical sliding window attention）** ✅
  arXiv 2504.18756 · https://arxiv.org/abs/2504.18756
  摘要明确点名 MS-TCN 的两种失败："rely on large receptive fields, that causes **over-segmentation**,
  or **under-segmentation**, where distinct actions are incorrectly aligned"。**摘要未报数字**。
- **ATBA: Efficient and Effective Weakly-Supervised Action Segmentation via Action-Transition-Aware Boundary Alignment** ✅
  arXiv 2403.19225 · https://arxiv.org/abs/2403.19225 —— 用 transcript 直接定位少量 **action transition**，
  避开逐帧串行对齐，又快又能并行。**摘要未报数字**。

---

### B3 类不平衡 / 稀有类 / 标签噪声

#### 28. Cost-Sensitive Learning for Long-Tailed Temporal Action Segmentation ✅
- **Venue/Year**: **BMVC 2024** · **URL**: https://arxiv.org/abs/2503.18358
- **机制**（摘要逐字）：识别出 TAS 里的 **bi-level learning bias**：
  (1) **class-level bias**（类别不平衡偏向头部类）；(2) **transition-level bias**（常见转移被优先）。
  为二者定义 learning state，写成一个**约束优化问题**，并提出**加权交叉熵**，权重按"动作及其转移的学习状态"**自适应调整**。
- **报出的结果**：三个基准 × 多种框架上，"**significant improvements in both per-class frame-wise and segment-wise performance**"。
  **摘要未报具体数字**。
- **代码**：摘要未给 → ❌ 未核验。
- **适配**：★★★★★ **直接对着 CleanSight 的 `sbc`（18 段、类内中位时长 6.5 帧）和 `flush`/`withdraw`。**
  特别值得注意的是它把 **transition-level bias** 单列——CleanSight 的 `insert`/`withdraw` 正是**一对互为逆转移的类**，
  在数据里它们的转移模式高度对称，这会造成"transition 层面的系统性混淆"，
  而 CleanSight 已经试过的"损失权重"很可能只是 class-level 的，**没有碰 transition-level**。

#### 29. Mitigating Surgical Data Imbalance with Dual-Prediction Video Diffusion Model (SurgiFlowVid) ✅
- **URL**: https://arxiv.org/abs/2510.07345 （2025-10）
- **机制**：用**稀疏可控视频扩散**生成**欠表示类**的手术视频，双预测模块同时去噪 RGB 与光流以注入运动先验。
  在三个手术数据集 × 多个下游任务上验证。
- **适配**：★☆☆☆☆ 对 CleanSight **不可用**：我们的输入源只有已固化的检测输出（`frames/` 下只有 `txt`、无图像），
  **连像素都没有，生成视频无从谈起**（这一点与项目内 §19.1 的核查一致）。
  列出来是为了**明确排除**"用生成模型补稀有类"这条路在我们当前数据上不可行。

#### 30. Boundary-Centric Clip-Budgeted Active Learning for TAS (B-ACT) ✅
- **URL**: https://arxiv.org/abs/2604.15173 （2026-04，v2 2026-06）
- **机制**：把标注预算集中投到**边界区域**：两级循环——先按预测不确定性排序选视频，
  再在视频内用 boundary score（邻域不确定性 + 类歧义 + 时序预测动态）选 top-K 边界，
  **只在边界帧要标注**、但用边界中心 clip 训练。
- **报出的结果**：GTEA / 50Salads / Breakfast 上在稀疏预算下超过代表性 AL 基线；
  摘要明确说"**Gains are largest on datasets where performance is highly sensitive to boundary placement**"。
- **适配**：★★☆☆☆ **对 CleanSight 的方向是相反的**：它优化边界，而 CleanSight 的 78% 错误在段内。
  但它的**评测框架**（在稀疏预算下比较 metric 敏感度）对我们做"哪些类值得补标注"的决策有参考价值。

#### 31. Two-Stage Active Learning for Efficient Temporal Action Segmentation ✅
- **Venue**: **ECCV 2024**（Springer LNCS）· **URL**: https://link.springer.com/chapter/10.1007/978-3-031-72970-6_10
- **机制**（摘要已核验）："Training a TAS model on long and untrimmed videos requires gathering framewise video annotations,
  **which is very costly**. We propose a two-stage active learning framework to efficiently learn a TAS model using only a small…"
  即用两阶段主动学习**用极少的逐帧标注**训出可用模型。
- **适配**：★★★☆☆ 与 CleanSight 的"标注太少"直接相关，但它的目标是**标注效率**，
  不是"用同样标注把段身份学对"。作为"再接数据时怎么挑视频标"的方法论参考。

#### 32. Rethinking Pseudo-Label Guided Learning for Weakly-Supervised TAL from the Perspective of Noise Correction ✅
- **URL**: https://arxiv.org/abs/2501.11124 （2025-01，v2 2025-04）
- **机制**：明确论证**伪标签里的噪声**会伤害监督头，并列出三类噪声症状：
  "(1) inaccurate boundary localization; (2) undetected short action clips; (3) **multiple adjacent segments incorrectly
  detected as one segment**"。提出两阶段噪声学习：context-aware 去噪精修边界 + online-revised 训练。
- **适配**：★★★☆☆ 弱监督设定与 CleanSight 不同，但**"short action clips 被漏掉"和"相邻段被并成一段"**
  与我们的 `sbc`（6.5 帧）和段数比 0.90 的现象对应，可作机制引证。

#### 33. Reducing the Label Bias for Timestamp Supervised Temporal Action Segmentation ⚠️
- **Venue**: CVPR 2023 · **URL**: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Liu_Reducing_the_Label_CVPR_2023_supplemental.pdf （仅取到 supplemental）
- **说明**：题录来自检索；**摘要未核验** → ⚠️。列出的理由是它把"**label bias**"当成 TAS 的独立问题来治，
  与"标签本身有问题"这条线索同向。

---

### B4 迭代精修 / 扩散 / 原型（用于改变"段身份"的生成式路线）

| # | 文献 | 机制（核验程度） | 适配度 |
|---|---|---|---|
| 34 | **ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation** ✅ NeurIPS 2024 · [abs](https://arxiv.org/abs/2412.04353) | 用**同一个扩散模型**同时做分割与预测；训练时用 **anticipative masking** 把视频后段 mask 成"不可见"，用可学 token 代替来学预测未来；分割与预测**双向互相受益** | ★★★☆☆ 它的 masking 训练天然支持"只看过去"→ 因果/流式友好；但扩散迭代采样与 p95 几毫秒的预算冲突（见 #36） |
| 35 | **HybridTAS: Learning Action Hierarchies via Hybrid Geometric Diffusion** ✅ **WACV 2026** · [abs](https://arxiv.org/abs/2601.01914) | 在扩散去噪里混合**欧氏 + 双曲几何**，用双曲空间的树状关系让**动作标签去噪"粗到细"**：大 timestep 受高层抽象类别影响，小 timestep 用细粒度类精修 | ★★★★☆ **这是对"段标签同一性"最有想法的一篇**——它把"先判断这是哪一大类、再判断细类"显式做进去噪过程，正是针对"整段认错"。风险：扩散采样延迟 |
| 36 | **Faster Diffusion Action Segmentation** ✅ [abs](https://arxiv.org/abs/2408.02024) | 直陈扩散 TAS 的**采样步数**造成 "substantial computational burden, limiting their practicality in **real-time applications**"，并指 transformer encoder 长序列的 feature-smoothing 问题 | ★★☆☆☆ 这篇本身就是"扩散路线不适合低延迟"的可引用证据 |
| 37 | **MSBATN** ✅ [abs](https://arxiv.org/abs/2504.18756) | 见 §27 | ★★☆☆☆ |
| 38 | **MMF-TAS: Multi-Modal Few-Shot Temporal Action Segmentation** ⚠️ **ICCV 2025** · [poster](https://iccv.thecvf.com/virtual/2025/poster/118) · 代码 https://github.com/ZijiaLewisLu/ICCV2025-MMF-TAS | 少样本 TAS；**摘要未逐字核验** | ★★★☆☆ 小数据方向 |
| 39 | **An Efficient Framework for Few-shot Skeleton-based TAS** ✅ [abs](https://arxiv.org/abs/2207.09925) | 面向小规模数据集：运动插值做数据增强 + 在骨架 TAS 网络上**拼一个 CTC 层**；论文称 CTC "can enhance the temporal alignment between prediction and ground truth and further **improve the segment-wise metrics**" | ★★★★☆ **与我们最相关的少样本工作**：它明确说 CTC（一个**段级/序列级对齐目标**）改善的是**段级指标**而非帧级 |
| 40 | **Do we really need temporal convolutions in action segmentation? (TUT)** ✅ [abs](https://arxiv.org/abs/2205.13425) | 纯 Transformer（无时序卷积）+ temporal sampling 的 U-Transformer | ★★☆☆☆ 反证 TCN 未必必需 |
| 41 | **DIR-AS / FACT / Permutation-Aware** | 见 B1 | — |

---

## 1. Top-5 排序短名单（针对 CleanSight）

排序依据四项：(a) 是否攻击**段身份混淆**；(b) 流式可行性与延迟代价；
(c) **小数据/稀有类风险**；(d) 实现成本。

### 🥇 第 1 名：CAD —— 约束感知 Viterbi 解码（`Improving TAS via Constraint-Aware Decoding`）
https://arxiv.org/abs/2605.10149 （ICPR 2026）
- **(a) 攻击段身份**：**最强**。它改的正是"逐帧 argmax"这一步。CleanSight 的症状是"段内 78% 的帧错、且整段多数标签错"，
  这在数学上恰恰是**逐帧独立决策**的典型失败：单帧后验把 `flush` 判成 `water_injection` 的概率略高，
  逐帧 argmax 就整段翻车；而 Viterbi 在**序列层面**最大化联合概率 + 转移先验 + 逐类时长先验，
  一个"时长明显偏短、转移明显违反流程"的整段翻转会被直接否决。**这是唯一一个直接对准"整段认错"这个目标函数层面的手段。**
- **(b) 流式与延迟**：**极低**。推理期改动、不重训、不加参数（论文原话 "without retraining or added model complexity"）。
  滑窗 32 帧 × ~10 类的 Viterbi 是 32×100 次乘加——**远低于你现有 GRU 的 0.64 ms**。
  ⚠️ **唯一需要自己设计的地方**：滑窗内 Viterbi 只保证窗内一致，跨窗的段身份一致性要靠
  "把上一窗末段的累计代价作为下一窗的初始条件"这类工程手段接起来——这会和 `temporal_feed_mode` 的对比口径直接产生交互，
  必须一起测。
- **(c) 小数据/稀有类风险**：**低，但有一个陷阱**。先验（转移置信度、逐类时长）是从标注里统计出来的，
  在 `sbc` 只有 18 段的情况下，时长先验方差很大。**建议对稀有类做先验收缩（shrinkage）或只对出现 ≥N 次的转移施加约束**，
  否则会反过来把稀有类压死。
- **(d) 实现成本**：**最低之一**。核心是一个 ~100 行的 Viterbi + 从现有 `labels/*.txt` 直接统计的先验表。
  代码仓库 404，**必须自己实现**——但这也回避了 license 问题。

### 🥈 第 2 名：段级结构先验解码 —— task-graph 软惩罚（ProTAS）＋ graph-constrained beam search 硬剪枝（VidParse）
- ProTAS / CVPR 2024：https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf
- VidParse / ECCV 2026：https://arxiv.org/abs/2608.27562
- **(a) 攻击段身份**：**强，且是唯一有实测段级增益的一组**。
  ProTAS：EgoProceL Edit 42.5→48.8、EgoPER 48.9→60.7，**而 Acc 基本不动（72.7→70.2 甚至下降）**。
  **这个"edit 涨、acc 不涨"的指纹和 CleanSight 的观察完全同构**——说明段级先验作用在"段怎么排"这条独立通路上，
  正是 §21.2 里你测出来的那条通路。
  VidParse 把同一想法推到更硬的一端：beam search 直接**剪掉不可能的轨迹**（"prune impossible trajectories"），
  摘要报 **复合多步解析准确率最高 10× 提升**（注意：是解析准确率的倍数，不是 F1 的倍数），且**零梯度更新**。
- **(b) 流式与延迟**：**极低**。ProTAS 的约束是 logit 惩罚（一次查表 + 减法），天然因果；
  VidParse 的 beam search 在滑窗上是个极小宽度的搜索（K≈10 类、beam 宽度个位数）——但**beam search 的延迟是随 beam 宽度线性增长**，
  需要实测并纳入 p95 统计，不能想当然。
  ProTAS 的 CAS 设计（把 MS-TCN 改成因果）同时就是 CleanSight 流式化 MS-TCN++ 的说明书。
- **(c) 小数据/稀有类风险**：**中**。从训练视频学 task graph 在 8 个测试视频的量级上会很稀疏；
  但这里有个**关键优势**：清洁流程的合法性是**领域知识**，可以直接手工写成一个很小的图（甚至不需要学），
  相当于把领域先验零成本注入。⚠️ 硬剪枝版本（VidParse）对稀有类（`sbc` 18 段、6.5 帧）更危险——
  一个在训练集里少见的合法转移可能被误剪，**建议先上软惩罚，硬剪枝只对高置信转移启用**。
- **(d) 实现成本**：低-中——CAD 是纯解码器替换，这两项还要**同时**做因果化改造 + 图构建。
  ProTAS 代码无 license（GitHub API 显示 license = None）→ **不能拷代码，只能读算法**；VidParse 未给代码。

### 🥉 第 3 名：FACT 的 action-token matching loss（段-原型强绑定）
https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html （CVPR 2024，代码 MIT）
- **(a) 攻击段身份**：**强且是"训练目标层面"的**。matching loss 强制"一个 action token 唯一编码一个段"，
  即**在表示空间里把"类别原型"与"实际段"绑死**——这直接对治"一个类被拆成多段、或一段被赋予错类"。
  与 CAD/ProTAS（推理期）互补：一个治训练目标，一个治解码。
- **(b) 流式与延迟**：**中**。frame branch 是卷积（可因果化）；action branch 是 transformer，但 token 数 = 类别数（~10），
  注意力开销与序列长度无关；cross-attention 需要**滑窗内的 action token**，因果化是可行的。
  比 TeSTra 更容易落地到 144 维输入。
- **(c) 小数据/稀有类风险**：**中**。matching loss 需要"段"这个单位，稀有类只有 18 段 → 匹配监督极稀疏，
  可能对 `sbc` 帮助有限甚至加重不平衡。**建议只在非稀有类上启用 matching loss，稀有类只吃 CE + cost-sensitive 权重。**
- **(d) 实现成本**：中——要加双分支 + cross-attention + 匹配损失，属于"改模型"级别；
  但 MIT license + 113 stars，可读可借鉴。

### 4️⃣ 第 4 名：Cost-Sensitive Learning for Long-Tailed TAS —— **transition-level** 重加权
https://arxiv.org/abs/2503.18358 （BMVC 2024）
- **(a) 攻击段身份**：**中-强，且是针对稀有类的唯一有文献支撑的手段。** 它把偏差拆成 **class-level** 与 **transition-level**。
  CleanSight 已经"穷尽损失权重"——但那很可能只是 class-level。`insert` / `withdraw` 是一对**逆转移**，
  在 2×3 ROI × presence/count/max_area 的特征里**没有方向信息**（见 §2.2），
  所以模型在转移层面几乎必然把两者混在一起；**transition-level 重加权是 CleanSight 尚未试过的那个旋钮。**
- **🔥 为什么这一条从"值得试"升级为"高置信度值得试"**：这篇论文的 Table 2 **直接复现了 CleanSight 的失败**——
  在 MS-TCN 上叠加 **+CB（+0.9/+0.7/+0.3）、+LA（+1.0/+1.1/+0.1）、+Focal（+0.2/−0.3/−1.2）、
  +τ-norm（−1.1/−1.0/−1.0）**——**标准逐类重加权在长尾 TAS 上就是 ≈0 甚至负收益**；
  **而他们的 transition-aware cost-sensitive 方案拿到 +8.1/+8.1/+5.7。**
  也就是说：**CleanSight "权重旋钮已穷尽"的结论，在文献上被证实为"穷尽了错误的那一半"。**
- **(b) 流式与延迟**：**零成本**。权重是训练期的事，推理图不变。
- **(c) 小数据/稀有类风险**：**低**。它就是为长尾设计的（Breakfast 20 head / 28 tail 类，Assembly101 31/171）。
  文献量化了尾部类处境：MS-TCN 在 Breakfast 上 head Acc 65.1 vs **tail 37.7**，Assembly101 head 33.9 vs **tail 4.7**。
  ⚠️ 但它的尾部类仍有几十个实例，**不是 18 段**——对 `sbc` 仍需配合先验收缩。
- **(d) 实现成本**：**最低**。改损失函数即可，与现有框架完全兼容。**这一条应当是下一个立刻做的实验。**

### 5️⃣ 第 5 名：DIR-AS 的解耦（逐帧识别 vs 段集合预测）—— 作为**诊断性改造**而非直接套用
https://arxiv.org/abs/2304.02110
- **(a) 攻击段身份**：**概念上最强**。它的论文动机就是 CleanSight 的处境原话的英文版：
  "frame-wise supervision ... **cannot effectively tackle the evaluation metrics with different granularities**"。
  把 head 拆成"帧分类" + "段集合预测"两个，**让段级指标由一个专门的 head 负责**，
  理论上直接打通"edit/F1 卡住"的因果关系。
- **(b) 流式与延迟**：**中**。集合预测头是 DETR 式的（在滑窗内输出定数个段），需要 Hungarian 匹配（小矩阵，CPU 上微秒级）。
  但**段的起止时间在流式里只能预测"已结束的段"或"进行中的段"**，需要重新定义输出空间——这是真正的工作量。
- **(c) 小数据/稀有类风险**：**高**。集合预测在 3 万帧 / 18 段的稀有类上很可能学不动，
  且需要 tuning 集合大小上限。**这是短名单里风险最高的一项。**
- **(d) 实现成本**：**高**（新 head + 匹配 + 流式语义重定义）。代码未公开。
- **为什么仍进前五**：因为它提供了**最强的诊断价值**——如果"帧分类 head 的 acc 不动、段集合 head 的 edit 能动"，
  就证明 CleanSight 的瓶颈确实在"段级目标缺失"而不是数据；反之如果两个 head 都不动，
  就几乎坐实了 §2 的"数据/标注问题"结论。**它是一次决定性的判决性实验，而不是一个稳妥的提分方案。**

### 落选但值得知道
- **TeSTra（ECCV 2022）**：如果哪天真要重启 Transformer 家族，**只应该用这一版**（常数缓存/常数时间流式、Apache-2.0）。
  但在 CleanSight 当前"Transformer 已显著更差"的证据下，优先级低于上面 5 项。
- **OnlineTAS（NeurIPS 2024, MIT）**：adaptive memory + 在线过分割后处理，是 **A 类里最值得直接对标**的在线基线，
  MIT license 也友好；排在 5 名外是因为它的增益方向（抗过分割）与"整段认错"并非同一问题。
- **扩散/迭代精修（ActFusion、HybridTAS、Faster Diffusion）**：HybridTAS 的"粗到细标签去噪"很有想法，
  但扩散采样与 p95 几毫秒预算直接冲突（Faster Diffusion 那篇自己就是这条路的延迟证据），**不适合生产流式**。
- **Mamba/SSM**：文献稀薄、无 TAS 主干先例（§A3-14），属于探索性押注，不建议作为主线。

---

## 2. "证据反方"（Evidence Against）：这是不是根本上的**数据/标注**问题？

这一节是诚实的对立面：**文献里有相当强的证据表明，CleanSight 的段身份混淆更像数据与可观测性问题，而不是架构问题。**

### 2.1 逐帧监督本身喂不出段级指标（机制层的反对）

**DIR-AS**（https://arxiv.org/abs/2304.02110）的原话：既有工作"most of them take advantage of frame-wise supervision,
**which cannot effectively tackle the evaluation metrics with different granularities**"。
言下之意：**用逐帧损失去追段级指标，是一条先天不通的路径**，需要**额外**的段集合预测目标。

**Combining Boundary Supervision and Segment-Level Regularization**（CVPR 2026 Workshop，https://arxiv.org/abs/2604.01859）
给出同向实测：段级正则**提升 F1 与 Edit，但 "Frame-wise accuracy remains largely unchanged"**。
这与 CleanSight §21.2 的配对实验（h32→h128：edit 36.00→46.45，acc 53.73 vs 53.66 持平）**指纹一致**。

⚠️ **但这一条只对"同族架构内加段级损失"成立，不能推广成"架构无关"——见 §2.4 的纠正。**

---

### 2.2 特征/可观测性的信息上限（对 CleanSight 最强的一条反证）

**HeiChole 基准**（Medical Image Analysis 2023, DOI 10.1016/j.media.2023.102770；
我通过 NCBI eutils 抓取了 PubMed 记录 `id=36889206`）——**这是整个调研里最锋利的一个数字**：
33 例腹腔镜胆囊切除、**12 支国际团队、同一份数据**、同一套指标：

| 任务 | F1 |
|---|---|
| 阶段识别（phase, n=9） | 23.9–67.7% |
| **器械存在（instrument presence, n=8）** | **38.5–63.8%** |
| **动作识别（action recognition, n=5）** | **21.8–23.3%** |

**解读**：只要标签是"器械在不在"（物体级），模型做得不错；一旦标签需要"**动作动词**"，
**五支独立团队的 F1 全部塌到 ~22%**。
这正是 CleanSight 的处境：144 维特征是 `presence / count / max_area`——
**它是"器械在不在、有多少、多大"，也就是 HeiChole 里那个 38.5–63.8% 的物体级信号；
而 CleanSight 要预测的是 insert / withdraw 这类"动词级"标签。**
HeiChole 说明：**从物体级特征推动词级标签，是一个被独立验证过的、跨团队一致的性能断崖**，
不是"再多训练几轮"能跨过去的。

**TAS 综述**（Ding, Sener, Yao, *Temporal Action Segmentation: An Analysis of Modern Techniques*, IEEE TPAMI 2024,
DOI 10.1109/tpami.2023.3327284；我抓取了 https://arxiv.org/abs/2210.10352 与 ar5iv 全文）明确写着：
预计算特征（IDT 手工特征、I3D）"**tend to favour static cues, e.g., scene components, within frames**"，
并且——这句话值得原样引用——
"To the best of our knowledge, **no empirical research has compared utilizing pre-computed features to
training TAS models from raw images end-to-end**."
即：**整个 TAS 社区从未验证过"预计算/检测派生特征是否信息充分"这件事**。

**Cost-Sensitive Learning for Long-Tailed TAS**（https://arxiv.org/abs/2503.18358，全文 v1 已抓）给出最直接的机制陈述：
尾部动作欠学是因为"the learning of tail is suppressed due to the temporal continuity of frame representation…
**Distinctly separating two consecutive actions, one being head and the other tail, is challenging as
they share similar frame representations**, especially at segment boundaries."

**方向信息本身**：CleanSight 的特征契约里**没有方向项**（`insert`/`withdraw` 只差运动方向），
而项目自己的 §19.3 读数 3 已写明"`insert`/`withdraw` 是**方向信息不存在**"。
**⚠️ 我必须诚实说明**：子代理在 arXiv 与 PubMed 上专门检索 **insert-vs-withdraw / 方向依赖动作**，
**没有找到任何可核验的专门论文**。所以这一条是"机制上显然 + HeiChole 类比的间接支持"，
**不是有文献直接证明的结论**。相关旁证：

- **LaDy**（arXiv 2603.24097，https://arxiv.org/abs/2603.24097 ✅）：忽略物理动力学
  "limits inter-class discriminability between actions with **similar kinematics but distinct dynamic intents**"。
- **Spectral Scalpel**（arXiv 2603.24134，https://arxiv.org/abs/2603.24134 ✅）：
  把问题归因为"**limited inter-class discriminability** … insufficient distinction of spatio-temporal patterns
  between adjacent actions"。
- **Polyphony**（arXiv 2605.31115，https://arxiv.org/abs/2605.31115 ✅）：点名
  "**semantic ambiguity in fine-grained actions**"，对策是引入**结构化语言描述**——
  即当低维特征不够时，需要**额外模态/额外语义**，而不是更大模型。

---

### 2.3 标注本身：这是证据最硬、也最不舒服的一条

子代理从 PubMed 抓到了**手术视频标注一致性**的直接测量。这些是**同一批专家、有明确标注手册**的前提下的数字：

| 研究 | 设定 | 结果 |
|---|---|---|
| **LRYGB 胃旁路术本体论** — *Surgical Endoscopy* 2023（DOI 10.1007/s00464-022-09745-2，PubMed `36289088`） | 2 位委员会认证外科医生，12 阶段 / 46 步骤本体论，131 段视频 | **Cohen's kappa：阶段 95.9±4.3%，但步骤只有 80.8±10.0%**；组内准确率 98.4%（阶段）vs **88.1%（步骤）** |
| **ESG 术式三标注者研究** — *International Journal of Surgery* 2026（DOI 10.1097/JS9.0000000000005019，PubMed `42682465`） | 3 位标注者、专家手册、40 段视频 | 两两 kappa 仅 **0.44–0.58** |
| **手术过程标注系统综述** — *Surgical Endoscopy* 2023（DOI 10.1007/s00464-023-10041-w，PubMed `37157035`） | 2806 篇筛选出 34 篇 | 原文结论："**Surgical video annotation lacks a rigorous and reproducible framework**"；"description of the surgical procedures was highly variable" |
| **SAGES Delphi 手势分类共识** — *Surgical Endoscopy* 2026（DOI 10.1007/s00464-026-12906-2，PubMed `42141188`） | 专家小组 | 明确指出存在 "**predictable ambiguities among semantically proximate actions（例如 cut vs seal；grasp vs clamp；dissect vs spread）**"，并把"给这些手势定义时间边界"列为 next critical step |
| **垂体腺瘤切除动作分类** — *IJCARS* 2026（PubMed `41964780`） | 独立标注者 vs 本体论 | kappa 0.69–0.95 —— **反例**：粗粒度"器械-动词-靶点"三元组**可以**被稳定标注 |

**对 CleanSight 的含义**：`flush`（20/21 段全错）与 `withdraw`（25/27 段全错）的形态——
**同一批专家、有标注手册、kappa 仍只有 0.44–0.58，且明确点名"语义邻近的动作对存在可预测的歧义"**——
与"这两个类在标注口径上不可分"的假说**高度一致**。
尤其注意 LRYGB 那条：**连"阶段 vs 步骤"这种粒度差异都能让 kappa 从 95.9% 掉到 80.8%**，
而 CleanSight 的 `flush` / `water_injection` 之间的粒度差异比"阶段 vs 步骤"更细。

**⚠️ 必须声明的边界（子代理核实）**：
- **TAS 领域本身没有多人标注一致性研究**。arXiv 检索（`"temporal action segmentation" annotator agreement`、
  `action segmentation label noise robustness`、`annotation ambiguity surgical workflow`）**全部无结果**。
  所以**不能**从文献直接断言"CleanSight 的标签有歧义"——只能说"**同粒度的手工标注在邻域任务上被证明不可靠，
  而 TAS 从未做过这个测量**"。
- 垂体那条是**反例**，说明歧义是**粒度与本体论依赖**的，不是普适的。

---

### 2.4 数据量：一条干净的学习曲线（以及一个被纠正的过度断言）

**Learning from a tiny dataset of manual annotations: a teacher/student approach for surgical phase recognition**
（arXiv 1812.00033，https://arxiv.org/abs/1812.00033，ar5iv 全文已抓）
——Cholec80 上，**精度随"已标注视频数"的变化**：

| 已标注视频数 | 1 | 3 | 5 | 10 | 20 | 80 |
|---|---|---|---|---|---|---|
| Accuracy | **40.1%** | 71.1% | 76.2% | 78.5% | 84.1% | **89.5%** |
| avg F1 | 21.1 | 55.3 | 65.8 | 69.7 | 75.8 | 82.5 |

**这是本次调研里关于"数据量才是主变量"最干净的一条曲线**：从 1 段到 3 段视频就跳 +31 个点。
**诚实边界**：这是**手术阶段识别**，不是 TAS；且**文献里没有** MS-TCN/MS-TCN++ 在
Breakfast/50Salads/GTEA 上随训练集比例变化的公开学习曲线——
所以正确的表述是"**没有人证明过 3 万帧够**"，而**不是**"3 万帧被证明不够"。

**规模参照**（TAS 综述 Table II）：50Salads = 5.5 小时 / 50 视频 / **0.9K 段**；
Breakfast = 77 小时 / 1712 视频 / **11K 段**。
**CleanSight：~3 万帧；测试 split 的真值段数按项目 §19.3 的表反推为每轮 73 段
（idle 40 + flush 7 + insert 11 + withdraw 9 + sbc 6 = 73；该表的 219 段是 3 seed 汇总、584 段是 8 seed 汇总）。
即单轮 73 段 vs 50Salads 的 ~900 段——低一个数量级还多。**

---

### 2.5 架构到底是不是次要变量？—— 我原来写错了，这里纠正

我在初稿里写了"架构买结构、不买帧级正确率"。**子代理拿到的同特征同划分对照数据纠正了这个过度断言**：

**Cost-Sensitive Learning… （arXiv 2503.18358）在 Breakfast 上、同样的 I3D 特征与划分**：

| 模型族 | per-class Acc | per-class F1@10 | global Acc | global Edit |
|---|---|---|---|---|
| MS-TCN（TCN） | 49.1 | 48.1 | 67.7 | 66.6 |
| ASFormer（Transformer） | 52.3 | 57.9 | 72.4 | 74.5 |
| DiffAct（diffusion） | **56.2** | **63.3** | — | — |

**帧级准确率确实随架构族变化**（per-class Acc +14.5% 相对提升）。所以正确说法是：

> **架构是"二阶杠杆"（second-order lever）：它同时推动帧级与段级，
> 但段级指标对架构的相对敏感度约是帧级的 2–3 倍**
> （per-class Acc +14.5% vs per-class F1@10 +31.6%）。

**ASFormer 论文自己的 Table 8 是最好的控制实验**：在**完全相同**的 ASRF 精修模块下，
只把 MS-TCN 骨架换成 ASFormer，得到 **+1.4 Acc / +2.6 Edit / +1.9 F1@10**
（50Salads 84.5/79.3/84.9 → 85.9/81.9/86.8）。
即：**读出模块固定后，换骨架对帧级只值 +1.4 点，对段级值 +2.6 点。**

**这直接修正了对 CleanSight 的解读**：
CleanSight 观察到"MS-TCN++ > GRU > Transformer"**并不违背文献**——
但也要注意，标准基准上的 Transformer（ASFormer）**是比 MS-TCN 更好的**，
说明 CleanSight 上 Transformer 失利的**原因大概率不是"Transformer 不行"，
而是 ASFormer 论文自己点名的那条**："the lack of inductive biases with small training sets"**
（https://arxiv.org/abs/2110.08568）——在极小数据上，**先验是否匹配任务**比"族"更重要。

---

### 2.6 指标本身在骗人：53.98 这个数要重新解读

这是我**没有预料到**、但对 CleanSight 影响最大的一条。

**TAS 综述（TPAMI 2024）**对 MoF（frame accuracy）的原文批评：
"**The MoF metric can be problematic under dataset imbalance, i.e. if frequent and long action classes dominate.**
… models with similar MoF scores may have large qualitative differences,
suggesting that **class-balanced metrics may be more appropriate though this is currently not adopted in the literature**.
Furthermore, MoF, as a per-frame calculation, **does not capture segment quality**;
the score can be high even when the segments are fragmented."
同一综述给出 **Imbalance Ratio**：GTEA 24、50Salads 6、**Breakfast 639**、Assembly101 2604。

**通胀的精确量化**（arXiv 2503.18358 的表 2 与表 8 对照）：

| 模型（Breakfast） | global frame Acc | per-class frame Acc | 差值 |
|---|---|---|---|
| MS-TCN | **67.7%** | **49.1%** | **−18.6** |
| ASFormer | **72.4%** | **52.3%** | **−20.1** |

**对 CleanSight 的直接含义**：报告的 **frame acc 53.98 是 global（逐帧池化）口径**。
文献显示，在长尾数据上 global 口径会比 per-class 口径**高约 19–20 个点**。
CleanSight 的类别分布是长尾的（idle 120 段 vs sbc 18 段，且 idle 段连续长）。
**因此 53.98 几乎肯定显著高估了"逐类平均正确率"。**
建议立刻补一个 **per-class frame accuracy** 口径——项目 §19.4 已经提出"应把主指标换成逐类帧级召回 + 每类 precision"，
**文献完全支持这个决定**，而且 §19.4 的判断可以改成有引用的结论，而不再只是内部直觉。

**⚠️ 诚实边界**：子代理**没有**找到任何专门批评 **F1@IoU** 的论文
（检索 `action segmentation evaluation metric critique` 等 26+9 条命中，无一相关）。
所以"F1@IoU 有偏"如果要在报告里说，**必须标注为我们自己的论证，不能引用**。

---

### 2.7 稀有类：文献证实"你试过的重加权方式就是不会有效"

**Cost-Sensitive Learning…（arXiv 2503.18358）Table 2** —— 这一条**直接给 CleanSight 的失败清单做了文献背书**：
在 MS-TCN 上叠加标准的类别不平衡补救手段（+CB / +LA / +Focal / +τ-norm），
Breakfast 上的逐类指标变化是——

- **+CB**：+0.9 / +0.7 / +0.3
- **+LA**：+1.0 / +1.1 / +0.1
- **+Focal**：+0.2 / **−0.3** / **−1.2**
- **+τ-norm**：**−1.1 / −1.0 / −1.0**
- **只有他们的 transition-aware cost-sensitive 方案**：**+8.1 / +8.1 / +5.7**

**解读**：CleanSight 报告"损失权重已穷尽而不奏效"——
**文献表明这就是预期结果**：CB / LA / Focal / 温度归一化这类**逐类**重加权在长尾 TAS 上是**≈0 甚至负收益**。
**但 transition-aware 的重加权是唯一有效的**（+8.1）。
这与 CleanSight 从未试过 transition-level 这件事**正好互补**——**这是 §1 第 4 名从"值得试"升级为"高置信度值得试"的文献依据。**

同一篇的组别结果：MS-TCN 在 Breakfast 上 head Acc **65.1 vs tail Acc 37.7**；
在 Assembly101 上 head **33.9 vs tail 4.7**。尾部类"第一个阵亡"是被量化的。

---

### 2.8 反方小结（诚实版，已按子代理证据修正）

> **一、架构不是无关变量，但它是二阶杠杆。** 在标准基准同特征同划分下，
> 帧级 per-class Acc 随架构族走 49.1→52.3→56.2（+14.5% 相对），
> 而段级 per-class F1@10 走 48.1→57.9→63.3（+31.6% 相对）——
> **段级对架构的相对敏感度约为帧级的 2–3 倍**；在固定读出模块的对照里，换骨架只值 +1.4 Acc 但 +2.6 Edit。
> 所以 CleanSight "架构是唯一动过的杠杆"这个观察**是对的**，但"架构能救 51.6% 段标签错"**没有文献支持**。
>
> **二、最硬的证据是"数据 + 标注"，而不是"模型不够大"。**
> (i) 一条干净的 Cholec80 学习曲线：**1 段标注视频 40.1% → 3 段 71.1% → 80 段 89.5%**；
> (ii) **同一批专家 + 明确手册**下，细粒度步骤的 kappa 只有 **80.8%**（阶段 95.9%），
> 三位标注者之间只有 **0.44–0.58**，专家共识明确点名"语义邻近动作对存在可预测歧义（cut vs seal；grasp vs clamp）"；
> (iii) 系统综述结论："surgical video annotation lacks a rigorous and reproducible framework"。
> ⚠️ 但这三条都在**手术**域，**TAS 域没有任何一致性研究**——所以这是"高度同构的旁证"，不是"同一问题的证明"。
>
> **三、特征的信息上限是 CleanSight 特有的、且可能无法用模型跨过的一堵墙。**
> **HeiChole：12 支团队、同一份数据、器械存在 F1 38.5–63.8%，但动作识别 F1 只有 21.8–23.3%。**
> CleanSight 的 144 维特征（presence / count / max_area）在信息类型上等价于"器械存在"，
> 而要预测的 insert / withdraw 是"动作动词"。
> 加上 TAS 综述承认"预计算特征倾向于静态线索、且**从未有人验证它们是否信息充分**"，
> 这一条构成对"继续调架构能解决段身份错"的最强反驳。
> ⚠️ 但方向依赖动作（insert vs withdraw）**没有**找到专门文献，此点属机制推断。
>
> **四、报告的 53.98 frame acc 几乎肯定是虚高的。**
> 长尾数据上 global 与 per-class 帧级准确率实测相差 **18.6–20.1 个点**（MS-TCN/ASFormer on Breakfast）。
> 建议立刻补 per-class 口径——§19.4 的直觉被文献证实。
>
> **五、但反方不等于"放弃模型侧"。** 上面的结论针对**帧级可分性**；
> CleanSight 的**主指标是段级**（edit / F1@IoU / insert recall）。§1 前 5 名
> **作用在"段级"这条通路上**，而且 §2.1 与 ProTAS 表 1 都显示：**段级指标可以独立于帧级移动**。
> 换句话说：**段级先验值得做，但不要指望它把 51.6% 的段标签错降到 0——它更可能把"逐帧噪声"整合成"更合法的段序列"。**
>
> **六、真正能改那三类的只有三件事**（与项目 §19.4 的结论一致，现在有文献支撑）：
> **(1) 往特征里补方向/运动信息**（HeiChole 类比）；**(2) 补图像源重跑检测（救 `sbc`）**；
> **(3) 人工审计 `flush`/`withdraw` 的标签时间轴**（kappa 0.44–0.58 的旁证）。

## 3. 建议的下一步（按性价比排序）

| 序 | 动作 | 预期作用通路 | 成本 | 依据 |
|---|---|---|---|---|
| **0** | **立刻补 `per_class_frame_accuracy` 口径**（现有 53.98 是 global 口径） | **度量本身** | **零** | TAS 综述 TPAMI 2024；arXiv 2503.18358 表 2/8（global 67.7 vs per-class 49.1） |
| 1 | 从现有 `labels/*.txt` 统计**转移矩阵 + 逐类时长分布**，做**滑窗内 Viterbi 后处理**（稀有类先验收缩） | 段级（edit / F1@IoU） | 极低（不重训） | CAD, ICPR 2026 |
| 2 | **transition-level** 重加权（区别于已试过的 class-level 权重） | 段级 + 稀有类帧级 | 极低（只改损失） | Cost-Sensitive Long-Tailed TAS, BMVC 2024（+8.1 vs 逐类重加权 ≈0） |
| 3 | 手工/统计一个**很小的合法流程任务图**，推理期做 logit 惩罚（软）；确认后再试 beam-search 硬剪枝 | 段级 | 低 | ProTAS, CVPR 2024；VidParse, ECCV 2026 |
| 4 | 因果化 MS-TCN++（causal dilated conv）+ 与 `temporal_feed_mode` 对齐的训练采样 | 流式一致性 | 中 | TeCNO, MICCAI 2020；O-TALC |
| 5 | 加一个 **CDF 段级正则**（单通道 + 两个损失项，architecture-agnostic） | 段级 | 低-中 | arXiv 2604.01859, CVPR 2026 W |
| 6 | 判决性实验：加**段集合预测 head**，看是否只有 edit 动 | 诊断 | 高 | DIR-AS |
| — | **（并行、非模型侧）** 特征补方向/运动信息；补图像源重跑检测救 `sbc`；人工审计 `flush`/`withdraw` 标签时间轴 | 帧级可分性（唯一能改那三类的路径） | 高（人力/数据） | §2.2 / §2.3 / §2.6 |

第 0 项成本为零却可能改变所有后续解读——**它应当排在所有实验之前**。
第 1、3 项**不需要重训**，第 2、5 项只改损失，都可以在现有 8-seed 口径下直接做配对检验。

---

## 4. 未核验清单（再次明确）

- `LUNAProject22/CAD` 代码仓库 **404**（2026-09-27 抓取）→ license 与可用性未知。
- ProTAS 仓库 **无 license**；UVAST 为 **AGPL-3.0**（强 copyleft，工业闭源慎用）。
- 本报告中**绝大多数论文的具体数字我没有逐表核验**；只有 ProTAS 表 1、DIR-AS 摘要、以及各摘要中明示的数字
  （TeSTra 6×、FACT 3×、MS-TCN++ 数据集名）是我确证的。
- **仍标记为 ⚠️（摘要未逐字核验）**：IEEE 的 *Global and Local Fusion Mamba for Skeleton-based TAS*
  （IEEE Xplore 页面收费墙/反爬）、arXiv 2603.24097（LaDy）。
  （已核验并已从本清单移出：AAAI Backtrace Mamba/MOAD、arXiv 2604.09164、2608.27562、2608.24043、
  Two-Stage Active Learning ECCV 2024。）
- **HMM/CRF 平滑在 TAS 的现代代表作**：未找到可核验的强项，未列表。
- **"ICF"**：未能在 TAS 文献中定位，未做任何推测性归属。

### 4.2 文献中的**空白**（不是"我没找到"，而是子代理在 arXiv + PubMed 上专门检索后确认的空白）

| 空白 | 检索范围 | 含义 |
|---|---|---|
| **TAS 的多人标注一致性研究不存在** | arXiv `"temporal action segmentation" annotator agreement`、`action segmentation label noise robustness`、`annotation ambiguity surgical workflow` | 不能从文献直接断言"CleanSight 标签有歧义"，只能用手术域的旁证 |
| **insert-vs-withdraw 类方向依赖动作研究不存在** | arXiv `surgical action recognition insertion withdrawal direction`、`motion direction action recognition limitation`；PubMed `surgical instrument insertion withdrawal recognition`、`surgical action recognition motion direction instrument trajectory` | 方向信息上限这一条**只能作为机制推断**，无直接文献 |
| **TAS 模型的数据量学习曲线不存在** | 无 MS-TCN/MS-TCN++ 随训练集比例变化的公开曲线 | "3 万帧不够"是**未被证明**的；只能说"没人证明够" |
| **F1@IoU 的专门批评文献不存在** | `action segmentation evaluation metric critique`、`evaluation of temporal action segmentation metrics`（26 命中）、`temporal action localization evaluation metric limitations`（9 命中） | 若要说 F1@IoU 有偏，**必须标注为我们的论证，不可引用**。可引用的只有对 MoF 的批评 |

### 4.3 §2 新增引用源（全部由子代理实际抓取验证）

| 文献 | Venue | 我抓取的 URL / DOI |
|---|---|---|
| Temporal Action Segmentation: An Analysis of Modern Techniques | IEEE TPAMI 2024 | https://arxiv.org/abs/2210.10352 ；DOI 10.1109/tpami.2023.3327284（Crossref 核验）；全文 https://ar5iv.labs.arxiv.org/html/2210.10352 |
| HeiChole benchmark | Medical Image Analysis 2023 | NCBI eutils `db=pubmed&id=36889206`；DOI 10.1016/j.media.2023.102770 |
| Learning from a tiny dataset of manual annotations（Cholec80 学习曲线） | arXiv 预印本（**无 Crossref 记录 → venue 未核验**） | https://arxiv.org/abs/1812.00033 ；全文 https://ar5iv.labs.arxiv.org/html/1812.00033 |
| LRYGB 手术本体论（kappa 95.9% phases / 80.8% steps） | Surgical Endoscopy 2023 | eutils `id=36289088`；DOI 10.1007/s00464-022-09745-2 |
| ESG 三标注者研究（kappa 0.44–0.58） | International Journal of Surgery 2026 | eutils `id=42682465`；DOI 10.1097/JS9.0000000000005019 |
| 手术过程标注系统综述 | Surgical Endoscopy 2023 | eutils `id=37157035`；DOI 10.1007/s00464-023-10041-w |
| SAGES Delphi 手势分类共识（cut vs seal 类歧义） | Surgical Endoscopy 2026 | eutils `id=42141188`；DOI 10.1007/s00464-026-12906-2 |
| 垂体腺瘤动作分类（**反例**，kappa 0.69–0.95） | IJCARS 2026 | eutils `id=41964780` |
| ASFormer（Table 8 骨干替换对照） | BMVC 2021 | https://arxiv.org/abs/2110.08568 ；DOI 10.5244/c.35.49（Crossref 核验） |
| Distill and Collect for Semi-Supervised TAS | arXiv 预印本 | https://arxiv.org/abs/2211.01311 |
| Disentangling Static and Dynamic Information for Reducing Static Bias | arXiv 预印本 | https://arxiv.org/abs/2509.23009 |

子代理的完整证据文件（含更细的表号与原文引文）：`.tas_evidence_tmp/evidence_against_architecture.md`

# Adversarial survey: is the segment-identity ceiling a data/label/feature problem rather than an architecture problem?

**Target regime under test.** Per-frame input = 144-dim hand-crafted feature (YOLO detections over a 2×3 ROI grid: per (class, region) [presence, count, max_area]). Tiny dataset: tens of thousands of frames total, 2,639 test frames across **8 test videos**, ~9 classes + idle, dense per-frame labels. Business metrics are segment-level: edit distance of the predicted segment-label sequence, F1@k at IoU 0.1/0.25/0.5, insertion recall, frame accuracy. Best model: MS-TCN++ → edit 51.08, F1@0.1 40.01, frame acc 53.98. Linear probe: edit 17.86 / F1@0.25 5.31.

**Failure mode under test.** 51.6% of ground-truth segments receive a wholly wrong label; 78% of frame errors are deep inside segments (not at boundaries); insensitive to capacity, feature-set changes, loss weights, augmentation.

**Evidence tiers used below**

| Tag | Meaning |
|---|---|
| **VERIFIED-P** | I fetched the primary source myself in this session and read the quoted text/table. Local copy named in `raw/`. |
| **VERIFIED-D** | A delegated sub-investigator fetched the primary source; raw HTML/PDF text is in `raw/` and I inspected the extracted text. |
| **DERIVED** | My own arithmetic or interpretation *on top of* a paper's own numbers. **Not a claim the paper makes.** |
| **UNVERIFIED** | Could not confirm; what I tried is stated. |

A note on honesty: I was asked to find evidence that architecture *cannot* break a segment-identity ceiling. **I found substantial evidence that it can.** Section A.4 and B.1 report it in full. The genuine, well-supported conclusion is narrower: *capacity* is not the lever in this regime; *output formulation, the decode-time structural prior, and the class-balance of the objective* are — and in this project's data regime the measurement noise is large enough to swamp most architecture deltas.

---

# A. Does architecture matter, or do features/labels dominate?

## A.1 The TPAMI 2023 TAS survey — what it actually claims

**Source.** Guodong Ding, Fadime Sener, Angela Yao, "Temporal Action Segmentation: An Analysis of Modern Techniques", **IEEE TPAMI** (arXiv `Comments:` field: "19 pages, 9 figures, 8 tables, TPAMI 2023"). URL: https://arxiv.org/abs/2210.10352 · full text https://arxiv.org/html/2210.10352v5
**Venue independently confirmed** via Crossref `10.1109/TPAMI.2023.3327284` → container-title "IEEE Transactions on Pattern Analysis and Machine Intelligence", authors [Ding, Sener, Yao], published 2024-02 (issue). **VERIFIED-P.**

**Finding A1a — the survey explicitly frames pre-computed features as the thing that makes architecture comparisons interpretable, and as a *confound* in the other direction.**
§III-A "Frame-Wise Representations", first paragraph:
> "The standard practice in TAS is to use pre-computed frame-wise features, without end-to-end learning, as inputs. This convention is due to the heavy computational demands of learning video features. **Using pre-computed features has a key advantage in that it allows a dedicated comparison of the proposed architectures without the confounding influences of improved frame-wise feature representations.**"

**Finding A1b — the survey states that nobody has done the end-to-end-vs-precomputed comparison, and that pre-computed features bias toward static cues.**
§V "Conclusions and Outlook", subsection "Input Features":
> "Nonetheless, as pointed out by [161, 162], **pre-computed characteristics tend to favour static cues, e.g., scene components, within frames.** To the best of our knowledge, **no empirical research has compared utilizing pre-computed features to training TAS models from raw images end-to-end** due to the high demands in terms of training efficiency and GPU memory requirements."

This is the closest the field's own survey comes to a statement that the feature side is under-studied. Note it is an *absence* statement, not evidence that features dominate. **Confidence: high for the quote; do not read it as "features are the bottleneck".**

**Finding A1c — the survey says the field's frame-wise formulation is the wrong objective, on the field's own reading.** §V, "Segment-Level Modeling":
> "Exploring how to add sequence-based losses, such as edit scores that penalize segment-wise mistakes, into the learning process is an interesting but under-explored direction. Segment-level losses readily coincides with the first interpretation of the TAS task (Eq. 1) while the majority of existing techniques follow frame-wise prediction (Eq. 2). **We recommend a greater emphasis on solving the task at the segment level** and anticipate that it can greatly reduce the current issues with oversegmentation."

**Finding A1d — the survey explicitly flags boundary-label uncertainty as a performance-relevant open problem.** §V, "Forms of Supervision":
> "An additional component of supervision to consider is the inherent uncertainty of the action boundaries, as actions occurring in time are frequently not as distinct as an object in space. According to [55], **these uncertainties in action boundaries can have a significant impact on model performance.** It is, therefore, worthwhile to investigate how to define/label action boundaries."

**Finding A1e — the survey confirms that sparse (timestamp) supervision is *nearly as good* as dense supervision, i.e. dense frame labels carry large redundancy.** §V, "Forms of Supervision":
> "Procedural video sequences feature **enormous temporal redundancy in the supervisory signals** due to the significant similarity between successive video frames of the same motion. Such redundancy is been proven by the comparable performance of using timestamp supervision [136] vs. fully-supervised setting [136, 57]."

**Finding A1f — the survey reports EM-TSS's finding that boundary frames are the *worst* place to put a label** (repeated in §IV-B/weak-supervision discussion):
> "EM-TSS also shows that **selecting the boundary frames as timestamps for each action segment impairs performance compared to using random or middle frames, highlighting the ambiguity of labels at the boundaries.**"

**Finding A1g — the survey reports class imbalance ratios and that global metrics hide it.** Table IV ("Imbalance Ratio (IR) on four TAS datasets"): GTEA **24**, 50Salads **6**, Breakfast **639**, Assembly101 **2604**. Fig. 4 caption: "The head class 'fry pancake' is **639× more frequent** than the tail class 'take butter'." §II-D1:
> "**The MoF metric can be problematic under dataset imbalance**, i.e. if frequent and long action classes dominate. The long-tailed nature of current datasets (see Section II-B6) implies that **models with similar MoF scores may have large qualitative differences**, suggesting that class-balanced metrics may be more appropriate though this is currently not adopted in the literature."

**Finding A1h — the survey's own definition establishes that edit score is an *ordering/identity* metric that explicitly forgives boundary shift.** §II-D1, Fig. 5 caption:
> "(a) MoF estimates how accurate frame wise predictions are. **(b) Edit score tolerates small boundary shifts, as long as the sequence order is correct.** (c) F1 score is the harmonic mean of the precision and recall of the detected action segments."
And: "As a metric, the Edit score can assess how well a model predicts **the sequence of actions without requiring exact frame-wise correspondence to the ground truth**."

**This is directly load-bearing for the project.** Edit score is normalised edit distance: `Edit = (1 − e/ max(|X|,|Y|)) · 100` (survey Eq. 7). **DERIVED:** with |X| ≈ |Y| and errors dominated by substitutions, an edit score of **51.08** implies a substitution rate of ≈ 48.9% of segment slots — which is almost exactly the project's independently measured **51.6% wholly-wrong-segment rate**. The two numbers are mutually consistent with a *single* failure mode: the model finds the segments but gives ~half of them the wrong identity. This is the internal-consistency argument for treating this as an identity problem, not a boundary problem. **Confidence: high** (arithmetic is elementary and the survey supplies the exact formula; the agreement of 48.9% vs 51.6% is strong).

---

## A.2 Controlled feature-swap evidence: hand-crafted vs I3D on Breakfast

This is the single most directly relevant question for a project running on a 144-dim hand-crafted feature. Three independent sources converge, and the direction is **not** what the "upgrade your features" intuition predicts.

### A2a. MS-TCN (CVPR 2019) — same architecture, same dataset, IDT vs I3D

**Source.** Yazan Abu Farha, Juergen Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", **CVPR 2019** (arXiv `Comments:` "CVPR 2019 Camera Ready"). https://arxiv.org/abs/1903.01945 · full text https://arxiv.org/html/1903.01945v2. **VERIFIED-P.**

**Exact table values — Table 10, Breakfast section** (columns F1@{10,25,50} | Edit | Acc):

| Row | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---|---|---|---|---|---|
| MS-TCN (IDT) | **58.2** | **52.9** | **40.8** | 61.4 | 65.1 |
| MS-TCN (I3D) | 52.6 | 48.1 | 37.9 | **61.7** | **66.3** |

**The paper's own conclusion** (§4.9 "Comparison with the State-of-the-Art"):
> "Note that all the reported results are obtained using the I3D features. To analyze the effect of using a different type of features, we evaluated our model on the Breakfast dataset using the improved dense trajectories (IDT) features, which are the standard used features for the Breakfast dataset. As shown in Table 10, **the impact of the features is very small.** While the frame-wise accuracy and edit distance are slightly better using the I3D features, **the model achieves a better F1 score when using the IDT features compared to I3D.** This is mainly because I3D features encode both motion and appearance, whereas the IDT features encode only motion. For datasets like Breakfast, **using appearance information does not help the performance since the appearance does not give a strong evidence about the action that is carried out.** ... **The video frames share a very similar appearance. Additional appearance features therefore do not help in recognizing the activity.**"

**Reading.** Replacing a hand-crafted *motion-only* feature with a 2048-dim Kinetics-pretrained deep spatio-temporal feature changed Edit by **+0.3**, Accuracy by **+1.2**, and made all three F1 scores **worse** (F1@10 **−5.6**, F1@25 −4.8, F1@50 −2.9). **Confidence: high** (I read Table 10 and §4.9 directly).

### A2b. Souri et al. — a dedicated study of feature extractors for TAS, on the same dataset

**Source.** Yaser Souri, Alexander Richard, Luca Minciullo, Juergen Gall, "On Evaluating Weakly Supervised Action Segmentation Methods", arXiv `Comments:` **"Technical Report"** (no peer-reviewed venue printed). https://arxiv.org/abs/2005.09743 · full text https://arxiv.org/html/2005.09743v3. **VERIFIED-P.**

**Abstract:**
> "Furthermore, our investigation on feature extraction shows that, **for the studied weakly-supervised action segmentation methods, higher-level I3D features perform worse than classical IDT features.**"
§1: "**Unexpectedly, higher-level I3D features do not perform better than low-level IDT features.**"

**Exact table values — Table 2** ("Average MoF is computed by running the training and testing of each method 5 times on split 1 of the Breakfast dataset"):

| Approach | IDT | I3D | PCA-I3D |
|---|---|---|---|
| NNV | 40.6 | **11.4** | 23.2 |
| CDFL | 48.9 | **34.9** | 38.0 |
| MuCon | 40.2 | **48.3** | 47.7 |

**Two things follow, and they cut in opposite directions:**

1. **The feature swap can be enormous — up to −29.2 MoF** (NNV). That is a far bigger number than any architecture delta in Table V of the survey. So "features don't matter" is **false**.
2. **The sign is architecture-dependent** (−29.2, −14.0, **+8.1** for the three methods), and the paper explicitly refuses to explain it:
   > "This indicates that current methods that perform well for weakly supervised action segmentation on Breakfast using IDT features cannot be directly applied to other features like I3D. Although the experiments do not fully reveal why this is the case, there are three possible explanations: a) I3D features do not work well on Breakfast in general, b) **the existing methods need to be adapted to I3D**, c) there is a need for different methods that benefit from I3D features."
   And the closing line: "we observed that current methods do not benefit from 'better' features. **However, it remains an open research question why this is the case.**"

**DERIVED:** Table 3 in the same paper shows a *preprocessing* choice — the input feature temporal window — moving NNV's Breakfast MoF from 23.2 (window 21) to **33.9** (window 1). That is **+10.7 MoF from feature-window engineering alone**, on the same features and same architecture, which is larger than most published architecture deltas on Breakfast. **Confidence: high for the numbers; the comparison to "architecture deltas" is my framing.**

### A2c. CLIP features are catastrophically worse than I3D on 50Salads

**Source.** LTContext, "How Much Temporal Long-Term Context is Needed for Action Segmentation?", **ICCV 2023** (arXiv `Comments:` "ICCV 2023"). https://arxiv.org/abs/2308.11358 · full text https://ar5iv.labs.arxiv.org/html/2308.11358, Appendix C "Other Features" / Table 12. **VERIFIED-P** (I re-fetched this myself after the sub-investigator reported it).

**Exact quote:**
> "In order to evaluate the impact of using vision-language models, we extract features using CLIP [43] from 50Salads and report the result of action segmentation in Table 12. **Without additional fine-tuning, the features do not perform well.**"

**Table 12 (Results are on 50Salads):**

| Features | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---|---|---|---|---|---|
| CLIP | 65.8 | 57.6 | 44.2 | 64.2 | 62.4 |
| I3D | 89.4 | 87.7 | 82.0 | 83.2 | 87.7 |

**DERIVED deltas:** CLIP vs I3D = **−23.6 F1@10, −19.0 Edit, −25.3 Acc**. Swapping *to* a modern vision-language feature on the same architecture and dataset cost ~19–25 points. **Caveat stated by the authors**: "Without additional fine-tuning". **Confidence: high** that these are the paper's numbers; **medium** on generalising — a fine-tuned CLIP feature is a different experiment, and I found no verified TAS result doing that on 50Salads/Breakfast.

### A2d — the inverse signature: fragmentation = high accuracy, low F1/Edit

From the same appendix (Appendix D, "Alternative Efficient Attentions"):
> "These types of attention focus on sparseness and result in **fragmented segments, which is indicated by high accuracy, but very low F1 and Edit scores.**"

**DERIVED:** this is the *mirror image* of the project's failure mode and a useful diagnostic. High-accuracy/low-F1-Edit = fragmentation (boundary problem). The project has the opposite signature — its frame accuracy is low while its edit score is not relatively worse — consistent with identity error. **Confidence: medium** (the quote is verified; the diagnostic mapping is my inference).

### A2e — evidence that a feature swap DID help, on a different domain

**Source.** M2R2, "M2R2: MultiModal Robotic Representation for Temporal Action Segmentation", https://arxiv.org/abs/2504.18662 (arXiv `Comments:` "8 pages, 6 figures, 2 tables" — **no venue printed**; venue **UNVERIFIED**). Full text https://arxiv.org/html/2504.18662, Tables I/III/IV. **VERIFIED-D.**

On REASSEMBLE (fine-grained labels) with the TAS model held fixed at DiffAct, replacing the vision-only extractor (Br-Prompt) with multimodal M2R2 features moves F1@10 **12.0 → 78.1** (+66.1) and Edit **21.4 → 68.7** (+47.3). Holding *features* fixed and swapping architecture (MS-TCN / ASRF / DiffAct) moves F1@10 only 83.1 / 83.5 / 78.1 — a spread of **5.4**. **DERIVED** framing (+66 vs +5.4) is the sub-investigator's arithmetic.

**Honest counter-note from the same paper's Table IV:** on REASSEMBLE, `only proprio` = 78.0/75.4/69.2 vs `all modalities` = 78.1/74.9/68.7 — i.e. on that dataset the vision and audio modalities add essentially **nothing** over proprioception alone. **DERIVED reading:** this is a proprioception-dominated benchmark, not proof that visual features are decisive. **Domain caveat: robotic TAS with force/torque/pose modalities — not video-only TAS, and not this project's regime.**

### A2f. The cleanest controlled feature-vs-architecture answer: a 14-encoder sweep at fixed pipeline

This is the strongest single piece of evidence in the whole feature-vs-architecture question, and it also independently reproduces the project's own "insensitive to capacity" observation from a different direction.

**Source.** "Exploring Vision-Language Models for Open-Vocabulary Zero-Shot Action Segmentation", arXiv `Comments:` **"ICRA 2026"** (per the sub-investigator; I did not re-check the Comments field myself — **venue VERIFIED-D**). https://arxiv.org/abs/2602.21406 · full text https://arxiv.org/html/2602.21406v1. **VERIFIED-P** (I re-fetched and read Tables V–VIII and §V-1 myself).

**Setup.** A **training-free, zero-shot** "segmentation-by-classification" pipeline (OVTAS) with two stages — FAES (frame–action embedding similarity) and SMTS (similarity-matrix temporal segmentation). The pipeline is held fixed and **only the VLM encoder is swapped**, across **14 VLMs from 4 families**. `Avg` = mean of F1@10, F1@25, F1@50, Edit, Accuracy.

**Table VI — encoder-choice effect (Avg), holding the pipeline fixed** (n=14 encoders):

| Dataset | best VLM (Avg) | worst VLM (Avg) | **DERIVED spread** |
|---|---|---|---|
| Breakfast | 46.6 (CLIP-M2) | 44.4 (PECore-M2) | **2.2** |
| 50 Salads | 41.7 (SigLIP-M1) | 34.9 (PECore-M2) | **6.8** |
| GTEA | 23.9 (SigLIP-M2) | 15.0 (PECore-M2) | **8.9** |

**Table VII — pipeline-stage ablation (removing one stage), holding the encoder fixed** (Avg drop vs Best):

| Dataset | remove Stage1 (↓Avg) | remove Stage2 (↓Avg) |
|---|---|---|
| Breakfast | −32.76 | **−40.56** |
| 50 Salads | −31.41 | **−38.80** |
| GTEA | −15.76 | **−20.50** |

**DERIVED — the headline comparison: method-stage removal costs 15.8–40.6 Avg points; the entire 14-encoder choice spans 2.2–8.9. The method is worth roughly 1.8× to 15× the encoder.** On this evidence the answer to "does swapping the feature extractor buy more than changing the architecture?" is **no, by a wide margin.**

**And the paper independently reproduces the "capacity doesn't matter" result — for encoders, not for TAS backbones** (§V-1, exact quote):
> "The plot in Fig. 4 shows that **simply scaling VLMs up does not yield better action-segmentation performance using OVTAS—in all VLM families. Larger checkpoints underperform their smaller counterparts.**"

**DERIVED, and this is the striking part:** per Table V the 14 encoders span **149.62M (CLIP/OpenCLIP ViT-B/16) to 2419.27M (PECore-M3) parameters — a 16× range** — and the **best Breakfast result is CLIP-M2 at 149.62M, the smallest model in the study**. A 16× parameter increase buys nothing; the ordering is, if anything, inverted. **This is a capacity-independence result on the input side that parallels the project's capacity-independence result on the model side.**

**Bonus finding from the same paper, directly relevant to a small densely-labelled test set** (§V-2):
> "For GTEA, where each video contains **many short segments (mean ∼36)**, the model struggles compared to Breakfast, where the mean is ∼5 segments. 50 Salads lies between these two extremes. This shows that **the number of fine-grained action boundaries strongly influences performance, with dense sequences of short actions being particularly challenging.**"
Table X (segment durations): GTEA mean **1.94 s**, 50 Salads **18.59 s**, Breakfast **20.95 s**. Table VIII: performance degrades monotonically with video length. **DERIVED: GTEA has ~7× more segments per video than Breakfast and the worst Avg (23.9 vs 46.6).** **The project's regime — ~9+1 classes, dense per-frame labels, many segments — is the GTEA-like end of this axis, which is the hardest configuration in this study.**

**Caveat, stated plainly: OVTAS is zero-shot and training-free.** The "architecture spread" here is a *pipeline-stage* spread, not a spread over trained TAS backbones, so it does not by itself refute the trained-architecture gains in A.3 (MS-TCN→DiffAct). It does decisively settle the *encoder-choice* question in the regime where features are the only thing varying.

---

## A.3 How much does architecture move the needle? (the honest counter-evidence)

I have to report this clearly: **on the standard benchmarks, architecture changes after MS-TCN moved segment metrics a lot.**

**Source.** Survey Table V ("Performance of supervised TAS methods on GTEA and Breakfast"), compiled by the TPAMI survey from the original papers. **VERIFIED-P.**

**Breakfast column only** (F1@10 | F1@25 | F1@50 | Edit | MoF):

| Method | Year | Feature | F1@10 | Edit | MoF |
|---|---|---|---|---|---|
| MS-TCN | 2019 | IDT | 58.2 | 61.4 | 65.1 |
| MS-TCN | 2019 | I3D | 52.6 | 61.7 | 66.3 |
| MS-TCN++ | 2020 | I3D | 64.1 | 65.6 | 67.6 |
| ASFormer | 2021 | I3D | 76.0 | 75.0 | 73.5 |
| UVAST | 2022 | I3D | 76.9 | 77.1 | 69.7 |
| DTL + ASFormer | 2022 | I3D | 78.8 | 77.7 | 75.8 |
| DiffAct + ASFormer | 2023 | I3D | **80.3** | **78.4** | **76.4** |

**DERIVED:** MS-TCN (I3D) → DiffAct + ASFormer is **+16.7 Edit** and **+27.7 F1@10** on Breakfast, all at *fixed I3D features*. By contrast the IDT→I3D swap at fixed architecture is **+0.3 Edit** (and negative on F1). On Edit, architecture is worth ~55× the feature swap on this table; on F1@10 the feature swap is *negative* while architecture is +27.7.

**MS-TCN's own ablation tables make the same point, harder.** From MS-TCN (CVPR 2019), **VERIFIED-P**:

- Table 1 (50Salads, stages): SS-TCN 27.0 F1@10 → MS-TCN 2 stages 55.5 → 3 stages 71.5 → 4 stages **76.3** → 5 stages 76.4 (saturates).
- Table 2 (50Salads): SS-TCN (48 layers, deep single-stage) 49.0 F1@10 vs MS-TCN **76.3**.
- Table 3 (50Salads, loss): `L_cls` only 71.3 F1@10 / 64.2 Edit → `L_cls + λ·L_T-MSE` (smoothing) **76.3 / 67.9**.
- Table 9 (GTEA, §4.8 "Impact of Fine-tuning the Features"): MS-TCN w/o FT F1@10 85.8 / Edit 79.0 / Acc 76.3 → **with FT 87.5 / 81.4 / 79.2**. So **feature fine-tuning buys +2.4 Edit**, while §4.8's own text says:
  > "Fine-tuning improves the results, but **the effect of fine-tuning for action segmentation is lower than for action recognition. This is expected since the temporal model is by far more important for segmentation than for recognition.**"

**DERIVED, and this is the important synthesis:** within a *single* paper the architecture delta (SS-TCN→MS-TCN, **+24.0 Edit**) is roughly **10×** the feature delta (no-FT→FT, **+2.4 Edit**), and the IDT→I3D swap is **~0**. Point 3 of the premise — *"architecture changes give small gains in this regime"* — is **NOT** supported for the standard benchmarks.

### A3b — but capacity specifically is not the lever; the *output formulation* is

This is the most useful nuance I found, and it is verified.

**Source.** Nadine Behrmann, S. Alireza Golestaneh, Zico Kolter, Juergen Gall, Mehdi Noroozi, "Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation" (**UVAST**), **ECCV 2022** (arXiv `Comments:` verified; survey Table V lists it as ECCV 2022). https://arxiv.org/abs/2209.00638 · full text https://arxiv.org/html/2209.00638v2. **VERIFIED-P.**

**The paper's central claim:**
> "In contrast to current state-of-the-art frame-level prediction methods, we view action segmentation as a **seq2seq translation task**, i.e., mapping a sequence of video frames to a sequence of action segments."

**And the crucial parameter-matched comparison** (§5, comparing to ASFormer):
> "As expected, **ASFormer achieves in general a better frame-wise accuracy while UVAST achieves a better Edit score.** Since ASFormer uses a smoothing loss and multiple refinement stages to address over-segmentation similar to MS-TCN, **it has ∼1.3M learnable parameters, whereas our proposed model has ∼1.1M parameters.**"

**DERIVED:** UVAST beats ASFormer on Edit (Breakfast 77.2 with its alignment decoder / 77.1 with FIFA vs ASFormer's 75.0; 50Salads 83.9 vs ASFormer's 79.6 — UVAST Table 1) with **fewer parameters**, while losing on frame accuracy (Breakfast 68.2 vs ASFormer's 73.5). **The gain comes from changing the prediction target (a segment/transcript sequence) rather than from capacity.** This is the single most actionable architecture result in this survey: capacity is not the lever, but the **objective/output space** is. **Confidence: high** for the quotes and the parameter counts.

**UVAST's own ablation quantifies it** — Table 4, Breakfast split 1 (F1@10 | F1@25 | F1@50 | Edit | Acc):

| Configuration | F1@10 | Edit | Acc |
|---|---|---|---|
| Vanilla Transformer (segment-level output only) | 48.1 | 52.9 | 35.0 |
| + frame-wise auxiliary loss | 70.7 | **73.9** | 59.1 |
| + alignment decoder | 77.1 | **78.2** | 71.7 |

**DERIVED:** the frame-wise auxiliary loss alone is worth **+21.0 Edit**; the alignment decoder adds **+4.3**. Conversely Table 5 shows the segment loss alone gives Edit 55.6 (Breakfast) while the frame loss alone gives Edit **14.1** — *both* are required. **Caveat: the "Vanilla Transformer" baseline is a deliberately weak starting point, by the authors' own admission** ("this seemingly natural approach does not immediately perform well on the action segmentation task by itself"). **Confidence: high for the table; do not read `+21` as "add a frame loss to any model and get +21".**

**Two further UVAST statements that matter for this project:**

1. **On why a pure segment-level model fails on small data** (§1):
   > "In contrast to language translation, action segmentation typically involves **long input sequences of very similar frames** opposed to short output sequences of action segments. This difference **together with the relatively small amount of training videos**, makes it challenging for the encoder and decoder to keep track of the full information flow that is necessary to predict the high-level segmentation alone."
2. **On the frame-accuracy / segment-metric decoupling** (§3.1):
   > "the encoder performs frame-wise classification with **high localization performance, i.e., high frame-wise accuracy, but low discrimination performance**, i.e., over-segmentation with **low Edit distance** to the ground truth."

**That last sentence is the literature's clearest statement of the localization-vs-discrimination decomposition** — exactly the axis on which this project is failing. **Confidence: high.**

---

## A.4 Data scaling: what happens at low data?

### A4a. Learning curves over % of training data — three verified ones

**Source 1.** C2F-TCN: A Framework for Semi and Fully Supervised Temporal Action Segmentation, Singhania, Rahaman, Yao, https://arxiv.org/abs/2212.11078 (arXiv `Comments:` only an admin note; **venue UNVERIFIED**). Table XIII. **VERIFIED-D.**

Breakfast (F1@10 / Edit / MoF):

| Label budget | Supervised-only | Semi-supervised (ICC4) |
|---|---|---|
| ≈5% videos | 15.7 / 19.8 / 26.0 | 60.2 / 56.6 / **65.3** |
| ≈10% videos | 35.1 / 36.3 / 40.3 | 64.6 / 61.9 / 68.8 |
| 100% | 70.8 / 67.5 / **74.3** | — |

The paper's own summary (§VI-F2):
> "**with just 5% of labeled videos, there is only 9% less in MoF in the Breakfast actions compared to fully supervised (100%).**"

**DERIVED and important:** the *supervised-only* 5% baseline collapses to MoF 26.0 / F1@10 15.7, while a *semi-supervised* method at the same 5% reaches MoF 65.3. The gap is not "more data" — it is **how the unlabelled data is used**. Note also the paper's caveat: "Using less than 5% (3 videos for 50Salads and GTEA) for training videos does not ensure coverage of all the actions." **Confidence: high for the table; the semi-supervised reading is DERIVED.**

**Source 2.** "Distill and Collect for Semi-Supervised Temporal Action Segmentation", https://arxiv.org/abs/2211.01311, Table 3, Breakfast Split 3: 20% → MoF 53.6, 35% → 60.7, 45% → 64.6, 100% → 69.7. **VERIFIED-D.**

**Source 3.** SMC-NCA, arXiv 2312.12347, `Comments:` **"Accepted to IEEE Transactions on Multimedia"**, Table VIII (accuracy at 5/10/40/100% of data): 50Salads 68.9 → 73.6 → 80.1 → 87.0; GTEA 73.9 → 77.5 → 79.5 → 82.6; Breakfast 68.9 → 69.7 → 73.1 → 76.4. **VERIFIED-D.**

**A4b — the strongest low-data result in this survey, and it is about *labels*, not model size.**

**Source.** Hilde Kuehne, Alexander Richard, Juergen Gall, "A Hybrid RNN-HMM Approach for Weakly Supervised Temporal Action Segmentation", arXiv `Comments:` "15 pages, **preprint for IEEE TPAMI**" → **TPAMI**. https://arxiv.org/abs/1906.01028 · full text https://ar5iv.labs.arxiv.org/html/1906.01028. **VERIFIED-P.**

**Table IV, Breakfast MoF, "fraction indicates how many frames of the data were labeled":**

| Fraction of frames labelled | Breakfast MoF |
|---|---|
| 0.0 | 36.7 |
| **0.0025** | **56.0** |
| 0.01 | 58.8 |
| 0.1 | 60.9 |
| 1.0 (full) | 61.3 |

**DERIVED:** labelling **0.25% of frames** moves Breakfast MoF from 36.7 to **56.0** (+19.3), and going from 0.25% to 100% (a **400× increase in labels**) buys only a further **+5.3**. **Confidence: high** (I read Table IV directly).

**And this is on a 64-dim hand-crafted feature** (§V-B "Features"): IDT + Fisher vectors, reduced from 426→64 by PCA for the IDT input, and the final 8,192-dim FV representation reduced **to 64 dimensions** by PCA. Table VII (fully supervised Breakfast MoF): HMM-BOW 28.8, HMM-FV 56.3, TCFPN 52.0, **GRU w/o length prior 60.2, GRU + length prior 61.3**.

**DERIVED — this is directly reassuring for a 144-dim hand-crafted setting:** a 64-dim hand-crafted feature plus a GRU+HMM reaches **Breakfast MoF 61.3**, which is *comparable to* MS-TCN on IDT (65.1) and *better than* TCFPN (52.0) and HMM-FV (56.3). Low dimensionality per se is not the ceiling; the project's 144-dim feature is not obviously the binding constraint.

**A4c — few-shot TAS is thin, and the one pointer paper is a different modality.**

The task's suggested pointer, arXiv 2207.09925, is titled "An Efficient Framework for **Few-shot Skeleton-based** Temporal Action Segmentation" — skeleton, not RGB/detection features. Its Table 6 is a genuine learning curve over 2/4/6/8 training sequences and is **non-monotonic** (6 < 4 on every metric). **VERIFIED-D.** Few-shot *segmentation* literature is thin; searches for few-shot TAS return mostly few-shot recognition/localisation. **Confidence: medium-high.**

---

## A.5 Label noise, annotation ambiguity, annotator variance

### A5a. VERIFIED-NEGATIVE: the "pour milk vs pour coffee is ambiguous" premise is only half true

I was asked to check the Breakfast and 50Salads papers for their own statements about fine-grained class ambiguity. The honest answer is that **the taxonomy exists but the stated ambiguity does not.**

**Source.** H. Kuehne, A. Arslan, T. Serre, "The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities", **CVPR 2014**. Open-access PDF: https://openaccess.thecvf.com/content_cvpr_2014/papers/Kuehne_The_Language_of_2014_CVPR_paper.pdf. **VERIFIED-D.**

- **Yes, there are 10 near-identical "pour X" classes.** Fig. 4 enumerates: `pour milk`, `pour juice`, `pour cereals`, `pour water`, `pour coffee`, `pour oil`, `pour dough2pan`, `pour egg2pan`, `pour sugar`, `pour flour` — plus `spoon flour`, `spoon sugar`, `spoon powder`, `stir milk`, `stir cereals`, `stir coffee`, `stir egg`, `stir fruit`, `stir dough`, `stirfry egg`.
- **But the paper never states these classes are visually indistinguishable**, and it measures **no inter-annotator agreement** between its 3 coarse annotators or its 15 fine annotators.
- Its own ambiguity language is **activity-level, not unit-level**: "related activities like the preparation of drinks vs. food tend to be more often confused... Closely related activities such as drinking or eating tend to be more easily confused." A unit-level confusion matrix exists as Fig. 8 but only as an image; I could not recover its cells as text. **UNVERIFIED — I fetched the CVPR PDF text; Fig. 8 cells are not extractable.**
- The paper's stated main limitation is annotation **cost**: "One of the main limitations of the system is the need for annotations at the level of individual action units, which is a long and tedious task for annotators."

**50Salads explicitly claims the opposite of ambiguity.** Stein & McKenna, UbiComp 2013: "**Annotations are therefore unambiguous and repeatable.**" Its only confusability admission is `mix dressing` vs `mix ingredients` ("the exhibited motion pattern... is similar"). **VERIFIED-D.**

**Conclusion for the project: do NOT cite Breakfast/50Salads as evidence that their authors consider the classes ambiguous. That claim would be fabricated.** The near-identical-class problem is real as a *taxonomy* fact; its status as a *visual* impossibility is not established by these papers. Similarly, **no agreement metric is reported in the Breakfast, 50Salads, or Assembly101 papers themselves.** **CORRECTION to the task brief: Assembly101 is arXiv 2203.14712** (2203.15110 is "The State of Fortran").

### A5b. VERIFIED: inter-annotator boundary variance measured *on Breakfast*, and it is large

**Source.** R. Rahaman, D. Singhania, A. Thiery, A. Yao, "A Generalized & Robust Framework For Timestamp Supervision in Temporal Action Segmentation", **ECCV 2022** (venue confirmed from the ECVA proceedings PDF `eccv_2022/papers_ECCV/papers/136640276.pdf`; the arXiv abs page has no `Comments:` field). https://arxiv.org/abs/2207.10137. Appendix 0.D, "Details of user study". **VERIFIED-D.**

> "For full-supervision annotations, we found that the **start times of the same action segments, marked by different annotators, had a standard deviation of ≈1.5 seconds or 23 frames (breakfast containing frames at 15fps).** This further gives us an indication of the **ambiguity in annotating the boundary frames.**"

**DERIVED:** at 15 fps, σ ≈ 23 frames of disagreement on where a segment *starts*. Breakfast segments are on the order of tens of frames. So a substantial fraction of ground-truth boundary frames are contested *between human annotators*.

**And it moves metrics a lot.** §5.4:
> "**Ambiguity in Boundary Frames.** ... There is a **≥20% difference in scores** when trained with boundary frames (start/end frame) versus frames selected at (Random) or from the middle (Mid) of the action segment. This indicates ambiguity in boundary frame annotation."

Table 5 right sub-table (50Salads, Naive baseline, F1@25 | F1@50 | Edit | MoF): Random 39.4 / 29.3 / 34.2 / 69.9 · Mid 47.7 / 34.9 / 40.9 / 69.7 · **Start 12.5 / 3.2 / 17.5 / 31.9** · End 12.2 / 2.7 / 21.4 / 29.1.

**DESERVEDLY SURPRISING RESULT — EM-TSS beats full supervision on 50Salads.** Main text Table 2 (50Salads): full supervision F1@25/50 = 67.7 / 58.6, Edit 63.8, MoF 77.8; **EM-TSS = 75.4 / 63.7, Edit 70.9**, MoF 77.3. **DERIVED but the paper says it too:** the authors explain this via boundary ambiguity, and their §4.1 motivates soft labels because "The assignment of hard labels to each frame may lead to some ambiguities at the action boundaries and deteriorate performance". **If a method trained on *less* information beats full supervision, the ground-truth labels themselves are part of the error budget.** **Confidence: high for the numbers** (verified from the paper's Table 2 by the sub-investigator; I flag it as VERIFIED-D).

**Missing-segment (annotation *omission*) errors also collapse performance.** Abstract: "performance rapidly collapses under subtle violations of the annotation assumptions." §5.2: "existing TSS methods suffer from poor performance with missing segments, with a **sharp drop in performance at 5% and at 20% missing segments.**" **VERIFIED-D.**

**Important scoping caveat the paper itself makes (§2):**
> "Note that **this sort of annotation error differs from the typical annotation error of mislabelling** as it emerges from the difficulty of adhering to annotation constraints."

So Rahaman et al. measure **boundary** noise and **omission** noise, not **class-label (identity) noise**. That distinction matters for interpreting the project's failure mode.

### A5c. Boundary-annotator agreement, general dense-video annotation

- **Alwassel et al., ECCV 2018** (arXiv 1807.10706): "**168 Turkers** to re-annotate temporal boundaries of actions from ActivityNet... Turkers exhibited an **agreement score of 64.1%**." **VERIFIED-D.** *(Note: their own conclusion is that this is **not** a roadblock to benchmarking — dissenting evidence, reported below.)*
- **Sigurdsson et al., ICCV 2017** (arXiv 1708.02696): agreement "**72.5%** IOU... for Charades and **58.7%** IOU in MultiTHUMOS"; masking boundary regions raises consensus 72.5 → **79.8%** and mAP 9.6 → **10.9%**. **VERIFIED-D.**
- **Moltisanti PhD thesis (Bristol, 2019)** — readable primary source for "Trespassing the Boundaries" (the ICCV-2017 CVF PDF uses outline fonts and yields no extractable text, so the thesis is the accessible version). BEOID: 5 annotators avg IOU **0.62** (start SD 0.62 s); 100 AMT annotators **0.57**; rubric-guided "Rubicon Boundaries" raises avg IOU to **0.81** (start SD 0.14 s). Perturbing only *test* boundaries drops top-1 accuracy: BEOID 2SCNN **93.5 → 83.8**, IDT 85.3 → 75.4. Better labels raise accuracy **61.2 → 65.6** on GTEA Gaze+. **VERIFIED-D.** **DERIVED:** the last two are the cleanest available natural experiment that *label quality*, not model capacity, gates accuracy.
- **EPIC-KITCHENS-100 (Damen et al., IJCV 2022):** "increased the number of annotators per segment to 5, compared to 4... This resulted in **higher agreements between annotators**." **VERIFIED-D.** (Annotator-count → agreement dose-response.)
- **Assembly101 (CVPR 2022, arXiv 2203.14712):** screens annotators ("annotators who were slow or made many mistakes were not selected to continue"), used 21 annotators / 213 hours, and **reports no agreement metric at all**. **VERIFIED-D.**

### A5d. UNVERIFIED / could not find (label-noise track)

- **UNVERIFIED — no paper found that injects class-label noise (mislabelling) into Breakfast/50Salads/GTEA and measures the TAS effect.** What I tried: arXiv HTML search for "label noise action segmentation", "action segmentation annotation", "annotator", "boundary ambiguity action segmentation", "noisy labels temporal action segmentation"; arXiv API; Semantic Scholar; OpenAlex. **The arXiv search channel returned HTTP 429 for essentially the entire session and the arXiv API returned HTTP 406, so treat this as *failed queries*, not as negative evidence.**
- **UNVERIFIED — SEDT** ("Stacked encoder–decoder transformer with boundary smoothing", Electronics Letters 58(25):972–974, 2022): metadata only via Crossref; the Wiley page returned 403 and the code repo 404. **Do not cite SEDT for content.**
- **UNVERIFIED — the exact numeric table of "Trespassing the Boundaries" (ICCV 2017) as published**; only the thesis version (A5c) was extractable.
- **UNVERIFIED — any inter-annotator agreement figure for Breakfast, 50Salads, or Assembly101 printed in those datasets' own papers.** (A5b provides a third-party measurement on Breakfast instead.)

---

## A.6 Summary of section A

| Claim | Verdict | Confidence |
|---|---|---|
| "Upgrade hand-crafted features → deep features (I3D) fixes TAS" | **NOT SUPPORTED on Breakfast.** Three independent sources: IDT ≥ I3D for MS-TCN, and I3D much worse for NNV/CDFL. | HIGH |
| "Encoder choice matters more than the method" | **REFUTED at fixed pipeline.** 14-VLM sweep: encoder spread 2.2–8.9 Avg vs method-stage removal 15.8–40.6 Avg (ICRA 2026). | HIGH |
| "Bigger encoders are better encoders" | **REFUTED.** 16× parameter span; smallest model wins Breakfast; "Larger checkpoints underperform their smaller counterparts." | HIGH |
| "Architecture changes give only small gains in TAS" | **FALSE on standard benchmarks.** MS-TCN(I3D)→DiffAct+ASFormer = +16.7 Edit / +27.7 F1@10 on Breakfast at fixed features. | HIGH |
| "Capacity is the lever" | **NOT SUPPORTED.** UVAST beats ASFormer on Edit with *fewer* parameters by changing the output formulation. | HIGH |
| "Features can matter enormously — but the sign depends on the method" | **SUPPORTED.** NNV −29.2 MoF on I3D; MuCon +8.1; feature window ±10.7 MoF. | HIGH |
| "Modern features (CLIP) help" | **REFUTED on 50Salads** without fine-tuning: −19.0 Edit. | MEDIUM-HIGH |
| "TAS scales steeply with labelled data at low budgets" | **SUPPORTED.** 0.25% of frames → +19.3 MoF over 0% (RNN-HMM); 5% videos ≈ 9% less MoF than 100% (C2F-TCN). | HIGH |
| "A 64-dim hand-crafted feature can reach near-SOTA MoF" | **SUPPORTED.** RNN-HMM on IDT+FV PCA-64: Breakfast MoF 61.3. | HIGH |
| "Breakfast/50Salads' own papers call their classes ambiguous" | **REFUTED — they do not.** 50Salads claims the opposite. Do not cite this. | HIGH |
| "Boundary labels are noisy between annotators on Breakfast" | **SUPPORTED (third-party).** σ ≈ 1.5 s ≈ 23 frames. | HIGH |

---

# B. Is whole-segment identity confusion a known, documented phenomenon?

**Short answer: yes as a *phenomenon*, no as a *quantified metric*.** Nobody I could find reports "fraction of GT segments with a wholly wrong label." What exists is (i) explicit localization-vs-discrimination decomposition language, (ii) documented whole-class (rare-class) failure, (iii) per-class/head-tail tables where tail classes sit at ~0, and (iv) explicit statements that global metrics hide exactly this.

## B.1 VERIFIED: the localization-vs-discrimination decomposition (the core framing)

**Source.** UVAST, ECCV 2022, §3.1. https://arxiv.org/abs/2209.00638 (full text verified). **VERIFIED-P.**

> "the encoder performs **frame-wise classification with high localization performance, i.e., high frame-wise accuracy, but low discrimination performance**, i.e., over-segmentation with **low Edit distance** to the ground truth."

**And the fix is discriminative features for *action identity*, supplied by a different prediction head:**
> "This immediate auxiliary supervision signal allows the decoder to learn **more discriminative features for different actions**."

**DERIVED:** "localization" = knowing where the segments are; "discrimination" = knowing which action each one is. The project's failure mode (78% of frame errors deep inside segments, 51.6% of segments wholly mislabelled, edit score barely better than frame accuracy) is a pure **discrimination** failure with localization largely intact. To my knowledge this is the closest the literature comes to naming the project's exact failure mode, and it names it as a *documented* property of the frame-wise-prediction paradigm **when trained on small datasets** — UVAST attributes it to "the relatively small amount of training videos".

**Corroborating inverse signature** (LTContext, ICCV 2023, Appendix D): fragmentation shows up as "**high accuracy, but very low F1 and Edit scores**". The project has the *opposite* signature (low frame accuracy, edit not relatively worse) — **DERIVED**, consistent with identity rather than segmentation error.

## B.2 VERIFIED: rare classes are *entirely* mislabelled — and global metrics hide it

**Source.** Zhanzhong Pang, Fadime Sener, Shrinivas Ramasubramanian, Angela Yao, "Cost-Sensitive Learning for Long-Tailed Temporal Action Segmentation", https://arxiv.org/abs/2503.18358 (**arXiv `Comments:` field is empty — venue UNVERIFIED**; v2 does not exist, HTTP 404). Full text https://arxiv.org/html/2503.18358v1. **VERIFIED-P.**

**The headline fact** (§1, referring to their Fig. 1(a) and Supplementary):
> "Despite this, state-of-the-art methods often overlook the long-tail, failing to recognize tail actions. For example, **AsFormer (Yi et al., 2021) and DiffAct Liu et al. (2023) exhibit zero accuracy on 5 and 4 out of 48 actions on Breakfast** (see Fig. 1 (a) and Supplementary)."

> "**The long-tail issue in action segmentation remains unexplored** (Ding et al., 2022; Farha and Gall, 2019; ...) due to the **widespread use of global evaluation metrics across all samples which obscure the poor performance on tail actions.**"

And in §4.1 "Evaluation metrics":
> "Three commonly used metrics are: frame-wise accuracy (Acc.), segment-wise edit score (Edit), and F1 score with IoU thresholds of 0.10, 0.25 and 0.50. **Conventionally, these metrics are tabulated globally over all the frames, obscuring the performance of tail actions.**"

**Their group-wise table quantifies the head/tail gap.** Table 3 ("Group-wise result summary across datasets and backbones"), Breakfast:

| Backbone | Head Acc | **Tail Acc** | Head F1@25 | **Tail F1@25** |
|---|---|---|---|---|
| MS-TCN | 65.1 | **37.7** | 53.3 | **38.7** |
| ASFormer | 69.7 | **39.8** | 69.9 | **43.9** |

Assembly101 (202 coarse classes):

| Backbone | Head Acc | **Tail Acc** | Head F1@25 | **Tail F1@25** |
|---|---|---|---|---|
| MS-TCN | 33.9 | **4.7** | 26.3 | **3.9** |
| ASFormer | 35.2 | **5.7** | 29.0 | **4.8** |

**DERIVED:** on Breakfast, tail classes are ≈30 accuracy points behind head classes; on Assembly101, tail-class accuracy is **4.7–5.7%** — statistically indistinguishable from never getting the class right. **This is the literature's verification of "whole classes get the wrong label", and it is severe.** Confidence: high.

**Their causal explanation is a *representation*-level claim, and it is directly relevant to a low-dim feature setting** (§3):
> "we observe **under-learnt tail actions** in temporal segmentation. This is because the learning of tail is suppressed due to the **temporal continuity of frame representation**... Distinctly separating two consecutive actions, one being head and the other tail, is challenging as **they share similar frame representations**... This similarity in representation hinders independent learning of tail actions without adversely affecting head actions."

**And the class prior is a large part of the mechanism** — their reweighting term is literally `G = 1/π_i + λ·1(T>0)/π_i`, where `π_i` is the **action prior** (Eq. 9). **DERIVED:** this says the residual error on rare classes is driven by the class prior, which is a data-statistics problem with a known cheap correction.

**Their transition-level finding is arguably the most striking claim in this survey, and it is about *context-dependent identity recognition*** (§3):
> "Variations in action transitions introduce a transition-level learning bias... for action 'take_eggs', the transition distribution from 'pour_oil' or 'take_bowl' to 'take_eggs' is skewed... **'take_eggs' is more easily detected when preceded by 'pour_oil' compared to 'take_bowl'.**"

**DERIVED:** the *same* action, with different predecessors, has different detectability. That is a label-conditional structure effect, not a capacity effect — and it is precisely the kind of thing a transition prior / grammar at decode time can exploit. **Confidence: high for the quote.** *(Caveat: the paper reports no significance test and this comparison is qualitative, from Fig. 1(b).)*

## B.3 VERIFIED: TAS evaluation practice has no error decomposition at all

- **No paper found reporting "fraction of GT segments with a wrong label."** **UNVERIFIED — I searched for "segment-level recall action segmentation", "segment level identity errors temporal action segmentation", "per-class analysis temporal action segmentation", "action ordering temporal action segmentation", "out-of-context errors action segmentation"; arXiv search was HTTP 429 for most of the session.** State this as *absent from my search*, not as *absent from the literature*.
- **No verified paper quantitatively asserts "identity errors dominate boundary errors."** The closest is UVAST's localization/discrimination framing (B.1), which is qualitative.
- **The standard tool for identity/ordering analysis exists and is under-used.** The survey defines an **order variation score** `v = 1 − e(X,Y)/max(|X|,|Y|)` (Eq. 4) computed as normalised edit distance between GT sequences across the dataset — a *dataset-level* measure of how tightly ordered the activity is. Survey Table III: 50Salads v = 0.02, Breakfast v = 0.15, Assembly101 v = 0.05; repetition r = 0.08 / 0.11 / 0.18. **VERIFIED-P.** **DERIVED:** a *low* `v` means the label sequences are nearly identical across videos, i.e. the activity is highly scripted and **a strong sequence prior is available for free**. Breakfast's v = 0.15 is the loosest of the three. **A `take_eggs`-style analysis — whether the same action's detectability depends on its predecessor — is a cheap diagnostic the project can run on its own 8 videos, and no published TAS paper has done it for this failure mode.**
- **"Out-of-context errors" is the community's term for identity/ordering errors**, used in both Activity Grammars and DTL: Activity Grammars §1 — "deep neural networks... often face **out-of-context errors that reveal the lack of capacity to capture the intricate structures of human activity**, and **the scarcity of annotated data exacerbates the issue in training**." **VERIFIED-P** (I read this in `raw/gram.txt`).

## B.4 VERIFIED: segment *density* is itself a documented difficulty axis

**Source.** OVTAS, ICRA 2026, arXiv 2602.21406, §V-2, Tables VIII–X. **VERIFIED-P** (I read these myself).

> "We also study the effect of the number of ground-truth action segments per video (Table IX). For GTEA, where each video contains **many short segments (mean ∼36)**, the model struggles compared to Breakfast, where **the mean is ∼5 segments**. 50 Salads lies between these two extremes. This shows that **the number of fine-grained action boundaries strongly influences performance, with dense sequences of short actions being particularly challenging.**"

Table X (segment durations): GTEA mean **1.94 s**; 50 Salads **18.59 s**; Breakfast **20.95 s**.
Table VIII: performance degrades monotonically with video duration (Breakfast `Avg` 62.48 for <60 s clips → **36.83** for ≥120 s).

**DERIVED:** GTEA has ~7× more segments per video than Breakfast and the worst `Avg` (23.9 vs 46.6) — i.e. **the densest-labelling benchmark is the hardest one.** The project's regime (dense per-frame labels over ~9+1 classes, 2,639 frames) sits at the GTEA-like end of this axis. **This is a verified, non-architectural explanation for why the regime is hard: it is the regime in which the segment-identity task is densest.** **Confidence: high for the quotes and table values; the "project sits at this end" mapping is DERIVED.**

## B.5 Where the identity-fix evidence actually lives (preview of C)

The papers that report large *identity*-metric gains are overwhelmingly **decode-time / structure** papers, not capacity papers:

| Method | Mechanism | Breakfast Edit | 50Salads Edit | Source |
|---|---|---|---|---|
| ASFormer (baseline) | frame-wise | 75.0 (as published) / 75.6 (reproduced) | 75.0 (published) / 76.5 (reproduced) | Activity Grammars Tables 3–4 |
| + KARI grammar + BEP parser | PCFG constraint | **77.8** | **79.9** | same |
| ASFormer + constrained Viterbi | duration + transitions + start/end | (Table 4 row) | 50Salads baseline 45.9 → **48.9** (semi-sup) | CAD |
| VidParse (graph-constrained) | task-graph hard constraints | — | GTEA recipe-specific F1 21.30 → **80.47** | VidParse |

All four are **training-free or near-training-free at inference**. **Confidence: high for each individual number; the table's framing is DERIVED.**

---

# C. Cheap structural / non-architectural fixes

## C.1 Constraint-aware Viterbi decoding — the strongest single verified result

**Source.** Yeo Keat Ee, Debaditya Roy, Chen Li, Hao Zhang, Basura Fernando, "Improving Temporal Action Segmentation via Constraint-Aware Decoding" (**CAD**), arXiv `Comments:` **"accepted to ICPR 2026"**. https://arxiv.org/abs/2605.10149 · full text https://arxiv.org/html/2605.10149v1 (license: CC BY 4.0, stated in the HTML). Code: https://github.com/LUNAProject22/CAD — **UNVERIFIED: the repo returns 404 from the GitHub API (not public at time of check 2 fetches).** **VERIFIED-P.**

**What it does.** Extracts three constraints directly from the training annotations and enforces them inside a modified Viterbi decode:
1. **transition confidence** `Conf(A→B) = Count(A→B)/Count(A)` — all observed transitions form the valid-transition set `T`;
2. **valid start/end actions**;
3. **normalised per-class duration bounds** `[d_min_c, d_max_c]` relative to video length.

> "This algorithm enforces valid transitions, segment durations, and boundary conditions **at inference time without changing the TAS model.** Unlike approaches requiring extra training or neural Viterbi modules, our method is **training-free, low-cost**, and suitable for large-scale or real-time use."

**Results — fully-supervised setting, Table 1** (backbone ASFormer; Breakfast | 50Salads, F1@{10,25,50} / Edit / Acc):

| Method | Breakfast F1@10/25/50 | Breakfast Edit | Breakfast Acc | 50Salads F1@10/25/50 | 50Salads Edit | 50Salads Acc |
|---|---|---|---|---|---|---|
| w/o constraints | 74.1 / 68.7 / 55.5 | 72.8 | 72.4 | 83.4 / 80.8 / 74.6 | 75.7 | 85.0 |
| KARI* (grammar) | 77.3 / 71.6 / 57.0 | 78.3 | 74.5 | 83.8 / 81.8 / 74.7 | 76.4 | 83.3 |
| **Ours** | **78.8 / 73.1 / 58.0** | **77.9** | **74.6** | **84.8 / 83.1 / 76.0** | **78.6** | 82.9 |

**DERIVED deltas vs the unconstrained backbone: Breakfast +4.7 F1@10, +5.1 Edit; 50Salads +1.4 F1@10, +2.9 Edit.** Model-agnostic (Table 4 repeats it on a FACT backbone: 81.3→81.8 F1@10, Edit 79.7→81.4).

**Semi-supervised setting (5% labelled videos), Table 1:**

| Method | Breakfast F1@10/25/50 | Breakfast Edit | 50Salads F1@10/25/50 | 50Salads Edit |
|---|---|---|---|---|
| w/o constraints | 57.1 / 51.2 / 34.6 | 54.6 | 51.8 / 47.7 / 37.0 | 45.9 |
| **Ours** | **58.5 / 52.4 / 36.0** | **56.4** | **55.8 / 51.4 / 38.5** | **48.9** |

**DERIVED:** on 50Salads at 5% labelled data, **+4.0 F1@10 and +3.0 Edit from a training-free decode.**

**Component ablation, Table 5 (Breakfast, fully supervised):**

| Constraints | F1@10 | Edit | Acc |
|---|---|---|---|
| w/o any | 74.1 | 72.8 | 72.4 |
| w/o start-end | 77.9 | 76.0 | 73.1 |
| w/o transition | 75.1 | 73.8 | 72.3 |
| w/o duration | 75.0 | 75.5 | 74.5 |
| 0.3 tra. + 0.7 dur. | 78.8 | 78.8 | 74.6 |
| 0.7 tra. + 0.3 dur. | **79.0** | **78.9** | **74.7** |
| **Soft** (all, soft) | 75.6 | 74.6 | 72.4 |
| **Hard** (all, hard) | 78.8 | 77.9 | 74.6 |

**DERIVED, and this is a real design warning: the default `Hard` configuration beats `Soft` by +3.3 Edit (77.9 vs 74.6) and +3.2 F1@10 (78.8 vs 75.6)** — i.e. permitting constraint violations as soft penalties is substantially worse here. Re-weighting transitions vs duration matters little (78.8 vs 78.9 Edit).

**Which constraint carries the weight?** Reading the ablation against the `w/o any` row (Edit 72.8 / F1@10 74.1), each constraint added *alone* gives: **start-end → 76.0 Edit** (+3.2), **duration → 75.5** (+2.7), **transition → 73.8** (+1.0). All three together → **77.9** (+5.1). **DERIVED: the start/end boundary condition is the single most valuable constraint, and the three are complementary rather than redundant.** **Confidence: high for the table; the ranking interpretation is DERIVED.**

**Efficiency, Table 2:** average inference time on 20 Breakfast videos with the same backbone — KARI (grammar + BEP parser) **148.9 s** vs CAD (constrained Viterbi) **2.81 s**. Complexity `O(C·T)`, linear in T, vs `O(T³)` worst case for the grammar parser. **DERIVED: ~53× faster.** **Verified from Table 2.**

## C.2 VERIFIED-NEGATIVE, and important: a naive Viterbi / transition prior alone can HURT

This is a genuine trap and two independent papers document it.

**(a) CAD's own ablation, Table 3 (50Salads, semi-supervised setting):**

| Method | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---|---|---|---|---|---|
| w/o constraints (Table 1 baseline) | 51.8 | 47.7 | 37.0 | 45.9 | 62.3 |
| **Classical Viterbi** | **45.5** | **40.2** | **30.1** | **39.2** | **60.1** |
| Ours (constrained Viterbi) | 55.8 | 51.4 | 38.5 | 48.9 | 63.4 |

The paper states (§4.5): "The classical Viterbi relies on TAS action logits for each time step and **transition confidences** for sequence prediction."

**DERIVED: adding only a transition prior made things WORSE than no decoding at all (−6.3 F1@10, −6.7 Edit).** Only when start/end + duration bounds are added does decoding beat the plain baseline. **Read this before adding a transition-matrix post-processor.** Confidence: high.

**(b) UVAST reports the same trap.** Timestamp-supervised setting, Breakfast, Table 2: MS-TCN F1@10 56.1 / **Edit 61.7** / Acc 62.5 → **MS-TCN + Viterbi 43.3 / 43.5 / 35.9**. And in the fully supervised setting, ASFormer Edit 75.0 → ASFormer + Viterbi **74.5** (slightly worse).

UVAST's explanation (§5):
> "**ASFormer and MSTCN do not benefit from the Viterbi algorithm in this case. This is due to the relatively lower Edit distance of these methods.** Namely, Viterbi hurts MSTCN on Breakfast as it achieves significantly lower Edit distance compared to ours."

And more fundamentally (§5):
> "This is due to the fact that **the objective function that is minimized by FIFA/Viterbi does not optimize the evaluation metrics directly, i.e., the global optimum of the Viterbi objective function does not guarantee the global optimum of the evaluation metrics.** This observation is consistent with the results reported in [33]."

**DERIVED synthesis:** Viterbi helps *only* when the underlying frame-wise posteriors already have good segment-identity structure. It is a **sharpener, not a fixer**. If the posteriors are wrong about identity, imposing transition/duration structure can lock in wrong segments. **This directly predicts that adding a transition prior to the project's MS-TCN++ will not by itself fix the 51.6% wrong-segment rate — and may worsen it.** What the CAD ablation says to add instead is **duration bounds + valid start/end + hard (not soft) transitions, together.** **Confidence: high** (two independent verified sources).

## C.3 Grammar-based / probabilistic parsing

### C3a. KARI — Activity Grammars (NeurIPS 2023)

**Source.** Dayoung Gong, Joonseok Lee, Deunsol Jung, Suha Kwak, Minsu Cho, "Activity Grammars for Temporal Action Segmentation", arXiv `Comments:` **"Accepted to NeurIPS 2023"**. https://arxiv.org/abs/2312.04266 · full text https://arxiv.org/html/2312.04266v1. Code: https://github.com/gongda0e/KARI — **license MIT**, 14 stars, last push 2024-06-14 (**checked via GitHub API**). **VERIFIED-P.**

**What constraints it imposes on the segment label sequence.** It induces a **probabilistic context-free grammar** over action sequences with two rule types (Eqs. 1–2):

> "AND: `V → α` where `V ∈ V` and `α ∈ (Σ ∪ V)*` ... **The AND rule replaces a head variable V with a sequence of variables and terminals α, determining the order of the terminals and variables.**"
> "OR: `V → V₁[p₁] | V₂[p₂] | ... | Vₙ[pₙ]` ... **the OR rule converts a head variable V to a sub-variable Vᵢ with the probability pᵢ, providing multiple alternatives for replacement.**"
> "Recursive rules are indispensable for expressing complex and realistic structures found in action phrases and activities... The proposed grammar induction enables **recursive rules with flexible temporal orders**, which leads to powerful generalization capability."

Induction uses **key actions** ("those consistently present in every action sequence from the training dataset") as reference points, splitting each sequence into sub-sequences; within a sub-sequence, temporally dependent actions become AND rules and temporally independent ones become OR rules. A **generalized parser (BEP)** then maps frame-level class predictions `Y ∈ R^{T×|A|}` to a grammar-consistent action sequence `a*`, followed by a segmentation-optimisation step for the lengths. Hyperparameters: `N_key = 4` (Breakfast), `3` (50Salads); BEP queue size 20.

**So the imposed constraints are: (i) which actions must occur, (ii) their partial/total order, (iii) which alternatives are admissible, (iv) recursion/repetition structure.** It is *not* a duration model — durations are handled separately by the segmentation-optimisation step.

**Results — Table 4 (Breakfast) and Table 3 (50Salads)**, columns Edit | F1@10 | F1@25 | F1@50 | Acc:

| Model | reprod. | refinement | grammar | Breakfast Edit | Breakfast F1@10 | 50Salads Edit | 50Salads F1@10 |
|---|---|---|---|---|---|---|---|
| ASFormer | — | — | — | 75.0 | 76.0 | 75.0 | 76.0 |
| ASFormer | ✓ | — | — | 75.6 | 77.3 | **76.5** | **83.8** |
| ASFormer | ✓ | ✓ | ADIOS-AND | 69.2 | 69.8 | 58.3 | 70.0 |
| ASFormer | ✓ | ✓ | ADIOS-OR | 70.3 | 71.8 | 61.1 | 72.0 |
| ASFormer | ✓ | ✓ | **KARI** | **77.8** | **78.8** | **79.9** | **85.4** |
| MS-TCN | — | — | — | 61.7 | 52.6 | 67.9 | 76.3 |
| MS-TCN | ✓ | — | — | 69.7 | 70.7 | 62.4 | 69.5 |
| MS-TCN | ✓ | ✓ | **KARI** | **74.9** | **74.6** | 66.7 | 75.1 |

**DERIVED:** KARI's gain over the *reproduced* baseline is **+2.2 Edit / +1.5 F1@10** (Breakfast ASFormer) and **+3.4 Edit / +1.6 F1@10** (50Salads ASFormer). Real but modest. **Compare that to the reproduction gap in the same table.** Note also that the weaker ADIOS grammars *hurt* on 50Salads ASFormer (58.3 and 61.1 Edit vs 76.5 baseline) — **a bad grammar is worse than no grammar.** Table 10 confirms: on 50Salads, Kuehne et al.'s grammar gives Edit 62.9 and Richard et al.'s 63.1, vs 76.5 with no grammar. **Confidence: high.**

**Ablation Table 5 (50Salads, ASFormer):** full KARI 79.9 Edit → removing recursive rules 69.2 (**−10.7**) → removing key actions too 62.6 (**−17.3**). So recursive structure is doing most of the work. **DERIVED.**

**Efficiency cost (from CAD Table 2): 148.9 s per 20 Breakfast videos vs 2.81 s for constrained Viterbi — ~53× slower.**

### C3b. Kuehne et al., "Weakly supervised learning of actions from transcripts", and grammar/HTK decoding

**Source (the accessible journal version).** Hilde Kuehne, Alexander Richard, Juergen Gall, "A Hybrid RNN-HMM Approach for Weakly Supervised Temporal Action Segmentation", **IEEE TPAMI** (arXiv `Comments:` "preprint for IEEE TPAMI"). https://arxiv.org/abs/1906.01028. **VERIFIED-P.**

It uses an HTK-style hierarchical HMM: each action is a sequence of latent **subactions** represented as HMM states, populated from the transcript's action order. §III:
> "At top level, we model each temporal sequence as a combination of basic actions... Each of those actions is represented by a respective probabilistic graph model, in this case an **HMM, which models each action as a combination of subactions**."
> "To ensure the sequential peculiarity of human actions within the state graph, we use a **feed-forward topology, allowing only self-transition or transitions to the next state.**"

**So the constraints imposed are: the transcript's action order, a left-to-right HMM topology per action, and a per-state length prior.** See C.4 for the numbers.

### C3c. DTL — differentiable temporal logic (ordering constraints)

**Source.** Z. Xu, Y. S. Rawat, Y. Wong, M. Kankanhalli, M. Shah, "**Don't pour cereal into coffee**: Differentiable temporal logic for temporal action segmentation", **NeurIPS 2022**. Verified via the reference list of Activity Grammars [44] and of the TPAMI survey [113]. **VERIFIED-P (existence + venue via secondary reference lists; I did not fetch DTL's own full text).**

**UNVERIFIED — DTL's own numbers.** I tried the arXiv HTML and title search; both failed (arXiv search HTTP 429, and the paper does not appear in arXiv title search). The survey reports DTL's result on Breakfast as **DTL + ASFormer: F1@10 78.8, Edit 77.7, MoF 75.8** (survey Table V, compiled from the original). Activity Grammars cites DTL as "a **model-agnostic framework to give temporal constraints to neural networks**". **Do not quote DTL's internals — I did not read them.**

## C.4 HMM / HSMM with explicit duration modelling

### C4a. VERIFIED: the length prior, and how much duration structure is worth

**Source.** Kuehne/Richard/Gall, TPAMI (arXiv 1906.01028). **VERIFIED-P.**

**Motivation for explicit duration modelling — the degenerate-state failure:**
> "the proposed length prior serves as an **additional regularization factor to enforce a meaningful length of the single states**. We will show that the length prior **helps to prevent degenerated states during inference** and thus to improve recognition accuracy in general."
And on why modelling duration via state count alone fails:
> "depending on the observation prior, a number of states will **aggregate all frames of an action during inference**, thus undermining the original idea of representing variable length actions by adapting the number of states only."

**Table III — length-model ablation (Breakfast MoF / Hollywood-Extended IoU):**

| Length model | Breakfast MoF | Hollywood Ext. IoU |
|---|---|---|
| **No length model** | **32.6** | 11.5 |
| Box function | 36.7 | 9.9 |
| Linear decay | **37.0** | 10.5 |
| Half Poisson | 35.7 | 11.1 |
| Half Gaussian | 36.7 | **12.3** |

**DERIVED:** an explicit duration prior buys **+4.4 MoF** on Breakfast (32.6 → 37.0), with box/linear/Gaussian all roughly equivalent and the choice mattering less than having one. **Note the sign flips on Hollywood Extended: the box function is worse than no length model (9.9 vs 11.5).** So a length prior is data-dependent, not free.

**Table V (Breakfast MoF, weak/semi setting):** HTK 25.9, ECTC 27.7, GRU-RNN 33.3, **GRU + length prior 36.7**, TCFPN 38.4.
**Table VII (Breakfast MoF, fully supervised):** HMM-BOW 28.8, HMM-FV 56.3, TCFPN 52.0, **GRU w/o length prior 60.2, GRU + length prior 61.3.** So under full supervision the length prior is worth only **+1.1** — and the paper says so:
> "Additionally, it shows that the length model also improves the segmentation accuracy in case of fully supervised training. This is important as in this case, we can assume that all other temporal factors, such as the number of states are already optimal. Thus, even in this case, **a temporal prior can improve the overall recognition of the system.**"

**The most striking duration result — Table II (oracle lengths):**

| Configuration | Breakfast MoF |
|---|---|
| GRU no subactions | 22.4 |
| GRU w/o reestimation | 28.8 |
| GRU + reestimation | 33.3 |
| **GRU + GT length** | **51.3** |

**DERIVED: giving the model the ground-truth action lengths adds +18.0 MoF (33.3 → 51.3)** — nearly five times the gain from the learned length prior. **This says the remaining headroom is dominated by getting segment *extents* right, and that a duration prior is a weak proxy for it.** **Confidence: high for the table.**

### C4b. VERIFIED: HSMM/HMM over frame-wise posteriors in surgical phase recognition — EndoNet + HHMM

**Source.** Andru P. Twinanda, Sherif Shehata, Didier Mutter, Jacques Marescaux, Michel de Mathelin, Nicolas Padoy, "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos", **IEEE TMI 2017** (arXiv 1602.03012). Full text fetched via https://ar5iv.labs.arxiv.org/html/1602.03012. **VERIFIED-P.**

**Input features and dataset size (asked for explicitly):**
- **Dataset:** "we build a large dataset of cholecystectomy videos containing **80 videos** recorded at the University Hospital of Strasbourg" (= Cholec80). **7 phases** (Table I: Preparation, Calot triangle dissection, Clipping and cutting, Gallbladder dissection, Gallbladder packaging, Cleaning and coagulation, Gallbladder retraction). Evaluation on **40 videos**. A second dataset (EndoVis) is used for generalisation.
- **Features compared** (§V, Table III): **`Tool binary`** (binary tool-presence vector); **`Handcrafted`** ("bag-of-word of SIFT", colour/texture/shape); **`Handcrafted+CCA`**; **`AlexNet`** (fc7 of ImageNet-trained AlexNet); **`PhaseNet`** (single-task CNN); **`EndoNet`** (multi-task AlexNet fine-tuned for phase + tool presence).
- **The temporal model:** a **two-level Hierarchical HMM (HHMM)**, 7 top-level states = 7 phases, data-driven number of bottom-level states, mixture of 5 Gaussians per feature, diagonal covariance. Viterbi/"most likely path" inference. Neither the CNN nor the HHMM is trained end-to-end (§VI: "the HHMM is trained separately from the EndoNet fine-tuning process").

**Table III — before HHMM (mean ± std over videos), Cholec80, Accuracy:**

| Feature | Accuracy |
|---|---|
| Handcrafted | 44.0 ± 1.8 |
| Handcrafted+CCA | 39.0 ± 0.6 |
| **Tool binary** | **48.2 ± 2.7** |
| AlexNet | 59.2 ± 2.4 |
| PhaseNet | 73.0 ± 1.6 |
| **EndoNet** | **75.2 ± 0.9** |

**Table IV — after HHMM (offline), Cholec80, Accuracy:**

| Feature | Accuracy (offline) |
|---|---|
| Handcrafted | 36.7 ± 7.8 |
| Handcrafted+CCA | 61.3 ± 8.3 |
| **Tool binary** | **69.2 ± 8.0** |
| AlexNet | 76.2 ± 6.3 |
| PhaseNet | 89.1 ± 5.4 |
| **EndoNet** | **92.0 ± 1.4** |

**DERIVED — the two effects, side by side:**
- **Feature swap (handcrafted → EndoNet): +31.2 accuracy** (44.0 → 75.2). This is a *large* feature effect — but note the task is single-frame phase classification with 7 classes on 80 videos, not dense per-frame segmentation of 9+1 classes.
- **Structural post-processing (adding the HHMM): +16.8 accuracy for EndoNet** (75.2 → 92.0), **+21.0 for Tool binary** (48.2 → 69.2), **+22.3 for Handcrafted+CCA** (39.0 → 61.3). The paper: "By comparing the results from Tab III-a and Tab IV-a, we can see **the improvement that the HHMM brings, which is consistent across all features.**"
- **So on this task the temporal-structure prior is worth roughly 2/3 of the entire feature upgrade — and it is the cheaper of the two.** Confidence: high for the tables.
- **Non-monotonic exception worth flagging:** plain `Handcrafted` *got worse* under the HHMM (44.0 → 36.7). **DERIVED: a strong structural prior can amplify a weak observation model in the wrong direction** — the same lesson as C.2.

**Also relevant: the project's 144-dim YOLO-detection feature is conceptually closest to `Tool binary` (a symbolic object-presence vector), which is the *second-best* feature in Table III at 48.2 ± 2.7, ahead of handcrafted appearance features at 44.0.** **That is the single most encouraging comparison I found for a detection-based feature set.** **Confidence: medium** — the analogy is mine; the numbers are the paper's.

**The paper's own stated limitation** (§VII):
> "the phase recognition still relies on the HHMM, which is **required to enforce the temporal constraints in the phase estimation**. Thus, **the features learnt by EndoNet do not include any temporal information** present in the videos."

**And note the reporting practice:** every number here is **mean ± std over videos**, with accuracies' σ typically 0.6–8.3 points. This is far more disciplined than the main video-TAS benchmarks (see C.6).

## C.5 Class prior and transition-matrix post-processing, measured directly

**(a) Class prior as an explicit term in the decode.** The TPAMI survey gives the standard formulation (Eqs. 10–19), decomposed into a **context model** `∏ p(c_{n+1}|c_n)`, a **length model** `∏ p(ℓ_n|c_n)` and a **visual model** `∏ p(x_t|c_n)`; and:

> "The prior `p(c_n)` can be estimated either **empirically, based on the fraction of frames with label c_n** [13] or simply as a uniform distribution [56]. The posterior `p(c_n|x_t)` is then approximated by the output of an action classification network supervised by the action annotations."
with the correction (Eq. 16): `p(x_t|c_n) ∝ p(c_n|x_t) / p(c_n)`.

**This is a one-line, free class-prior correction on top of any frame-wise model** — and the survey notes the length model is Poisson-distributed with `λ` fitted by constrained optimisation (COBYLA), and that "**The explicit modeling of lengths is necessary to avoid producing unreasonably long action segments.**" **VERIFIED-P.** **UNVERIFIED — I found no paper that isolates the gain from the `p(c|x)/p(c)` correction alone on Breakfast/50Salads.**

**(b) Segment-level relabelling (S-NCM) — a direct attack on segment identity.** From the long-tailed TAS paper (§3.3), **VERIFIED-P**:
> "we make predictions using frame representations based on **NCM** (Nearest Class Mean), which involves computing mean representations for each class and performing nearest neighbor search... Applying frame-level NCM, however, disregard the temporal continuity and **lead to over-segmentation**. We propose **Segment Nearest Class Mean (S-NCM)**... we first leverage the classifier's predictions `ŷ` to **detect segment boundaries** and then utilize frame-wise NCM predictions `v̂` for **labelling each segment through major voting**, namely **the frames in each segment share the same prediction**."

**DERIVED: this is exactly the right shape of fix for a wrong-identity problem** — freeze the boundaries the model already gets right, and re-decide the identity of each whole segment by a different classifier (nearest class mean, i.e. without the biased linear head). It is model-agnostic and cheap.

**Measured effect (Table 5, AsFormer, Breakfast, per-class | global F1@25):**

| Objective | Constraint | NCM | S-NCM | F1@10 | F1@25 | F1@50 | Acc | G_F1@25 |
|---|---|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | ✗ | 57.9 | 54.7 | 45.4 | 52.3 | 69.9 |
| ✓ | ✗ | ✗ | ✗ | 58.4 | 55.6 | 46.4 | 52.9 | 69.8 |
| ✓ | ✓ | ✗ | ✗ | 60.6 | 57.3 | 48.5 | 54.4 | **70.8** |
| ✓ | ✓ | ✓ | ✗ | 59.7 | 56.4 | 46.7 | 55.0 | 67.9 |
| ✓ | ✓ | ✗ | **✓** | **61.0** | **57.9** | **49.0** | **55.1** | 70.3 |

**DERIVED:** naive frame-level NCM alone *reduces* global F1 (67.9 vs 70.8) because it fragments — exactly as the authors say. **S-NCM recovers and exceeds it.** Full stack: **+3.1 F1@10, +3.2 F1@25, +3.6 F1@50, +2.8 Acc, +0.5 G_F1** over AsFormer baseline on Breakfast, and on Assembly101 the **+Objective+Constraint** steps raise per-class F1@10 from 9.2 → 12.4 (+35% relative). **Confidence: high for the table; the "right shape of fix" framing is DERIVED.**

**Important trade-off the authors state explicitly** (§4.1/§4.4):
> "Empirically, over-emphasizing the per-class performance will hurt the global performance."
> "Balancing global and balanced results is challenging due to the trade-off: improving tail often boosts per-class results at the expense of head performance, resulting in the drop in global results."

**DERIVED: this is a decision the project must make deliberately.** If the business metric is segment-level identity (edit distance), optimising a global frame metric will *not* get there, and vice versa.

**Alternatives that gave ~nothing (Table 2, deltas over baseline, Breakfast per-class F1@10):** CB (class-balanced reweighting) +0.9, LA (logit adjustment) +1.4, Focal +1.0, τ-norm −1.1 for MS-TCN; +0.8 / +1.4 / +1.0 / 0.0 for ASFormer. **DERIVED: generic long-tail losses transfer poorly to TAS** — the sequence-aware cost-sensitive method is the one that works (ours +8.1 / +3.1).

**(c) The strongest structural-decoding result I found anywhere: VidParse.**

**Source.** "VidParse: Online Parsing of Egocentric Procedures Like a Pro", **ECCV 2026** (arXiv `Comments:` "Accepted at ECCV 2026"). https://arxiv.org/abs/2608.27562 · full text https://arxiv.org/html/2608.27562. **VERIFIED-P** (I re-fetched and read the key passage myself).

**Mechanism** (§1, §3.3): MAFs = frozen DINOv2 descriptors masked by a hand-object detector; boundaries from a temporal similarity matrix + Gaussian-tapered checkerboard kernel (no learned boundary predictor); then **graph-constrained beam search over an induced procedural task graph** that "strictly enforces transitions defined in the induced Task Graph. **Transitions that violate the procedural structure... are assigned infinite cost and removed.**" "training-free" is defined as "requires no task-specific training or fine-tuning on the downstream procedure parsing dataset".

**The decisive passage** (Appendix Table 1, GTEA, recipe-specific task-graph setting):
> "Even when the ProTAS baseline is adapted to utilize this recipe-specific structure alongside MAF representations, **it plateaus at an overall F1 score of 21.30**. In contrast, VidParse leverages **the same MAFs** within our strict graph-constrained inference to achieve an **overall F1 of 80.47**. ... This underscores that **while strong features are beneficial, our training-free, structurally constrained decoding is uniquely suited** for exact procedural understanding over complex, recipe-specific trajectories."

**DERIVED: same features, no gradient updates, +59.17 F1.** **Caveats, stated plainly:** this is a recipe-specific protocol on GTEA (not the standard 4-split GTEA protocol), the baseline is a particular online method (ProTAS), and the task graph embeds recipe identity that a general TAS model does not receive. **Do not present this as "graph decoding gives +59 F1 on any TAS benchmark."** Presented correctly, it is the clearest existence proof that **when the label sequence is strongly constrained, decode-time structure — not features, not capacity — is what closes the gap.**

**And the same paper's feature-swap numbers are a caution:** in the *protocol-matched* rows, ProTAS with I3D = Acc 73.96 / Edit 71.81 / F1 56.53 vs ProTAS with DINO CSL = 71.90 / 71.47 / 56.50 — i.e. on GTEA, swapping to modern DINOv2 features moved F1 by **−0.03**. **DERIVED.** Consistent with A.2 and the "small dataset" theme.

## C.6 Small-test-set statistical fragility

This is, in my judgement, the strongest and most under-appreciated finding in the survey for this project.

### C6a. VERIFIED: run-to-run standard deviation in TAS is 1–2.5 points

**Source.** Souri, Richard, Minciullo, Gall, "On Evaluating Weakly Supervised Action Segmentation Methods", **Technical Report**, arXiv 2005.09743. **VERIFIED-P.**

> "we train each method on the Breakfast dataset **5 times** and provide **average and standard deviation** of the results. Our experiments show that **the standard deviation over these repetitions is between 1 and 2.51%, and significantly affects the comparison between different approaches.**"
> "The variance in the performance of weakly supervised approaches is high when running the training and testing multiple times with different random seeds. **It is therefore necessary to report the average and standard deviation over multiple runs** to evaluate the performance of an approach."
> "For NNV and CDFL, it is about 2.5. This shows that the methods for weakly supervised learning are more sensitive to the random model initialization and the random sampling of mini-batches compared to fully supervised approaches."

**Table 1 — reported vs reproduced MoF, Breakfast, mean over all splits and 5 runs:**

| Model | MoF *reported* | MoF *reproduced* (mean ± std) | Max | Min | **DERIVED: max−min spread** |
|---|---|---|---|---|---|
| ISBA | 38.4 | 36.4 ± 1.0 | 37.6 | 35.1 | 2.5 |
| NNV | 43.0 | 39.7 ± 2.4 | 43.5 | 37.5 | **6.0** |
| CDFL | 50.2 | 48.1 ± 2.5 | 50.9 | 44.6 | **6.3** |
| MuCon | 48.5 | 48.5 ± 1.7 | 49.9 | 45.6 | 4.3 |

**DERIVED, and this is the punchline: the run-to-run spread within a single method (up to 6.3 MoF) is comparable to or larger than the gaps between the methods in the table (38.4 → 50.2, a 11.8-point range across four methods).** And NNV's *reported* number (43.0) is 3.3 points above its own reproduced mean (39.7) — larger than its reported margin over ISBA. **Confidence: high.**

### C6b. VERIFIED: the canonical TAS baselines report NO standard deviation at all

- **MS-TCN (CVPR 2019)**, full text: "we use five-fold cross-validation and report the average"; "we use the standard 4 splits ... and report the average". **`grep -c "±"` on the full text = 0.** **VERIFIED-D.**
- **ASFormer (BMVC 2021)**, full text: same pattern; **`grep -c "±"` = 0.** **VERIFIED-D.**
- **Long-tailed TAS paper (2503.18358)** Table 2 caption says "over 3 runs" — one of the few to do so, though it does not print per-run std in the main table.

**DERIVED: the SOTA table the project is benchmarking against is a table of single-split-point-estimate means with no error bars.** Comparisons between methods differing by 1–2 points in that table are not statistically meaningful.

### C6c. VERIFIED: the exceptions that DO report ± — and their numbers

**Source.** C2F-TCN (arXiv 2212.11078), appendix §S2-E. **VERIFIED-D.**

> "Tab. T22 shows the deviations of our final results ... for **4 runs with different random seeds**. For each metric, we report the results in the format **mean ± std**... **For the smallest GTEA dataset, the deviation in the results is higher than in Breakfast and 50Salads.**"

**Table T22:** Breakfast 71.9±0.6 / 68.8±0.7 / 58.5±0.8 / Edit 68.9±1.3 / MoF 76.6±0.9 · 50Salads 84.3±0.7 / 81.7±0.4 / 72.8±0.7 / 76.3±0.8 / 84.5±0.8 · GTEA 92.3±1.1 / 90.1±0.7 / 80.3±0.9 / 88.5±1.5 / 81.2±0.4.

**Table T30 — σ from *which videos you label*, not from seeds** (5 different random selections of 5%/10% labelled videos): Breakfast ICC4 at 10% is 64.6 ± 2.1 F1@10, 59.0 ± 1.9 F1@25, 42.2 ± 2.5 F1@50, 61.9 ± 2.2 Edit, 68.8 ± 1.3 MoF; 50Salads ICC4 at 3 videos is 52.9 ± 2.2 / 49.0 ± 2.2 / 36.6 ± 2.0 / 45.6 ± 1.4 / 61.3 ± 2.3.

**DERIVED, and this is the key inference:** the σ arising from **which 5–10% of videos happen to be labelled** reaches **±2.5** on segmental F1 — *larger* than the σ from seed variation (±0.4–1.5) and larger than most published SOTA gaps. **Therefore: the labelled-subset choice alone can flip a comparison.** This is my inference, not a claim the paper makes, but it follows directly from their numbers. **Confidence: medium-high.**

### C6d. The reproduction gap on a TAS benchmark is larger than the method's own claimed gain

This is the single most dramatic number I found for statistical fragility, and it comes from a paper that simply reports its own reproduction honestly.

**Source.** Activity Grammars / KARI, NeurIPS 2023, Tables 3–4 and §4.4. **VERIFIED-P.**

> "The first row in Table 3 and Table 4 indicates **the performance from the original paper** [9,45], whereas the second row represents **the reproduced performance obtained using official codes.**"

**50Salads, ASFormer (F1@10 | F1@25 | F1@50 | Edit | Acc):**

| | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---|---|---|---|---|---|
| ASFormer **as published** | 76.0 | 70.6 | 57.4 | 75.0 | 73.5 |
| ASFormer **reproduced** by KARI's authors | **83.8** | **81.7** | **74.8** | **76.5** | **86.1** |

**DERIVED: the same architecture, same benchmark, same official code, reproduced by a different group — F1@50 differs by +17.4, frame accuracy by +12.6, F1@10 by +7.8.** By contrast, KARI's own contribution over the reproduced baseline is **+1.6 F1@10 / +3.4 Edit**. And on Breakfast, MS-TCN reproduced goes from published Edit 61.7 / F1@10 52.6 to reproduced **69.7 / 70.7** (+8.0 / +18.1). **On 50Salads, MS-TCN reproduced was *worse* (67.9 → 62.4 Edit).**

> "Since we apply the proposed method to the reproduced temporal action segmentation models, **we directly compare and evaluate the performance based on the reproduced results.**"

**DERIVED: cross-paper TAS numbers on these benchmarks are not comparable at the 1–10 point level.** Any claim that "architecture X beats architecture Y by 3 edit points" on 50Salads is, on this evidence, unsafe. **Confidence: high** for the table values, which I parsed from the paper's HTML myself.

### C6e. How few *segments* do the standard benchmarks actually measure?

- **50Salads:** 50 videos, 5-fold cross-validation (MS-TCN §4, ASFormer §5.1) → **10 test videos per fold**. MS-TCN §4: "On average, each video contains **20 action instances** and is 6.4 minutes long." **VERIFIED-D.**
- **DERIVED: ≈10 × 20 = ~200 segments per fold.** F1@k and Edit rest on O(10²) objects, not on the O(10⁵) frames.
- **Breakfast: 4 splits, and — correcting the working assumption — the test split is NOT ~10–13 videos.** The official Serre Lab page (via the Wayback snapshot; the live page is JS-rendered) gives the splits as participant ranges: "s1: P03 – P15, s2: P16 – P28, s3: P29 – P41, s4: P42 – P54" = 13 **participants** per split. Leave-one-split-out therefore implies **≈1712/4 ≈ 428 test videos** per evaluation, corroborated by C2F-TCN labelling "≈63 videos" as "5% of the training split". **VERIFIED-D; the 428 figure is DERIVED.** **The "13" is participants, not videos.**
- **DERIVED and decisive for this project:** the project evaluates on **8 test videos / 2,639 frames**. That is *smaller than the smallest standard TAS split* (50Salads at 10 videos ≈ 200 segments). With ~9+1 classes and an unknown segment count, the per-segment metric sample is likely **O(10¹)**. **At that sample size the ±1–2.5 point run-to-run σ measured on *Breakfast* (428 videos) will be much larger.**

### C6f. General methodology literature (all four verified to exist, with their own findings)

- **Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks"** (arXiv 2103.03098): "the variance due to data sampling is well explained by the limited statistical power in the test set"; "variance is not small compared to the differences between pipelines". **VERIFIED-D.**
- **Reimers & Gurevych, "Reporting Score Distributions Makes a Difference"** (EMNLP 2017, arXiv 1707.09861): 86 seeds — min 89.99% / max 91.00% F1, σ = 0.00241; median seed-difference 0.38% on CoNLL-2003 NER. **VERIFIED-D.**
- **Musgrave, Belongie, Lim, "A Metric Learning Reality Check"** (arXiv 2003.08505): "most papers do not present confidence intervals ... improvements ... often range in the low single digits". **VERIFIED-D.**
- **Picard, "torch.manual_seed(3407) is all you need"** (arXiv 2109.08203): Table 1, 10⁴ seeds — min 89.01 vs max 90.83, a "**1.82% difference ... just the effect of finding a lucky/cursed seed**". **VERIFIED-D.**

**DERIVED: the general ML literature's own guidance is that single-run, single-split numbers differing by low single-digit points carry no information.** The project's model-selection decisions are currently being made inside that noise band.

### C6g. UNVERIFIED (statistical-fragility track)

- **UNVERIFIED — no TAS paper found that performs a formal significance test on TAS metrics.** This is an absence of evidence from a search channel that was broken for most of this session (arXiv search HTTP 429, arXiv API HTTP 406), **not proof of absence**.
- **UNVERIFIED — no TAS paper reporting error bars on all three standard benchmarks** except C2F-TCN's appendix.
- **UNVERIFIED — exact per-split variance on Breakfast/50Salads as reported by a paper's own tables.** I found run-to-run σ (C6a), seed σ (C6c) and labelled-subset σ (C6c), but not a tabulated per-split breakdown.

---

# Honest bottom line

1. **"Features don't matter" is false, and "architecture doesn't matter" is also false.** Both are large effects. On Breakfast specifically, the controlled IDT→I3D swap at fixed MS-TCN is **+0.3 Edit / +1.2 Acc but −5.6 F1@10** — three independent sources agree the feature swap is ≈0 and I3D is *worse* for F1 — while MS-TCN(I3D)→DiffAct+ASFormer at fixed I3D is **+16.7 Edit / +27.7 F1@10**. In the Souri et al. study the sign of the feature effect flips per method (−29.2 to +8.1 MoF), and a CLIP swap cost −19.0 Edit on 50Salads. **So "swap the feature extractor" is not a supported fix, and "the literature proves features are the bottleneck" is not supported either.**
2. **The premise "architecture changes give small gains in this regime" is NOT supported for the standard benchmarks, and I will not pretend otherwise.** Published TAS architecture gains on Breakfast are large (+16.7 Edit / +27.7 F1@10 over four generations). **The defensible version of the claim is narrower and better:** *capacity* is not the lever — UVAST matches/beats ASFormer on Edit (Breakfast 77.2 vs 75.0; 50Salads 83.9 vs 79.6) with **fewer** parameters (1.1M vs 1.3M) by changing the output from frame labels to a segment/transcript sequence — and the project has already ablated capacity independently. **The cleanest controlled evidence is the 14-encoder sweep (ICRA 2026, arXiv 2602.21406): at a fixed pipeline, choosing among 14 VLMs spanning 16× in parameters moves `Avg` by only 2.2–8.9, while removing one pipeline stage costs 15.8–40.6 — the method is worth 1.8–15× the encoder. And "larger checkpoints underperform their smaller counterparts" in all four VLM families, with the smallest model winning Breakfast.** That is the encoder-side analogue of the project's own capacity-insensitivity.

3. **The single most on-point literature framing is UVAST's localization-vs-discrimination split**: frame-wise models have "**high localization performance, i.e., high frame-wise accuracy, but low discrimination performance ... low Edit distance**", which they attribute to "**the relatively small amount of training videos**". This is the project's exact failure mode, named, in a peer-reviewed paper. **Edit 51.08 vs the survey's own formula implies ≈48.9% substitution rate, matching the measured 51.6% wrong-segment rate — one failure mode, not two.**

4. **Whole-class identity failure is documented and severe, and global metrics are known to hide it.** AsFormer and DiffAct have "**zero accuracy on 5 and 4 out of 48 actions on Breakfast**"; tail-class accuracy on Assembly101 is 4.7–5.7%; the long-tail paper states outright that global TAS metrics "**obscure the performance of tail actions**". **Action: report per-class and head/tail metrics for all 9 classes + idle, not just global edit/F1. The project's own 51.6% figure is a per-class statistic that the standard protocol would have hidden.**

5. **There are cheap, training-free structural fixes with verified, sizeable gains — and they target identity, not boundaries.** Constrained Viterbi with per-class **duration bounds + valid start/end + hard observed transitions** gives **+5.1 Edit / +4.7 F1@10 on Breakfast** and **+3.0 Edit on 50Salads at 5% labels**, model-agnostically, at `O(C·T)` (2.81 s vs 148.9 s for a grammar parser). VidParse gets **F1 21.30 → 80.47 with identical features and no gradient updates** under a strongly constrained label graph (**caveat: recipe-specific GTEA protocol, not a standard TAS split — do not generalise the +59 figure**). Segment-level relabelling by nearest-class-mean (**S-NCM**) gives +3.1 F1@10 on Breakfast by re-deciding each whole segment's identity after boundaries are fixed — **the exact shape of the project's problem.**

6. **But a naive transition prior alone HURTS — verified twice, independently.** On 50Salads, "classical Viterbi" (logits + transition confidences) scored **45.5 F1@10 / 39.2 Edit**, *worse* than doing no decoding at all (51.8 / 45.9); and UVAST reports MS-TCN + Viterbi dropping Breakfast Acc 62.5 → 35.9. UVAST's reason: "**the global optimum of the Viterbi objective function does not guarantee the global optimum of the evaluation metrics**". **Viterbi is a sharpener, not a fixer. Add duration + start/end + hard transitions together, or don't add it.**

7. **The label/data side is where the largest *cheap* wins actually are.** Labelling **0.25% of frames** moved Breakfast MoF 36.7 → **56.0** (+19.3), while going 0.25% → 100% (400× more labels) bought only +5.3 more. Fewer labels used *better* (semi-supervised ICC at 5%: MoF 65.3 vs supervised-only 5%: 26.0) beats "more labels". And an oracle action-length model adds **+18.0 MoF** — segment *extent* structure carries most of the remaining signal. **A 64-dim hand-crafted feature reaches Breakfast MoF 61.3 in the RNN-HMM paper, so 144 dims of hand-crafted detection features are not plausibly the binding constraint.**

8. **On the specific "our fine-grained classes are ambiguous" hypothesis, the literature does NOT say what the brief expects — I must report this as a negative.** Breakfast does have **10 near-identical `pour X` classes** as a taxonomy fact, but **neither the Breakfast nor the 50Salads paper claims those classes are visually indistinguishable**, and 50Salads explicitly asserts the opposite ("**Annotations are therefore unambiguous and repeatable**"). Nobody reports inter-annotator agreement in those papers. **What IS verified is boundary noise**: a third-party 5-annotator study on Breakfast found segment start times agreed with **σ ≈ 1.5 s ≈ 23 frames at 15 fps**, and training on boundary frames collapses 50Salads F1@50 from 39.4 to **3.2**. **Do not cite Breakfast/50Salads as endorsing class ambiguity — that would be fabrication.** Measure the project's own annotation ambiguity instead.

9. **The measurement noise in the project's regime probably exceeds most of the architecture deltas being chased.** Verified: run-to-run σ = **1–2.5 MoF** over 5 runs on Breakfast, with a **max−min spread of up to 6.3** within one method; labelled-subset choice alone gives **σ up to 2.5 F1**; MS-TCN and ASFormer report **zero "±" in their entire papers**; and reproducing ASFormer on 50Salads with official code shifted F1@50 by **+17.4**. The project's **8 test videos** are smaller than the smallest standard split (50Salads ≈ 10 videos ≈ 200 segments). **Multi-seed/multi-split reporting with error bars is a prerequisite for believing any of the project's own ablations, including the negative ones.**

10. **What is genuinely absent, and should be stated as absent.** I found **no** verified paper that: (a) reports "fraction of GT segments with a wholly wrong label"; (b) quantitatively shows identity errors dominate boundary errors; (c) injects class-label noise into Breakfast/50Salads/GTEA; (d) performs a significance test on TAS metrics; or (e) benchmarks feature extractors for video TAS in one controlled study. **Items (a) and (b) are therefore plausible *novel contributions* for the project, not settled literature** — and the project already has the diagnostic numbers to report them.

---

## Appendix: local artifacts

| Path | Contents |
|---|---|
| `notes/adversarial.md` | this file |
| `raw/findings_datascale.md` | delegated findings: data scaling, few-shot TAS, split/std reporting, variance methodology (670 lines) |
| `raw/findings_features.md` | delegated findings: feature-swap vs architecture-swap across domains (≈420 lines) |
| `raw/findings_labelnoise.md` | delegated findings: annotation ambiguity, annotator variance, label noise (833 lines) |
| `raw/survey.{html,txt}` | Ding/Sener/Yao TPAMI 2023 survey |
| `raw/mstcn.{html,txt}` | MS-TCN CVPR 2019 |
| `raw/cad.{html,txt}` | Constraint-Aware Decoding, ICPR 2026 |
| `raw/gram.{html,txt}` | Activity Grammars / KARI, NeurIPS 2023 |
| `raw/eval.{html,txt}` | Souri et al. evaluation report |
| `raw/lt.{html,txt}` | Cost-Sensitive Learning for Long-Tailed TAS |
| `raw/endonet.{html,txt}` | EndoNet, IEEE TMI 2017 |
| `raw/rnn_hmm.{html,txt}` | Kuehne/Richard/Gall RNN-HMM, TPAMI |
| `raw/uvast.{html,txt}` | UVAST, ECCV 2022 |
| `raw/ltctx.{html,txt}` | LTContext, ICCV 2023 |
| `raw/vidparse.{html,txt}` | VidParse, ECCV 2026 |
| `raw/ovtas.{html,txt}` | OVTAS 14-VLM study, ICRA 2026 |
| `mktext.py`, `tbl.py`, `asearch.py`, `minipdf.py` | HTML/PDF text and table extractors (pdftotext/pypdf unavailable in this environment) |

**Tooling note.** `pdftotext` and `pypdf` are not installed; all PDF-derived evidence used either an HTML full-text mirror (arXiv HTML / ar5iv / ECVA / CVF HTML) or a hand-written PDF text extractor (`minipdf.py`, which drops fi/ff/fl ligatures — quotations from that path are marked in the source findings files). The arXiv `/search/` endpoint returned **HTTP 429** and the arXiv API **HTTP 406** for a large part of this session, so several search-based nulls reflect failed queries rather than confirmed absence; those are marked `UNVERIFIED — <what was tried>` throughout.

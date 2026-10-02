# Recent TAS literature aimed at whole-segment identity / sequence structure (2024–2026)

Survey context: industrial TAS project on a **144-dim hand-crafted YOLO-ROI feature** (2x3 grid, per (class, region)
[presence, count, max_area]), tiny dataset (~2,639 test frames, 8 videos, ~9 classes + idle), dense per-frame labels,
segment-level business metric (Edit, F1@k, insertion recall, frame acc). Best model: MS-TCN++ (4x10, hidden 128,
~3.3M params) → Edit 51.08, F1@0.1 40.01, frame acc 53.98.

Core unsolved failure being targeted here: **51.6% of GT segments receive a wholly wrong label** (one class: 20/21
segments mislabeled); 78% of frame errors are deep inside segments. Capacity / feature / loss tweaks do not fix it.

**Evidence rule used below:** every number comes from a primary source I actually fetched (arXiv abs page, arXiv HTML
full text, or GitHub API), and says which table/section. Anything I could not verify is marked UNVERIFIED with what I
tried. No metric is quoted from memory or from a secondary source.

---

## Global verification caveat (affects the discovery section)

`https://arxiv.org/search/?searchtype=title&query=...&size=...&order=-announced_date_first` — the instructed recipe —
**returned HTTP 429 (rate-limited) on every attempt**: 6 attempts across ~5 minutes with 20–45 s backoff, all `HTTP:429 bytes:0`.
`https://export.arxiv.org/api/query` returned **HTTP 406**, and `https://api.semanticscholar.org/graph/v1/paper/search`
returned **HTTP 429**. `https://arxiv.org/abs/<id>` and `https://arxiv.org/html/<id>` worked reliably throughout (2–4 s spacing).
Discovery was therefore done via web search + a curated GitHub list, and **every candidate was then verified by fetching its
arXiv abs/HTML page directly**, which is the same primary-source standard. The title-field newest-first sweep itself is
**UNVERIFIED — attempted 6x, HTTP 429 every time**.

---

## 1. Improving Temporal Action Segmentation via Constraint-Aware Decoding

- **Venue/year**: ICPR 2026 (verified: arXiv abs page `Comments: accepted to ICPR 2026`; submitted 11 May 2026, cs.CV).
  Authors: Yeo Keat Ee, Debaditya Roy, Chen Li, Hao Zhang, Basura Fernando.
- **URL**: https://arxiv.org/abs/2605.10149 — full text fetched from https://arxiv.org/html/2605.10149v1
- **Code**: paper states https://github.com/LUNAProject22/CAD — **fetched GitHub API → `{"message":"Not Found"}`**; the
  author's repo list (`LUNAProject22` user repos, 15 repos) contains **no CAD repo**. So the announced code was **not
  publicly available at fetch time**. License: paper is **CC BY 4.0** (verified in HTML page footer).
- **Core mechanism**: Extracts three **statistical structural priors directly from the annotated training set** — (a) a
  transition-confidence graph over consecutive action pairs (A→B), all observed transitions forming the valid set `T`;
  (b) valid start-action and valid end-action sets; (c) **per-class normalized duration bounds** `[d_min_c, d_max_c]`
  normalized as a fraction of the whole activity length. These are enforced inside a **modified Viterbi decoder** with a
  duration tracker `D ∈ R^{T×C}`: a self-transition is legal only if `D[t-1,c]/T < d_max_c`; a switch `c'→c` is legal
  only if the outgoing segment has already reached `D[t-1,c']/T ≥ d_min_c'` **and** `(c'→c) ∈ T`; the final timestep must
  land on a class in the valid end set. It is **training-free at inference** and adds no model capacity; decoding
  complexity is `O(CT)`, linear in T (vs `O(T^3)` worst case for the grammar-based KARI baseline it compares to).
  In the semi-supervised setting the same constrained decoder is used to **re-label pseudo-labels between ICC iterations**.
- **Problem targeted**: **Sequence structure / legal-order + duration plausibility** — not boundary refinement, not backbone
  capacity. Explicitly framed against "over-segmentation, poor temporal coherence" and "temporally implausible transitions
  between semantically unrelated actions" (Sec. 4.3). It directly suppresses whole-sequence structural errors, including
  transitions that violate the observed action grammar.
- **Datasets & metrics** (Table 1), feature backbone **not stated** — I grepped the full text for `i3d` → **0 matches**;
  the paper reuses the ICC [21] / C2F-TCN [22] / ASFormer setups, so backbone features are inherited, not restated
  (`not stated` in this paper's own text).
  - **50Salads, semi-supervised (5% labels), backbone C2F-TCN** — w/o constraints: F1@10 51.8, F1@25 47.7, F1@50 37.0,
    Edit 45.9, Acc 62.3 → **Ours: 55.8, 51.4, 38.5, Edit 48.9, Acc 63.4**.
  - **Breakfast, semi-supervised (5%)** — w/o: 57.1, 51.2, 34.6, Edit 54.6, Acc 64.2 → **Ours: 58.5, 52.4, 36.0, Edit 56.4, Acc 63.9**.
  - **Breakfast, fully supervised, backbone ASFormer** — w/o: 74.1, 68.7, 55.5, Edit 72.8, Acc 72.4; KARI* [5] reproduced:
    77.3, 71.6, 57.0, Edit 78.3, Acc 74.5; **Ours: 78.8, 73.1, 58.0, Edit 77.9, Acc 74.6**.
  - **50Salads, fully supervised, ASFormer** — w/o: 83.4, 80.8, 74.6, Edit 75.7, Acc 85.0; KARI*: 83.8, 81.8, 74.7,
    Edit 76.4, Acc 83.3; **Ours: 84.8, 83.1, 76.0, Edit 78.6, Acc 82.9**.
  - **Ablation (Table 3, 50Salads, semi-supervised)**: classical Viterbi (logits + transition confidences only)
    45.5, 40.2, 30.1, Edit 39.2, Acc 60.1 → constrained Viterbi **55.8, 51.4, 38.5, Edit 48.9, Acc 63.4**. This is the
    single most informative row in the paper: **most of the gain comes from the hard duration/start/end/transition
    constraints, not from Viterbi decoding as such.**
  - **Runtime (Table 2)**, 20 Breakfast videos, ASFormer, NVIDIA A5000: KARI (grammar + BEP parser) **148.9 s** vs Ours
    (constrained Viterbi) **2.81 s**.
  - **Honest caveat I read off Table 1**: on fully-supervised Breakfast **Edit, Ours (77.9) is slightly *below* KARI (78.3)**,
    and on Breakfast fully-supervised Acc the gain over baseline is only 72.4 → 74.6. Gains are real but not uniform.
- **Params / compute**: **no model parameters added** (inference-time decoding only); decoding `O(CT)`; 2.81 s per 20 videos
  (Table 2). Model-side parameter count **not stated**.
- **Small-data & low-dim-feature relevance**: **HIGH and unusually well-matched.** The mechanism is *feature-agnostic*
  (it consumes per-frame class posteriors, whatever produced them), adds **zero** parameters, needs no retraining, and
  its priors are estimated from the annotated **segment sequences** — which is exactly the regime where a tiny dataset's
  frequent whole-segment identity errors look like illegal transitions. **Caveat specific to this project**: the duration
  prior is normalized to *total video length*, so with only 8 videos the per-class `[d_min, d_max]` intervals and the
  transition graph would be estimated from very few segment instances (and a class with 21 segments across the dataset
  gives a very noisy graph). Both priors are directly extractable from our labels, so this is testable in days, not weeks.
  No diffusion-like iteration, no added compute of concern.
- **Verification notes**: **FETCHED** — abs page + full HTML text (constraints in Sec. 3.1/3.2, all numbers from Tables 1–3,
  runtime Table 2). Code availability **verified negative** via GitHub API. Backbone features **verified absent** from the text.

---

## 2. Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation

- **Venue/year**: **CVPR 2026 Workshop** "AI-driven Skilled Activity Understanding, Assessment & Feedback Generation
  (SAUAFG)" (verified: arXiv abs page `Comments:`). Submitted 2 Apr 2026, cs.CV. Authors: Hinako Mitsuoka, Kazuhiro Hotta.
  **Note: workshop paper, not main conference.**
- **URL**: https://arxiv.org/abs/2604.01859 — full text fetched from https://arxiv.org/html/2604.01859v1
- **Code**: **none found** — I grepped the full text for `github`/`code is available`/`code will be`: only matches were
  arXiv page chrome ("Report GitHub Issue"). No repository link, no project page. License: **arXiv.org perpetual
  non-exclusive license** (verified in HTML page).
- **Core mechanism**: A **training-time dual-loss framework** adding **one class-agnostic boundary output channel** and two
  auxiliary losses on top of an unmodified backbone.
  1. **Boundary-regression loss `L_B`**: binary cross-entropy on the single extra boundary channel; it "encourages high
     responses around class transitions while maintaining low values elsewhere" (Sec. 3.1, Fig. 3).
  2. **CDF-based segment shape regularization loss `L_S`** (Sec. 3.2, Eq. 2–3) — the segment-level part. For each GT
     segment `S_i=[s_i:e_i]` with class `c_i`, a `δ`-frame margin is removed at both ends (to avoid ambiguous transition
     frames), giving length `L_i = e_i − s_i − 2δ`; segments with `L_i ≤ 0` are skipped. The **predicted** distribution is
     the L1-normalized probability of **that segment's own GT class**: `p̂_i = norm_l1(Ŷ[c_i, s_i+δ : e_i−δ]) ∈ R^{L_i}`.
     The **target** is a **uniform** vector `p_i = 1/L_i · 1`. The loss is the **squared ℓ2 distance between the two
     cumulative distributions**: `L_S = (1/N) Σ_i (1/L_i) Σ_j (CDF(p̂_i)[j] − CDF(p_i)[j])²`. The paper motivates this as
     the classic 1D-Wasserstein-via-CDF connection, and activates `L_S` only after a warm-up (`E_start = 20` epochs for MS-TCN).
  3. **Temporally decoupled loss assignment** (Sec. 3.3): to avoid conflicting gradients, `L_B` is applied **only in
     boundary regions** (±5 frames around GT transitions) and `L_S` **only in non-boundary regions**;
     `L_total = L_model + λ_B·L_B^(boundary) + λ_S·L_S^(non-boundary)`.
- **Critical technical reading (important for our problem) — what the CDF loss does and does NOT do**: it matches the
  predicted probability of the **already-correct class** `c_i` against a *uniform* profile over the segment interior.
  There is **no cross-class comparison anywhere in `L_S`** and no coupling between the predicted segment and a *set* of
  candidate labels. So it **cannot reassign identity**: if the model has already chosen the wrong class for a whole
  segment, `L_S` supplies almost no gradient to fix the choice — it only flattens/smooths the confidence of whatever class
  the segment was assigned. **"Why it produces coherent within-segment structure"** is exactly this: it penalizes
  probability mass that drifts within a segment (fragmentation, over-segmentation, mid-segment dips) by forcing the
  per-frame cumulative mass of the chosen class to grow linearly, i.e. a constant per-frame probability over the segment.
  It is a **within-segment shape prior, not a segment-identity prior.**
- **Problem targeted**: Primarily **boundary quality + within-segment coherence (over-segmentation)**. The abstract's words
  "segment-level regularization" are accurate but the loss is not identity-aware. **Directly relevant negative result for
  our project**: this is the best-verified example of an architecture-agnostic "segment-level" loss, and its design shows
  why such losses cannot by themselves break a whole-segment-identity ceiling.
- **Datasets & metrics**: GTEA (28 videos, 7 activities, 11 classes, leave-one-subject-out), 50Salads (50 videos, 17
  classes, 5-fold), Breakfast (1,716 videos, 10 recipes, 48 classes, 4-fold). Features: **I3D**, 15 fps for GTEA/Breakfast,
  50Salads downsampled 30→15 fps (verified: Sec. 4.1.1 "For all datasets, we utilize I3D [4] features extracted from video
  frames as input"). Metrics: Acc, segmental Edit, F1@{10,25,50}.
  - **Table 2 (backbone MS-TCN), F1@{10,25,50} / Edit / Acc:**
    - **GTEA**: 86.47 / 83.73 / 71.20 / Edit 79.92 / Acc 76.53 → **+Ours: 88.13 / 84.62 / 72.40 / Edit 83.23 / Acc 78.20**
    - **50Salads**: 73.96 / 71.42 / 62.12 / Edit 66.29 / Acc 79.11 → **+Ours: 77.44 / 75.22 / 67.31 / Edit 70.29 / Acc 81.17**
    - **Breakfast**: 50.61 / 46.41 / 36.58 / Edit 61.75 / Acc 66.96 → **+Ours: 55.96 / 51.39 / 40.78 / Edit 62.29 / Acc 67.46**
  - Also applied to C2F-TCN and FACT (Table 2): e.g. **C2F-TCN GTEA Edit 81.14 → 85.75**; **FACT GTEA Edit 89.89 → 91.34**.
  - Paper's own claim text (Sec. 4.2.1), which I **recomputed against Table 2 and confirmed**: MS-TCN 50Salads **+4.0 Edit**
    (70.29−66.29 = 4.00) and **+3.5 F1@10** (77.44−73.96 = 3.48); GTEA **+3.3 Edit** (83.23−79.92 = 3.31); largest
    F1@10 gain **+5.4** on Breakfast (55.96−50.61 = 5.35). **Frame accuracy largely unchanged** (and the paper is candid
    that some settings show marginal drops, e.g. F1@25 on 50Salads with C2F-TCN).
- **The two "lightweight / architecture-agnostic" claims — verified, not just asserted**:
  - **Lightweight: VERIFIED.** Table 3 (GTEA) reports MACs and params alongside metrics: MS-TCN **0.80 G / 0.80 M** →
    MS-TCN + Ours **0.80 G / 0.80 M** (unchanged at the reported precision). For contrast, same table: ASRF 1.30 / 1.30,
    **BCN 16.64 / 16.66**. So one extra output channel costs no reportable params/MACs. Head-to-head on GTEA (Table 3):
    MS-TCN+Ours **88.13 / 84.62 / 72.40 / Edit 83.23 / Acc 78.20** vs **BCN 88.07 / 85.88 / 75.73 / Edit 82.26 / Acc 79.95**
    — **Ours wins F1@10 and Edit, BCN (at ~20x the compute) wins F1@25, F1@50 and Acc.** The efficiency claim is therefore
    "comparable quality at a fraction of the cost", not "better than BCN". Reported honestly here rather than smoothed over.
  - **Architecture-agnostic: VERIFIED** (3 structurally different backbones — MS-TCN TCN, C2F-TCN coarse-to-fine, FACT
    transformer — with reported gains on each, and identical loss configuration logic).
- **Params / compute**: no backbone params added; **0.80 M / 0.80 G MACs** for MS-TCN ± Ours (Table 3). Loss weights and
  warm-up epochs selected by cross-validation per model/dataset (Table 1); "no additional model-specific tuning".
- **Small-data & low-dim-feature relevance**: **HIGH for feasibility, LOW for our specific failure.** Feasibility: it is a
  pure loss change on any backbone, no extra branch, and would drop into an MS-TCN++ training script; the CDF term is cheap
  and needs only GT segments we already have. But: (i) the boundary channel is class-agnostic and adds only transition
  localization; (ii) `L_S` is — as shown above — **identity-blind**, so it plausibly will *not* move the 51.6%
  whole-segment-wrong number; (iii) the CDF geometry is designed for I3D-15fps segments and would need re-derivation for
  our segment-length statistics. **Predicted effect on our project: improves fragmentation/frame accuracy inside correct
  segments, does not address whole-segment identity.** Worth a cheap ablation precisely because it is cheap.
- **Verification notes**: **FETCHED** — abs page + full HTML text; mechanisms from Sec. 3.1–3.3 (equations read verbatim),
  all metrics from Tables 2–3, dataset/feature details from Sec. 4.1.1. Code existence **verified negative** (grep of full
  text). The "architecture-agnostic/lightweight" claim was **independently checked** against Table 3's MACs/params.

---

## 3. Long-tail TAS: two papers by the same group (ECCV 2024 / BMCV 2024)

These two are the most direct academic treatment of "some classes have very few segments and get catastrophically
mislabeled" — the same regime as our project's `20 of 21 segments mislabeled` class.

### 3a. Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment

- **Venue/year**: **ECCV 2024** (verified: arXiv abs page `Comments: Accepted by ECCV 2024`).
  Pages: arXiv 2408.09919. Authors: Zhanzhong Pang, Fadime Sener, Shrinivas Ramasubramanian, Angela Yao.
- **URL**: https://arxiv.org/abs/2408.09919 — full text fetched from https://arxiv.org/html/2408.09919
- **Code**: https://github.com/pangzhan27/GTLA — **verified via GitHub API** (exists, 12 stars, pushed/active).
  **License: none found** — GitHub API `license: None`; repo contents listing shows no `LICENSE` file
  (`CB.py, Focal.py, LA.py, LDAM.py, README.md, activity_weighted_tla.py, bags.py, baseline.py, data, eval.py, seesaw.py, tau_norm.py`).
- **Core mechanism (G-TLA)**: Two coupled components that explicitly **encode action interdependencies, breaking the
  class-independence assumption of standard long-tail methods**:
  1. **Group-wise classification** — the frame feature is fed to **`n` classifiers, one per activity group** (groups are
     activity labels, or unsupervised clustering on action-frequency KL divergence). Each group has an auxiliary
     `others` class; a class shared across groups (e.g. *spoon sugar* in both "making tea" and "making coffee") is treated
     as **a different class per group**. This removes activity-incompatible classes from the label space by construction,
     which suppresses cross-activity false positives and "implausible predictions".
  2. **Temporal logit adjustment** — the per-class logit offset is made **per-class and per-frame** by multiplying in a
     temporal factor `T^{(k)}_{c,t}(X)` derived from **action-ordering priors / temporal bounds** (e.g. *stir tea* always
     follows *pour water* even with *spoon sugar* in between). Loss = `L_GTLA + λ·L_sm`. **Temporal logit adjustment is
     applied only in the target group during training and is NOT used at inference** (inference does argmax on `p̂`);
     the group label is inferred from the `others`-class predictions.
- **Problem targeted**: **Class imbalance / rare-class identity**, plus **sequence-order structure**. Not boundary refinement.
  The paper's framing is precisely "SOTA methods fail to recognize tail actions" and it introduces **new balanced
  per-class metrics** because standard global metrics mask tail performance.
- **Datasets & metrics**: Breakfast, **YouTube Instructional (YTI)**, Assembly101, GTEA, 50Salads. Backbones **MS-TCN** and
  **ASFormer** (plus **DiffAct** on Breakfast, Table 4). Features: **I3D** — "pre-extracted I3D features as the original
  papers" (Sec. 5.1). Metrics: **balanced per-class recall (frame-wise)** and **per-class F1@0.25 (segment-wise)**,
  averaged within **Head** and **Tail** groups plus their **harmonic mean (Hmean)**; global Acc/F1@25 reported in grey.
- **Imbalance is extreme (Table 1, imbalance ratio = most frequent class count ÷ least frequent)**: Breakfast 639,
  YouTube 558, **Assembly101 2,604**, 50Salads 6, GTEA 24.
  - **Table 2 (YouTube), AsFormer baseline → G-TLA deltas**: frame acc Head **53.1** → +2.3; Tail **17.2** → **+6.8**;
    Hmean **26.0** → **+7.5**. Segment F1@25 Head **47.6** → −0.3; Tail **20.2** → **+5.1**; Hmean **28.4** → **+4.6**.
    Global: Acc 69.8 → +0.1, F1@25 45.6 → +0.6.
  - **Table 2 (YouTube), MS-TCN baseline → G-TLA**: frame Head 46.0 → +2.7, Tail **15.5** → **+6.3**, Hmean 23.2 → **+6.8**;
    F1@25 Head 39.0 → +2.7, Tail 16.8 → +3.3, Hmean 23.5 → +3.6; global Acc 68.0 → −0.4, F1@25 39.1 → +1.1.
  - **Table 3 (Breakfast), AsFormer baseline → G-TLA**: frame Head 69.7 → +0.6, Tail **39.8** → **+3.4**, Hmean 50.7 → **+2.6**;
    F1@25 Head 69.9 → +1.8, Tail **43.9** → **+2.6**, Hmean 53.9 → **+2.6**; global 72.4 / 69.9.
  - Paper's own comparative text: with AsFormer it beats the next-best long-tail method by 2.2 / 1.5 points frame accuracy
    (YouTube / Breakfast) and 2.5 / 1.0 points F1@25; with MS-TCN by 4.1 / 2.6 and 1.7 / 4.1.
- **Does it show rare-class *segment* mislabeling is the dominant error? — YES, and it is the strongest primary evidence I
  found in the whole survey.** Two direct quotes/numbers:
  - Sec. 1: *"ASFormer [52], DiffAct [35] have **zero accuracy on 5 and 4 of 48 classes**, respectively"* (Breakfast).
    A class with **zero** frame accuracy is by definition a **whole-class identity failure**, not a boundary error — the
    model never emits that label at all. This is the literature analogue of our "20 of 21 segments mislabeled" case.
  - Table 2/3: tail frame recall (17.2 YouTube AsFormer; 15.5 MS-TCN) is **~3x lower than head** (53.1 / 46.0), and tail
    *segment* F1@25 is 20.2 vs 47.6 head. The gap is far larger at segment level than the paper can close, and G-TLA's
    own gains are much larger on tail than head (+6.8 tail vs +2.3 head frame, YouTube/AsFormer).
  - The paper also documents that standard global metrics **hide** this ("these metrics aggregate globally across all
    samples, masking tail class performance", Sec. 5.2) — relevant to how we should report our own numbers.
- **Secondary finding with a direct bearing on identity confusion (Appendix)**: **group identification** improved from
  **62.0% (baseline) to 83.0%** on Assembly101 with MS-TCN. That is a *coarse* identity decision (which activity) improved
  by a structural/group-wise design — evidence that coarse-to-fine identity disambiguation is where structural methods
  buy accuracy.
- **Params / compute**: **not stated** in the text I fetched (no param/FLOP table; grep for `param`/`million` found no
  model-size numbers). Cost is `n` classifiers over the final feature layer (one per activity group) — the only structural
  overhead, and small at our scale (~9 classes + idle would likely mean very few groups).
- **Small-data & low-dim-feature relevance**: **HIGH, and the single most portable idea for our project.** It consumes
  pre-extracted per-frame features and a per-frame classifier — **fully compatible with 144-dim features**. The two
  ingredients it needs, we already have: (i) **an activity/group label per video** (our 8 test videos are whole procedures —
  this is our analogue of G-TLA's groups, and it is a *hard, cheap prior* that forbids a segment in video A from taking a
  class that never occurs in that activity); (ii) **action-ordering/transition priors** from the label sequences (the same
  statistics paper 1 uses). The group-wise trick of **treating shared classes as separate per-group classes** and adding an
  `others` class is precisely a mechanism that can kill impossible whole-segment labels. Very low compute, no extra
  iteration. Our severe imbalance (one class: 21 segments) is exactly the tail regime where they report the largest gains.
  **Caveat**: with 8 videos, per-group class priors are extremely sparse; the paper itself notes that adding a small value
  to a zero conditional probability "does not solve the numerical problem — instead it converts activity-incompatible
  classes to tail classes", which is a failure mode we would hit.
- **Verification notes**: **FETCHED** — abs page + full HTML text (mechanism Sec. 3.1–4.3, Table 1 imbalances,
  Tables 2–4 numbers, Sec. 5.2 metric definitions, Appendix group-ID accuracy). Code + license via GitHub API.
  Param count **UNVERIFIED — grepped full text for `param`/`million`/model-size: none found.**

### 3b. Cost-Sensitive Learning for Long-Tailed Temporal Action Segmentation

- **Venue/year**: **BMCV 2024** (verified: arXiv abs page `Comments: BMCV 2024`). arXiv 2503.18358 (v1 March 2025).
  Same author group as 3a (Zhanzhong Pang, Fadime Sener, Shrinivas Ramasubramanian, Angela Yao).
- **URL**: https://arxiv.org/abs/2503.18358 — full text fetched from https://arxiv.org/html/2503.18358
- **Code**: https://github.com/pangzhan27/CSL_LT-TAS — **verified via GitHub API** (exists, 6 stars).
  **License: none found** — GitHub API `license: None`, and directory listing shows only
  `README.md, baselines, eval.py, transit-csl.py` (no `LICENSE`).
- **Core mechanism (CSL + S-NCM)**: Diagnoses a **bi-level learning bias** — (1) *class-level* bias from class imbalance,
  (2) *transition-level* bias from transition-frequency variation — and fixes both:
  1. **Learning states from a transition-based confusion tensor** `C` (its `ijk`-th entry counts/derives classifier
     accuracy): the learning state of a class `i` and of a transition `(k→i)` is their **accuracy relative to the average**,
     which labels an action/transition as **over- or under-learned**. The paper explicitly notes this differs from the
     usual *over-fitting* trend of tail classes elsewhere: here tail classes are **under-learned**.
  2. **Constrained optimization → cost-sensitive loss.** A balanced-accuracy objective plus temporal-transition constraints
     is reframed as a **Lagrangian min-max** problem, giving a **weighted cross-entropy** whose per-class weight is
     `G_{i,i,k} = (1/π_i) + λ_{i,k}·1(T_{i,k}>0)/π_i`, i.e. **action prior + transition learning state**; `λ_{i,k}` are
     learned multipliers that grow while a transition stays below average accuracy (table: transitions into *butter_pan*
     from *spoon_flour* keep increasing λ, while *stir_dough*→*butter_pan* decays to zero).
  3. **S-NCM (Segment Nearest Class Mean) — the segment-identity-relevant part.** At inference: use the classifier's
     predictions `ŷ` to **detect segment boundaries**, then **relabel each entire segment by majority vote of frame-wise
     NCM predictions** `v̂` (nearest class mean over **frame representations** by Euclidean distance to per-class means),
     so that **all frames in a segment share one label**. The paper states plain frame-level NCM "disregard[s] temporal
     continuity and lead[s] to over-segmentation", which S-NCM fixes.
- **Problem targeted**: **Class imbalance (primary) + segment-level identity & coherence (via S-NCM)**. Since our project's
  failure is "whole segment gets one wrong identity", S-NCM is the most on-target mechanism in this paper: it makes the
  identity decision **once per segment, in embedding space against class prototypes**, instead of per frame in logit space.
- **Datasets & metrics**: Breakfast (1,712 videos, 48 classes, avg 2.3 min), Assembly101 (4,321 videos, 202 coarse classes,
  avg 7.1 min), 50Salads (50 videos, 19 actions). Backbones: **MS-TCN**, **ASFormer**, and **DiffAct** (diffusion, Table 4).
  Metrics: **per-class F1@{10,25,50} + per-class accuracy + global F1@25 ("G_F1")**, plus **Head/Tail group** accuracy and
  F1@25. Feature backbone: **not stated in the text I fetched** (grep for `I3D` produced no hit) — the paper follows the
  original backbones' setups like 3a; `not stated`.
  - **Table 2 (backbone MS-TCN), per-class F1@{10,25,50} / Acc / G_F1, baseline → +ours(S-NCM) deltas**:
    - **Breakfast**: 48.1 / 44.8 / 36.9 / Acc 49.1 / G_F1 57.9 → **+8.1 / +8.1 / +5.7 / Acc +3.7 / G_F1 +6.1**
    - **50Salads**: 78.8 / 76.4 / 67.6 / Acc 75.6 / G_F1 75.9 → **+2.8 / +3.1 / +3.2 / Acc +1.7 / G_F1 +3.1**
    - **Assembly101**: 7.5 / 6.6 / 4.8 / Acc 8.3 / G_F1 27.2 → **+4.1 / +3.3 / +2.0 / Acc +2.6 / G_F1 +2.3**
  - **Table 3 (backbone MS-TCN), Head/Tail accuracy and F1@25 — baseline → +ours(S-NCM)**:
    - **Breakfast**: Acc 65.1 / **37.7** → 65.3 / **44.0**; F1@25 53.3 / **38.7** → **64.5** / **44.6**
    - **50Salads**: Acc 87.7 / 70.0 → 87.8 / 72.5; F1@25 85.7 / 72.1 → 87.7 / 75.7
    - **Assembly101**: Acc 33.9 / **4.7** → 34.1 / **8.7**; F1@25 26.3 / **3.9** → **31.7** / **7.6**
      — head/tail segment F1@25 of 26.3 / **3.9** is the starkest imbalance number in either paper: on a 202-class dataset,
      tail-class segments are essentially never segmented correctly, and CSL roughly **doubles** tail F1 (3.9 → 7.6).
  - **Table 3 (AsFormer) baseline I could read for cross-checking**: Breakfast head/tail Acc **69.7 / 39.8**, F1@25
    **69.9 / 43.9** — **identical to the AsFormer Breakfast baseline in paper 3a's Table 3**, a useful independent
    consistency check between the two papers.
  - **Transition-level evidence (Sec. 5, transition-accuracy analysis)**: baseline AsFormer detects **132 of 167**
    transitions; CSL detects **11 more**, with higher average transition accuracy **56.1% vs baseline 54.3%**.
  - **Paper's own claim I could NOT reproduce from its tables (flagged, not endorsed)**: "we surpass the second best model
    LA on F1 score by **8.3%, 3.5%, and 3.0%** for Breakfast, 50Salads, and Assembly101 respectively for MSTCN backbone."
    My arithmetic from Table 2 (subtracting the LA row deltas from the ours row deltas) gives ≈7.1/3.8/1.9 on F1@25.
    The percentages may be averaged over IoU thresholds and/or runs; the text does not say. **Reported here as the paper's
    claim, not as a verified delta.**
- **Does it show rare-class *segment* mislabeling is the dominant error?** It shows tail-class **segment** F1 is
  catastrophically low (Assembly101 3.9; Breakfast 38.7 vs head 53.3) and that tail classes are **under-learned** rather
  than over-fit — supporting the diagnosis, but **it does not attribute the error specifically to whole-segment identity
  confusion**; its framing is imbalance + transition frequency. It also reports **global** frame/segment behavior is
  preserved (it does not claim frame accuracy is sacrificed).
- **Params / compute**: **not stated** (no param/FLOP table; grep found none). Hyperparameters: `ε = 0.9` in the
  constraint formulation, multiplier learning rate `γ = 0.01`, threshold `τ` tuned per dataset (smaller for Assembly101's
  larger imbalance). Cost: a confusion-tensor bookkeeping pass + learned multipliers + one NCM prototype per class and a
  per-segment majority vote at inference — all negligible.
- **Small-data & low-dim-feature relevance**: **HIGH — and S-NCM is the most directly testable identity fix I found.**
  It needs only (a) frame representations (our 144-dim vector *is* one, or MS-TCN++ hidden 128), (b) segment boundaries
  (we can get them from our existing predictions), and (c) per-class means/majority vote — **no new backbone, no extra
  training loop, no extra data**. It attacks the exact failure mode: per-frame argmax errors that are *individually*
  near-tie get **outvoted** inside the segment by prototype distance in feature space, and the segment gets exactly one
  label. It also directly fits our metric (segment sequence identity). Risks specific to us: with only ~9 classes + idle
  and tens of thousands of frames, prototype means for a class with **21 segments** are estimated from very little data,
  and S-NCM inherits errors in the *boundary* estimate (wrong boundaries → wrong vote). The cost-sensitive reweighting is
  also directly applicable to our severe imbalance, but its "learning state" estimate needs enough instances per class
  to measure accuracy, which a rare class may not have.
- **Verification notes**: **FETCHED** — abs page + full HTML text (Sec. 3.1–3.3 mechanism, Tables 1–8, transition-accuracy
  analysis, hyperparameters). Code + license via GitHub API. The 8.3/3.5/3.0 claim is marked unreconciled.
  Backbone features and param counts **UNVERIFIED — grepped, not stated.**

---

## 4. Coherent Temporal Synthesis for Incremental Action Segmentation

- **Venue/year**: **CVPR 2024** (verified: arXiv abs page `Comments: ... accepted to CVPR 2024`). arXiv 2403.06102.
  Authors: Guodong Ding, Hans Golong, Angela Yao.
- **URL**: https://arxiv.org/abs/2403.06102 — full text fetched from https://arxiv.org/html/2403.06102
- **Code**: **none found** — no repository link in the full text I fetched (grep for `github` found only arXiv page chrome).
  License: **UNVERIFIED — did not fetch a license page.** (The abs page has a standard "view license" link which I did not
  follow for this paper.)
- **Core mechanism (TCA — Temporally Coherent Action)**: Replaces frame-exemplar replay with a **generative model of actions**
  for incremental TAS. A **conditional VAE** per task: encoder `E(x, a, c) = q_φ(z|x,a,c)` takes the frame feature `x`,
  one-hot action label `a`, and a **coherence variable `c`**, and the decoder `D(z, a, c) = p_θ(x|z,a,c)` reconstructs `x̂`.
  Train loss is `E_z log p_θ(x|z,a,c) − D_KL(q_φ(z|x,a,c) || p(z))` — i.e. standard CVAE + a temporal-coherence conditioning
  term. **`c` is defined as the relative temporal progression of a frame within its action segment** (for frame `i` of a
  segment of duration `ℓ`), so the decoder learns how action features *evolve* over time rather than averaging them.
  **Replay generation is top-down**: first sample a sequential structure (action sequence + segment durations), then for
  each segment sample **one shared latent `z ~ N(0,I)` for all its frames** and step `c` across the segment, so generated
  features change *gradually*; segments are then concatenated by timestamp into a full replay video.
- **Problem targeted**: **Incremental-learning forgetting** — but the *mechanism* is explicitly **segment-level temporal
  coherence**, and its reported diagnosis is relevant: storing static frame exemplars loses "temporal coherence in the
  natural progression of actions" and causes over-segmentation; the paper states that using **temporally evolving** segment
  features "alleviate[s] over-segmentation" (Sec. 4.3). It also reports its own residual failure mode as **confusion between
  semantically similar activities** ("scrambledegg" vs "pancake", both pan-cooking) — a fine-grained identity confusion
  statement, though it is about *activity*-level rather than segment-level confusion.
- **Datasets & metrics**: Breakfast and YouTube Instructional. Backbones **MS-TCN** and **ASFormer**; features are
  **pre-computed features such as I3D** (verified: Sec. 3 "x^t is typically provided as pre-computed features such as
  I3D [7] rather than raw RGB inputs"). 10-task and 5-task incremental setups. Metrics: Acc, segment-wise Edit,
  F1@{10,25,50}.
  - **Table 1 (10-task Breakfast, one backbone group), Acc / Edit / F1@10 / F1@25 / F1@50**:
    Finetune **15.7 / 16.1 / 16.9 / 15.8 / 13.2**; **Exemplar [3] 32.5 / 28.9 / 30.8 / 28.5 / 22.9**;
    **Ours 54.5 / 49.4 / 51.1 / 46.9 / 37.7**; **Original (upper bound, original features) 60.4 / 59.1 / 60.3 / 56.1 / 46.0**.
  - **Table 1 (10-task Breakfast, MS-TCN group)**: **Ours 29.4 / 25.9 / 26.3 / 23.5 / 17.7**, Original 43.1 / 41.1 / 41.2 / 37.6 / 29.5.
  - **Text (Sec. 4.3, MSTCN 10-task Breakfast)**: "static segments with stored exemplar features ... gaining an accuracy
    increase from **7.4% to 16.1%** with MSTCN. Our approach achieves a more substantial boost, reaching **29.4%**" — this
    matches the 29.4 in Table 1, and the abstract's "increases in accuracy for **up to 22%**" matches the 54.5−32.5 = 22.0
    Acc gain of the other backbone group. Remaining ~13.1% gap vs. original features acknowledged by the authors.
  - Consistent pattern across Table 1: **Edit and F1 gain much more than frame Acc** (e.g. 10-task, exemplar → ours:
    Edit 28.9 → 49.4 vs Acc 32.5 → 54.5; and in the 5-task block Acc can even drop slightly, 30.8 → 30.2, while Edit rises
    19.7 → 25.0). That is the same "segmental metrics improve before frame accuracy" signature seen in papers 1 and 2.
  - **Ablation (Table 3)**: authors report optimal results only when **both** segment diversity and temporal coherence are
    modeled ('SD'/'FD'/'TC' ablations). Replay size sweep (Table 4): M = 30/60/90/120 → 34.0 / 35.4 / ... best at M = 120.
    TCA trained on only **25%** of task data still beats the Exemplar baseline; training on the full task data is best
    (a ~6% drop with only a quarter of the data).
- **Honest caveat on my table reading**: Table 1 packs multiple backbone × task-count blocks, and the flattened HTML did not
  let me map *every* row to its backbone with certainty. I therefore report only the rows I could anchor: the text's
  explicit MS-TCN 10-task numbers (7.4 / 16.1 / 29.4) and the two 10-task blocks I could read in full. Rows I could not
  confidently attribute are **not** quoted.
- **Params / compute**: **not stated** — the paper argues qualitatively that a generative model represents actions "with a
  fixed model size" but reports **no parameter count or FLOPs** (grep for `param`/`million` found no model-size figure).
  Compute is a **CVAE training pass per task plus sample-based replay generation** — an extra generative stage we would have
  to train.
- **Small-data & low-dim-feature relevance**: **MODERATE, mostly conceptual.** Fully compatible in principle — it operates
  on **pre-computed per-frame feature vectors** (I3D in the paper), so it could ingest 144-dim features directly; nothing
  about a CVAE requires high dimensionality. But (i) its problem is **incremental learning / forgetting**, which is not our
  failure mode — we train a single model on a fixed label set; (ii) it requires training **one generative model per task**
  plus replay generation, which is real added machinery and, with only tens of thousands of frames, the CVAE itself would be
  data-starved; (iii) it does nothing to reassign a wrong whole-segment label. **The transferable idea is the diagnosis,
  not the method**: representing a segment as *one identity + a temporal evolution* rather than as a bag of independent
  frames is exactly the structural prior our model lacks. Reading it as a *design principle for a segment-level head*
  (one latent per segment, evolved over its duration) is more useful than porting the CVAE.
- **Verification notes**: **FETCHED** — abs page + full HTML text (mechanism Sec. 3.1–4.2 with equations, Table 1 numbers,
  Table 3/4/5 ablations, Sec. 4.3 analysis). Code **verified negative** by grep of full text. Param count **UNVERIFIED —
  grepped, not stated**. License **UNVERIFIED — not fetched**. Some Table 1 rows could not be attributed to a backbone
  (stated above); those are omitted rather than guessed.

---

## 5. DISCOVERY: additional 2025–2026 work on segment-level identity / coherence / order

Method: the instructed arXiv title-field newest-first sweep **failed (HTTP 429, 6/6 attempts)** — see the global caveat at
the top. Substitutes used: web search on the specific mechanisms requested (set/permutation-invariant segment prediction,
sequence-level or edit-distance-aware losses, segment prototype/clustering classification, Viterbi/constrained decoding,
explicit class-confusion disambiguation), plus a curated GitHub survey list
(`derkbreeze/AwesomeActionSegmentation`, raw README fetched). **Every candidate below was then verified by fetching its own
arXiv abs page and HTML full text.** Four papers survived verification as genuinely targeting segment-level
structure/identity rather than boundary refinement.

### 5.1 VidParse: Online Parsing of Egocentric Procedures Like a Pro  ← **most on-target find**

- **Venue/year**: **ECCV 2026** (verified: abs page `Comments: Accepted at ECCV 2026`). arXiv **2608.27562**.
  Authors: Anubhav Gupta, Archit Kambhamettu, Vatsal Agarwal, Pulkit Kumar, Abhinav Shrivastava (Univ. of Maryland).
- **URL**: https://arxiv.org/abs/2608.27562 — full text fetched from https://arxiv.org/html/2608.27562v1
- **Code**: https://github.com/learn2phoenix/VidParse — **verified via GitHub API** (exists, 6 stars; repo contains
  `LICENSE, configs, docs, run.py, scripts, tests, vidparse`). **License: MIT** (verified by fetching
  `raw.githubusercontent.com/.../LICENSE` → "MIT License, Copyright (c) 2026 Anubhav Gupta, ... University of Maryland").
- **Core mechanism** — three stages, **online and training-free** (no task-specific training or fine-tuning; relies on frozen
  pretrained components):
  1. **Boundary detection without a learned filter**: compute a **Temporal Similarity Matrix** over
     **Manipulation-Anchored Features (MAFs)** = frozen **DINOv2** descriptors masked by a **Hand-Object Detector**, in a
     sliding window; convolve with a **Gaussian-tapered checkerboard kernel**; emit boundaries at peaks. The paper's framing
     of *why* this works is the key insight: "continuous procedural actions naturally manifest as distinct, highly
     correlated **block-diagonal structures** within the matrix" — i.e. **a segment is detected as a region of self-similarity**.
  2. **Segment identity by non-parametric micro-prototype matching** (Sec. 3.4) — *this is the mechanism that directly
     attacks whole-segment identity*. A naive global per-class prototype is rejected because "procedural actions are often
     lengthy and multi-phasic; averaging an entire execution sequence blurs distinct sub-states". Instead: agglomerative
     clustering of training examples into **execution styles** → centroids; each clustered action sequence is **temporally
     sliced into overlapping short windows** (4 s / 2 s stride on EgoPER; 1.5 s / 0.5 s on GTEA) to build a **dense library
     of micro-prototypes `P_a` per action**. At inference, a detected segment's frames are HOD-masked, its descriptor
     averaged, and the label assigned by **nearest neighbor (minimum cosine distance) in micro-prototype space**. The
     matching cost enters a decoder energy weighted by segment duration.
  3. **Procedural structure as a hard constraint**: a **task graph** is induced from training sequences (immediate
     transitions, first-time visits vs revisits, prerequisite relationships); a **graph-constrained beam search** assigns
     **infinite cost to transitions that violate the graph**, so "pruning impossible trajectories" is a hard constraint, not
     a soft regularizer. A **gap-rectification** rule retroactively relabels a short background gap to the preceding action
     if followed by a high-confidence return; a **fixed-lag commitment** freezes predictions older than a few seconds.
- **Problem targeted**: **Explicitly whole-segment identity + sequence order.** The abstract names the failure as
  "severe over-segmentation and **structural collapse**", and the method's two distinct answers to it are
  **micro-prototype identity assignment** and **hard transition-graph constraints**. Its own evaluation of long-range
  structure is *n-step transition accuracy*, which is a sequence-level rather than boundary-level metric — the authors say
  so directly: standard segmentation metrics "primarily measure local boundary accuracy and **do not capture whether
  predicted action sequences preserve long-range procedural dependencies**" (Sec. 4.1).
- **Datasets & metrics**: GTEA (28 videos, 7 activities, 15 fps) and **EgoPER** (213 normal + 173 erroneous egocentric videos,
  5 recipes; evaluated on the 213 normal, same splits as [38], 10 fps). Metrics: frame Acc (without background), Edit,
  F1@{0.1,0.25,0.5}, plus the n-step transition metric.
  - **Table 1 (GTEA)**: MSTCN offline 79.35 / Edit 84.46 / F1@0.1 86.54 / F1@0.25 83.79 / F1@0.5 71.86;
    MSTCN **online 47.28 / 60.26 / 66.75 / 60.12 / 40.29**; ProTAS [38] online 73.19 / 71.81 / 72.89 / 68.94 / 54.87;
    **Ours (online, train-free) 89.1 / Edit 87.4 / 91.9 / 89.9 / 80.5**. (Consistent with the text's "+15% F1 across all
    thresholds", F1@0.5 gain = 80.5−54.87 = **+25.6**.)
  - **Table 1 (EgoPer)**: ProTAS 76.61 / 65.50 / 64.26 / 62.59 / 51.31; **Ours Acc 80.7, Edit 88.7** (text: "improvements of
    roughly 25% on F1 and Edit metrics while maintaining a 4% gain in frame-level accuracy").
  - **Long-range structure (Fig. 4)**: AUC for **5-step and 7-step transitions is up to 5x–10x higher** than the ProTAS
    baseline, with the gap widening for longer transitions — the paper's headline "preserves long-range state transitions".
  - **Recipe-specific task graph (Supplement, GTEA)**: ProTAS adapted to use the same MAF features **plateaus at overall
    F1 21.30**, while VidParse with the same features and strict graph-constrained inference reaches **overall F1 80.47** —
    the cleanest available isolation of *decoding structure* rather than *features* as the cause of the gain.
  - **Ablation (Table 4)**: centroid-based micro-prototypes give **Acc 80.69, F1@0.5 77.61**; varying prototypes per action
    from 3 to 11 changes Acc/Edit by **< 1.2%**; gap threshold `d` sensitivity reported.
  - **Efficiency (Table 3)**: **8.6 FPS** vs ProTAS **4.5 FPS**; parsing+decoding **32.8 s on an AMD EPYC 7443 CPU** vs
    ProTAS **706 s on an NVIDIA RTX A4000** (a ~21x wall-clock advantage on CPU).
  - **Stated limitations (Sec. 5, read directly)**: hard graph constraints mean the model "cannot recover if a user performs
    a completely novel — yet practically valid — action sequence. In such out-of-distribution executions, the decoder may
    prune the correct sequence, forcing an incorrect alignment to the known graph." Also HOD-dependent under occlusion.
- **Params / compute**: no task-specific trained parameters (frozen DINOv2 + HOD). Prototype library size and beam width are
  the only knobs (k = 3 clusters GTEA / 9 EgoPER; beam width B = 5 GTEA / 10 EgoPER). **An explicit model-parameter count is
  not stated** — the method is training-free.
- **Small-data & low-dim-feature relevance**: **SPLIT VERDICT — read carefully.** The *feature* half is **not** compatible:
  MAFs come from **frozen DINOv2 + a hand-object detector**, so this is not a drop-in for 144-dim YOLO-ROI vectors, and it
  assumes object/hand-centric egocentric imagery. The *structural* half is **exactly our problem and is feature-agnostic**:
  (i) **micro-prototype nearest-neighbor segment labeling** needs only per-frame feature vectors — our 144-dim vectors or
  MS-TCN++ hidden states qualify; (ii) **task-graph-constrained beam search with infinite cost on illegal transitions** is a
  drop-in decoder over our 9 classes + idle; (iii) **gap rectification** is a 5-line post-process that matches our "78% of
  frame errors are deep inside segments" symptom. Compute is trivial for us (8 videos, 2,639 frames). **Risks**: our task
  graph would be induced from only ~8 videos (and EgoPER-style micro-prototypes from 21 instances of a rare class are thin);
  the paper's own OOD-pruning limitation is a real risk if our procedures are loosely ordered. **This is the strongest single
  lead in the survey for breaking a whole-segment-identity ceiling.** Also note: **training-free** means it can be evaluated
  against our existing MS-TCN++ predictions **without retraining**.
- **Verification notes**: **FETCHED** — abs page, full HTML text (main + supplement), GitHub API + LICENSE file.
  All numbers from Table 1, Table 3, Table 4, Fig. 4 and the supplement's recipe-specific table.

### 5.2 Neural Finite-State Machines for Surgical Phase Recognition (NFSM)

- **Venue/year**: **UNVERIFIED venue** — arXiv 2411.18018, submitted **Nov 2024**, abs page has **no `Comments:` field**
  (so no venue given; cs.CV / eess.IV). **Date caveat: outside the strict 2025–2026 window** — included because it came up
  under the instructed `phase recognition` query and is structurally on-topic. Authors: Hao Ding, Zhongpai Gao,
  Benjamin Planche, Tianyu Luan, Abhishek Sharma, Meng Zheng, Ange Lou, Terrence Chen, Mathias Unberath, Ziyan Wu.
- **URL**: https://arxiv.org/abs/2411.18018 — full text fetched from https://arxiv.org/html/2411.18018v1
- **Code**: **none found** — no repository link in the full text I fetched. License: **arXiv.org perpetual non-exclusive
  license** (verified in HTML page footer).
- **Core mechanism**: A **plug-and-play module** that "enforces temporal coherence by integrating classical state-transition
  priors with modern neural networks". It uses **learnable global state embeddings as unique phase identifiers** plus
  **dynamic transition tables** that model phase-to-phase progression; a **future phase forecasting** mechanism pads with
  repeated current-frame embeddings to make `m` pseudo-future embeddings so the decoder anticipates upcoming transitions.
  Transition state probabilities come from a dot-product attention-like similarity between a dynamic state embedding and the
  global state embeddings. It integrates into an existing pipeline (demonstrated on **Surgformer**) by taking the model's
  final-layer features — "without changing core architectures". The motivating failure is verbatim: a baseline "exhibit[s]
  **fragmented predictions** within the surgical phase of packaging (P5), **misclassifying temporary movements as phase
  transitions**".
- **Problem targeted**: **Sequence/segment coherence via explicit state structure** — fragmented predictions, spurious
  transitions, long-horizon consistency. Not boundary refinement, not capacity. Its reported metrics include **phase-level**
  (i.e. segment-level) precision/recall/F1/mAP, so it is evaluated at segment granularity.
- **Datasets & metrics**: Cholec80 and AutoLaparo (surgical, 25 fps → 1 fps); plus a **non-surgical generalizability study on
  Breakfast** — specifically **Breakfast-Cereals**, "45 videos averaging 349.6 frames ... train/validate on 34 videos and test
  on 11", downsampled by 2. Metrics: Accuracy, Precision, Recall, Jaccard (phase-level). Backbone: **MViTv2-S/16** (Kinetics-400
  pretrained) as frame encoder with a RetNet long-term decoder, **768-dim embeddings, raw 224×224 images** (not precomputed
  features).
  - **Abstract's own numbers (BernBypass70)**: video-level accuracy **+0.9**, phase-level **precision +3.8, recall +3.1,
    F1 +3.3, mAP +4.1** — quoted as the paper's claim; I did not locate the underlying table.
  - **Appendix C.2 (Breakfast-Cereals, 11 test videos)**: baseline → **+inference only**: Acc 76.2 (**+0.9**),
    Precision 67.4 (**−1.9**), Recall 66.3 (**+4.9**), Jaccard 52.9 (**+3.9**); **+training+inference**: Acc 77.3 (**+2.0**),
    Precision 68.7 (−0.6), Recall 66.6 (**+5.2**), Jaccard 53.6 (**+4.6**). (Same table continues per Breakfast sub-activity:
    Coffee, Friedegg, Juice, Milk, Pancake, Salat, Sandwich, Scrambledegg.)
  - Consistent pattern again: **recall/Jaccard gain more than precision**, and precision can dip — the classic signature of a
    structural prior that recovers rare states at some cost in head-class precision.
- **Params / compute**: **not stated** as a parameter count. Compute context: **two NVIDIA A40 GPUs**; MViTv2-S/16 encoder on
  16-frame windows, RetNet decoder with history `n = 256`, future window `m = 128` (surgical) and `n = 48`, `m = 16`
  (Breakfast); 30 epochs (5-epoch warm-up) surgical, 100 epochs Breakfast. This is **heavy** relative to our setting.
- **Small-data & low-dim-feature relevance**: **LOW as-is, MEDIUM as a concept.** As-is it consumes **raw video frames through
  MViTv2** (768-dim), not precomputed per-frame features, so it is not compatible with a 144-dim input without replacing the
  encoder — and the demonstrated gains are surgical-domain. However the **NFSM module itself** operates on the model's final
  feature layer, and its idea — **a learned transition table + per-state embeddings that make "which segment am I in" an
  explicit, discrete state with legal successors** — is dimension-agnostic and portable onto an MS-TCN++ hidden state. Note
  the encouraging datum that the authors bothered to validate on **Breakfast-Cereals with only 11 test videos and ~350 frames
  per video**, which is closer to our data scale than the usual 1,700-video Breakfast protocol. Cost of adaptation is moderate
  (needs training with a transition loss), not free.
- **Verification notes**: **FETCHED** — abs page + full HTML text (mechanism Sec. 3–4, Breakfast-Cereals setup Sec. 4.1,
  Appendix C.2 numbers, training details). Venue **UNVERIFIED — abs page has no `Comments:` field**. Code existence and param
  count **UNVERIFIED/negative by grep**. The BernBypass70 figures are quoted from the abstract, not from a table I located.

### 5.3 Joint Self-Supervised Video Alignment and Action Segmentation (VASOT)

- **Venue/year**: **ICCV 2025** (verified via the curated list entry and confirmed by the abs page; arXiv 2503.16832,
  authors Ali Shah Ali, Syed Ahmed Mahmood, Mubin Saeed, Andrey Konin, M. Zeeshan Zia, Quoc-Huy Tran).
  *(Venue attribution on the abs page itself was not re-read as a `Comments:` string — the ICCV 2025 attribution comes from
  the fetched curated list and the arXiv ID/date are verified.)*
- **URL**: https://arxiv.org/abs/2503.16832 — full text fetched from https://arxiv.org/html/2503.16832
- **Code**: https://github.com/trquhuytin/VASOT-ICCV25 — **verified via GitHub API** (exists, 2 stars).
  **License: none found** — GitHub API `license: None`.
- **Core mechanism**: A **unified optimal-transport framework** for joint video alignment + action segmentation. The
  alignment module (**VAOT**) uses a **fused Gromov-Wasserstein (FGW) optimal transport** formulation **with a structural
  prior**, solving a frame-to-frame soft assignment `T* ∈ R^{N×M}` between two videos' frame embeddings; pseudo-labels from
  `T*` supervise the frame encoder via cross-entropy between normalized similarities and `T*`. The action-segmentation half
  (ASOT) uses **`K` learnable action centroids `A ∈ R^{D×K}`** (initialized by K-Means) and assigns frames to them; the
  joint model (**VASOT**) shares one encoder. Its prediction-to-ground-truth comparison uses **Hungarian matching between
  predicted and GT action clusters**.
- **Problem targeted**: **Sequence-level alignment / cross-video temporal correspondence** — a genuinely sequence-level
  objective (a coupling between *whole videos*), not boundary refinement. Relevant to us as the clearest 2025 instance of a
  **differentiable alignment loss over full sequences**, and as a **learnable-prototype frame-to-centroid assignment** scheme
  (the same family as S-NCM and VidParse micro-prototypes, but trained end-to-end).
- **Datasets & metrics**: video alignment on Pouring, Penn Action, IKEA ASM (Acc@{0.1,0.5,1.0}, Progress, τ, AP@{5,10,15});
  action segmentation on **Breakfast, YouTube Instructions, 50Salads (Mid/Eval), Desktop Assembly**, with
  **MoF, F1, mIoU** plus Hungarian matching. Encoders: **ResNet-50** for alignment; an **MLP encoder** for the action
  segmentation setting. **Features: not I3D** — the MLP encoder operates on frame-level features of the self-supervised
  benchmark suite; I grepped the full text for `I3D` → no hits.
  - Text claims VASOT "consistently achieves the best results across all metrics and datasets" against self-supervised action
    segmentation competitors (Table 3). I did **not** extract individual Table 3 values (the flattened HTML table was not
    reliably row-alignable), so **no metric:value is quoted here** rather than risk misattributing a number.
  - Stated efficiency (Supplement): VASOT needs **(108 MB, 116 min)** vs VAOT+ASOT **(216 MB, 162 min)** on Pouring with a
    ResNet-50 encoder; **(287 KB, 15 min)** vs **(571 KB, 21 min)** on Desktop Assembly with the MLP encoder.
- **Params / compute**: model sizes are given as **memory footprints (108 MB / 287 KB)** rather than parameter counts;
  GPU: NVIDIA 3090Ti. **No parameter count stated.**
- **Small-data & low-dim-feature relevance**: **LOW–MODERATE.** Positives: it is a **feature-level method** (MLP encoder over
  frame features), so 144-dim input is not an obstacle; the **learnable action centroids + Hungarian matching** idea speaks
  directly to segment identity. Negatives: (i) it is **self-supervised/unsupervised** and requires **`K` set to the ground-truth
  number of clusters** — an assumption we do not need and cannot exploit; (ii) Hungarian cluster matching means label
  *identity* is established only up to a permutation, which is unhelpful for a supervised 9-class + idle problem; (iii) FGW
  aligns *pairs* of videos, needing enough videos to pair — with 8 videos this is very thin; (iv) the segmentation metrics
  include mIoU/MoF rather than our Edit/F1@k. **Verdict: cite as evidence that differentiable sequence-level alignment is
  an active 2025 direction, but it is not a practical lever for our setting.**
- **Verification notes**: **FETCHED** — abs page + full HTML text (mechanism Sec. 3, metrics/eval Sec. 4–5, supplement
  efficiency numbers). Code + license via GitHub API. **No metric:value quoted** (see above). Feature backbone
  **verified not I3D** by grep.

### 5.4 Hierarchical Action Learning for Weakly-Supervised Action Segmentation (HAL)

- **Venue/year**: **CVPR 2026** (verified via the curated list entry; arXiv 2602.24275). Authors: Junxian Huang, Ruichu Cai,
  Hao Zhu, Juntao Fang, Boyan Xu, Weilin Chen, Zijian Li, Shenghua Gao.
- **URL**: https://arxiv.org/abs/2602.24275 — full text fetched from https://arxiv.org/html/2602.24275
- **Code**: https://github.com/DMIRLAB-Group/HAL — **verified via GitHub API** (exists, 2 stars).
  **License: none found** — GitHub API `license: None`.
- **Core mechanism**: Introduces a **hierarchical causal data generation process** in which a **high-level latent action
  variable governs the dynamics of low-level visual features**, and — the key observation — the two evolve at **different
  rates**: low-level visual variables change rapidly, high-level action variables evolve **slowly**, "making them easier to
  identify". Deterministic temporal-alignment processes tie the two layers together, a **hierarchical pyramid transformer**
  extracts both, and a **sparse transition constraint enforces the slower dynamics of the high-level action variables**.
  Under mild assumptions the paper **proves the latent action variables are strictly identifiable**.
- **Problem targeted**: **Over-segmentation / fragmented predictions caused by reliance on low-level visual cues**
  ("visual variations are easily mistaken for action transitions, resulting in false boundaries"). It also promises
  **identifiability of the action variable**, which is a theoretical statement about *identity* being recoverable — closer to
  our problem than most boundary work.
- **Datasets & metrics**: Breakfast, CrossTask, Hollywood, GTEA. Metrics in the tables are the weakly-supervised TAS set —
  **MoF, MoF-Bg, IoU, IoD** (not Edit / F1@k). Backbone: **ATBA** chosen as the backbone with Transformer encoder/decoder
  and linear classifiers (Supplement, architecture details Table 6); visual encoder is a **visual-transformer backbone over
  the video features** — **feature backbone not stated in a form I could verify as I3D** (no `I3D` hit).
  - Table 1 (Breakfast) baseline rows I read verbatim: **HMM+RNN 33.3** MoF; **TCFPN+ISBA 38.4 / 36.4±1.0**, MoF-Bg 38.4,
    IoU 24.2, IoD 40.6; **NN-Viterbi 43.0 / 39.7±2.4**; **D3TW 45.7**; **CDFL 50.2 / 48.1±2.5**, MoF-Bg 48.0, IoU 33.7,
    IoD 45.4; **DP-DTW 50.8**, IoU 35.6, IoD 45.1; **TASL 47.8**, IoU 35.2. I did **not** reliably row-align HAL's own row,
    so I do **not** quote a HAL metric:value.
  - Text claims HAL "significantly outperforms existing methods" and shows **denser, more compact T-SNE clusters** for
    high-level action variables than ATBA (Fig. 5), with high-level variables "more stable and align[ing] better with
    ground-truth segmentation".
- **Params / compute**: **not stated** (no parameter table found; grep for `param`/`million` gave none).
- **Small-data & low-dim-feature relevance**: **LOW–MODERATE.** It is **weakly supervised** (we are fully supervised, so we
  cannot use its reduced-annotation premise) and it is a **whole probabilistic generative framework** (ELBO, encoders/decoders
  for visual and action variables, pyramid transformer) — a large, data-hungry addition poorly suited to tens of thousands of
  frames. Ingesting 144-dim features is feasible in principle (it starts from an extracted feature `b_{1:T}`), but the model
  around them is heavy. Its **transferable idea** is the useful part: **two variables at different time-scales, with an
  explicit constraint forcing the action-level variable to be slow/piecewise-constant** — a principled way to make
  *segment-level* state decidable and identifiable, which is conceptually adjacent to our ceiling.
- **Verification notes**: **FETCHED** — abs page + full HTML text (mechanism Sec. 3–4, datasets/metrics, Table 1 baseline
  rows, Fig. 5 discussion, backbone details in supplement). Code + license via GitHub API. **No HAL metric:value quoted**
  (table rows not reliably alignable in flattened HTML). Venue attribution from the curated list (arXiv ID/date verified).

### 5.5 Also checked and *not* included (verified but off-target or blocked)

- **arXiv 2603.06201, "Point-Supervised Skeleton-Based Human Action Segmentation"** (URL fetched, abstract read): uses a
  **prototype-similarity method with constrained K-Medoids** — but for **generating pseudo-labels** under point supervision,
  on **skeleton** data. Its prototype machinery is for label *recovery*, not for fixing identity in a supervised model, and
  the modality (skeleton joint/bone/motion via a pretrained unified model) does not transfer to YOLO-ROI detections.
  **Verified via abs page; excluded as off-target.**
- **arXiv 2609.14624, "TTDF: A Two-Stage Framework for Reliable Surgical Phase Transition Detection"** (abs page fetched):
  a causal two-stage framework over a **frozen** online phase recognizer, applying a **minimum-duration requirement and a
  workflow-graph constraint** (a candidate is kept only if the target phase persists for a minimum duration and the ordered
  phase pair is in the workflow graph's allowed transition set), then verifying candidates with phase-posterior shifts and
  DINOv2 visual-change cues. **Mechanism is closely related to paper 1 and VidParse** (duration floor + legal-transition graph
  as a hard filter, training-free), but the task is **event-level transition detection**, the domain is surgical, and the
  metric is event-level. **Verified via abs page; noted here as corroboration that "duration floor + transition graph as a
  hard constraint" is a convergent 2026 idea, but not detailed further as it is off our segment-identity target.**
- **arXiv 2411.18018 NFSM** and **arXiv 2503.16832 VASOT** are detailed above; NFSM's date (Nov 2024) is outside the
  2025–2026 window and is flagged as such.
- **What I looked for and did NOT find (honest negatives)**:
  - **Set / permutation-invariant segment prediction** (DETR-style set loss over predicted segments, no NMS/order
    assumption) applied to TAS in 2025–2026: **NOT FOUND.** I searched web for exactly this phrasing plus `permutation
    invariant`, `set prediction`, `segments` — no 2025–2026 TAS paper surfaced. **UNVERIFIED — targeted web searches only;
    the instructed arXiv title sweep was unavailable (HTTP 429), so this negative is not exhaustive.**
  - **Sequence-level / edit-distance-aware training loss (soft-DTW, differentiable edit distance) in 2025–2026 TAS:**
    **NOT FOUND.** Soft-DTW surfaced only in the 2021/2022 literature (e.g. a CVPR 2021 discriminative-prototypes-with-DTW
    paper and an AAAI soft-DTW paper via search snippets) and in an action-*alignment* context (D³TW). In the 2025–2026 TAS
    papers I verified, all segment-level supervision is either **constraint/decoding-based** (papers 1, 5.1, 5.2) or
    **metric-derived at evaluation time only**. **UNVERIFIED as a complete negative — same search limitation.**
  - **"Insertion recall"** as a metric: did not appear in any paper I fetched (all use F1@k / Edit / MoF / IoU / Jaccard).
    **UNVERIFIED — no paper fetched uses that metric.**

---

## Bottom line for the project

Ranked by *plausibility of breaking the whole-segment-identity ceiling*, based only on the verified material above:

1. **VidParse (ECCV 2026) micro-prototype segment relabeling + graph-constrained beam search** — the only verified
   mechanism that decides **identity once per segment in feature space against a prototype library**, *and* forbids illegal
   transitions with infinite cost. Training-free, so it can be layered on our existing MS-TCN++ predictions without retraining.
   Caveat: its *features* (frozen DINOv2 MAF) are not our 144-dim vectors — take the two structural components, not the features.
2. **CSL's S-NCM (BMCV 2024)** — segment-level nearest-class-mean majority vote; the same "one label per segment from
   prototypes" idea, with a supervised cost-sensitive loss on top. Cheapest of all to try on our 128-dim hidden states.
3. **G-TLA (ECCV 2024) group-wise classification + ordering priors** — attacks identity by **shrinking the label space per
   activity** and forbidding impossible transitions, using an activity label we already have per video. Its own evidence
   (zero accuracy on 5/48 classes; tail recall 17.2 vs head 53.1) is the closest literature match to our failure, and its
   gains are concentrated exactly on the rare classes that currently get whole-segment-wrong labels.
4. **Constrained Viterbi (ICPR 2026)** — hard duration/start/end/transition constraints at inference; zero params, zero
   retraining, and its own ablation (39.2 → 48.9 Edit from constraints vs plain Viterbi) shows the constraint set is what
   matters. Cheapest sanity check on whether our 51.6% whole-segment-wrong rate is partly *structurally illegal* segments.

**What the evidence says about the ceiling itself**: no verified 2025–2026 paper reports a *sequence-level* training loss
(soft-DTW, differentiable edit distance, set/permutation-invariant segment prediction) for TAS. Every strong result I could
verify works by **constraining or re-deciding the label sequence at inference/training time from structural priors
(transitions, durations, prototypes, state machines)** — not by making the frame-level loss smarter. That asymmetry is
itself the main finding, and it argues that our ceiling is a **decoding/structure** problem rather than a
capacity/feature/objective-smoothness problem.

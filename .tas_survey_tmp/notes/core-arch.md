# Core TAS architectures — primary-source extraction

Survey scope: 5 papers (ASFormer, UVAST, C2F-TCN, MS-TCN, MS-TCN++) for an industrial TAS project.
Context for the "relevance" bullets: our input is a **144-dim hand-crafted per-frame vector** (YOLO detections
over a 2x3 ROI grid), **NOT** I3D-2048; tiny dataset (~2,639 test frames, 8 videos, ~9 classes + idle);
business metric is **segment-level** (Edit distance of the predicted segment-label sequence, F1@k at
IoU 0.1/0.25/0.5, insertion recall, frame accuracy). Current best: MS-TCN++ (4 stages x 10 layers,
hidden 128, ~3.3M params) -> Edit 51.08, F1@0.1 40.01, frame acc 53.98.

All numbers below were read from the papers' own text/tables in the fetched full text (arXiv HTML) or from the
authors' own code/repos. Anything I could not verify is marked `UNVERIFIED`.

---

### ASFormer: Transformer for Action Segmentation

- **Venue/year**: BMVC 2021. Verified from the arXiv abstract page `Comments:` field, which reads verbatim
  "Accepted by BMVC 2021", and independently from the official repo description
  ("Official repo for BMVC2021 paper ASFormer: Transformer for action segmentation").
- **URL**: https://arxiv.org/abs/2110.08568
- **Code**: https://github.com/ChinaYi/ASFormer — **MIT** (GitHub API `license.spdx_id = MIT`; `LICENSE` file present in repo root)
- **Core mechanism**: Encoder-decoder Transformer built for long sequences. The **encoder** projects the input
  with a fully connected layer (`d_model = 64`), then stacks blocks each of which = **dilated temporal convolution
  as the feed-forward layer (replacing the vanilla MLP)** + **single-head self-attention constrained to a local
  window of size `w = 2^i`** ("hierarchical representation pattern"), with the temporal-conv dilation rate doubled
  in step, residual connections, instance norm + ReLU, and **no positional encoding**. **Three decoders** perform
  incremental refinement; each decoder block uses **cross-attention where Q,K come from the concatenation of the
  encoder output and the previous layer's output, while V comes only from the previous layer** (so the refinement
  feature space is not disturbed by the encoder). Final config: 1 encoder + 3 decoders, 9 blocks each, dim 64.
- **Problem targeted**: Explicitly the three concerns of applying vanilla Transformers to TAS: (i) "lack of inductive
  biases with small training sets", (ii) inability to form an effective representation over thousands of frames,
  (iii) decoder architecture unable to exploit temporal relations among multiple action segments for refinement.
- **Datasets & metrics**: 50Salads / GTEA / Breakfast. **I3D features; the paper states explicitly
  "The dimension of the I3D feature for each frame is 2048-d."** Metrics = frame-wise accuracy (Acc.),
  segmental edit score (Edit), segmental overlap F1 at IoU thresholds k/100, denoted F1@{10,25,50}.
  - ASFormer row, paper's own Table 7 (50Salads | GTEA | Breakfast):
    - 50Salads: F1@10/25/50 = 85.1 / 83.4 / 76.0, Edit = 79.6, Acc. = 85.6
    - GTEA: 90.1 / 88.8 / 79.2, Edit = 84.6, Acc. = 79.7
    - Breakfast: 76.0 / 70.6 / 57.4, Edit = 75.0, Acc. = 73.5
  - Baselines in the same table for reference: MS-TCN (50Salads) 76.3/74.0/64.5, Edit 67.9, Acc 80.7;
    MS-TCN (GTEA) 85.8/83.4/69.8, Edit 79.0, Acc 76.3; MS-TCN (Breakfast) 52.6/48.1/37.9, Edit 61.7, Acc 66.3.
- **Params**: **1.134M** for ASFormer, **0.799M** for MS-TCN, with 6.80G vs 4.79G FLOPs and ~3.5G vs ~1.7G GPU
  memory — paper's own **Table 9**, measured on 50Salads. Cross-check/discrepancy: UVAST's text (Sec. 4.4) says
  ASFormer "has ~1.3 M learnable parameters", which disagrees with ASFormer's own 1.134M; I report both rather
  than pick one.
- **Small-data & low-dim-feature relevance**:
  - **Can ingest 144-dim**: yes, architecturally trivial. Paper: "The first layer of the encoder is a fully connected
    layer that adjusts the dimension of the input feature", so `D` is free. Official code exposes it as
    `features_dim` (default `2048` in `main.py`) and `Encoder(..., input_dim, ...)`. Change one constant.
  - **Small-data claims: the strongest of the five papers, and directly on point for us.** The whole motivation is
    lack of inductive bias on small training sets. Sec. 5.2 ablation on 50Salads (encoder only) — paper's own Table 3:
    | feed-forward layer | F1@10/25/50 | Edit | Acc |
    |---|---|---|---|
    | MLP + positional encoding (vanilla Transformer) | 27.6 / 25.3 / 19.9 | 20.0 | 74.2 |
    | dilated temporal conv (ours) | **53.1 / 51.4 / 47.0** | **43.3** | **85.7** |
    The paper's reading: "when MLP is used as the feed-forward layer, the performance drops greatly, especially on
    F1 scores and Edit score, which denotes that the model fails to model the temporal relationship among frames to
    produce smoother and consistent predictions." This is exactly the failure mode we observed (plain Transformer
    encoder lost to MS-TCN++), and ASFormer's diagnosis is that locality must be injected.
  - Sec. 5.3 (Table 4): non-hierarchical attention windows 64.2/61.5/55.1, Edit 59.5, Acc 76.8 vs hierarchical
    85.1/83.4/76.0, Edit 79.6, Acc 85.6 — the "freely" learned attention "cannot automatically learn a hierarchical
    pattern from data". Also a small-data/optimisation-inductive-bias argument.
  - Sec. 5.4 (Table 5): decoders "largely boost the performance compared to the encoder", best with 3 decoders.
  - Caveat: `d_model = 64` was tuned for a 2048-dim I3D input; with a 144-dim input the input projection is
    already a large relative compression (144 -> 64), so the "adjust the dimension" step is much less of a burden.
- **Verification notes**:
  - Fetched: `https://arxiv.org/abs/2110.08568` (comments/venue), `https://arxiv.org/html/2110.08568` (full text,
    abstract, Sec. 3.1/3.3, Sec. 5.2/5.3/5.4/5.6, Table 3/4/5/7/9), GitHub API for the repo license, repo
    `model.py` and `main.py` (config: `num_layers = 10`, `num_f_maps = 64`, `features_dim = 2048`,
    `num_epochs = 120`, `lr = 0.0005`, `channel_mask_rate = 0.3`, `0.5` for GTEA; `MyTransformer(3, ...)` = 3 decoders).
  - **Discrepancy found and reported honestly**: the paper says "each encoder and the decoder contains nine blocks"
    (`J = 9` chosen in the Sec. 5.5 ablation), but the released `main.py` default is `num_layers = 10`. I did not
    determine which one produced the published Table 7 numbers.
  - NOT verified: parameter count under a 144-dim / 10-class configuration (not reported by the paper; would need
    to instantiate the model).
  - The paper reports no `insertion recall`, no fragmentation metric, and no low-dimensional/hand-crafted-feature
    experiment.

---

### Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation

- **Venue/year**: **ECCV 2022 (Main Conference)** — the discrepancy is resolved in favour of 2022. Four independent
  primary checks, all from the authors' own material, agree:
  1. arXiv abstract page `Comments:` field, verbatim: "ECCV 2022 (Main Conference)".
  2. Official repo README line 3: "Official PyTorch implementation of the ECCV 2022 paper".
  3. Official repo README BibTeX: `@inproceedings{uvast2022ECCV, ..., booktitle={ECCV}, year={2022}}`.
  4. GitHub API repo `description` field: "... (ECCV 2022)".
  The arXiv v2 is dated 11 Oct 2022 and v1 is from Sep 2022; there is no 2023 date on any primary source I
  fetched. **I could not reproduce any primary source supporting "2023"** — the "2023" attribution appears to be
  a secondary-source error (UVAST is also unconnected to any 2023 venue I could verify). Treat ECCV 2022 as
  authoritative.
- **URL**: https://arxiv.org/abs/2209.00638
- **Code**: https://github.com/boschresearch/UVAST — **AGPL-3.0** (GitHub API `license.spdx_id = AGPL-3.0`;
  README: "This project is open-sourced under the AGPL-3.0 license"; `LICENSE` present). **Note: GitHub API reports
  `archived: true`.** **AGPL-3.0 is a strong network copyleft licence — this is a real industrial-adoption hazard
  and should be flagged to whoever owns licensing, unlike the MIT repos above.**
- **Core mechanism**: **Seq2seq translation.** Encoder = a modified ASFormer encoder (supplement Sec. 2 / **Table 7**, "Impact of Encoder Model": "we replace
  the RELU activation layers with GELU activation and add one more layer of dilated convolution at the end of each
  encoder block"), supervised by an auxiliary **frame-wise** loss, producing frame-level predictions. Decoder = a
  **standard autoregressive Transformer decoder (2 layers, single head)** that emits the **sequence of action
  segments (the "transcript")** rather than per-frame labels. Auxiliary losses: segment-level CE, **group-wise**
  frame and segment CE (averaged prediction per ground-truth class), and a **cross-attention loss** aligning
  encoder frames to decoder segments. A separate **non-autoregressive alignment decoder** (single layer, single
  head) computes a soft frame-to-segment assignment matrix `M_bar` with temperature `tau`; durations are the row
  sums `u_i = sum_t M_bar_{t,i}`. Optional Viterbi / FIFA decoding uses the predicted transcript.
- **Problem targeted**: The frame-classification formulation "suffers several drawbacks, such as **over-segmentation
  when trained on relatively small datasets**", and gives no direct handle on the segment-level transcript. UVAST
  targets the transcript (i.e. the Edit metric) directly, and unifies fully supervised + timestamp supervised
  training via a constrained k-medoids pseudo-labelling algorithm.
- **Datasets & metrics**: 50Salads / GTEA / Breakfast. **I3D features.** Sec. 4.2 states the metrics are exactly
  "frame-wise accuracy (Acc), segmental edit score (Edit), and the segmental F1 score at overlapping thresholds
  10%, 25%, and 50%, denoted by F1@{10,25,50}", and gives the rationale we care about: "**Edit measures the quality
  of the predicted transcript of the segmentation**, while F1 scores penalize over-segmentation and are also
  insensitive to the duration of the action classes."
  - UVAST rows, paper's own Table 1 (Breakfast | 50Salads | GTEA):
    - **w/o duration** (transcript only — only Edit is reported): Edit = **76.9** (Breakfast), **83.9** (50Salads),
      **92.1** (GTEA). This is the pure transcript-prediction number and it beats every baseline's Edit.
    - **+ alignment decoder**: Breakfast F1 76.7/70.0/56.6, Edit 77.2, Acc 68.2; 50Salads F1 86.2/81.2/70.4,
      Edit 83.9, Acc 79.5; GTEA F1 77.1/69.7/54.2, Edit 90.5, Acc 62.2.
    - **+ Viterbi**: Breakfast F1 75.9/70.0/57.2, Edit 76.5, Acc 66.0; 50Salads F1 89.1/87.6/81.7, Edit 83.9,
      Acc 87.4; GTEA F1 92.7/91.3/81.0, Edit 92.1, Acc 80.2.
    - **+ FIFA**: Breakfast F1 76.9/71.5/58.0, Edit 77.1, Acc 69.7; 50Salads F1 88.9/87.0/78.5, Edit 83.9,
      Acc 84.5; GTEA F1 82.9/79.4/64.7, Edit 90.5, Acc 69.8.
  - Baselines reproduced in UVAST Table 1 (useful cross-check because they match the original papers):
    MS-TCN++ Breakfast 64.1/58.6/45.9, Edit 65.6, Acc 67.6; 50Salads 80.7/78.5/70.1, Edit 74.3, Acc 83.7;
    GTEA 88.8/85.7/76.0, Edit 83.5, Acc 80.1. ASFormer Breakfast 76.0/70.6/57.4, Edit 75.0, Acc 73.5;
    50Salads 85.1/83.4/76.0, Edit 79.6, Acc 85.6; GTEA 90.1/88.8/79.2, Edit 84.6, Acc 79.7.
- **Params**: paper's own **Table 10**, per dataset, stage 1 (encoder-decoder): **1.109M** (Breakfast),
  **1.103M** (50Salads), **1.102M** (GTEA); the stage-2 alignment decoder is **0.166M**. Sec. 4.4 text agrees
  ("our proposed model has ~1.1 M parameters"). Table 10 also lists encoder `d = 2048` input feature dim,
  `d' = 64`, 10 encoder layers, 2 decoder layers, `tau'` (stability temperature) 0.001, dropout used, GELU,
  training <= 800 epochs, Adam, lr 5e-4, batch 1, split-segment value 0.17 (Breakfast) / 0.15 (50Salads) /
  0.17 (GTEA).
- **Small-data & low-dim-feature relevance**:
  - **Can ingest 144-dim**: yes. Table 10's `d` is the input feature dimension and the encoder is a Transformer
    whose first operation is a projection; the paper's abstraction is `T x D`. The released code would need the
    feature-dim/hyperparameter table changed. Note `d = 2048` vs `d' = 64` means the model projects *down* 32x —
    with a 144-dim input that ratio collapses, so the bookkeeping (and possibly the optimal `d'`) changes.
  - **Small-data claims, and they are honest about the failure mode**: "it is important to note that **Transformers
    are very data-hungry and training them on small datasets can be challenging**" (Sec. 4.4), and "**the small size
    of the GTEA dataset hinders the training of the alignment decoder**" (GTEA Acc 62.2 with the alignment decoder
    vs 80.2 with Viterbi — a visible small-data breakdown). Also Sec. 4.3: duration prediction explicitly "does not
    work out of the box" partly because of "the relatively small number of training videos".
  - The **split-segment** strategy is a concrete small-data/class-imbalance device worth stealing: long GT segments
    are split into shorter ones during training so that segment durations are more uniformly distributed (value =
    max segment length as a fraction of total video frames, e.g. 0.1); **inference uses the full video unsplit**
    (supplement Sec. 2 / **Table 8**). Directly applicable to our 51.6% whole-segment-wrong-label failure.
  - The **segment-level and group-wise losses** (`L_segment`, `L_g-segment`) are the only explicitly
    segment-identity-oriented loss terms among the five papers, and they are what makes Edit high while Acc drops
    ("Lower Acc and higher Edit/F1 scores indicate that UVAST localizes action boundaries ... less accurately").
- **Verification notes**:
  - Fetched: `https://arxiv.org/abs/2209.00638` (comments/venue), `https://arxiv.org/html/2209.00638` (full text +
    supplement: Sec. 4.2/4.3/4.4, Tables 1/6/10/11), repo README `main` branch, GitHub API repo metadata, and the
    HTML table of Table 1 parsed cell-by-cell (the naive tag-stripping scrambles multi-row tables, so I parsed the
    `<table>` elements directly to avoid misreading a number).
  - **Discrepancy recorded**: the arXiv page has **no `Journal reference:` field** for UVAST (only the DataCite
    arXiv DOI), so the venue rests on `Comments:` + the authors' README/BibTeX/API metadata — all four agree.
  - NOT verified: whether the released code's encoder can accept a non-2048 `d` without further edits (I did not
    run it); the exact per-dataset `# Parameters` effect of changing the number of classes.
  - The paper reports no `insertion recall`.

---

### C2F-TCN: A Framework for Semi and Fully Supervised Temporal Action Segmentation

- **Venue/year**: **TPAMI 2023** (IEEE Transactions on Pattern Analysis and Machine Intelligence). The arXiv
  `Comments:` field does **not** state a venue (it only carries the admin note "text overlap with
  arXiv:2112.01402"). Verification comes from (a) the paper's own Sec. I: "Part of this work was first published
  in [26] ... **In this journal extension**, we comprehensively show and evaluate the design of the C2F-TCN
  architecture ...", and (b) the authors' official repo README, verbatim: "Code for full supervsion version of
  'C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation' [link](https://ieeexplore.ieee.org/abstract/document/10147035) **published in TPAMI-2023**."
  (sic, typo in README). Note the earlier conference paper (arXiv 2112.01402, "Coarse to Fine Multi-Resolution
  Temporal Convolutional Network") is the CVPR 2022 version that this arXiv entry extends; the repo README links
  the conference paper to arXiv 2105.10859 — an inconsistency in the authors' own README that I did not resolve.
- **URL**: https://arxiv.org/abs/2212.11078
- **Code**: https://github.com/dipika-singhania/C2F-TCN — **MIT** (GitHub API `license.spdx_id = MIT`;
  `LICENSE` present). The paper's own text contains **no** GitHub URL (I grepped for `github.com/...` in the full
  text and found none), so the repo was located via the GitHub API search on the author's account. Related repo:
  https://github.com/dipika-singhania/ICC-Semi-Supervised-TAS (also MIT).
- **Core mechanism**: A **U-Net-style 1D encoder-decoder TCN** ("Coarse-to-Fine TCN") whose novelty over prior
  ED-TCNs is a **"coarse-to-fine" ensemble of the decoder outputs**: each successive decoder layer restores a finer
  temporal resolution and its per-frame class probabilities are ensembled, which the paper reports "significantly
  improves segmentation performance by **reducing over-segmentation**, i.e. highly fragmented segmentation
  outputs" and mitigates over-confidence. Encoder/decoder blocks are `double_conv` = Conv1D(k=5)+BatchNorm+ReLU
  twice; resolution is halved by MaxPool1D(2) per encoder stage and restored by Upsample1D(2) with skip-concat.
  Bottleneck fuses multiple max-pool windows. A second contribution is a **model-agnostic temporal feature
  augmentation** (FA): stochastic max-pooling of segments of the input features, with window `w` drawn from a
  distribution parameterised by a base window `w0` (w0 sampled with prob 0.5, else uniform in [w0/2, 2*w0]).
- **Problem targeted**: (i) over-segmentation / fragmentation, and (ii) the fact that MS-TCN's multi-stage
  probability-refinement is "not well-suited to representation learning" — hence the return to ED-TCN. Also
  introduces the first **semi-supervised** TAS setting ("Iterative-Contrast-Classify", ICC).
- **Datasets & metrics**: 50Salads / GTEA / Breakfast **with Kinetics-pretrained I3D features that are NOT
  fine-tuned**; plus Assembly101 with TSM features fine-tuned on Epic-Kitchens. Metrics: "Mean-over-frames (MoF),
  segment-wise edit distance (Edit) and F1-scores with IoU thresholds of 0.10, 0.25 and 0.50
  (F1@{10,25,50})". **MoF here = the same frame-accuracy computation as MS-TCN's "Acc"** (confirmed in the code,
  see the shared metric section).
  - C2F-TCN rows, paper's own Table VI (Breakfast | 50Salads | GTEA):
    - C2F-TCN alone: Breakfast F1 64.9/60.6/49.7, Edit 63.2, MoF 70.2; 50Salads 75.6/72.7/61.2, Edit 69.1,
      MoF 79.6; GTEA 89.9/88.3/75.9, Edit 86.8, MoF 79.6.
    - **C2F-TCN + FA**: Breakfast 71.9/69.0/58.5, Edit 68.9, MoF 76.6; 50Salads 84.3/81.7/72.8, Edit 76.3,
      MoF 84.5; GTEA 92.3/90.1/80.3, Edit 88.5, MoF 81.2.
    - **MS-TCN + FA** (same FA applied to MS-TCN, showing FA transfers): Breakfast 70.8/67.6/56.8, Edit 67.7,
      MoF 71.3; 50Salads 82.8/80.4/72.2, Edit 76.1, MoF 84.2; GTEA 88.8/85.7/74.1, Edit 82.8, MoF 79.0.
  - Assembly101 (**Table VII**): MS-TCN F1 17.1/14.1/8.7, Edit 21.0, MF 21.2 vs C2F-TCN 20.2/16.6/10.8, Edit 22.3, MF 22.5.
  - Semi-supervised ICC (abstract + Sec. VI-F): "with 5% labeled videos; with 40% labeled videos, we almost match
    full supervision (see Fig. 9, Tab. XIV)".
- **Params**: **"Our model has a total of ≈ 6 million trainable parameters"** (Appendix S1, after Table T16) for the
  base 6-layer kernel-5 model. Also (Fig. 7 discussion) "Reducing the convolution kernel size from 25 to 5 within
  our 6 layered C2F-TCN model leads to a reduction in the parameters from ≈ 20M to ≈ 6M with similar results."
  No per-dataset parameter table is given.
- **Small-data & low-dim-feature relevance**:
  - **Can ingest 144-dim**: yes, but the architecture is the **least feature-agnostic of the five** — Appendix
    Table T16 hardcodes the first block as `double_conv(2048, 256)` with input `T_in x 2048`. That single `in_c`
    is the only 2048 dependency; after that stage all channels are 256/128. So a 144-dim input means changing
    `in_c=2048` -> `144` in stage Phi_0, and the parameter count drops slightly. Note this also removes most of
    the "narrowing" the network does from I3D; with 144 dims the first double_conv is nearly shape-preserving in
    channel count, so the representation-learning premise ("huge gains over the input I3D", Table IX) does not
    transfer.
  - **Small-data claims**: yes, but they are about *label* efficiency rather than inductive bias. The ICC
    semi-supervised scheme is explicitly motivated by "a fully supervised setting requires frame-wise labels for
    every single video", and the reported result is that **40% of labeled videos nearly matches full supervision,
    and 5% labeled videos still gives "noteworthy segmentation performance"**. That is the most directly usable
    small-data result for us if our bottleneck is annotation, not raw frames. The FA strategy is reported as
    improving accuracy *and* calibration and reducing fragmentation "across various TCNs" — model-agnostic, so it
    is portable to MS-TCN++ regardless of feature dim.
  - Caveat for us: the unsupervised/contrastive half of the framework "leverag[es] the clustering capabilities of
    the input I3D features" — it assumes a rich, semantically clustered feature space. A 144-dim
    presence/count/area vector is a much weaker clustering substrate, so the representation-learning half of
    C2F-TCN should be treated as non-transferable; the supervised C2F ensemble + FA are the transferable parts.
- **Verification notes**:
  - Fetched: `https://arxiv.org/abs/2212.11078` (title/authors/comments), `https://arxiv.org/html/2212.11078`
    (full text: Sec. I, Sec. III-B/III-C architecture, Sec. VI setup, Appendix S1 with Table T16 and the ≈6M param
    statement, Tables I/III/V/VI/VII/IX/T16/T21), repo README (`main` branch), GitHub API repo metadata + root file listing,
    and the repo's `eval.py` + `utils.py` (metric implementations).
  - **Discrepancy recorded honestly**: the paper's own Table VI labels its first baseline "MS-TCN [6]" with
    Breakfast F1 64.1/58.6/45.9, Edit 65.6, MoF 67.6 — but those exact numbers appear in MS-TCN++'s own paper as
    **MS-TCN++**, and MS-TCN's own paper reports 52.6/48.1/37.9 / Edit 61.7 / Acc 66.3 for Breakfast (I3D). The
    baseline label in C2F-TCN Table VI appears to be mislabelled; I did not resolve which model produced those
    numbers.
  - NOT verified: the exact per-stage parameter breakdown; whether the FA base window `w0` needs retuning for a
    very different frame rate; the Assembly101/TSM feature dimension (not stated in the text I read).
  - The paper reports no `insertion recall`.

---

### MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation

- **Venue/year**: **CVPR 2019**. Verified from the arXiv abstract page `Comments:` field, verbatim
  "CVPR 2019 Camera Ready", and from the official repo's own citation block ("In IEEE Conference on Computer Vision
  and Pattern Recognition (CVPR), 2019").
- **URL**: https://arxiv.org/abs/1903.01945
- **Code**: https://github.com/yabufarha/ms-tcn — the root contains a `LICENSE` whose first line is
  "**MIT+CC License**" and whose body is the MIT permission text. **GitHub's API reports
  `license.spdx_id = NOASSERTION`** because of that non-standard header; a licence file *is* present. Treat as
  MIT-with-an-additional-CC-notice and have it read by counsel before industrial use.
- **Core mechanism**: A **multi-stage TCN**. The single-stage TCN (SS-TCN) is a `1x1` convolution that projects the
  input feature dim to 64 channels, then **ten dilated residual 1D convolution layers** with kernel size 3 and a
  dilation factor **doubled at each layer** (`1,2,4,...,512`), ReLU, dropout after every layer, and residual
  connections; a final `1x1` conv + softmax emits per-frame class probabilities. The multi-stage model stacks four
  such stages, **where stage `s` receives only the frame-wise probabilities of stage `s-1` and no features**
  ("the input to the next stage is just the frame-wise probabilities without any additional features"). Training
  loss per stage = cross-entropy + a **truncated MSE smoothing loss** on frame-wise log-probabilities with
  `lambda = 0.15` and `tau = 4`, summed over stages.
- **Problem targeted**: Over-segmentation errors produced by per-frame models, and the fact that per-frame accuracy
  does not reflect them ("long action classes have a higher impact than short action classes on this metric and
  over-segmentation errors have a very low impact"). The multi-stage refinement plus the T-MSE smoothing loss are
  the two mechanisms aimed at it.
- **Datasets & metrics**: 50Salads (50 videos, 17 classes) / GTEA (28 videos, 11 classes incl. background) /
  Breakfast (1,712 videos, 48 actions). **"For all datasets, we extract I3D features for the video frames and use
  these features as input to our model."** 15 fps (50Salads downsampled from 30 fps). The paper **never states the
  I3D dimension** — I grepped the full text for `2048` and there are zero matches. Metrics: frame-wise accuracy
  (Acc), segmental edit distance, segmental F1 at IoU thresholds 10%/25%/50%.
  - MS-TCN's own headline rows, paper's own **Table 10** (SOTA comparison):
    - 50Salads: F1 76.3 / 74.0 / 64.5, Edit 67.9, Acc 80.7
    - GTEA: F1 85.8 / 83.4 / 69.8, Edit 79.0, Acc 76.3 (with I3D fine-tuning, **Table 9**: 87.5/85.4/74.6, Edit 81.4, Acc 79.2)
    - Breakfast: **MS-TCN (I3D)** F1 52.6 / 48.1 / 37.9, Edit 61.7, Acc 66.3; **MS-TCN (IDT)** F1 58.2 / 52.9 / 40.8,
      Edit 61.4, Acc 65.1
- **Params**: **not stated in the MS-TCN paper** (`grep -i param` finds only qualitative discussion; there is no
  parameter table). Two later, primary companion sources give it for 50Salads: MS-TCN++'s **Table XIII** = **0.80M**,
  and ASFormer's **Table 9** = **0.799M** — mutually consistent. The paper's config is 4 stages x 10 layers x
  64 filters, kernel 3.
- **Over-segmentation analysis (explicitly requested)**:
  - **The analysis is not a dedicated fragmentation metric.** MS-TCN does not report a count of predicted segments,
    an over-segmentation ratio, or a fragmentation rate anywhere — I grepped for "number of segments",
    "fragmentation", "# segments", "predicted segments" and found zero. Over-segmentation is instead inferred
    from Edit / F1@k plus **qualitative figures**, and this is stated as the reason for preferring those metrics:
    "the segmental F1 score [is used] as a measure of the quality of the prediction".
  - The quantitative evidence is the **stage-count ablation (Table 1, 50Salads)** and the **loss ablation (Table 3)**:
    - Table 1, single-stage vs multi-stage: SS-TCN F1 27.0/25.3/21.5, **Edit 20.5**, Acc 78.2 vs MS-TCN 4-stage
      F1 76.3/74.0/64.5, **Edit 67.9**, Acc 80.7. Note Acc barely moves (78.2 -> 80.7) while Edit moves 47 points:
      the paper's exact point is that "all of these models achieve a comparable frame-wise accuracy. Nevertheless,
      the quality of the predictions is very different ... the single-stage model produces a lot of
      over-segmentation errors, as indicated by the low F1 score."
    - 2 stages 55.5/52.9/47.3, Edit 47.9, Acc 79.8; 3 stages 71.5/68.6/61.1, Edit 64.0, Acc 78.6;
      5 stages 76.4/73.4/63.6, Edit 69.2, Acc 79.5.
    - Table 3 (loss ablation, 50Salads): CE only F1 71.3/69.7/60.7, Edit 64.2, Acc 79.9; CE + KL smoothing
      71.9/69.3/60.1, Edit 64.6, Acc 80.2; **CE + T-MSE 76.3/74.0/64.5, Edit 67.9, Acc 80.7**. Text: "the proposed
      loss achieves better F1 and edit scores with an absolute improvement of 5%".
    - **Table 5** (Effect of passing features to higher stages, 50Salads) — "Probabilities and features"
      56.2/53.7/45.8, Edit 47.6, Acc 76.8 vs "Probabilities only" 76.3/74.0/64.5, Edit 67.9, Acc 80.7.
      **Relevant warning for us**: re-injecting raw features into refinement stages *hurts a lot* in their setting.
  - Two findings with direct bearing on our numbers:
    - **Lower temporal resolution improves the segmental metrics** — **Table 6** (Impact of temporal resolution,
      50Salads): MS-TCN at 1 fps F1 77.8/74.9/64.0, **Edit 70.7**, Acc 78.6 vs 15 fps F1 76.3/74.0/64.5,
      **Edit 67.9**, Acc 80.7. The paper's explanation: "Operating on a low temporal resolution makes MS-TCN
      **less prone to the over-segmentation problem**, which is reflected in the better edit and F1 scores."
    - **Fewer layers per stage is clearly worse** — **Table 7** (Effect of the number of layers L, 50Salads):
      L=6 -> Edit 46.2, L=8 -> 60.1, L=10 -> 67.9, L=12 -> 69.6. So our 10-layer choice is at the low end of what
      they tested, and 12 would likely be better *if* we had the data for it.
    - **Duration buckets** — **Table 8** (GTEA): short videos give much better segmental scores (Edit 82.5 for
      < 1 min vs 71.8 for >= 1.5 min).
- **Small-data & low-dim-feature relevance**:
  - **Can ingest 144-dim**: yes, and it is the cheapest of the five to adapt. The code's
    `SingleStageModel(num_layers, num_f_maps, dim, num_classes)` starts with
    `self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)` — `dim` is the only place the input width appears, and the
    paper says the `1x1` conv "adjusts the dimension of the input features to match the number of feature maps in
    the network". With 64 filters it maps 2048 -> 64 and 144 -> 64 equally well; the parameter count is essentially
    unchanged (the 1x1 conv contributes 64*144 vs 64*2048, i.e. ~122K fewer params — negligible against ~0.8M).
    **This is why MS-TCN/MS-TCN++ is a sane baseline at 144 dims, and it is consistent with our finding that it
    currently wins.**
  - **Small-data claims**: indirect but real. (a) Dropout is credited with "preventing the model from over-fitting
    the training data". (b) They show that **more is not better on small data**: going from 4 to 5 stages degrades
    performance, and the paper attributes it to "an over-fitting problem as a result of increasing the number of
    parameters". (c) The duration-bucketed **Table 8** (GTEA) shows the model is much stronger on short videos
    (<1 min: Edit 82.5) than long ones (>=1.5 min: Edit 71.8). (d) The IDT-vs-I3D comparison on Breakfast
    (**Table 10**, IDT **beats** I3D on F1: 58.2 vs 52.6) is direct evidence from a primary source that **a richer 2048-dim feature
    does not automatically beat a weaker feature** — the temporal model dominates.
- **Verification notes**:
  - Fetched: `https://arxiv.org/abs/1903.01945` (comments/venue), `https://arxiv.org/html/1903.01945` (full text:
    Sec. 3.1/3.3/3.4, Sec. 4 metrics, all result tables incl. 1/3/12/13/16/18/19/20/21/22), GitHub API repo
    metadata + root file listing, repo `LICENSE`, `main.py`, `model.py`, `eval.py`.
  - NOT verified: the I3D feature dimension as used by *this* paper (**the paper does not state it**; `grep 2048`
    = 0 matches). Statements that MS-TCN "uses 2048-dim features" are true of the *released code and the ecosystem*,
    not of anything written in the MS-TCN paper.
  - NOT verified: whether `ends.append(i+1)` (see metric section) was the version used for the published numbers.
  - The paper reports no `insertion recall`.

---

### MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation

- **Venue/year**: **IEEE TPAMI** (journal version; arXiv admin note: "substantial text overlap with
  arXiv:1903.01945"). The arXiv abstract page `Comments:` field names the journal verbatim ("IEEE Transactions on
  Pattern Analysis and Machine Intelligence") but gives **no year**, and there is **no `Journal reference:` field**.
  The year 2020 comes from the authors' own repo, whose title line reads
  "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (**TPAMI 2020**)" and whose README
  cites "S. Li, Y. Abu Farha, Y. Liu, MM. Cheng, and J. Gall. MS-TCN++ ..." (arXiv v1 was submitted 16 Jun 2020).
  If a precise volume/issue/page citation is needed, that is **UNVERIFIED — the arXiv page carries no journal-ref
  and I did not fetch the IEEE Xplore record.**
- **URL**: https://arxiv.org/abs/2006.09220
- **Code**: https://github.com/sj-li/MS-TCN2 — **MIT** (GitHub API `license.spdx_id = MIT`; `LICENSE` present).
- **Core mechanism**: Extends MS-TCN by (i) **decoupling the prediction-generation stage from the refinement
  stages** and (ii) introducing a **dual dilated layer (DDL)**: instead of one dilated convolution, each DDL layer
  runs **two parallel convolutions with dilation factors `2^l` and `2^(L-l)`** (kernel 3), concatenates their
  outputs, and projects back with a `1x2D->D` convolution — so every layer sees both a small and a large receptive
  field. Final architecture: **4 stages = 1 prediction-generation stage (11 layers, with DDL) + 3 refinement stages
  (10 dilated residual layers each)**; refinement stages still consume only the previous stage's probabilities.
  Refinement-stage parameters **can be shared**, giving a much smaller model. Same CE + T-MSE loss (`tau=4`,
  `lambda=0.15`), dropout 0.5 after each layer, Adam lr 5e-4.
- **Problem targeted**: Same as MS-TCN (over-segmentation, long-range dependency modelling) plus a specific
  limitation of MS-TCN: "lower layers still suffer from a small receptive field" because the dilation factor only
  grows with depth. Also model compactness via parameter sharing.
- **Datasets & metrics**: 50Salads / GTEA / Breakfast, **I3D features, "we use the I3D features without
  fine-tuning"**, 15 fps. Metrics identical to MS-TCN: Acc, segmental edit distance, F1@{10,25,50} with IoU
  thresholds. The paper **never states the I3D dimension** (zero `2048` matches in the full text; the only
  appearance of 2048 elsewhere in these papers is in third-party tables).
  - MS-TCN++ rows (paper's own tables):
    - 50Salads (Table XIII, and Table XII for the stage ablation): **MS-TCN++** F1 80.7 / 78.5 / 70.1,
      **Edit 74.3**, Acc 83.7, **0.99M params**; **MS-TCN++(sh)** (shared refinement params) F1 78.7 / 76.6 / 68.3,
      Edit 70.7, Acc 82.2, **0.66M params**; **MS-TCN** F1 76.3 / 74.0 / 64.5, Edit 67.9, Acc 80.7, **0.80M**.
    - GTEA (Table XV, without I3D fine-tuning): MS-TCN++ F1 87.0 / 85.2 / 73.5, Edit 82.0, Acc 78.7;
      MS-TCN++(sh) F1 87.8 / 86.2 / 74.4, Edit 82.6, Acc 78.9; MS-TCN 85.8 / 83.4 / 69.8, Edit 79.0, Acc 76.3.
    - Breakfast: MS-TCN++ (I3D) F1 64.1 / 58.6 / 45.9, Edit 65.6, Acc 67.6; MS-TCN++(I3D)(sh) 63.3 / 57.7 / 44.5,
      Edit 64.9, Acc 67.3.
- **Params (explicitly requested)**: Paper's own **Table XIII** gives `# param.(m)` on 50Salads:
  **MS-TCN = 0.80M, MS-TCN++ = 0.99M, MS-TCN++(sh) = 0.66M**, and the text says sharing "reduces the total number
  of parameters to roughly 66% of the total parameters in the original model". These are for **64 filters** and
  **50Salads' 17 action classes** — the output-layer size depends on the class count, so our 9+idle setting will
  differ slightly. Our 3.3M at **hidden 128** is consistent with scaling: 0.80M at 64 filters is dominated by
  quadratic-in-channels 3x3xDxD conv weights, so doubling D roughly quadruples the bulk (~3.2M), which matches the
  ~3.3M we measure.
- **Config details (explicitly requested)**: 4 stages; prediction-generation stage = **11 layers with DDL**;
  refinement stages = **10 dilated residual layers** each (dilation doubled per layer, `1,2,4,...,512`);
  **64 filters in all layers**; kernel size **3**; dropout **0.5**; `tau = 4`, `lambda = 0.15`; Adam lr `0.0005`.
  **Stage/layer ablation (Tables VIII/IX/X/XII)**: refinement depth L_r = 6/8/10/11/12 -> Edit 69.6 / 73.4 / 74.3 /
  72.6 / 71.3 (best at 10); prediction-generation depth L_g = 6/8/10/11/12 -> Edit 67.8 / 70.3 / 72.5 / 74.3 / 70.8
  (best at 11); MS-TCN's own best L is 12 (Edit 69.6, **Table VIII**). **Number of refinement stages N_r** (**Table XII**):
  0 -> Edit 40.4, 1 -> 62.0, 2 -> 69.4, **3 -> 74.3**, 4 -> 73.1. So nearly all the Edit gain comes from the
  refinement stages, and the 4th refinement stage starts to hurt.
- **Over-segmentation analysis (explicitly requested)**: as in MS-TCN, there is **no dedicated quantitative
  fragmentation metric** (zero matches for "fragmentation"/"number of segments"/"predicted segments"). The evidence
  is (a) the stage-count ablation above (N_r=0 Edit 40.4 -> N_r=3 Edit 74.3 with Acc nearly flat), (b) the loss
  ablations, and (c) **qualitative figures** — Fig. 5 (number of stages), Fig. 6 (loss functions), Fig. 8 (passing
  features to higher stages), Fig. 9 (impact of DDL), Fig. 10 (all datasets). The paper's own framing: "the
  predictions suffer from over-segmentation errors" and the multi-stage architecture "helps in reducing the
  over-segmentation errors"; the T-MSE loss "reduces the over-segmentation errors more" than KL.
  Two extra primary observations:
  - Passing features (not just probabilities) to higher stages produces "a huge drop of the F1 score" — again the
    warning against re-injecting raw features into refinement stages.
  - Table XIV: at **1 fps** MS-TCN++ Edit 73.3, Acc 81.1 vs at 15 fps Edit 74.3, Acc 83.7; MS-TCN at 1 fps Edit 70.7
    vs 15 fps 67.9. Consistent with MS-TCN: **downsampling helps MS-TCN's Edit but hurts MS-TCN++'s**.
- **Small-data & low-dim-feature relevance**:
  - **Can ingest 144-dim**: yes, same `1x1` projection mechanism as MS-TCN; the released `main.py` exposes
    `--features_dim` with default `2048`, and `--num_f_maps` default `64`, `--num_layers_PG`, `--num_layers_R`,
    `--num_R`. So the input width and the whole stage/layer/depth configuration are command-line parameters —
    the cheapest path to a 144-dim adaptation of any of the five papers.
  - **Small-data claims — the most directly usable of the five**:
    - **Parameter sharing is justified by small-data overfitting, in the paper's own words**: "Note that without
      fine-tuning, sharing parameters achieves better results on GTEA. This is mainly due to the **reduced number
      of parameters, which prevents the model from over-fitting the training data, especially for small datasets
      like GTEA**." And Table XV shows MS-TCN++(sh) *beating* MS-TCN++ on GTEA F1 (87.8 vs 87.0) and Edit (82.6 vs
      82.0) — i.e. **on the smallest dataset the smaller model is better on the segmental metrics.** That is a
      concrete, citable prior that a *smaller* MS-TCN++ may beat our current 3.3M model on our tiny dataset.
    - Adding the 5th stage/4th refinement stage degrades performance, attributed to overfitting from more
      parameters; DDL added to the *refinement* stages drops accuracy "due to overfitting".
    - Duration analysis (**Table XI**, GTEA): MS-TCN++ Edit 84.4 on videos <1 min vs 76.1 on videos >=1.5 min — the model is
      markedly better when sequences are short, which is relevant given our 8 short test videos.
    - Fine-tuning I3D gives a small gain ("the effect of fine-tuning for action segmentation is lower than for
      action recognition. This is expected since **the temporal model is by far more important for segmentation
      than for recognition**") — supporting the hypothesis that our weak 144-dim features need not be the blocker.
- **Verification notes**:
  - Fetched: `https://arxiv.org/abs/2006.09220` (comments/venue, no journal-ref), `https://arxiv.org/html/2006.09220`
    (full text: Sec. III-A/III-C/III-F, Sec. IV metrics, Tables I/XII/XIII/XIV/XV/XX/XXI/XXII/XXIII/XXIV/XXV/XXVI/XXVII,
    figure captions), GitHub API repo metadata + root listing, README, `main.py`.
  - NOT verified: TPAMI **year/volume** from an authoritative bibliographic source (arXiv has no journal-ref; the
    2020 year is from the authors' README title only). The 0.80M/"MS-TCN" param count is from MS-TCN++'s own Table
    XIII (50Salads); MS-TCN's paper itself does not report parameters.
  - NOT verified: a per-dataset parameter table for GTEA/Breakfast (only 50Salads, Table XIII).
  - The paper reports no `insertion recall`.

---

## Metric definitions (shared) — how `Edit`, `MoF`/`Acc`, and `F1@k` are actually computed

This is the part that matters most for us, because we report `edit` and are optimizing a
*segment-level* objective. All of ASFormer, MS-TCN, MS-TCN++, C2F-TCN and UVAST report the **same** three
metrics, and the implementations are near-identical clones of one ancestor. I read the actual code in three
independent official repos (`yabufarha/ms-tcn`, `sj-li/MS-TCN2`, `ChinaYi/ASFormer`, `dipika-singhania/C2F-TCN`)
and they agree on the definitions. **The metric definitions are nowhere in the papers' prose** — the papers only
name the metrics and cite each other; the operational definition lives in `eval.py`.

### `Edit` — segment-level edit distance (the normalized Levenshtein score)
From `eval.py` in all four repos:

1. **Frames -> segment label sequence.** `get_labels_start_end_time(frame_wise_labels, bg_class=["background"])`
   collapses runs of identical per-frame labels into a list of labels. Consecutive duplicate labels become **one**
   symbol, so the sequence is the *transcript*.
2. **Background frames are dropped from both sequences** (default `bg_class = ["background"]`). A background run
   splits the transcript, but contributes no symbol and, importantly, **"background" is the only string treated
   this way**.
3. **Levenshtein distance** between the predicted label sequence `P` and the ground-truth label sequence `Y`, with
   **insertion, deletion and substitution all costing 1** (standard DP; matches = free).
4. **Normalization**: `score = (1 - D[-1,-1] / max(m_row, n_col)) * 100`, where `m_row = len(P)` and
   `n_col = len(Y)`. So the reported `Edit` is a **0-100 similarity score (higher is better, 100 = perfect)**,
   normalized by **`max(#predicted segments, #GT segments)`** — *not* by the number of GT segments.
5. **Averaging**: per-video `Edit` values are accumulated and divided by the number of test videos
   (`edit = (1.0*edit)/len(list_of_videos)`) — i.e. **macro-averaged over videos, not pooled over segments or
   frames**. ASFormer additionally averages over the dataset's splits.

**Implications we must be careful about:**
- Because the normalizer is `max(|P|, |Y|)`, **over-segmentation actively penalizes Edit**: a model that
  fragments a segment inflates `|P|` and therefore lowers the score even if the label sequence is otherwise
  right. Under-segmentation is penalized via the deletion term. This is *different* from the plain
  "1 - edit_distance/|Y|" definition sometimes quoted in secondary sources.
- Because background-only labels are excluded by an **exact string match on `"background"`**, the `Edit` value we
  compute depends critically on what our idle class is *called*. If our idle label is not literally the string
  `"background"`, idle segments count as ordinary segments and inflate `|P|` and `|Y|` — changing the normalizer
  and the Levenshtein distance. If our current 51.08 was computed with a different convention than the literature,
  it is **not directly comparable** to the numbers in this survey. Worth an explicit check of our eval script.
- One unit of Edit is one whole segment identity (or one spurious/omitted segment), regardless of segment length,
  which is precisely why it is the right metric for our "51.6% of GT segments get a wholly wrong label" failure:
  each such segment costs 1 substitution in `D`.

### `Acc` (MS-TCN, MS-TCN++, ASFormer, UVAST) vs `MoF` (C2F-TCN)
- **They are the same computation**: `correct frames / total frames * 100`. No background exclusion, no
  segment-level weighting. Confirmed: `eval.py` counts `if gt_content[i] == recog_content[i]: correct += 1` over
  all frames and prints `100*correct/total`; C2F-TCN's `utils.calculate_mof` computes
  `n_correct * 100.0 / n_frames` and prints it as "frame accuracy".
- The papers' shared critique of it (quoted in MS-TCN, MS-TCN++, UVAST, C2F-TCN): long classes dominate and
  "over-segmentation errors have a very low impact". **This is why our 53.98 frame accuracy is a misleadingly
  optimistic headline and should always be reported next to Edit/F1.**
- Caveat: the frame-level comparison is done 1:1, so **the prediction file must be at the same temporal resolution
  and length as the ground truth.** MS-TCN's repo notes that for 50Salads features are sampled at 30 fps and the
  output is up-sampled; ASFormer's `main.py` uses `sample_rate = 2` for 50Salads. Any resampling/off-by-one here
  silently corrupts `Acc*` and shifts every IoU.

### `F1@k` (k = 0.10 / 0.25 / 0.50, IoU thresholds)
Also identical across the four repos (`f_score`), and **pooled, not per-video-averaged**:
- Predicted and GT segments are extracted as in the `Edit` step (background excluded).
- For each predicted segment `j`, IoU is computed against **all** GT segments as
  `intersection = min(p_end[j], y_end) - max(p_start[j], y_start)` and
  `union = max(p_end[j], y_end) - min(p_start[j], y_start)`, then **zeroed wherever the labels differ**
  (`* [p_label[j] == y_label[x] ...]`). The best-scoring GT segment is taken; `tp += 1` iff `IoU >= threshold`
  **and that GT segment has not already been matched** (`hits[idx]`), else `fp += 1`. `fn = len(y_label) - sum(hits)`.
- `tp/fp/fn` are **accumulated across all test videos**, then precision, recall and
  `F1 = 2*P*R/(P+R)` are computed once globally. Papers call this "segmental F1"; note it is a **micro/F1 on
  segments**, so a single long segment and a single short segment count the same.
- The label-match requirement means F1@k *does* penalize whole-segment identity confusion (a wrongly-labelled
  segment can never be a true positive), so our F1@0.1 = 40.01 is already reflecting the 51.6% mislabeling.

### Practical metric-implementation warnings (found while reading the code)
- **Off-by-one discrepancy between the two MS-TCN-lineage repos.** For the *final* segment of a video,
  `yabufarha/ms-tcn` appends `ends.append(i + 1)` while `sj-li/MS-TCN2`, `ChinaYi/ASFormer` and C2F-TCN append
  `ends.append(i)`. Interior segment ends are `i` (exclusive) in all of them. This makes the last segment of each
  video one frame longer in MS-TCN than in MS-TCN++/ASFormer, which shifts `IoU` slightly and can flip a segment
  across a `0.10`/`0.25`/`0.50` threshold. **The exact metric version used by our current baseline must be fixed
  and documented**, or replays will not match the literature.
- All repos use `np.float` (removed in modern NumPy) and the MS-TCN original is Python 2 — the eval scripts need
  patching before they will run at all.
- `Edit` is reported as a 0-100 number; `F1@k` likewise. Our values (Edit 51.08, F1@0.1 40.01, acc 53.98) are on
  the same scales, so they are comparable **provided** the background convention matches.
- **`insertion recall` is reported by none of these five papers** (grep for "insertion" returns 0 matches in all
  five full texts). If we report it, it is our own metric and has no literature comparator from this set.

---

## Cross-cutting answers to the two questions asked of every paper

**1. Does it report results with I3D features (a mismatch for us)?**

| Paper | Primary features | I3D dim stated in the paper? | Any non-I3D experiment? |
|---|---|---|---|
| ASFormer (BMVC 2021) | **I3D** for all three datasets | **Yes, explicitly "2048-d"** (Sec. 4) | No |
| UVAST (ECCV 2022) | **I3D** for all datasets ("the same I3D features that were used in many previous works") | Dim appears only as `d = 2048` in Table 10 (supplement), not as a sentence | No |
| C2F-TCN (TPAMI 2023) | **I3D** (Kinetics-pretrained, **not** fine-tuned) for 50Salads/GTEA/Breakfast | **Yes, as a hardcoded tensor shape** `T_in x 2048` in Appendix Table T16 | **Yes** — Assembly101 uses TSM features fine-tuned on Epic-Kitchens (dim not stated in the text I read) |
| MS-TCN (CVPR 2019) | **I3D** ("For all datasets, we extract I3D features") | **No — never stated** (0 matches for `2048`) | **Yes** — Breakfast with **IDT** features, and IDT beats I3D on F1 there |
| MS-TCN++ (TPAMI) | **I3D**, explicitly **without fine-tuning** | **No — never stated** | No (Table XVI compares I3D vs IDT for MS-TCN only) |

So **yes, all five are I3D-first**, and only two of them even write the number 2048 in the paper. The useful
counter-example for us is MS-TCN's own Breakfast result: IDT (weak, motion-only) features gave a *better* F1
(58.2/52.9/40.8) than I3D (52.6/48.1/37.9) with the same model.

**2. What input feature dimension does it assume?**

| Paper | Assumed input dim (paper) | Input dim in released code | Cost to accept 144-dim |
|---|---|---|---|
| ASFormer | free (`T x D`), 2048 used in experiments | `features_dim = 2048`; model takes `input_dim` | One constant |
| UVAST | `d = 2048` (Table 10) | per-dataset config | Config edit; note 2048->64 projection collapses to 144->64 |
| C2F-TCN | **2048 baked into the architecture table** (`double_conv(2048, 256)`) | first block `in_c = 2048` | Change one `in_c`; but the representation-learning half assumes a rich clustered feature space |
| MS-TCN | not stated ("I3D features") | `nn.Conv1d(dim, num_f_maps, 1)`, `dim` free | One constant |
| MS-TCN++ | not stated ("I3D features") | `--features_dim` default `2048` | One CLI flag |

---

## Bottom line for our project

1. **All five models are architecturally feature-dim-agnostic at the input**, because every one of them starts
   with a pointwise (1x1 / FC) projection. None of them needs 2048 semantically; 2048 is just what I3D provides.
   Feeding 144 dims is a one-line change in each case.
2. **The published parameter counts are all in the 0.66M - 1.2M range** (MS-TCN 0.80M, MS-TCN++ 0.99M / 0.66M
   shared, ASFormer 1.134M, UVAST ~1.10M + 0.166M, C2F-TCN ~6M) with 64 model channels. **Our current best is
   ~3.3M at hidden 128 — 3-5x larger than any of the paper-configured models.** MS-TCN++ explicitly attributes
   the superiority of its *shared*, 0.66M model on the smallest dataset (GTEA) to reduced overfitting, and shows
   both 5-stage MS-TCN and DDL-in-refinement-stages degrading from overfitting. **The strongest single
   evidence-backed recommendation from this survey is to try hidden 64 (and shared refinement parameters) before
   trying anything wider.**
3. **For the whole-segment identity confusion failure, UVAST is the most relevant design in this set**, because it
   is the only one that optimizes the transcript (our `Edit` metric) directly via segment-level and group-wise
   losses, and it reports the highest Edit of any method here (up to 92.1 on GTEA). Its own warning is that
   Transformers are data-hungry and small datasets hurt — so the realistic transfer is the *losses*
   (segment/group-wise CE, cross-attention alignment, split-segment), not the architecture.
4. **For the plain-Transformer-loses result, ASFormer's Sec. 5.2 ablation is the citable explanation**: swapping
   the temporal-conv feed-forward for an MLP + positional encoding costs 25.5 F1@10 and 23.3 Edit points on
   50Salads, with frame accuracy moving only 74.2 -> 85.7. Locality is the missing inductive bias. If we revisit
   attention, it must come with a local-connectivity prior.
5. **Our `Edit` numbers are only comparable to these papers if our eval uses the same background convention**
   (exact-match exclusion of a label literally named `"background"`) and the same `max(|P|, |Y|)` normalizer, and
   the same last-segment end-index convention. Verify our `eval.py` against `sj-li/MS-TCN2`'s before drawing any
   comparative conclusion.

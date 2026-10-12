# Closest-precedent survey: TAS on LOW-DIMENSIONAL / hand-crafted features, and TAS with SMALL datasets

Scope: primary-source verification of the papers assigned to this thread, plus a discovery pass on
object-detection-derived features, explicit-duration (semi-Markov) models, and industrial/assembly TAS datasets.

**Evidence rules applied**: every number below was read out of a document fetched in this session; the fetch
method is recorded per paper. Anything not verified is marked `UNVERIFIED — <what was tried>`. Table numbers are
named so they can be re-checked.

---

## HEADLINE FINDING (read this first)

**The belief that Lea et al.'s TCN used ~64-dim Fisher-vector / IDT features is NOT supported by the primary
sources.** Neither the CVPR 2017 paper (1611.05267) nor the workshop version (1608.08242) uses IDT or Fisher
vectors as the TCN input. The string `fisher` occurs **0 times** in both full texts (verified by grep on the
extracted full text). IDT appears only as a *baseline table row* and in related-work discussion.

**However, a much better and directly relevant precedent exists in the same two papers**: the TCN workshop
version evaluates the TCN on **low-dimensional accelerometer and robot-kinematics sensor streams** and wins.
That is the real precedent for our setting, and it is stronger than the IDT story would have been.

**And the strongest single piece of evidence found in this thread is negative in an important way**: on
robotic/assembly/surgical data, low-dimensional *proprioceptive* features massively beat large pretrained
video features, while on the same data a purely *vision*-derived feature (including object-centric ones) can
collapse to near-useless (M2R2, Table I: F1@50 of 5.1–8.4 for vision-only vs 82.4 for multimodal). This cuts
both ways for us and is dissected in the **Feature-side evidence** section at the end.

**And the single most decision-relevant result is a controlled feature swap that says hand-crafted ≥ deep.**
MS-TCN (CVPR 2019), same architecture, only the input feature changed, on Breakfast: **IDT F1@10/25/50 =
58.2/52.9/40.8 vs I3D 52.6/48.1/37.9** — the hand-crafted motion-only descriptor wins segment-level F1. So
"solve this by moving to I3D" is **not** supported by the literature. What *is* well-supported is that adding
**temporal content** to an instantaneous per-frame representation is what moves Edit and F1@k: Funke et al.
(MICCAI 2019) went from Edit 41.4 to 64.0 at *identical* frame accuracy (79.9) purely by replacing a
spatially-only per-frame representation with a spatio-temporal one. Our 144-dim vector is instantaneous.

---

### Temporal Convolutional Networks for Action Segmentation and Detection

- **Venue/year**: CVPR 2017, pp. 156–165 — verified from the CVF open-access record
  (`openaccess.thecvf.com/content_cvpr_2017/html/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.html`,
  fetched; page reads "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
  2017, pp. 156-165"). — **URL**: https://arxiv.org/abs/1611.05267 — **Code**: none found in the paper (the
  only GitHub link in the full text is https://github.com/fchollet/keras, a library) — license: none found
- **Core mechanism**: Encoder–decoder TCN (ED-TCN): a hierarchy of 1D temporal convolutions with pooling in the
  encoder and upsampling in the decoder, capturing low/intermediate/high-level time scales. A second variant,
  Dilated TCN, replaces pooling with dilated convolutions. Reported as "over a magnitude faster to train than
  competing LSTM-based Recurrent Neural Networks" (abstract).
- **Input features** — *this is the critical question, and the answer is not IDT/Fisher*:
  - **50 Salads**: "We use the spatial CNN features of Lea et al. [15] as input into our models. This is a
    simplified VGG-style model trained solely on 50 Salads." Data downsampled to ~1 frame/second.
    **Dimensionality NOT stated.**
  - **MERL Shopping**: "We use the features from Singh et al. [27] as input. Singh's model consists of four
    VGG-style spatial CNNs: one for RGB, one for optical flow, and ones for cropped versions of RGB and optical
    flow. We stack the four feature-types for each frame and use Principal Components Analysis with **50
    components** to reduce the dimensionality." → **50-dim**, sampled at 2.5 fps.
  - **GTEA**: spatial CNNs trained from scratch; "a simplified VGG-style network where the input for each frame
    is a pair of RGB and motion images"; motion image = concatenation of difference images.
    **Dimensionality NOT stated.**
  - **No Fisher vector and no IDT** are used as TCN input anywhere in this paper.
- **Datasets & size** (Section 4.3, "Datasets"): 50 Salads — "contains 50 sequences of users making a salad",
  5–10 min each, "around 30 instances of actions", 9 higher-level classes + background, 17 mid-level classes,
  5-fold cross-validation. MERL Shopping — "106 surveillance-style videos", 5 actions + background.
  GTEA — "28 videos of 7 kitchen activities", 4 subjects, "about 19 (non-background) actions per video",
  ~1 min videos, 11 classes, leave-one-user-out.
- **Metrics** (Table 2, 50 Salads):
  - higher-level: ED-TCN F1@{10,25,50} = **76.5, 73.8, 64.5**, Edit **72.2**, Acc **73.4**; Bi-LSTM 72.2/68.4/57.8, Edit 67.7, Acc 70.9; Spatial CNN (the *input* feature model) 35.0/30.5/22.7, Edit 25.5, Acc 68.0.
  - mid-level: ED-TCN 68.0, 63.9, 52.6, Edit 59.8, Acc 64.7; IDT+LM baseline 44.4, 38.9, 27.8, Edit 45.8, Acc 48.7.
  - Table 3 (MERL Shopping, causal): ED-TCN F1 82.1, 79.8, 64.0, mAP 64.2, Acc 74.1; Dilated TCN 72.7, 70.6, 56.5, mAP 72.2, Acc 73.0.
  - Table 4 (GTEA): ED-TCN F1 72.2, 69.3, 56.0, Acc 64.0; Spatial CNN 41.8, 36.0, 25.1, Acc 54.1.
- **Params**: **not stated** (grep for `param` returns only "hyper-parameter" and "Parameters of our model were
  learned using…"). LSTM baseline uses 64 latent states per direction.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: The task-level precedent is real
  but the *feature* precedent the project assumed is not. What this paper does establish is that a TCN applied
  to a **50-dimensional PCA-reduced per-frame vector** (MERL Shopping) and to a per-frame CNN embedding of
  unknown but modest width (50 Salads, GTEA) is competitive with or better than LSTM/CRF baselines. Our 144-dim
  z-scored vector is in the same regime as MERL's 50-dim PCA vector. Note however that MERL used **~106 videos /
  2.5 fps** — our ~2,639 test frames over 8 videos is a materially smaller and noisier regime.
- **Verification notes**: fetched — arXiv abs page, arXiv HTML full text, CVF record page. Full text read
  end-to-end for Sections 4.3–4.4 and Tables 2–4.

---

### Temporal Convolutional Networks: A Unified Approach to Action Segmentation

- **Venue/year**: ECCV 2016 Workshop "Brave new ideas for motion representations in videos" — verified from the
  arXiv `Comments:` field, which reads: "Submitted to the ECCV workshop on "Brave new ideas for motion
  representations in videos"". — **URL**: https://arxiv.org/abs/1608.08242 — **Code**: none found — license:
  none found
- **Core mechanism**: The earlier/short version of the TCN. Same encoder–decoder temporal-convolution design,
  but here explicitly positioned as working on **"a sensor signal (e.g. accelerometers) or latent encoding of a
  spatial CNN applied to each frame"**. Layer config: `L=3` layers with `F_l = {32, 64, 96}` filters; filter
  duration `d` = mean segment duration of the shortest class (e.g. `d=10` seconds for 50 Salads).
- **Input features**: two distinct modalities are evaluated:
  - **Video**: "the input, X_t, is the first fully connected layer computed in a spatial CNN trained solely on
    each dataset" — i.e. a learned CNN embedding, **dimensionality not stated**.
  - **Sensor (the low-dim precedent)**: "Our sensor results used the features from [9] which are **the absolute
    values of accelerometer values**." For JIGSAWS, "synchronized robot kinematics (position, velocity, and
    gripper angle) for each robot end effector".
- **Datasets & size** (Section 3, "Evaluation"):
  - 50 Salads — "contains 50 sequences of users making a salad"; "This dataset includes video and synchronized
    accelerometers attached to **ten objects** in the scene, such as the bowl, knife, and plate." 5-fold CV on
    the "eval" granularity, **10 action classes**.
  - JIGSAWS — "Leave One User Out cross validation on the suturing activity, which consists of **39 sequences
    performed by 8 users about 5 times each**… as well as corresponding action labels with **10 action classes**.
    Sequences are a few minutes long and typically contain around 20 action instances."
  - GTEA — "28 videos of 7 kitchen activities", "about 30 actions per video", 11 classes, leave-one-user-out
    (results reported for user 2).
- **Metrics** (Table 1) — **the key rows for us**:
  - **50 Salads, sensor-based: LC-SC-CRF 50.2 Edit / 77.8 Acc; LSTM 54.5 / 73.3; TCN `65.6` Edit / `82.0` Acc.**
  - 50 Salads, video-based: Spatial CNN (the input features) 28.4 / 68.6; ST-CNN 55.5 / 74.2; TCN 61.1 / 74.4;
    `[8] IDT` 16.8 / 54.3; `[8] VGG` 7.6 / 38.3.
  - **JIGSAWS, sensor-based: [9] LC-SC-CRF 76.8 Edit / 83.4 Acc; [2] Bidir LSTM 81.1 / 83.3; [17] SD-SDL 83.3 / 78.6; TCN `85.8` Edit / `79.6` Acc.**
  - JIGSAWS, vision-based: [19] MsM-CRF Acc 71.7; [8] IDT 8.5 / 53.9; [8] VGG 24.3 / 45.9; Seg-ST-CNN 66.6 / 74.7;
    Spatial CNN 37.7 / 74.0; ST-CNN 68.0 / 77.7; TCN 83.1 / 81.4.
  - GTEA, video-based: [3] Hand-crafted Acc 47.7; EgoNet 57.6; TDD 59.5; EgoNet+TDD 68.5; TCN 58.8 Edit / 66.1 Acc.
  - **Important caveat stated by the authors in the Table 1 caption**: "(1) Results using VGG and Improved Dense
    Trajectories (IDT) were intentionally computed **without a temporal component** for ablative analysis, hence
    their low edit scores." → The very low IDT/VGG rows are **not** a fair feature-vs-feature comparison. Do not
    cite them as evidence that IDT features are bad.
- **Params**: **not stated**.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **This is our closest genuine
  precedent.** A ~21-channel accelerometer stream (see the 50 Salads entry below) fed to a TCN beats both an
  LC-SC-CRF and an LSTM on 50 Salads (Edit 65.6 vs 50.2/54.5). On JIGSAWS, low-dim robot kinematics + TCN gives
  Edit 85.8 / Acc 79.6, beating the LSTM/CRF baselines. Two honest caveats: (i) the sensor modality is
  *physically direct* (it measures the manipulated object), whereas our 144-dim vector is a *derived, lossy
  summary* of detections; (ii) 50 Salads is 50 videos at ~1 fps, not 8 videos. The precedent is encouraging for
  "low-dim can work", but it does not demonstrate that *detection-derived* low-dim features can work.
- **Verification notes**: fetched — arXiv abs page and arXiv HTML full text; Section 3 and Table 1 read in full.

---

### TricorNet: A Hybrid Temporal Convolutional and Recurrent Network for Video Action Segmentation

- **Venue/year**: `UNVERIFIED — the arXiv abs page for 1705.07818 has no Comments: and no Journal reference:
  field (checked the raw HTML table cells); the paper's own HTML full text declares no venue; the paper does not
  appear in the CVPR 2017 open-access index (fetched openaccess.thecvf.com/CVPR2017, size 1,645,918 bytes,
  grepping for "TricorNet" returned no matches).` Widely cited as CVPR 2017 but I could not confirm that from a
  primary source in this session. — **URL**: https://arxiv.org/abs/1705.07818 — **Code**: none found (only the
  keras GitHub link) — license: none found
- **Core mechanism**: Encoder = a hierarchy of 1D temporal convolutional kernels that "capture the local motion
  changes of different actions"; decoder = "a hierarchy of recurrent neural networks", specifically Bi-LSTMs,
  that "learn and memorize long-term action dependencies" after the encoding stage. `K=2` layers, each encoder
  layer has `32+32i` filters, `H=64` hidden states per Bi-LSTM direction. Three variants (high/low/full) vary the
  number of hidden units per stage.
- **Input features**: **Not specified dimensionally.** "we use the output features of their spatial CNN as the
  input to our TricorNet" (i.e. the same Lea et al. spatial-CNN features used by TCN), and for JIGSAWS "Features
  and cross-validation splits are provided by [13]". The paper never states the vector width. **No Fisher, no
  IDT** (`fisher` occurs 0 times in the full text). IDT+LM appears only as a baseline row.
- **Datasets & size** (Section 4, "Experiments"):
  - 50 Salads — "captures 25 people preparing mixed salad two times each", 5–10 min videos, **mid-level 17
    classes**, 5 splits.
  - GTEA — "seven types of daily activities… Each activity is performed by four different people, thus totally
    **28 videos**", "about 20 fine-grained action instances" per video, ~1 min.
  - JIGSAWS — "consists of **39 videos** capturing eight surgeons performing elementary surgical tasks… Each
    surgeon has about five videos with around 20 action instances… totally 10 different action classes. Each
    video is around two minutes long. In this work, we use the videos of suturing tasks."
- **Metrics** (Tables 1/2/3 as extracted from the arXiv HTML):
  - 50 Salads (Mid, Table 1): TricorNet Acc **67.5**, Edit **62.8**, F1@{10,25,50} **70.1, 67.2, 56.6**;
    ED-TCN 64.7 / 59.8 / 68.0, 63.9, 52.6; Spatial CNN 54.9 / 24.8 / 32.3, 27.1, 18.9.
  - GTEA (Table 2): TricorNet (low) Acc **64.7**, F1 **77.3, 73.4, 62.9**; ED-TCN Acc 64.0, F1 72.2, 69.3, 56.0;
    EgoNet+TDD Acc 68.5.
  - JIGSAWS (Table 3): TricorNet Acc **82.9**, Edit **86.8**; TricorNet (low) 82.2 / 84.9; TCN 81.4 / 83.1;
    ST-CNN 77.7 / 68.0; Spatial CNN 74.0 / 37.7.
- **Params**: **not stated**.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: Little new. It confirms that the
  entire Lea-et-al. lineage (TCN → TricorNet) rests on **unspecified-width learned CNN embeddings**, not on
  hand-crafted low-dim descriptors. Note the paper's own remark that "TricorNet (low) also has a good
  performance, which may due to the relatively small size of the dataset" — i.e. reduced capacity helped on the
  39-video JIGSAWS. That is weak but real support for the small-data-regime intuition.
- **Verification notes**: fetched — arXiv abs page, arXiv HTML full text; Tables 1–3 parsed directly from the
  HTML `<table>` elements. Venue UNVERIFIED as described.

---

### Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks (MS-GCN)

- **Venue/year**: `UNVERIFIED — arXiv preprint (arXiv:2202.01727v2 [cs.CV], 09 Oct 2022). The abs page has no
  Comments: or Journal reference: field, and the paper header says only "Submitted on x August 2022".`
  — **URL**: https://arxiv.org/abs/2202.01727 — **Code**: https://github.com/BenjaminFiltjens/MS-GCN (stated in
  the paper: "We publicly release our code and trained models at: https://github.com/BenjaminFiltjens/MS-GCN") —
  license: **none found** (GitHub API returns `"license": null`; `LICENSE`, `LICENSE.md`, `LICENSE.txt` all
  return HTTP 404 on the master branch)
- **Core mechanism**: MS-GCN "replaces the initial stage of temporal convolutions with spatial graph
  convolutions and dilated temporal convolutions", i.e. the prediction-generation stage of MS-TCN is swapped for
  modified ST-GCN layers that model "the spatial hierarchy among the joints", while MS-TCN's refinement stages
  are kept. Hyperparameters follow MS-TCN: 64 filters, temporal kernel size 3, 1 prediction-generation stage +
  3 refinement stages, 10 layers each, dilations 1,2,4,…,512, acausal, Adam lr 0.0005, 100 epochs, batch 4,
  τ=4, λ=0.15.
- **Input features**: A MoCap sequence is `f ∈ R^{T×N×C}` — `N` joints, `C` channels each. Per dataset the
  paper states (Section IV-F, "Graph representations"): "For HuGaDB we used the 3-axis accelerometer and 3-axis
  gyroscope data, for LARa the 3-axis limb position and 3-axis orientation, and for PKU-MMD v2, TUG, and
  FOG-GAIT we computed the 3-axis displacement and 3-axis relative coordinates (with respect to the root node)".
  → **6 channels per node** in every case (my arithmetic from the paper's own channel list; the paper does not
  print the total width). With `N` from Table I this gives **PKU-MMD v2: 25×6 = 150-dim**, **LARa: 19×6 =
  114-dim**, **FOG-GAIT: 9×6 = 54-dim**, **TUG: 19×6 = 114-dim**, **HuGaDB: 6×6 = 36-dim**.
- **Datasets & size** (Table I, "Dataset characteristics" — columns: Partitions / SR / #N / #Trials(test/train) / #L):
  - PKU-MMD v2: 3/10 subjects, 30 Hz, 25 nodes, **234/775 trials**, 52 classes. ("contains 1009 short video
    sequences in 52 action categories, performed by 13 subjects")
  - HuGaDB: 4/18, 60 Hz, 6 nodes, **69/307 trials**, 12 classes. ("a total of 18 subjects… 364 IMU trials in 12
    action categories")
  - LARa: 4/14, 50 Hz, 19 nodes, **113/264 trials**, 8 classes. ("The dataset contains 377 MoCap trials in 8
    action categories"; "All subjects participated in a total of 30 recordings of 2 minutes each")
  - FOG-GAIT: LOSO, 50 Hz, 9 nodes, **127/127 trials**, 5 classes. (proprietary; 7 PwPD subjects)
  - TUG: LOSO, 50 Hz, 19 nodes, **30/30 trials**, 6 classes. ("the data of only 10 participants were available…
    resulting in a total of 30 recordings")
- **Metrics** (Table II, F1@50 / Acc):
  - PKU-MMD v2: MS-GCN **51.6 / 68.5**; MS-TCN 46.3 / 65.5; ST-GCN 15.5 / 64.9; TCN 13.8 / 61.9; Bi-LSTM 22.7 / 59.6.
  - HuGaDB: MS-GCN **93.0 / 90.4**; MS-TCN 89.9 / 86.8.
  - LARa: MS-GCN **43.6 / 65.6**; MS-TCN 39.6 / 65.8; ST-GCN 25.8 / 67.9; Bi-LSTM 32.3 / 63.9.
  - FOG-GAIT: MS-GCN 95.0 / 90.1; TUG: MS-GCN 97.9 / 93.6.
  - Note the paper's own framing: "the addition of the refinement stages (i.e., multi-stage models) significantly
    reduces the number of segmentation errors", and graph convolutions helped "sample-wise accuracy more than…
    F1@50".
- **Params**: **not stated** (no parameter count in the paper; only the 64-filter / 10-layer-per-stage config).
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **Two important corrections and
  one genuinely useful data point.**
  - **Correction**: MS-GCN is **not** evaluated on JIGSAWS. `grep -c -i jigsaws` on the full text returns **0**.
    Our project assumption was wrong; the JIGSAWS low-dim precedent is MS-TCRNet and the TCN workshop paper, not
    MS-GCN.
  - **Useful**: MS-GCN operates on **54–150-dim** per-frame vectors — the same order of magnitude as our 144-dim
    input — and reports segment-level F1@50 on datasets as small as **30 test trials** (TUG) and **113 test
    trials** (LARa). On LARa (the closest analogue to an industrial, warehouse/assembly scenario) MS-GCN gets
    F1@50 43.6 / Acc 65.6 — **a segment-level F1 in the 40s, structurally similar to our F1@0.1 = 40.01**. So
    "low-dim input + deep multi-stage net → F1@50 in the low 40s" is a *known, published operating point*, not an
    anomaly. The paper also shows big gains from modelling structure *within* the low-dim vector (treating the 6
    channels as a spatial graph over nodes) — a direct hint that our 144-dim vector's (class × region) structure
    is being thrown away by a flat TCN.
- **Verification notes**: fetched — arXiv abs page, arXiv HTML full text; Table I and Table II read in full;
  GitHub API + LICENSE probes.

---

### MS-TCRNet: Multi-Stage Temporal Convolutional Recurrent Networks for Action Segmentation Using Sensor-Augmented Kinematics

- **Venue/year**: **Submitted to Pattern Recognition** (preprint) — verified from the arXiv `Comments:` field:
  "41 pages, 7 figures. Submitted to Pattern Recognition". No journal reference, so publication is not confirmed.
  — **URL**: https://arxiv.org/abs/2303.07814 — **Code**: https://github.com/AdamGoldbraikh/MS-TCRNet (stated in
  the paper) — license: **none found** (GitHub API `"license": null`; LICENSE probes return 404)
- **Core mechanism**: A TCN prediction generator with "intra-stage regularization" (prediction heads attached to
  intermediate Dual Dilated Residual Layers, not just the last one) followed by RNN-based refinement stages
  (BiLSTM → L-MS-TCRNet, BiGRU → G-MS-TCRNet) that operate at a *different sampling rate* than the generator
  ("the optimal frequency for action segmentation of RNN-based networks for kinematic data is significantly
  distinct from the optimal input frequency of TCN-based networks"). Plus two geometry-aware augmentations:
  **World Frame Rotation (WFR)** and **Hand Inversion (HI)**.
- **Input features** — very concrete, and very small:
  - **VTS/BRS**: six electromagnetic sensors (NDI trackSTAR 180) on index, thumb and wrist of both hands.
    "Each sensor provided three spatial coordinates and three Euler angles per timestamp, totaling **36 kinematic
    variables**." Original rates 179.695 Hz (VTS) / 100 Hz (BRS), downsampled to 30 Hz. The actual network input
    is the derived **linear and angular velocity** stream ("Based on previous studies, velocities are the most
    beneficial input to an algorithm for analyzing kinematic data").
  - **JIGSAWS**: "for each manipulator, we used only three position variables, calculated three Euler angles
    based on the rotation matrix data, and the gripper angle, in **total 14 kinematic variables**" (two PSMs →
    14-dim), sampled at 30 Hz.
- **Datasets & size**:
  - VTS: "Eleven medical students, thirteen attending surgeons, and one resident completed **100 procedures**,
    each lasting 2-6 minutes"; 6 suturing gestures. (96 procedures × 8 seeds used in reporting.)
  - BRS: "**255 porcine enterotomy repair procedures**" collected in total, "Among participants, **52 performed a
    single-layer repair** on the large hole, with only the suturing part annotated, resulting in **52 sequences**
    in this dataset"; 5 surgical maneuvers (suture throws, instrument knot ties, hand knot ties, thread cuts,
    background).
  - JIGSAWS: "eight surgeons… performed **five repetitions** of three elementary surgical tasks… our work focuses
    on the suturing task, which involves **ten distinct gestures**"; leave-one-user-out. (Sequence count not
    restated here; see the TCN workshop paper, which states 39 suturing sequences.)
- **Metrics**:
  - **BRS (Table 2)**: MS-TCN++ F1-macro 64.4, Acc 76.7, **Edit 45.8**, F1@{10,25,50} 52.7 / 49.4 / 37.9.
    G-MS-TCRNet F1-macro 64.8, Acc 77.4, **Edit 75.8**, F1@10/25/50 **77.4 / 72.9 / 57.6**.
    L-MS-TCRNet 65.9 / 77.8 / Edit 75.1 / 77.3 / 73.1 / 57.1.
    The paper: "the standard MS-TCN++ has significant difficulty with segmental metrics on this dataset, but both
    our architectures performed significantly better."
  - **JIGSAWS (Table 7)**: G-MS-TCRNet with WFR+HI — F1-macro **81.8 ± 11.2**, Acc **86.4 ± 7.3**, Edit
    **90.5 ± 11.4**, F1@10 **94.1 ± 7.5**, F1@25 **93.6 ± 8.1**, F1@50 **87.0 ± 13.1**. Baseline (no aug,
    G-MS-TCRNet): 79.2 / 85.0 / 87.7 / 91.4 / 90.6 / 85.1.
  - **Kinematics-only vs video-only vs multimodal on JIGSAWS (Table 10, supplementary 7.5)** — parsed from the
    HTML table preserving the Kin/Vid columns:
    - Video-only: C3D-MTL-VF* Acc 82.1 / Edit 86.6; MS-TCN* 78.9 / 85.8; TCN+RL* 81.4 / 88.0; TDRN* **84.6 /
      90.2**; RL+Tree* 81.7 / 88.5.
    - Multimodal (Kin+Vid): MRG-Net* 87.9 / 89.3; Fusion-KV* 86.3 / 87.2; MA-TCN 86.8 / 91.4.
    - **Kinematics-only: G-MS-TCRNet+WFR+HI 86.4 / 90.5.**
    - **Caveat stated by the authors**: "Note, some networks compared used original JIGSAWS labels corrected in
      [2] since 2020." The `*` rows use the *original* labels, so this is not a perfectly clean comparison.
- **Params**: **L-MS-TCRNet 6.3×10⁶** and **G-MS-TCRNet 8.4×10⁶** — from Table 9 ("Number of parameters"),
  described in text as "achieved a mean F1-Macro of 80.9 on the validation set with 6.3×10⁶ parameters and the
  obtained G-MS-TCRNet attained a mean F1-Macro of 81.2 with 8.4×10⁶ parameters."
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **This is the single most
  encouraging precedent in the whole thread, and the most directly actionable.**
  - A **14-dimensional** per-frame input (JIGSAWS kinematics) reaches **Edit 90.5 / Acc 86.4 / F1@50 87.0** with
    a ~8.4M-param model, and **beats every video-only baseline** in the paper's own comparison table. Low input
    dimensionality is demonstrably not a hard ceiling.
  - BRS is the closest analogue to **our failure mode**: on a 52-sequence dataset, MS-TCN++ (our current model
    family) collapses to **Edit 45.8** while the architecture change lifts it to **75.8** — a **+30 Edit** swing
    with the *same* 36-dim input. That is direct evidence that in the small-data/low-dim regime the bottleneck
    was **architecture + augmentation**, not feature dimensionality.
  - **Actionable mechanisms to steal**: (i) *intra-stage regularization* — extra prediction heads on intermediate
    layers, which the paper credits with better frame-wise performance and which "focus on small areas, minimizing
    long-history impact"; (ii) running the refinement stage at a **different sampling rate** from the generator;
    (iii) **geometric data augmentation**. Our features are z-scored counts/areas over a ROI grid — augmentation
    is less obvious, but segment-level resampling/augmentation is plausible.
  - **Caveat**: full frames — VTS/BRS are 100 Hz downsampled to 30 Hz over 2–6 min procedures, so these datasets
    are far larger in frame count than our ~tens of thousands of frames.
- **Verification notes**: fetched — arXiv abs page, arXiv HTML full text; Tables 1, 2, 7, 8, 9, 10 read; Table 10
  parsed from raw HTML to recover the Kin/Vid column alignment that plain text flattening loses.

---

### An Efficient Framework for Few-shot Skeleton-based Temporal Action Segmentation

- **Venue/year**: `UNVERIFIED — arXiv preprint, "Submitted on 20 Jul 2022" (arXiv:2207.09925). The abs page
  carries no Comments: or Journal reference: field.` — **URL**: https://arxiv.org/abs/2207.09925 — **Code**: none
  found — license: none found
- **Core mechanism**: Two independent contributions, both aimed squarely at insufficient data.
  (1) **Motion-interpolation data augmentation**: "First, divide the existing action sequences into multiple
  action primitives by labels. Second, separate the action primitives by category. Third, we select one action
  primitive in each category and use slerp to concatenate action primitives in the order they appear in the
  action sequence." Spherical linear interpolation (slerp) on quaternions blends the boundary between
  concatenated primitives. "In this way, training data can increase at an exponential rate by synthesizing new
  action sequences": with `M` sequences of `n` primitives the synthetic count is `M·2^n` — "it is sufficient to
  use only two action sequences (M=2) as training data. Each action sequence contains ten action primitives
  (n=10). After data augmentation, training data augment to **1024 (2^10) samples**."
  (2) **CTC head**: "we concatenate a Connectionist Temporal Classification (CTC) layer with a network designed
  for skeleton-based TAS", adding a CTC loss term to the CE + truncated-MSE objective, to improve
  sequence-level alignment. The base network is explicitly MS-GCN ("we quote Multi-Stage Spatial-Temporal Graph
  Convolutional Networks (MS-GCN) proposed in [28] as a simple baseline architecture"), with 1 prediction stage +
  3 refinement stages, 10 layers each, 64 filters, kernel 3, τ=4, λ=0.15, μ=0.0005, Adam lr 0.0005.
- **Input features**: skeleton joint sequences. Tai Chi: "The skeleton with **18 joints** is used in the
  experiment"; data "collected by a wearable inertial sensor system". UTD-MHAD2: "**25 joints** were recorded"
  (Kinect v2). PKU-MMD phase 2: "The skeleton consists of **25 joints**". The paper's feature map is
  `f_in ∈ R^{C_in × T × N}` (channels × frames × joints). No explicit per-joint channel width is stated.
- **Datasets & size**:
  - **Tai Chi** (self-collected, tiny): "twenty action sequences performed by a male subject and a female
    subject. Each action sequence has ten action primitives, and each subject repeats the action sequence ten
    times." For augmentation: "we select **2 action sequences** performed by subject 1 as training data… The
    remaining 18 action sequences are used as test data." → **training set = 2 sequences**, 10 classes.
  - **UTD-MHAD2**: "ten actions collected by Kinect v2… six subjects… five times." Assembled into "a total of 30
    (5×6) action sequences"; "Without data augmentation, we select an action sequence from subject 1 and 4
    respectively as training data, and the remaining **28 action sequences** are used as test data." → training
    set = 2 sequences, 11 classes.
  - **PKU-MMD** phase 2 (large): "2000 short videos in 49 action categories, performed by 13 subjects"; "we set
    775 samples of 10 subjects… as training data and **234 samples** of other 3 subjects as test data." (Note:
    these 234 PKU-MMD test samples are numerically identical to the PKU-MMD test-trial count in MS-GCN's Table I.)
- **Metrics** (Table 2, "Performance of Data Augmentation and CTC on the three datasets"):
  - **Tai Chi** — no aug / no CTC: Acc **0.776**, F1@{10,25,50} **0.735 / 0.697 / 0.606**.
    aug + CTC: Acc **0.895**, F1 **0.926 / 0.914 / 0.846**.
    (CTC alone: Acc 0.782, F1 0.726 / 0.689 / 0.611. Augmentation alone: Acc 0.878, F1 0.891 / 0.874 / 0.801.)
  - **UTD-MHAD2** — no aug / no CTC: Acc **0.484**, F1 **0.603 / 0.542 / 0.421**.
    aug + CTC: Acc **0.893**, F1 **0.837 / 0.831 / 0.786**.
  - The augmentation term dominates; CTC alone barely moves Tai Chi (0.776 → 0.782 Acc) and slightly hurts
    UTD-MHAD2 (0.484 → 0.442 Acc).
- **Params**: **not stated**.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **Highly relevant, and probably
  the most transferable idea in this document.** Our real constraint is not frames-in-total but *heterogeneity*:
  8 test videos, ~9+1 classes, dense labels. This paper trains a deep multi-stage net on **2 sequences** by
  synthesising new sequences from recombined action primitives, and roughly doubles segment-level F1
  (Tai Chi F1@10 0.735 → 0.926; UTD-MHAD2 F1@10 0.603 → 0.837). The mechanism needs *segment-level* labels,
  which we have. Our feature space is a hand-crafted detection summary rather than skeletons, so slerp on
  quaternions does not transfer literally — but the abstract recipe (decompose into segments → recombine in
  class-consistent orders → interpolate the seam) does, and directly targets our "51.6% of segments get a wholly
  wrong label" failure by *increasing label-sequence diversity* rather than model capacity.
- **Verification notes**: fetched — arXiv abs page, arXiv HTML full text; Sections 3–5 and Table 2 read in full.

---

## Discovery pass

### MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation (IDT-vs-I3D controlled swap)

- **Venue/year**: **CVPR 2019** — `PARTIALLY VERIFIED`: I fetched arXiv `1903.01945v2` and read Table 10 and the
  surrounding text; the arXiv abs page's venue field was not separately confirmed in this session.
  — **URL**: https://arxiv.org/abs/1903.01945 — **Code**: not checked — license: not checked
- **Core mechanism**: Multi-stage TCN — a prediction-generation stage followed by refinement stages, each with
  dilated residual layers, plus a truncated-MSE "smoothing" loss to suppress over-segmentation. This is the
  direct ancestor of the project's current MS-TCN++ baseline.
- **Input features**: I3D by default ("For all datasets, we extract I3D features…"), used "without fine-tuning".
  **For one controlled comparison the authors swap the feature type on the Breakfast dataset** to IDT.
- **Datasets & size**: 50 Salads, GTEA, Breakfast. (Dataset sizes not re-extracted here — out of scope for this
  thread; the point of this entry is the feature swap.)
- **Metrics — Table 10, Breakfast, SAME model, only the input feature differs**:
  - **`MS-TCN (IDT)`: F1@{10,25,50} = 58.2 / 52.9 / 40.8, Edit 61.4, Acc 65.1**
  - **`MS-TCN (I3D)`: F1@{10,25,50} = 52.6 / 48.1 / 37.9, Edit 61.7, Acc 66.3**
  - The authors' own reading, verbatim: *"the impact of the features is very small. While the frame-wise accuracy
    and edit distance are slightly better using the I3D features, the model achieves a better F1 score when using
    the IDT features compared to I3D. This is mainly because I3D features encode both motion and appearance,
    whereas the IDT features encode only motion. For datasets like Breakfast, using appearance information does
    not help the performance since the appearance does not give a strong evidence about the action that is
    carried out."*
- **Params**: not extracted.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **This is the most directly
  decisive result in the whole survey, and it argues AGAINST the "swap in deep features" fix.** It is the one
  controlled, same-architecture feature-type ablation I could verify anywhere in this thread, and the
  hand-crafted motion-only descriptor (IDT) **beats** the deep spatio-temporal feature (I3D) by **+5.6 F1@10,
  +4.8 F1@25, +2.9 F1@50** on segment-level metrics, losing only 0.3 Edit and 1.2 Acc. Caveat: Breakfast is a
  large dataset of visually homogeneous cooking actions, so the "appearance does not help" conclusion is
  dataset-specific. It does not prove our detection-derived features are adequate — but it does demolish the
  presumption that moving to I3D-style features is the obvious way to raise segment-level F1.
- **Verification notes**: fetched arXiv HTML full text (`arxiv.org/html/1903.01945v2`); Table 10 parsed directly
  from the HTML `<table>` element; comparison paragraph quoted verbatim from the extracted text.

---

### Using 3D Convolutional Neural Networks to Learn Spatiotemporal Features for Automatic Surgical Gesture Recognition in Video

- **Venue/year**: **MICCAI 2019** — verified from the arXiv `Comments:` field ("Accepted at MICCAI 2019. Source
  code will be made available") and the title. — **URL**: https://arxiv.org/abs/1907.11454 — **Code**: stated as
  forthcoming; not located — license: not checked
- **Core mechanism**: A 3D ResNet-18 whose max-pooling is adapted so "downsampling is only performed along the
  spatial dimensions", producing **dense per-frame** predictions from 16-frame video snippets, applied in a
  sliding-window fashion. Two initialisations are compared: Kinetics-pretrained (K) and bootstrapped from an
  ImageNet-pretrained 2D ResNet-18 (B).
- **Input features**: **raw video**, learned end-to-end. The comparison of interest is **2D ResNet-18 on single
  frames** (a purely spatial per-frame representation) versus **3D ResNet-18 over 16-frame snippets** (a
  spatio-temporal per-frame representation). Neither is hand-crafted.
- **Datasets & size**: JIGSAWS suturing — "We evaluate our approach on **39 videos** of robot-assisted suturing
  tasks performed on a bench-top model". Evaluated at 5 fps and 10 fps.
- **Metrics — Table 2, JIGSAWS suturing, evaluation at 5 fps** (Acc / Avg F1 / Edit / F1@10):
  - **2D ResNet-18: 79.9 / 73.3 / 41.4 / 55.4**
  - **3D CNN (B): 79.9 / 73.7 / 64.0 / 75.2**
  - 3D CNN (K): 81.8 / 75.8 / 58.7 / 71.1
  - 3D CNN (B) + sliding window: 84.0 / 78.4 / 80.7 / 87.2
  - (At 10 fps, for comparability with prior work: 2D ResNet-18 Edit 30.6 / F1@10 44.2; 3D CNN (B) Edit 49.5 /
    F1@10 62.8; S-CNN + TCN Edit 68.2 / F1@10 77.9; S-CNN + TCN + Deep RL Edit 88.0 / F1@10 92.0.)
  - The authors: the 3D variant "yields comparable or better frame-wise evaluation results (accuracy and average
    F1) and **considerably better segment-based evaluation results (edit score and F1@10)** compared to the 2D
    counterpart."
- **Params**: not extracted.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **The sharpest evidence that
  frame-level *representation content* — not model capacity or input width — governs segment-level metrics.**
  Going from a spatially-only per-frame representation to a spatio-temporal one leaves frame accuracy **exactly
  unchanged (79.9 → 79.9)** yet moves **Edit 41.4 → 64.0 (+22.6)** and **F1@10 55.4 → 75.2 (+19.8)**. Our project's
  two headline numbers are Edit 51.08 and F1@0.1 40.01 — i.e. we sit in the same band as the *2D per-frame*
  condition, and the published gain to Edit 64 / F1@10 75 came from giving the representation temporal context.
  **Actionable**: our 144-dim vector is an instantaneous per-frame detection summary with **no motion/temporal
  derivative** — the exact deficiency this paper isolates. Adding short-window temporal deltas (or frame
  differences) to the 144-dim vector is a cheap, directly-motivated experiment. A sliding window adds a further
  Edit 64.0 → 80.7.
- **Verification notes**: fetched arXiv abs page and arXiv HTML full text; Table 2 parsed directly from the HTML
  `<table>` element.

---

### M2R2: Multimodal Robotic Representation for Temporal Action Segmentation

- **Venue/year**: `UNVERIFIED — arXiv:2504.18662, Comments: "8 pages, 6 figures, 2 tables". No journal reference
  on the abs page.` — **URL**: https://arxiv.org/abs/2504.18662 — **Code**: none found in the paper — license:
  none found
- **Core mechanism**: A **modular, reusable multimodal feature extractor** for TAS, decoupled from the temporal
  model. Per-modality encoders (ActionCLIP image encoder for RGB; Audio Spectrogram Transformer for audio;
  learned linear projection + temporal embedding for each proprioceptive sensor) are fused by a Transformer
  encoder layer + MLP into a single per-frame feature. Training objectives: (i) an action-**order** contrastive
  loss matching a window embedding against a text sentence describing the action order; (ii) a **boundary
  regression** network. After training, the fusion transformer / boundary net / text encoder are discarded and
  per-frame features are extracted once and reused with **any** downstream TAS model.
- **Input features**: fused multimodal. **Embedding size `D_e = 512`**: "We use a Visual Transformer initialized
  with ActionCLIP weights [13], to produce the image representation `I_i ∈ R^{D_e}`, where **`D_e` = 512 is the
  M2R2 embedding size**." Modalities: vision (RGB), audio (64 mel filter banks, 16 kHz, window 400), and
  proprioception (force/torque, end-effector pose **and twist**, gripper width). So the TAS model sees a 512-dim
  vector, and — critically — the *ablation shows it would not need the vision part at all* (see below).
- **Datasets & size**:
  - **REASSEMBLE**: "**149 recordings** with a median of 36 actions per recording and strong visual similarity
    between objects"; "**69 fine-grain labels** (activity+object, e.g. Insert USB) and 4 coarse-level labels".
  - **(Im)PerfectPour**: "**544 demonstrations** of a robot performing bartending tasks."
  - **JIGSAWS**: suturing task, Leave-One-User-Out; modalities "pose, twist, and gripper width of both arms".
- **Metrics**:
  - **Table I — REASSEMBLE, fine-grain labels** (F1@{10,25,50} / Acc / EDIT):
    - Vision-only (Br-Prompt, modular): `BRP+MSTCN` 12.9 / 10.9 / **8.4** | Acc 7.9 | EDIT 18.4;
      `BRP+ASRF` 12.8 / 11.1 / **8.6** | Acc 6.4 | EDIT 19.4;
      `BRP+DiffAct` 12.0 / 10.1 / **5.1** | Acc 5.7 | EDIT 21.4.
    - Multimodal (M2R2, V+A+P+FT+G): `M2R2+ASRF` 83.5 / 83.4 / **82.4** | Acc 82.7 | EDIT 82.5;
      `M2R2+MSTCN` 83.1 / 82.7 / **80.8** | Acc 82.4 | EDIT 79.3;
      `M2R2+DiffAct` 78.1 / 77.7 / **74.6** | Acc 74.9 | EDIT 68.7.
    - Unsupervised robotic baselines: BOCPD (F/T only) F1@50 **12.8**, Acc 19.7, EDIT 17.4; AWE (pose only)
      F1@50 **35.8**, Acc 54.6, EDIT 22.1.
    - **Ablation (Table, REASSEMBLE, DiffAct head)**: *only vision* F1@50 21.6 / Acc 21.4 / EDIT 28.9;
      *only audio* 36.4 / 34.5 / 37.5; **only proprio 74.5 / 75.4 / 69.2**; *all* 74.6 / 74.9 / 68.7.
  - **Table II — (Im)PerfectPour** (F1@{10,25,50} / Acc / EDIT): `BRP+DiffAct` 81.6 / 79.9 / **71.8** |
    Acc 74.4 | EDIT 83.1; `M2R2+DiffAct` 90.5 / 89.7 / **89.7** | Acc 93.3 | EDIT 83.0.
    The paper: "Our method outperforms vision-only baselines by 16.1% in F1@50."
  - **Table III — JIGSAWS** (F1@{10,25,50} / Acc / EDIT): `BRP+DiffAct` (V) 59.6 / 49.6 / **30.2** |
    Acc 44.5 | EDIT 81.9; `M2R2+DiffAct` (V,P,G) 94.3 / 93.0 / **89.4** | Acc 87.1 | EDIT 91.7.
- **Params**: **not stated** (neither the extractor nor the downstream TAS models).
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: M2R2 is the closest published
  *architecture philosophy* to what we are doing — it accepts that a compact, purpose-built, non-I3D,
  modality-structured per-frame vector is the right input for procedural/industrial TAS, and it validates that
  such a vector, fed to a plain MS-TCN-family model, can reach very high segment-level scores. Two transfers:
  (i) **decouple feature extraction from the temporal model** — worth testing whether our 144-dim vector is
  itself improvable (e.g. adding temporal context or normalisation per video) independently of any TCN change;
  (ii) the paper's own *negative* result for vision-only features on small-object assembly ("TAS models trained
  with vision-only features frequently confuse object categories… because many REASSEMBLE objects are poorly
  visible due to their small size") is a strong warning that our detection-derived features may be losing exactly
  the object-identity signal that separates our 9 action classes. That is a testable hypothesis about our
  "whole-segment identity confusion" failure, not just a stylistic observation.
- **Verification notes**: fetched — arXiv abs + HTML full text; all four result tables parsed from raw HTML
  `<table>` elements. Venue UNVERIFIED.

---

### How Object Information Improves Skeleton-based Human Action Recognition in Assembly Tasks

- **Venue/year**: **IJCNN 2023** — verified from the paper's own author block: "in: IEEE Int. Joint Conf. on
  Neural Networks (IJCNN), Queensland, Australia, pp. 01-09, IEEE 2023,
  https://doi.org/10.1109/IJCNN54540.2023.10191686". — **URL**:
  https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2023/Aganian-IJCNN-2023.pdf
  (fetched) — **Code**: none found in the paper (the only GitHub link is to Microsoft's View-Adaptive Neural
  Networks, used as a baseline) — license: none found
- **Core mechanism**: Treat **object centers as additional skeleton joints**. Object masks come from a Mask
  R-CNN instance-segmentation model (ResNet-50 / ResNet-101 / ResNeXt-101 / Swin-Tiny backbones) fine-tuned on
  IKEA ASM; each mask is converted to its **center of mass** and inserted "in the appended lines below the
  skeleton joints", with coordinates set to zero for absent object classes. Two backbones are enhanced: a 2D-CNN
  (VA-CNN) and PoseConv3D (additional heatmaps per object class).
- **Input features**: skeleton joints + object center points. For PoseConv3D, "the skeleton joints of a single
  frame are embedded in heatmaps — one heatmap for each joint class — which are stacked along the channel axis,
  resulting in a heatmap tensor of size `J × H × W`, where `J` is the number of distinct joint classes"; the
  extension adds "Additional Heatmaps for Object Classes". The paper reports **7 object categories**. No total
  input dimensionality is stated.
- **Datasets & size** — **IKEA ASM**, stated in the paper: "Each assembly process is captured by three different
  Kinect V2 cameras, resulting in **1113 videos and 35 hours of footage at 24 fps**. For human action recognition,
  there are **17K labeled action instances, distributed over 33 atomic classes**. We use the official splits
  provided in [12], where about 2/3 is in training and 1/3 in testing. The pieces of furniture consist of **seven
  different object categories**." Object instances: "1% of the data were manually labeled. The remaining
  annotations were obtained by overfitting several models".
- **Metrics**: mean class accuracy (mAcc) and top-1 (top1). **Table I, skeleton-only baseline**: 2D-CNN mAcc
  37.7 / top1 70.3; PoseConv3D mAcc 39.5 / top1 75.0. The paper: "the much more complex PoseConv3D is also
  significantly better on only skeleton sequences". Adding object information raises these substantially (best
  PoseConv3D-with-all-objects values appear as mAcc 58.5 / top1 85.4).
  `PARTIALLY UNVERIFIED — my text extraction of Table I loses column/row alignment (the PDF was extracted with a
  hand-written zlib stream decoder because pdftotext is unavailable), so the exact cell-to-row pairing beyond the
  skeleton-only baseline row and the best reported values should be re-checked against the PDF before being
  quoted. The paper's own qualitative claim is unambiguous: "our approach massively improves the performance of
  action recognition on the IKEA ASM dataset [12]."`
- **Params**: **not stated**.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **Directly on point, with two
  caveats.** This is a verified precedent for **object-derived information improving a low-dimensional temporal
  model in an industrial assembly setting**, and it is the paper whose framing best matches ours.
  - Its argument for *why* object information is needed is essentially our failure mode, stated in their words:
    "skeleton-based models have the drawback of not being able to process objects involved in actions"; and
    "actions in which workers pick up similar-sized assembly parts, such as a cabinet side panel or a cabinet
    back panel, **cannot be distinguished based on skeletons**." Our whole-segment identity confusion is the
    same pathology: without a discriminative object/region signal, whole segments collapse onto the wrong class.
  - It also states the small-data premise we rely on: "Due to the difficulties in creating action recognition
    datasets for assembly processes, these datasets tend to be **smaller** compared to other areas such as object
    detection. As a result, RGB-based approaches trained on these datasets tend to **overfit** and exhibit poor
    generalization capabilities."
  - **Caveat 1**: this is action **recognition on pre-trimmed clips**, not temporal action segmentation. The
    authors explicitly note that segmentation methods "typically use models trained on action recognition of
    pre-trimmed clips as the backbone", so it is a backbone-level result, not a segmentation-metric result.
  - **Caveat 2**: the paper only *adds* object information to a skeleton baseline; it does not test a
    object-only feature vector. It therefore supports "object info helps", not "object-only features suffice".
- **Verification notes**: fetched the IJCNN 2023 PDF and extracted text via a hand-written zlib-based PDF stream
  decoder (pdftotext/pypdf unavailable). Venue, dataset statistics and qualitative claims verified; Table I cell
  alignment PARTIALLY UNVERIFIED as noted.

---

### 50 Salads dataset (Stein & McKenna, "Combining Embedded Accelerometers with Computer Vision for Recognizing Food Preparation Activities")

- **Venue/year**: UbiComp 2013 — verified from the dataset's own description document, which states the required
  citation: "Sebastian Stein and Stephen J. McKenna, Combining Embedded Accelerometers with Computer Vision for
  Recognizing Food Preparation Activities, The 2013 ACM International Joint Conference on Pervasive and
  Ubiquitous Computing (UbiComp 2013), Zurich, Switzerland, 2013." — **URL** (fetched):
  https://core.ac.uk/download/589171319.pdf
- **Input features** (the point of fetching this): **"3-axis accelerometer data at 50 Hz of devices attached to
  a knife, a mixing spoon, a small spoon, a peeler, a glass, an oil bottle, and a pepper dispenser."** That is
  **7 sensor devices**, and the CSV format is "a single accelerometer sample per row. Each row consists of (from
  left to right): 1. data info field ("ACCEL") 2. timestamp 3. device ID 4. sequence number 5. x-acceleration in
  g 6. y-acceleration in g 7. z-acceleration in g". → **7 devices × 3 axes = 21 raw accelerometer channels**
  (my arithmetic from the paper's own list).
  Note a **discrepancy between primary sources**: the TCN workshop paper says "synchronized accelerometers
  attached to **ten** objects in the scene, such as the bowl, knife, and plate", whereas the dataset document
  lists **seven** devices and does not include a bowl or plate. I record both; the dataset document is the more
  authoritative for the raw channel count.
- **Size**: "It captures **25 people preparing 2 mixed salads each** and contains over **4h** of annotated
  accelerometer and RGB-D video data." RGB video 640×480 at 30 Hz; depth 640×480 at 30 Hz.
- **Metrics**: none reported in the dataset document.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: This pins down the *actual*
  dimensionality of the low-dim sensor precedent in the TCN workshop paper. Lea et al. used "the absolute values
  of accelerometer values" from these devices, i.e. a signal derived from **~7–21 raw channels** — **an order of
  magnitude narrower than our 144-dim vector** — and still reached Edit 65.6 / Acc 82.0 on 50 Salads. So the
  "low dimensionality" objection to our feature set is empirically weak: the field has published strong TAS
  results on vectors narrower than ours. What differs is not width but **signal quality**: those channels
  physically measure the manipulated object, whereas ours measure YOLO detections of it.
- **Verification notes**: fetched the PDF and extracted text with the hand-written decoder; the accelerometer
  sentence and CSV schema were read verbatim from the extracted text.

---

### State Duration and Interval Modeling in Hidden Semi-Markov Model for Sequential Data Analysis

- **Venue/year**: `UNVERIFIED — arXiv:1608.06954v2 [cs.AI], 13 Feb 2019 (Narimatsu & Kasai). No journal
  reference field on the abs page.` — **URL**: https://arxiv.org/abs/1608.06954 — **Code**: not checked —
  license: not checked
- **Core mechanism**: Explicit **state-duration and interval modelling inside an HSMM** — a generative model that
  models how long a state persists (and how intervals between states behave), rather than relying on the
  geometric-duration implicit in a plain HMM.
- **Input features**: **Not applicable / not stated** — this is a general sequential-data methodology paper.
- **Datasets & size**: **not TAS**; no TAS benchmark.
- **What this implies for our 144-dim / tens-of-thousands-of-frames setting**: **Weak, methodology-only.** I
  fetched this to answer the "hidden (semi-)Markov models or explicit segment-duration models on low-dim
  features" discovery question. I could **not** verify, from primary sources within this session, a TAS paper
  that uses an HSMM/explicit-duration model on low-dimensional non-visual features and reports segment-level
  metrics. This paper establishes that explicit duration modelling is a mature, well-understood technique — it is
  *not* evidence that it beats deep nets in our regime.
- **Verification notes**: fetched the arXiv PDF and extracted text with the hand-written decoder. Included for
  completeness; treat as background, not as precedent.

---

### Industrial / procedural-activity TAS datasets (delegated verification)

`ATTRIBUTION NOTE — these facts were verified by a delegated parallel check whose raw notes are at
.tas_survey_tmp/notes/_industrial_datasets_raw.md. That check reports fetching each primary source directly
(arXiv abs/HTML and the CVF open-access PDFs, extracted with pypdf). I did NOT independently re-fetch these
three dataset papers myself, so treat them as second-hand-verified: high confidence, but re-check the cited
table before quoting in anything external.`

- **Assembly101** (CVPR 2022, arXiv 2203.14712 — id confirmed by title match): **362** disassembly/assembly
  sequences × 12 viewpoints = **4,321 videos / 513 hours**; **1,013,523** fine-grained + **104,759** coarse
  segments; **1380 fine / 202 coarse** classes; **53 participants** (Tables 1–2, Sec. 4). TAS baseline input,
  quoted: *"using frame-wise features extracted from TSM [25] trained for action recognition on Assembly as
  input"* — i.e. deep RGB video features; **feature dimensionality never stated**. Reported TAS baseline
  (Table 7): MS-TCN++ F1@10 31.6 / F1@25 27.8 / F1@50 20.6, Edit 30.7, MoF 37.1; C2F-TCN 33.3 / 29.0 / 21.3,
  Edit 32.4, MoF 39.2. Total frame count never reported.
- **IKEA ASM** (WACV 2021, arXiv 2007.00394): **371 assemblies**, **1113 RGB + 371 depth videos**,
  **3,046,977 frames (~35.27 h)**, **33 atomic actions**, **48 subjects**. Baselines are end-to-end fine-tuned
  I3D/P3D/C3D/ResNet on RGB — no frozen pre-computed features, no stated dims. Table 2 frame accuracy: P3D 60.4,
  I3D 57.58, C3D 45.73, ResNet50 30.38, ResNet18 27.06. **Notably, IKEA ASM has no TAS (F1@/Edit) benchmark at
  all** — it is an action-recognition/localisation dataset, which is why Aganian et al. (above) is the paper that
  actually uses it for a recognition comparison.
- **MECCANO** (WACV 2021, arXiv 2010.05654; extended 2209.08691): **20 videos / 299,376 frames / 20 participants
  / 61 action classes / 12 verbs / 8857 segments**. Baselines are PySlowFast C2D/I3D/SlowFast and ResNet-50,
  fine-tuned end-to-end on RGB clips; dims not stated. Table 3: SlowFast Top-1 42.85 / F1 41.05; I3D 42.51 / 38.88.
  **No TAS (F1@/Edit) benchmark.** Flagged inconsistency: the paper text states "47.82 Top-1" for SlowFast while
  its own Table 3 reports 42.85.

**Interpretation for our setting**: none of these three industrial datasets supports a low-dimensional feature
pipeline — the largest two benchmark with **deep video features (TSM) or end-to-end RGB**, and IKEA ASM/MECCANO
have **no segment-level TAS benchmark at all**. Their real use to us is as a **scale benchmark that shows how
unusual our regime is**: Assembly101 is ~1M fine-grained segments and 1380 classes; we have ~9 classes and a
few hundred segments, and our published-competitor segment-level F1 (0.40 at IoU 0.1) is in the same range as
the *much larger* Assembly101 MS-TCN++ baseline (F1@10 31.6) despite a ~1000× smaller dataset. That is a sign
our task is intrinsically harder per-class, not merely data-starved.

---

## Could NOT verify (explicit gaps)

- `UNVERIFIED — TAS / surgical phase recognition using YOLO-style object-detection outputs (presence/count/area
  over a ROI grid) as the temporal model's input.` The closest verified analogue is Aganian et al. (object
  centres appended to skeleton joints, IJCNN 2023 — action recognition, not segmentation) and M2R2
  (proprioceptive + audio + vision fusion, not object detections). No paper matching our exact
  detection→low-dim-vector→TAS pipeline was confirmed.
- `UNVERIFIED — a TAS benchmark comparison in which replacing hand-crafted/low-dim features with deep video
  features produced a LARGE GAIN, holding everything else fixed.` **This hypothesis is not just unverified — the
  one controlled test found points the other way.** MS-TCN (Table 10, Breakfast, same architecture, feature type
  swapped) reports IDT F1@50 40.8 versus I3D 37.9, and states "the impact of the features is very small". A
  delegated check additionally reports that the TPAMI TAS survey (arXiv 2210.10352v5) states: *"To the best of our
  knowledge, no empirical research has compared utilizing pre-computed features to training TAS models from raw
  images end-to-end"* — `second-hand, not independently re-fetched by me`. See the Feature-side evidence section.
- `VERIFIED BY DELEGATED CHECK (not re-fetched by me) — Assembly101, IKEA ASM (dataset paper), Meccano dataset
  sizes and baseline input features.` Confirmed by a parallel check whose raw notes are at
  `.tas_survey_tmp/notes/_industrial_datasets_raw.md` and summarised in the "Industrial / procedural-activity
  TAS datasets" entry above. Verified arXiv ids: Assembly101 = 2203.14712, IKEA ASM = 2007.00394,
  MECCANO = 2010.05654 (extended 2209.08691), M2R2 = 2504.18662 (confirmed correct).
- `UNVERIFIED — venue for TricorNet (arXiv 1705.07818), MS-GCN (2202.01727), few-shot skeleton TAS (2207.09925),
  M2R2 (2504.18662).` All four arXiv abs pages carry no Comments/Journal-reference field; I did not find these
  in the CVF open-access index (searched the CVPR 2017 index for TricorNet, no match).
- `UNVERIFIED — parameter counts for TCN (1611.05267), TricorNet (1705.07818), MS-GCN (2202.01727), few-shot
  skeleton TAS (2207.09925), M2R2 (2504.18662).` Each was grepped for `param`/`million`; no parameter counts are
  printed in these papers.
- **Tooling note**: `arxiv.org/search/` and `export.arxiv.org/api/query` became rate-limited/blocked during this
  session (HTTP 429 and 406 respectively, the latter requiring a browser User-Agent; both then hard-throttled).
  Discovery therefore relied on `web_search` plus direct `arxiv.org/abs|html/<id>` fetches, which continued to
  work. For PDF-only sources I wrote a zlib-based PDF stream decoder (`raw/lowdim/pdftext.py`); it works but
  **loses table layout**, so any table whose correctness depends on row/column alignment is flagged above.
  Correcting the brief: `pypdf` **is** importable in this environment (a delegated check verified
  `from pypdf import PdfReader` works); only `pdftotext`, `mutool`, `gs`, `qpdf`, `pdfinfo`, `fitz`, `pdfminer`
  and `pdfplumber` are missing. Future PDF work should prefer pypdf over my hand-written decoder.

---

## Feature-side evidence

Verified evidence bearing on the question *"is the frame-level feature representation, rather than the temporal
architecture, the bottleneck?"*

1. **Low-dimensional non-visual features can outperform large pretrained video features.**
   - **MS-TCRNet**, Table 10 (supplementary 7.5), JIGSAWS: **kinematics-only 14-dim** input reaches Acc 86.4 /
     Edit 90.5, while the best **video-only** baseline (TDRN) gets Acc 84.6 / Edit 90.2 and MS-TCN (video) gets
     Acc 78.9 / Edit 85.8. The paper's own summary: "our model outperforms all previous video-based algorithms
     and remains competitive with multi-modal networks." https://arxiv.org/abs/2303.07814
     *(Caveat: `*`-marked rows use uncorrected JIGSAWS labels — the authors flag this.)*
   - **TCN (workshop)**, Table 1: on 50 Salads the **~7–21-channel accelerometer** stream gives TCN Edit 65.6 /
     Acc 82.0 versus LC-SC-CRF 50.2 / 77.8 and LSTM 54.5 / 73.3. https://arxiv.org/abs/1608.08242
   - **M2R2**, ablation table (REASSEMBLE, DiffAct head): **proprioception-only** F1@50 74.5 / Acc 75.4 /
     EDIT 69.2 versus **vision-only** F1@50 21.6 / Acc 21.4 / EDIT 28.9. Adding vision to proprioception barely
     changes anything (all-modalities: F1@50 74.6 / Acc 74.9 / EDIT 68.7). The authors' explanation is that on
     REASSEMBLE "actions and objects can often be identified from proprioceptive cues alone", while vision-only
     "frequently confuse[s] object categories… because many REASSEMBLE objects are poorly visible due to their
     small size". https://arxiv.org/abs/2504.18662

2. **But a purely vision-derived feature can also collapse catastrophically — and this is the case closest to
   ours.** In M2R2's Table I (REASSEMBLE, fine-grain, 69 classes), all three vision-only modular pipelines score
   **F1@50 between 5.1 and 8.6** and Acc between 5.7 and 7.9, while multimodal reaches F1@50 80.8–82.4. The stated
   cause is loss of object identity under small-object / low-visibility conditions. Since our 144-dim vector is
   a *derived vision summary* (presence/count/max-area of detections per class×region), this is the most direct
   published warning that a detection-derived feature can be information-starved for object-identity
   discrimination — exactly the axis on which our model fails (51.6% of ground-truth segments wholly mislabelled).

3. **Object information measurably improves a low-dimensional temporal model in an industrial assembly setting.**
   **Aganian et al., IJCNN 2023** on IKEA ASM: appending object centres to skeleton joints raises mean-class
   accuracy and top-1 for both a 2D-CNN and PoseConv3D ("our approach massively improves the performance").
   https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2023/Aganian-IJCNN-2023.pdf
   *(Table cell alignment partially unverified — see that paper's entry.)*

4. **The one controlled feature-type swap found anywhere in this thread says hand-crafted ≥ deep.**
   **MS-TCN**, Table 10, Breakfast, identical architecture, only the input feature swapped:
   **IDT F1@10/25/50 = 58.2 / 52.9 / 40.8, Edit 61.4, Acc 65.1** versus **I3D 52.6 / 48.1 / 37.9, Edit 61.7,
   Acc 66.3**. Hand-crafted motion-only features win the segment-level metrics by +2.9 to +5.6 F1 while losing
   0.3 Edit and 1.2 Acc. The authors: *"the impact of the features is very small… the model achieves a better F1
   score when using the IDT features compared to I3D."* https://arxiv.org/abs/1903.01945
   *(Dataset-specific caveat: Breakfast is visually homogeneous, so appearance genuinely does not help there.)*

5. **Frame-level representation *content* governs segment-level metrics far more than model capacity does.**
   **Funke et al., MICCAI 2019**, Table 2, JIGSAWS suturing (39 videos), same 3D-ResNet-18 family, comparing a
   purely spatial per-frame representation (2D ResNet-18 on single frames) against a spatio-temporal one
   (3D CNN over 16-frame snippets): **frame accuracy is identical — 79.9 vs 79.9 — yet Edit moves 41.4 → 64.0
   (+22.6) and F1@10 moves 55.4 → 75.2 (+19.8)**. Adding a sliding window lifts Edit further, to 80.7.
   https://arxiv.org/abs/1907.11454
   *This is the strongest single signal that our architecture work may be misdirected: our numbers (Edit 51.08,
   F1@0.1 40.01) sit in the band of the **spatially-only** per-frame representation, and our 144-dim vector is
   likewise instantaneous — it carries no motion/temporal derivative.*

6. **In the small-data, low-dim regime, changing architecture and augmentation moved a segment-level metric far
   more than the input representation could.** **MS-TCRNet** on BRS (52 sequences, 36 kinematic variables):
   MS-TCN++ Edit **45.8** → G-MS-TCRNet Edit **75.8**, F1@50 **37.9 → 57.6**, with the input unchanged. The paper
   states MS-TCN++ "has significant difficulty with segmental metrics on this dataset". https://arxiv.org/abs/2303.07814

7. **Direct data augmentation of labelled segment sequences roughly doubled segment-level F1 from a 2-sequence
   training set** (few-shot skeleton TAS, Table 2): Tai Chi F1@10 0.735 → 0.926, UTD-MHAD2 F1@10 0.603 → 0.837,
   with CTC adding little. https://arxiv.org/abs/2207.09925

8. **Explicitly-flagged contrary signal (assertion, not measurement).** TricorNet states "Since our approach uses
   simpler spatial features as [10], it is **very likely** to improve our result by using the state-of-the-art
   features", and TCN's related work notes that on GTEA the best accuracy came from EgoNet+TDD features (Acc 68.5)
   versus their own learned spatial features. These are *hypotheses voiced by the authors*, not controlled
   feature-swap ablations. **They should not be cited as evidence that our features are the bottleneck** — and
   the one real controlled swap (item 4) contradicts them.

### Net reading for the project

The literature does **not** support "low-dimensional non-visual features are a known dead end" — the opposite.
Published TAS systems reach high segment-level scores on **14–36-dim kinematics** and **~7–21-channel
accelerometry**, and beat video-feature baselines on the same data (items 1). The one controlled feature-type
swap that exists found **IDT ≥ I3D** on segment-level F1 (item 4). Meanwhile the largest controlled gains in the
verified record came from:

- **giving the per-frame representation temporal content** (item 5: Edit +22.6 at *identical* frame accuracy);
- **architecture + augmentation changes** in the small-data low-dim regime (item 6: Edit +30 on a 52-sequence
  dataset with an unchanged 36-dim input);
- **segment-level data augmentation** from tiny training sets (item 7: F1@10 nearly doubled from 2 sequences).

So the evidence points at **three concrete, cheap experiments** rather than at replacing our feature pipeline:
1. **Add temporal context to the 144-dim vector** (frame deltas / short-window derivatives), the single
   best-motivated change, directly targeted at our Edit and F1@k failure.
2. **Port MS-TCRNet's intra-stage regularization + a different refinement sampling rate**, plus segment-level
   augmentation — because that is where a +30 Edit swing was actually demonstrated on a comparably small dataset.
3. **Only then** consider enriching the feature vector for object-identity signal (Aganian's appended object
   centres; M2R2's decoupled extractor) — motivated not by "features are the bottleneck" in general, but
   specifically by our 51.6% whole-segment identity-confusion failure, for which M2R2's vision-only collapse
   (F1@50 5.1–8.6 on small similar objects) is the closest published analogue.

**Do not** justify a move to I3D / deep video features on survey grounds: no verified evidence supports it, and
the only controlled test found argues against it.

---

*Files fetched for this thread are cached under `.tas_survey_tmp/raw/lowdim/` (arXiv abs/HTML pages, extracted
full texts, PDFs, GitHub API responses, and `pdftext.py`).*

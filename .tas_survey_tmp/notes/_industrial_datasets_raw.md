# Industrial / procedural-activity TAS datasets — strict-evidence raw notes

Session date: 2025-09-27. All facts below come from pages/PDFs **fetched in this session** with
`curl -sL -A "Mozilla/5.0"`. URLs are given per item.

**Method note (important):** the brief said `pdftotext` and `pypdf` are NOT installed. I verified this
in-session: `pdftotext`, `mutool`, `gs`, `qpdf`, `pdfinfo`, `fitz`, `pdfminer`, `pdfplumber` are all
MISSING, **but `pypdf` IS importable (`from pypdf import PdfReader` works)**. I therefore used
`pypdf` to extract the full text (including tables) of the CVF open-access PDFs. arxiv.org/abs/<id>
and arxiv.org/html/<id> were used for arXiv papers and both worked (no 429/406 encountered).
Raw fetched artifacts are kept in `.tas_survey_tmp/raw/`.

---

## 1. Assembly101

### Assembly101
- **Venue/year**: CVPR 2022 — verified from the CVF open-access landing page (author list, "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022") and the PDF first page header/footer ("21096" CVPR page number). arXiv id **2203.14712** verified by fetching the abs page (title matches exactly: "Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities").
  - **URL**: https://openaccess.thecvf.com/content/CVPR2022/papers/Sener_Assembly101_A_Large-Scale_Multi-View_Video_Dataset_for_Understanding_Procedural_Activities_CVPR_2022_paper.pdf
  - **URL (id verification)**: https://arxiv.org/abs/2203.14712
- **Size** (Sec. 4 "Dataset Statistics", Table 1, Table 2, Table 4, Sec. 3.2):
  - **Sequences / videos**: "Assembly101 features 362 disassembly-assembly sequences; each sequence is recorded from 12 viewpoints, totalling 4321 videos and 513 hours of footage." (Sec. 4.1)
  - **Hours**: 513.0 h total (Table 1 and Table 2); egocentric subset 167.0 h (Table 1, row "Assembly101 (ego)").
  - **Avg video length**: "The average sequence or video duration is 7.1±3.4 minutes" (Sec. 4.1); Table 1/2 report avg. 7.1 min.
  - **Temporal segments**: Table 1 (fine-grained) = **1,013,523** segments; Table 2 (coarse) = **104,759** segments. Abstract: "more than 100K coarse and 1M fine-grained action segments, and 18M 3D hand poses."
  - **Action classes**: "spanning 1380 fine-grained and 202 coarse action classes" (Sec. 1 / Abstract). Fine-grained = 90 objects (incl. 5 tools + "hand") × 24 interaction verbs → 1380 labels (Sec. 4.2). Coarse = "202 coarse actions composed of 11 verbs and 61 objects" (Sec. 4.3).
  - **Subjects/participants**: "We recruited 53 adults (28 males, 25 females) to disassemble and assemble 'take-apart' toy vehicles." (Sec. 3.2). Table 1 and Table 2 both list `#participants` = **53**.
  - **Other**: 101 unique toys from 15 categories; 8 static RGB (1920×1080) + 4 egocentric monochrome (640×480) cameras (Sec. 3.1); 21 3D keypoints per hand.
  - **Total frame count**: NOT stated in the paper. Table 1 reports a *percentage* column "labelled frames" = 81.4%, not a frame count. → **UNVERIFIED — searched the full extracted text of the CVPR PDF for a total-frame figure; the paper reports hours, videos and segments, never a total frame count.**
- **Input features** (baseline for temporal action segmentation, Sec. 5.1, p. 6):
  - Direct quote: *"We apply two competing state-of-the-art temporal convolutional networks: MS-TCN++ [24] and C2F-TCN [44], **using frame-wise features extracted from TSM [25] trained for action recognition on Assembly as input**."*
  - So: **pre-computed frame-wise TSM (Temporal Shift Module) features**, i.e. deep RGB-video CNN features — not RGB frames, not pose, not object detections, and not a sensor/IMU stream.
  - **Dimensionality**: **UNVERIFIED — the paper never states the TSM feature dimension.** I grepped the full extracted text for "2048", "1024", "512-dim", "dimensional", "dim": the only hit is the unrelated phrase "poses are low-dimensional common representations" (Sec. 5.5). The supplementary comparison table (Table 9 caption) also names only TSM, no dimension.
- **Metrics** (paper's own tables; F1@{10,25,50}, Edit = segmental edit distance, MoF = mean frame-wise accuracy):
  - **Table 7 — "Baselines of temporal action segmentation; unless specified, results are from C2F-TCN"**:
    - MS-TCN++ [24], all: F1@10 **31.6**, F1@25 **27.8**, F1@50 **20.6**, Edit **30.7**, MoF **37.1**
    - C2F-TCN [44], all: F1@10 **33.3**, F1@25 **29.0**, F1@50 **21.3**, Edit **32.4**, MoF **39.2**
    - Fixed views: 35.5 / 31.2 / 23.2, Edit 33.9, MoF 41.3; Egocentric: 28.7 / 24.4 / 17.5, Edit 29.2, MoF 34.8
    - Seen Disassembly: 35.8 / 31.1 / 22.2, Edit 31.7, MoF 39.8; Unseen Disassembly: 31.9 / 26.6 / 17.0, Edit 27.9, MoF 38.9
    - Seen Assembly: 33.0 / 28.6 / 22.7, Edit 30.0, MoF 42.5; Unseen Assembly: 29.9 / 26.2 / 19.8, Edit 32.0, MoF 34.8
  - **Table 9 — "Frame-wise features are extracted from TSMs pre-trained on various datasets. Action recognition is performed by TempAgg [38] trained on these features."** (verb / object / action Top-1):
    - Kinetics-400: 28.0 / 19.9 / **9.8**; SSv2: 28.7 / 18.8 / **10.2**; EPIC-KITCHENS-100: 44.0 / 25.2 / **17.3**; Assembly101: 65.9 / 50.5 / **40.5**; 3D pose MS-G3D w/ context: 65.7 / 36.3 / 28.7
    - Related in-text statement: *"TSM features pre-trained on EPIC-KITCHENS perform significantly better than the other datasets; though there is still a gap of 23.2% compared to pre-training on the native Assembly101."* (Sec. 5.5)
  - **Table 8 — Action recognition on 3D hand poses** (verb/object/action): 2s-AGCN 58.1/30.9/22.2; 2s-AGCN w/ context 64.4/33.9/26.7; MS-G3D w/ context 65.7/36.3/28.7; TSM egocentric (fuse 4 views) 59.0/46.5/33.8; Object GT 28.1/98.8/27.2.
  - Cross-check: EAST (below) reproduces C2F-TCN on Assembly101 as F1@10/25/50 = 33.3/29.0/21.3, Edit 32.4, Acc 39.2 — identical to Assembly101 Table 7, confirming both readings.
- **Verification notes**: **fetched** (CVF landing page HTML + CVF PDF via pypdf + arXiv abs 2203.14712). Total frame count: UNVERIFIED (not reported).

---

## 2. IKEA ASM

### IKEA ASM
- **Venue/year**: WACV 2021, pp. 847–859 — verified from the CVF landing page (explicit "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 847-859"). arXiv id **2007.00394** verified by fetching the abs page (title matches: "The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose").
  - **URL**: https://www.openaccess.thecvf.com/content/WACV2021/html/Ben-Shabat_The_IKEA_ASM_Dataset_Understanding_People_Assembling_Furniture_Through_Actions_WACV_2021_paper.html
  - **URL (id verification)**: https://arxiv.org/abs/2007.00394
- **Size** (Sec. 3 "The IKEA assembly dataset", subsection "Statistics"):
  - **Videos**: "The IKEA ASM dataset consists of 371 unique assemblies of four different furniture types (side table, coffee table, TV bench, and drawer) in three different colors (white, oak, and black). There are in total **1113 RGB videos and 371 depth videos** (top view)."
  - **Frames**: "Overall, the dataset contains **3,046,977 frames (~35.27h)** of footage with an average of **2735.2 frames per video (~1.89min)**."
  - **Action classes**: **33 atomic actions** — "annotations of 33 atomic actions" (Fig. 1 caption); "Each action class contains at least 20 clips"; "The dataset contains a total of **16,764 annotated actions** with an average of 150 frames per action (~6sec)."
  - **Subjects/participants**: "To collect our IKEA ASM dataset, we ask **48 human subjects** to assemble furniture in **five different environments**, such as offices, labs and family homes." (Data collection). (Also corroborated by Assembly101 Table 1 which lists IKEAASM `#participants` = 48 — secondary cross-check.)
  - **Capture**: three Kinect V2 cameras (front/side/top), synchronized, ~24 fps; 10 camera configurations (2 per environment).
  - **Splits**: "The trainset and testset consist of **254 and 117 scans**, respectively."
- **Input features** (Sec. 4.1 "Action recognition"):
  - Direct quote: *"We compare several state-of-the-art methods for action recognition, including **I3D [8], P3D ResNet [52], C3D [69], and frame-wise ResNet [23]**. For each we start with a pre-trained model and **fine-tune it on the IKEA ASM dataset** using parameters provided in the original papers."*
  - So: **raw RGB video clips fed to end-to-end fine-tuned 3D/2D CNNs** (I3D, P3D, C3D) and **single-frame 2D ResNets**. NOT pre-computed frozen features. Sec. 4.2 additionally uses **human pose** (HCN, ST-GCN) and **depth** streams.
  - **Dimensionality**: **UNVERIFIED — no feature dimensionality is stated anywhere for any baseline.** (Full-text grep for "2048"/"1024"/"dimensional"/"dim": no feature-dimension statements.)
- **Metrics** (paper's own tables; all values are frame-wise (%) for frame-wise accuracy / top-1 / top-3 / macro-recall / mAP):
  - **Table 2 — "Action recognition baseline frame-wise accuracy, macro-recall, and mean average precision results."**
    - ResNet18: 27.06 / 55.14 / 21.95 / 11.69
    - ResNet50: 30.38 / 56.1 / 20.03 / 9.47
    - C3D: 45.73 / 69.56 / 32.48 / 21.98
    - P3D: **60.4** / 81.07 / 45.21 / 29.86
    - I3D: 57.58 / 76.72 / 39.03 / 28.77
  - **Table 3 — "Action recognition frame-wise accuracy, macro-recall, and mean average precision results for multi-view/modal inputs."**
    - RGB top view: 57.58 / 76.72 / 39.03 / 28.77; front view: 60.75 / 79.3 / 42.67 / 32.73; side view: 52.16 / 72.21 / 36.59 / 26.76; combined views: 63.09 / 80.54 / 45.23 / 32.37
    - Human pose HCN: 39.15 / 65.37 / 28.18 / 22.32; ST-GCN: 43.4 / 66.29 / 26.54 / 18.56
    - combined RGB+pose: **64.25** / 80.58 / 46.33 / 33.08; Depth top view: 35.43 / 59.48 / 21.37 / 14.4; combined all: 64.02 / 81.45 / 44.61 / 31.45
  - In-text: *"I3D, for example, has an FA score of 57.57% compared to 68.4% on Kinetics and 63.64% on Drive&Act dataset."*
  - **No temporal action segmentation benchmark**: the paper's benchmarked tasks are frame-wise action recognition, object instance segmentation/tracking, and human pose estimation. It reports **no F1@{10,25,50} / edit-distance segmentation metrics**. (Full-text grep for "F1@", "temporal action segmentation", "edit distance", "MS-TCN": no hits.)
- **Verification notes**: **fetched** (CVF landing page HTML + CVF PDF via pypdf + arXiv abs 2007.00394). Feature dimensionality: UNVERIFIED (not reported).

---

## 3. MECCANO

### MECCANO
- **Venue/year**: WACV 2021 (CVF open-access paper). arXiv id **2010.05654** verified by fetching the abs page (title matches: "The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain"). A later extended/multimodal version exists as **arXiv:2209.08691** ("MECCANO: A Multimodal Egocentric Dataset for Humans Behavior Understanding in the Industrial-like Domain"), verified by fetching its abs page.
  - **URL**: https://www.openaccess.thecvf.com/content/WACV2021/papers/Ragusa_The_MECCANO_Dataset_Understanding_Human-Object_Interactions_From_Egocentric_Videos_in_WACV_2021_paper.pdf
  - **URL (id verification)**: https://arxiv.org/abs/2010.05654 ; https://arxiv.org/abs/2209.08691
- **Size** (Table 1, Table 2, Sec. 3.1, Sec. 3.2):
  - **Participants**: "MECCANO has been acquired by **20 participants**" (Abstract); "MECCANO was collected by 20 different participants in **two countries (Italy and United Kingdom)**" (Sec. 1); Table 1 `Participants` = **20**.
  - **Videos/sequences**: **20** — "Each video corresponds to a complete assembly of the toy model starting from the 49 pieces placed on the table." (Sec. 3.1); Table 1 `Sequences` = 20.
  - **Frames**: **299,376** (Table 1 `Frames`). Also: "we labeled a total of **64349 frames**" for active-object bounding boxes (Sec. 3.2, Stage 2 — this is the number of *bbox-annotated* frames, not the video total).
  - **Duration**: avg 20.79 min (Table 1); "The average duration of the captured videos is **21.14 min**, with the longest one being 35.45 min and the shortest one being 9.23min." (Sec. 3.1)
  - **Action classes**: **61** (Table 1 `Action classes` = 61; "Starting from the temporal annotations, we defined **61 action classes**. Each action is composed by a verb and one or more objects, for example 'align screwdriver to screw'" — Sec. 3.2). Verb vocabulary = **12** verbs (take, put, check, browse, plug, pull, align, screw, unscrew, tighten, loosen, fit). Object classes = **20** (Table 1).
  - **Segments**: **8857** annotated video segments (Sec. 3.2: "With this procedure, we annotated 8857 video segments."); 1401 (15.82%) of them overlap with another segment.
  - **Object bboxes**: 64,349 (Table 1 `Object BBs`).
  - **Capture**: Intel RealSense SR300 head-mounted, 1920×1080 @ **12 fps** (Sec. 3.1). Splits (Table 2): Train 11 videos / 55%, Val 2 / 10%, Test 7 / 35%.
- **Input features** (Sec. 4.1 "Action Recognition" → "Baselines"):
  - Direct quote: *"We considered 2D CNNs as implemented in the **PySlowFast library [20] (C2D), I3D [7] and SlowFast [23] networks**, which are state-of-the-art methods for action recognition. In particular, for all baselines we used the PySlowFast implementation based on a **ResNet-50 [32] backbone pre-trained on Kinetics [33]**."*
  - So: **trimmed RGB video segments classified end-to-end by 3D CNNs (C2D / I3D / SlowFast, ResNet-50 backbone)**. No pre-computed frozen feature vectors, no pose/skeleton, no sensor/IMU.
  - **Dimensionality**: **UNVERIFIED — no feature dimensionality is stated.** (Full-text grep for "2048"/"1024"/"dimensional": no feature-dimension statement.)
- **Metrics** (paper's own table):
  - **Table 3 — "Baseline results for the action recognition task."** Columns: Top-1 Acc / Top-5 Acc / Avg Class Precision / Avg Class Recall / Avg Class F1:
    - C2D [20]: 41.92 / 71.95 / 37.6 / 38.76 / 36.49
    - I3D [7]: 42.51 / 72.35 / 40.04 / 40.42 / 38.88
    - SlowFast [23]: **42.85** / **72.47** / **42.11** / **41.48** / **41.05**
  - In-text: *"SlowFast obtained the best results with a Top-1 accuracy of 47.82 and an F1-score of 41.05."* — NOTE: the "47.82 Top-1" in the text does **not** match the Table 3 SlowFast entry (42.85); quoting it verbatim as the paper's own wording, but the table value is 42.85.
  - Other tables (not TAS): Table 4 active-object detection (best AP(IoU>0.5) 38.14% for Hand Object Detector + Objs re-training (All dist.)); Table 5 active object recognition.
  - **No temporal action segmentation (frame-wise) benchmark**: MECCANO's four tasks are Action Recognition (on trimmed segments), Active Object Detection, Active Object Recognition, EHOI Detection. No F1@{10,25,50} / edit distance. (Full-text grep: no "F1@", no "MS-TCN", no "temporal action segmentation".)
- **Verification notes**: **fetched** (CVF PDF via pypdf; arXiv abs 2010.05654 and 2209.08691 for id verification). Feature dimensionality: UNVERIFIED (not reported).

---

## 4. M2R2 (arXiv 2504.18662) — **id verified correct**

### M2R2
- **Venue/year**: arXiv preprint, **arXiv:2504.18662v3** (v3 dated 29 Apr 2026 per the HTML header "arXiv:2504.18662v3 [cs.RO] 29 Apr 2026"). Title verified: **"M2R2: MultiModal Robotic Representation for Temporal Action Segmentation"** — **the given id 2504.18662 is CORRECT**. (Note: the ar5iv search-result snippet rendered it as "MulitModal" — that is a typo in the snippet, not the title.) Authors: Daniel Sliwowski, Dongheui Lee (TU Wien / DLR). No journal reference or venue comment is present on the abs page.
  - **URL (id verification)**: https://arxiv.org/abs/2504.18662
  - **URL (full text)**: https://arxiv.org/html/2504.18662
- **Type**: this is a **METHOD/feature-extractor paper, not a dataset paper**. Its "dataset sizes" are those of the three datasets it evaluates on (its own text, Sec. IV-A):
  - **REASSEMBLE [22]**: *"The dataset consists of **149 recordings** with a median of **36 actions per recording** and strong visual similarity between objects... In total, the REASSEMBLE dataset contains **69 fine-grain labels** (activity+object, e.g. Insert USB) and **4 coarse-level labels** where the object names are discarded (activity, e.g. Insert)."*
  - **(Im)PerfectPour [2]**: *"contains **544 demonstrations** of a robot performing bartending tasks."*
  - **JIGSAWS [23]**: *"is a surgical gesture recognition dataset. Following prior work, we use the **Suturing** task with a **Leave-One-User-Out** evaluation."*
  - **Cross-check against REASSEMBLE's own dataset paper** (arXiv:2502.05086, fetched): *"The dataset contains **4,551 demonstrations, of which 4,035 were successful, spanning a total of 781 minutes**."*; *"REASSEMBLE includes **four actions** (pick, insert, remove, and place) involving **17 objects**."*; *"The REASSEMBLE dataset includes four actions and 17 different objects, resulting in a total of 68 unique action-object pairs, or **69 when including the 'Idle' action**."* — the 69-label figure matches M2R2 exactly. The **"149 recordings" grouping is NOT stated in REASSEMBLE's own paper** → **UNVERIFIED in the primary dataset source** (verified only as M2R2's own description).
  - Number of subjects/participants: **UNVERIFIED — M2R2 never reports a subject/participant count for any of the three datasets, and REASSEMBLE is a robot-teleoperation dataset (no human subjects reported in its abstract/experiment text).**
- **Input features** (Sec. III-C "M2R2 Feature Extraction"; Sec. IV-C "Baselines"):
  - M2R2 fused multimodal feature, **512-dim**, quoted verbatim: *"**Images:** We use a Visual Transformer initialized with ActionCLIP weights [13], to produce the image representation I_i ∈ R^{D_e}, where **D_e = 512 is the M2R2 embedding size**."* ; *"...proprioceptive features... we project the sensor data into a D_e-dimensional space using a learned matrix W_p^s..."* ; output stored as *"a feature set X ∈ R^{T×D_e}, where T is the number of frames and D_e is the M2R2 feature dimension."*
  - Modalities fused: RGB (ActionCLIP ViT), audio (Audio Spectrogram Transformer, 16 kHz, 64 mel filter banks), proprioception (force-torque, end-effector pose and twist, gripper width; resampled to f = 300 Hz).
  - Vision-only baseline: *"We adopt **Br-Prompt** as our baseline vision-only extractor... We train Br-Prompt with its default hyperparameters, **sampling 16-frame windows** during training."* (Paper also notes "*two commonly used models in TAS are I3D [12] and Br-Prompt [14]*".)
  - Boundary/heuristic baselines: **BOCPD** (6-DoF wrench) and **AWE** (position trajectories); deep TAS heads: **MSTCN, ASRF, DiffAct**.
  - Explicit hand-crafted remark (Motivation): *"Supervised methods rely on labelled data... These methods rely on **handcrafted features such as power and contact duration** [5]."* and *"Others depend on **handcrafted features that do not scale well to diverse tasks** [17]."*
- **Metrics** (paper's own tables; per-frame accuracy "Acc", EDIT, DR = detection rate, F1@{10,25,50}):
  - **TABLE I — REASSEMBLE** (fine-grain labels): BRP+MSTCN (vision-only) F1@10/25/50 = **12.9 / 10.9 / 8.4**, Acc 7.9, EDIT 18.4, DR 18.5; BRP+ASRF 12.8/11.1/8.6, Acc 6.4, EDIT 19.4, DR 24.8; BRP+DiffAct 12.0/10.1/5.1, Acc 5.7, EDIT 21.4, DR 19.9; **M2R2+ASRF 83.5 / 83.4 / 82.4**, Acc 82.7, EDIT 82.5, DR 95.1; M2R2+MSTCN 83.1/82.7/80.8, Acc 82.4, EDIT 79.3, DR 89.5; M2R2+DiffAct 78.1/77.7/74.6, Acc 74.9, EDIT 68.7, DR 81.4. (Coarse-label columns also present.)
  - **TABLE II — (Im)PerfectPour**: BRP+MSTCN F1@10/25/50 = 81.0/73.9/63.6, Acc 76.8, EDIT 88.9, DR 28.8; BRP+ASRF 84.1/82.1/73.6, Acc 77.7, EDIT 91.2, DR 28.3; BRP+DiffAct 81.6/79.9/71.8; M2R2+DiffAct 90.5/89.7/89.7, Acc 93.3, EDIT 83.0, DR 54.0; M2R2+MSTCN 92.6/89.8/84.6, Acc 93.9, EDIT 88.9, DR 70.5.
  - **TABLE III — JIGSAWS**: Lea et al. [16] Acc 74.2, EDIT 66.5 (F1 n/r); Weerasinghe et al. [17] 87.3/86.5/81.1, Acc 87.1, EDIT 83.9; BRP+DiffAct 59.6/49.6/30.2, Acc 44.5, EDIT 81.9, DR 43.3; **M2R2+DiffAct 94.3 / 93.0 / 89.4**, Acc 87.1, EDIT 91.7, DR 84.5.
  - **TABLE IV — modality ablation, REASSEMBLE, DiffAct** (fine-grain): only vision 27.2/26.1/21.6, Acc 21.4, EDIT 28.9, DR 44.3; only audio 39.2/38.7/36.4, Acc 34.5, EDIT 37.5, DR 80.1; only proprio 78.0/77.5/74.5, Acc 75.4, EDIT 69.2, DR 80.3; all 78.1/77.7/74.6, Acc 74.9, EDIT 68.7, DR 82.4.
  - In-text claims: *"M2R2 with ASRF... attaining an F1@50 score of 82.4%, surpassing the unsupervised robotic TAS models, by at least 46.6 percentage points."* ; *"Among all cases, the TAS trained with vision-only features perform the worst, with F1@50 scores ranging from 5.1% to 8.4%."* ; *"Our method outperforms vision-only baselines by 16.1% in F1@50"* [on (Im)PerfectPour].
- **Verification notes**: **fetched** (arXiv abs 2504.18662 = id verification; arXiv html 2504.18662 = full text incl. Tables I–IV; arXiv abs 2502.05086 + arXiv html 2502.05086 = REASSEMBLE cross-check). Subject/participant counts: UNVERIFIED. "149 recordings": UNVERIFIED in REASSEMBLE's own paper.

---

## 5. Feature-representation vs temporal-architecture evidence

**Bottom line: I found NO TAS paper that reports "replace hand-crafted / low-dimensional features with deep video features (I3D etc.)" producing a large gain. I found one paper that ran exactly that swap and got NO gain, and a survey that states the comparison has never been done. The large, controlled gains that do exist come from (a) end-to-end fine-tuning of the backbone rather than swapping a frozen extractor, and (b) adding spatiotemporal/multimodal information to a per-frame representation.**

### 5a. DIRECT COUNTER-EVIDENCE: hand-crafted (IDT) vs deep (I3D), same model — no gain
- **Paper**: MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation, **arXiv:1903.01945v2** (version string read from the fetched page).
  - **URL**: https://arxiv.org/html/1903.01945
- **Setup**: Sec. 4.8/4.9 — *"In our experiments, we use the I3D features without fine-tuning."* ; *"Note that all the reported results are obtained using the I3D features. To analyze the effect of using a different type of features, we evaluated our model on the Breakfast dataset using the **improved dense trajectories (IDT) features, which are the standard used features for the Breakfast dataset**."*
- **Numbers — Table 10, Breakfast section** (F1@10 / F1@25 / F1@50 / Edit / Acc):
  - **MS-TCN (IDT)** = **58.2 / 52.9 / 40.8 / 61.4 / 65.1**  ← hand-crafted features
  - **MS-TCN (I3D)** = **52.6 / 48.1 / 37.9 / 61.7 / 66.3**  ← deep features
- **The paper's own conclusion (verbatim)**: *"As shown in Table 10, **the impact of the features is very small**. While the frame-wise accuracy and edit distance are slightly better using the I3D features, **the model achieves a better F1 score when using the IDT features compared to I3D**. This is mainly because I3D features encode both motion and appearance, whereas the IDT features encode only motion."*
- **Reading**: replacing hand-crafted IDT with deep I3D on Breakfast changed F1@10 by **−5.6 points** (worse), F1@50 by −2.9 (worse), Edit +0.3, Acc +1.2. This is the opposite of a large gain. **This is the single most direct TAS test of the hypothesis and it fails it.**

### 5b. The TAS survey states the comparison has never been done
- **Paper**: Temporal Action Segmentation: An Analysis of Modern Techniques, **arXiv:2210.10352v5** (TPAMI survey; version read from the fetched page).
  - **URL**: https://arxiv.org/html/2210.10352
- **Sec. "Input Features" (verbatim)**: *"The majority works on TAS typically take visual feature vectors, **either hand-crafted (IDT) [73] or extracted from an off-the-shelf CNN backbone (I3D) [75]**, as input for each frame. Using pre-computed inputs is conventional practice for several other tasks as well... Nonetheless, as pointed out by [161, 162], pre-computed characteristics tend to favour static cues, e.g., scene components, within frames. **To the best of our knowledge, no empirical research has compared utilizing pre-computed features to training TAS models from raw images end-to-end due to the high demands in terms of training efficiency and GPU memory requirements.**"*
- The survey also defines IDT explicitly: *"The original and Improved Dense Trajectories (IDT) [72, 73] were **commonly used hand-crafted features** for action recognition and video understanding before the rise of deep learning."*

### 5c. PARTIAL POSITIVE: swapping the frozen feature extractor (deep→better deep), same frozen-backbone TAS model
- **Paper**: End-to-End Action Segmentation Transformer (EAST), **arXiv:2503.06316v3**. (arXiv preprint; the abs page carries no journal reference/venue comment.)
  - **URL**: https://arxiv.org/html/2503.06316
- **Dimensionality quote (verbatim)**: *"previous work uses pre-computed **I3D [6] or TSM [23] features with a dimensionality of 2048**"*; EAST's fine-tuned ViT-G frame features *"have a dimensionality of **1408**"*.
- **Method quote**: *"The SOTA methods use precomputed **I3D features** [6] for Breakfast and **TSM features** [23] for Assembly101. To ensure consistency in the frame features across the SOTA methods and EAST, we extracted **MAEv2 frame features** for all the methods, including ours, using the ViT-G backbone pretrained with VideoMAEv2 [36]."*
- **Table 1 — "Impact of feature representation on Breakfast"** (F1@10 / F1@25 / F1@50 / Edit / Acc), four-split averages, exact feature swap with the model held fixed:
  - MSTCN: I3D **52.6 / 48.1 / 37.9 / 61.7 / 66.3** → MAEv2 **59.9 / 55.1 / 43.9 / 65.0 / 68.5**  (F1@10 **+7.3**)
  - ASFormer: I3D 76.0 / 70.6 / 57.4 / 75.0 / 73.5 → MAEv2 78.7 / 73.6 / 60.8 / 76.8 / 75.0  (+2.7)
  - LTContext: I3D 77.6 / 72.6 / 60.1 / 77.0 / 74.2 → MAEv2 80.6 / 75.7 / 64.1 / 75.2 / 76.6  (+3.0)
  - FACT: I3D 81.4 / 76.5 / 66.2 / 79.7 / 76.2 → MAEv2 80.6 / 75.9 / 65.3 / 78.9 / 77.2  (**−0.8, worse**)
  - EAST (end-to-end, MAEv2 ViT-G): **85.6 / 81.5 / 71.6 / 83.5 / 82.2**
- **Table 2 — "Impact of feature representation on Assembly101"** (same frozen-backbone model, feature swap):
  - LTContext: **TSM 33.9 / 30.0 / 22.6 / Edit 30.4 / Acc 41.2** → **MAEv2 31.3 / 27.9 / 21.1 / Edit 27.8 / Acc 40.3**  (**worse on every metric**)
  - EAST: MAEv2 **42.3 / 39.4 / 32.8 / Edit 39.9 / Acc 48.4**
  - Paper's own wording: *"Tab. 1 shows that the SOTA methods **improve performance on Breakfast** when using precomputed MAEv2 features, while **LTContext with MAEv2 features fails to do so on Assembly101**. On both datasets, **end-to-end trained EAST** using the same pretrained ViT-G **achieves superior performance**."*
- **Table 13 — Action segmentation on Assembly101** (frozen backbones vs fine-tuned): MS-TCN++ 31.6/27.8/20.6, Edit 30.7, Acc 37.1; UVAST 32.1/28.3/20.8, Edit 31.5, Acc 37.4; ASFormer 33.4/29.2/21.4, Edit 30.5, Acc 38.8; C2F-TCN 33.3/29.0/21.3, Edit 32.4, Acc 39.2; LTContext 33.9/30.0/22.6, Edit 30.4, Acc 41.2; **ViT-G fine-tuned = EAST 42.3/39.4/32.8, Edit 39.9, Acc 48.4**. In-text: *"On Assembly101, EAST surpasses previous methods by **7.2 points in accuracy and 9.5 points in Edit score**. Furthermore, EAST demonstrates F1@50 score gains of **7.0, 3.6, 5.2, and 10.2 percentage points on the GTEA, 50Salads, Breakfast, and Assembly101** datasets, respectively."*
- **Reading**: swapping a *frozen* extractor is worth **+2.7…+7.3 F1@10 on Breakfast for 3 of 4 models, −0.8 for the 4th, and a uniform loss on Assembly101**. The large gain (Assembly101 F1@50 22.6 → 32.8) comes from **end-to-end fine-tuning the backbone**, not from swapping the frozen extractor.

### 5d. PARTIAL POSITIVE: per-frame spatial representation → spatiotemporal representation (surgical TAS)
- **Paper**: Funke et al., "Using 3D Convolutional Neural Networks to Learn Spatiotemporal Features for Automatic Surgical Gesture Recognition in Video", **arXiv:1907.11454** (MICCAI 2019).
  - **URL (id verification)**: https://arxiv.org/abs/1907.11454
  - **URL (full text)**: https://arxiv.org/html/1907.11454
- **Paper's own claim (abstract, verbatim)**: *"**Previous approaches mainly rely on frame-wise feature extractors, either handcrafted or learned, which fail to capture the dynamics in surgical video.** To address this issue, we propose to use a 3D Convolutional Neural Network (CNN) to learn spatiotemporal features from consecutive video frames... Our approach achieves high frame-wise surgical gesture recognition accuracies of more than 84%, **outperforming comparable models that either extract only spatial features or model spatial and low-level temporal information separately**."*
- **Dataset**: JIGSAWS suturing — *"We evaluate our approach on **39 videos** of robot-assisted suturing tasks... performed by **eight participants**... In total, G = 10 different gestures are used."*
- **Numbers — Table 2, "Experimental results on the suturing task of JIGSAWS"** (Acc / Avg F1 / Edit / F1@10), evaluation at 5 fps:
  - 2D ResNet-18 (per-frame spatial features): **79.9 / 73.3 / 41.4 / 55.4**
  - 3D CNN (B) (spatiotemporal): **79.9 / 73.7 / 64.0 / 75.2**  → **Edit +22.6, F1@10 +19.8 at identical frame accuracy**
  - 3D CNN (K): 81.8 / 75.8 / 58.7 / 71.1
  - 3D CNN (B) + window (3 s look-ahead): **84.0 / 78.4 / 80.7 / 87.2**
  - 3D CNN (K) + window: 84.2 / 78.4 / 80.0 / 87.1
  - Evaluation at 10 fps: 2D ResNet-18 79.5 / 73.1 / 30.6 / 44.2 ; 3D CNN (B) 79.5 / 73.6 / 49.5 / 62.8 ; 3D CNN (B)+window 84.0 / 78.6 / 80.6 / 87.0
- **Reading**: this is a large, clean gain from changing the *frame-level representation* (single-frame spatial → 16-frame spatiotemporal) **while frame-wise accuracy stays identical (79.9 → 79.9)** — i.e. the segment-level bottleneck is representation (dynamics), not the temporal head. Caveat: both representations are *learned* (2D vs 3D CNN), and the 3D CNN is trained end-to-end, not a frozen hand-crafted-vs-deep feature swap. The paper's Table 2 contains **no hand-crafted baseline numbers** — hand-crafted methods are only named in related work (*"early approaches compute bag-of-features histograms from feature descriptors extracted around space-time interest points or dense trajectories [13]"*), not evaluated.

### 5e. Dataset-paper evidence: frozen feature *source* matters a lot (Assembly101, this session's Table 9)
- Same architecture (TempAgg), same modality, only the pre-training corpus of the frozen TSM features changed — Assembly101 Table 9, action Top-1: Kinetics-400 **9.8** → SSv2 **10.2** → EPIC-KITCHENS-100 **17.3** → Assembly101 **40.5**. A **23.2-point** swing from the feature representation alone (paper's own wording: *"there is still a gap of 23.2% compared to pre-training on the native Assembly101"*). See §1 above.

---

## Consolidated UNVERIFIED list

1. **UNVERIFIED — Assembly101 total frame count.** Tried: full-text extraction of the CVPR 2022 PDF (pypdf) and greps for "frames", "labelled frames", "4321", "513". The paper reports 4,321 videos / 513 h / 1,013,523 fine + 104,759 coarse segments and an 81.4% *labelled-frames percentage*, but never a total frame count.
2. **UNVERIFIED — Assembly101 TSM feature dimensionality** for the Table 7 TAS baselines. Tried: full-text greps for "2048", "1024", "512", "dim", "dimensional". The paper names TSM but never its dimension. (For context, EAST — a different paper, fetched — states the *conventional* TAS feature dimension: *"previous work uses pre-computed I3D [6] or TSM [23] features with a dimensionality of 2048"*, but that is not Assembly101's own statement.)
3. **UNVERIFIED — IKEA ASM baseline feature dimensionality.** Tried: full-text greps of the WACV 2021 PDF. Baselines are end-to-end fine-tuned CNNs; no dimension (and no frozen-feature representation) is reported.
4. **UNVERIFIED — MECCANO baseline feature dimensionality.** Tried: full-text greps of the WACV 2021 PDF. Baselines are PySlowFast C2D/I3D/SlowFast with a ResNet-50 backbone trained end-to-end; no feature dimension reported.
5. **UNVERIFIED — M2R2 participant/subject counts** for REASSEMBLE / (Im)PerfectPour / JIGSAWS. Tried: full-text of arXiv:2504.18662 HTML (Sec. IV-A and all tables). M2R2 reports no human-subject counts (REASSEMBLE is a robot-teleoperation dataset). JIGSAWS's 8 participants / 39 videos / 10 gestures are reported by **Funke et al. (arXiv:1907.11454)**, not by M2R2.
6. **UNVERIFIED — M2R2's "149 recordings" for REASSEMBLE, in the primary dataset source.** Tried: full-text of REASSEMBLE's own paper (arXiv abs + HTML 2502.05086); greps for "149". REASSEMBLE's paper reports 4,551 demonstrations (4,035 successful) / 781 minutes / 4 actions / 17 objects / 69 labels incl. Idle, but never "149 recordings". The 149 figure is verified **only as M2R2's own description** of the dataset.
7. **UNVERIFIED — a TAS paper showing that replacing hand-crafted/low-dimensional features with deep video features (I3D etc.) produces a large performance gain.** Tried: (a) fetched and read MS-TCN arXiv:1903.01945v2, which performs exactly this swap and reports the opposite (§5a); (b) fetched the TPAMI TAS survey arXiv:2210.10352v5, which states no such end-to-end-vs-precomputed comparison exists (§5b); (c) fetched EAST arXiv:2503.06316v3, whose controlled feature swaps give mixed/modest gains and a loss on Assembly101 (§5c); (d) fetched Funke et al. arXiv:1907.11454, which gives a large gain but from per-frame→spatiotemporal *learned* features, not hand-crafted→deep (§5d); (e) three web_search rounds with phrasings such as "hand-crafted features" + "I3D" + ablation / "we replace" / "same architecture different input features". No paper reporting a large hand-crafted→deep TAS gain was found. **Conclusion: no such evidence found; the one direct test found is negative.**
8. **UNVERIFIED — arXiv ids for the CVF-only items.** Assembly101 = 2203.14712, IKEA ASM = 2007.00394, MECCANO = 2010.05654 (and extended 2209.08691) were all **found via web_search and then verified by fetching arxiv.org/abs/<id> and matching the exact title**. Note that guessing was actively dangerous here: 2203.13899 (= "Exact Matching: Algorithms and Related Problems"), 2012.09471 (nuclear physics) and 2011.05005 (= "Deep Multimodal Fusion by Channel Exchanging") are all WRONG ids that a guessing approach would have used. A direct arXiv search API could not be used, as stated in the brief.

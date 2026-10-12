# DRAFT — sections owned by lead (to be merged)

## 0. Method & verification discipline

- Survey date: 2026-09-27.
- Primary sources fetched directly (curl): arXiv abs pages, arXiv/ar5iv HTML full text, CVF Open Access pages, AAAI OJS PDFs (parsed with pypdf).
- Venue independently re-verified through the Crossref REST API (`query.bibliographic`), not from memory or from search snippets.
- Naming corrections (claims I could NOT verify are stated as such): see §2.

### Independent venue verification (Crossref)

| Claim in the task | Verified reality | DOI |
|---|---|---|
| ASFormer (Transformer-based, 2021) | ASFormer: Transformer for Action Segmentation — **BMVC 2021** | 10.5244/c.35.49 |
| UVAST **2023** | "Unified Fully and **Timestamp** Supervised Temporal Action Segmentation via Sequence to Sequence Translation" — **ECCV 2022** (LNCS), not 2023 | 10.1007/978-3-031-19833-5_4 |
| DiffAct (2024/2025) | "Diffusion Action Segmentation" — **ICCV 2023**; journal extension "DiffAct++: Diffusion Action Segmentation" — **IEEE TPAMI 2025** | 10.1109/iccv51070.2023.00930 ; 10.1109/tpami.2024.3509434 |
| LTContext 2025 | "How Much Temporal Long-Term Context is Needed for Action Segmentation?" — **ICCV 2023** | 10.1109/iccv51070.2023.00950 |
| BaFormer | "Efficient Temporal Action Segmentation via Boundary-aware Query Voting" — **NeurIPS 2024** (Advances in NeurIPS 37) | 10.52202/079017-1192 |
| FactFormer | **No TAS method by this name found.** "FactFormer" = *Factorized Transformer for PDE surrogate modelling*, arXiv 2305.17560 — a different field. Closest real TAS method is **FACT** (CVPR 2024). | 10.1109/cvpr52733.2024.01721 |
| ICF (iterative contrastive/refinement) | **No TAS method named "ICF" found.** Closest real work: **ICC — Iterative Contrast-Classify** (AAAI 2022, arXiv 2112.01402). | — |
| C2F-TCN | "C2F-TCN: A Framework for Semi- and Fully-Supervised Temporal Action Segmentation" — **IEEE TPAMI 2023** (arXiv 2212.11078) | 10.1109/tpami.2023.3284080 |
| SSTCN / sparse hierarchical | **No TAS method named "SSTCN" found.** Nearest real: LTContext's sparse full-video attention (ICCV 2023); BIT bi-level (arXiv 2308.14900); "Sparse Temporal Pooling Network" (arXiv 1712.05080) is weakly-supervised localization, not TAS. | — |
| MS-TCN / MS-TCN++ | MS-TCN — CVPR 2019 (10.1109/cvpr.2019.00369); MS-TCN++ — IEEE TPAMI (10.1109/tpami.2020.3021756) | |
| Additional verified venues | Activity Grammars for TAS — NeurIPS 2023 (10.52202/075280-3296); "Do We Really Need Temporal Convolutions in Action Segmentation?" — ICME 2023 (10.1109/icme55011.2023.00178); DIR-AS — arXiv-only, venue unresolved; "Temporal Segment Transformer for TAS" — arXiv-only, no Crossref record; TPAMI survey "TAS: An Analysis of Modern Techniques" — TPAMI 2024 (10.1109/tpami.2023.3327284); Long-Tail TAS w/ Group-wise Temporal Logit Adjustment — ECCV 2024 (10.1007/978-3-031-73404-5_19); SMC-NCA — IEEE TMM 2024 (10.1109/tmm.2024.3452980); ASQuery — ICME 2024 (10.1109/icme57554.2024.10687535); D3TW — CVPR 2019 (10.1109/cvpr.2019.00366) | |

### Additional verified items discovered during the survey (not in the task list)

- **ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation** — NeurIPS 2024 (10.52202/079017-2855). Second diffusion-based TAS line.
- **Surgical Activity Recognition with Frame-Action Cross-Attention Temporal Modeling** — Journal of Medical Robotics Research, 2026 (10.1142/s2424905x26500029, Kamabattula, Perreault, Bhattacharyya). Applies **FACT** to Cholec80 (7 phases), AutoLaparo (7 phases), MultiBypass140 (12 phases / 46 steps). Domain-adjacent evidence that FACT transfers to ~7–12-class procedure segmentation.
- **IMPACT: A Dataset for Multi-Granularity Human Procedural Action Understanding in Industrial Assembly** — arXiv 2604.10409 (ACM MM 2026). 112 trials, 13 participants, 39.5 h, five-view RGB-D, real angle-grinder assembly/disassembly. Step-level TAS (26 step categories) evaluated with Accuracy / Edit / F1@{10,25,50}; baselines LTContext, ASQuery, DiffAct, FACT on frozen I3D and VideoMAE-v2 features. Contains a directly relevant negative finding: rare-phase imbalance is "a major bottleneck rather than model capacity alone" (recovery spans 2.22% of frames; F1 near zero).
- **Timestamp Query Transformer for Temporal Action Segmentation** — WACV 2026 (10.1109/wacv61042.2026.00487).

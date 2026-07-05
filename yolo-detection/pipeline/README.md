# yolo_pipeline · YOLO 检测数据/训练/评测流水线

内镜清洗巡检目标检测的**自包含**流水线:数据拉取 → 转 YOLO(**稳定切分**)→ 训练 → **验证/验收**。面向数据集 + 模型集维护同学。

**自包含**:输入、脚本、产物、依赖全部落在 `yolo_pipeline/` 目录内,不引用任何上级目录或外部环境。所有命令都在 `yolo_pipeline/` 下执行(脚本从这里 `import utils`)。

范围仅**目标检测**(Label Studio `videorectangle` bbox);`timelinelabels` 时序标签本期不处理,但 `splits.yaml` 设计成可被时序模型共用。

---

## 仓库结构

结构约定:**顶层 `0X_*.py` 是按顺序执行的编排脚本,`utils/` 是它们共用的工具函数(不单独运行)**。

```
yolo_pipeline/
  00_status.py           # 对账 / 增量前置:列出待办;--assign 回填 split
  01_pull_data.py        # 从 LS 下视频到 raw/videos/ + 完整性抽查
  02_build_dataset.py    # 转 YOLO,按 splits.yaml 整段路由,并打印样本分布
  03_train.py            # 各组训练
  04_validate.py         # 验证集指标 + 验收判定 + 报告
  config.yaml            # 中央配置:分组 / 白名单 / 抽帧 / 切分 / 超参 / 验收阈值 —— 改这里
  splits.yaml            # 视频 -> split 划分清单,稳定切分的唯一真源(入库)
  requirements.txt
  utils/                 # 编排脚本共用的工具函数
    common.py            # 定位根目录、加载 config、白名单判断
    lsexport.py          # LS 解析共享核心:fps 对齐、关键帧插值、坐标转换
    split.py             # 稳定切分逻辑(读写 splits.yaml)
    stats.py             # 样本分布统计(训练帧粒度,扫描落盘 label)
  raw/
    exports/             # LS 导出的 JSON(入库);脚本取文件名排序最后一份
    videos/              # 01_pull 下载的原始视频(不入库)
  datasets/<组>/         # 02_build 产出的 YOLO 数据集(不入库)
  runs/<组>/             # 03_train / 04_validate 的权重与报告(不入库)
  .venv/                 # 本项目虚拟环境(不入库)
```

入库的只有:脚本、`config.yaml`、`splits.yaml`、`requirements.txt`、`raw/exports/` 里的导出 JSON。`raw/videos/`、`datasets/`、`runs/`、`.venv/` 及 `*.pt`/`*.mp4` 均由 `.gitignore` 排除。

### 各部分功能定位

| 部分 | 定位 | 何时用 |
|------|------|--------|
| `00_status.py` | **对账中枢**:比对"导出 / 磁盘 / splits / 白名单"四方,列出待办;`--assign` 把已质检视频确定性回填 split | 每次开工、每次增量前先跑 |
| `01_pull_data.py` | **取数**:按导出 JSON 引用从 LS 服务器下视频到 `raw/videos/`,并做完整性抽查 | 有"未下载"视频时 |
| `02_build_dataset.py` | **转换**:导出 JSON + 视频 → 标准 YOLO 数据集,按 `splits.yaml` 整段路由;生成后打印样本分布与告警 | 数据/切分变化后重建 |
| `03_train.py` | **训练**:各组一套独立权重,device 自动选 MPS/CUDA/CPU | 数据集就绪、要出/更新权重时 |
| `04_validate.py` | **验收**:验证集跑指标、对照阈值判 PASS/FAIL、写报告,任一组 FAIL 退出码非零 | 训练后、交付卡口 |
| `config.yaml` | **唯一改动入口**:分组、白名单、抽帧、切分参数、超参、验收阈值全在这 | 调任何行为先改它,别改脚本 |
| `splits.yaml` | **切分的唯一真源**:视频 stem → split,人工可改、入库 | 手工调整 train/val/test 归属 |
| `utils/` | 编排脚本共用逻辑,尤其 `lsexport.py`(fps 对齐/插值/坐标)集中一份 | 改脚本时复用,别各写一套 |

**产物**(均不入库):`datasets/<组>/`、`runs/<组>/weights/best.pt`、`runs/<组>/acceptance_report.md`。

---

## 环境

本项目自带虚拟环境,不复用任何外部 venv。拉取/转换(`00`/`01`/`02`)只需 `cv2`、`numpy`、`pyyaml`;训练/验证(`03`/`04`)另需 `ultralytics`(含 torch,大件)。

```bash
cd yolo_pipeline
python3 -m venv .venv
# 只做数据(无需 torch):装前三个即可
.venv/bin/pip install opencv-python-headless numpy pyyaml
# 要训练/验证:装全部依赖(含 ultralytics/torch)
.venv/bin/pip install -r requirements.txt
```

下文用 `.venv/bin/python` 跑脚本(数据阶段用系统 `python3` 也行,只要装了前三个包)。

---

## 使用流程

### 场景一:生成数据集(拉数据 → 转 YOLO)

```bash
cd yolo_pipeline
export LS_HOST=http://<LS地址>:8080 LS_TOKEN=<AccessToken>
# 把 LS 导出 JSON 放进 raw/exports/
.venv/bin/python 01_pull_data.py       # 1. 下视频到 raw/videos/
.venv/bin/python 00_status.py          # 2. 看对账;人工质检合格的视频加进 config.only_videos
.venv/bin/python 00_status.py --assign # 3. 给已质检视频确定性回填 split(写回 splits.yaml,提交它)
.venv/bin/python 02_build_dataset.py   # 4. 转 YOLO 数据集(按视频切 train/val),并打印样本分布
```

产出 `datasets/<组>/`,含 `images/{train,val}`、`labels/{train,val}`、`data.yaml`。
`02_build` 会打印逐组 × 逐 split × 逐类的帧数/框数,并对"每视频尾部覆盖 < 80%""某类 val 无样本"等给出告警——务必扫一眼。

样本分布可随时独立重算(不重建):

```bash
.venv/bin/python -c "from utils import stats; stats.main()"
```

### 场景二:训练与评估

```bash
.venv/bin/python 03_train.py           # 各组训练;权重落 runs/<组>/weights/best.pt
.venv/bin/python 04_validate.py        # 验证集指标 + 验收报告 runs/<组>/acceptance_report.md
```

- 训练:`config.train.model`(默认 `yolo11n.pt`),**各组一套独立权重**;超参在 `config.train`(epochs 100 / imgsz 640 / batch 16 / patience 20)。
- 评估:在 **val** 上跑 `ultralytics.val`,取逐类 P/R/mAP 对照 `config.acceptance` 判 PASS/FAIL;**任一组 FAIL → 退出码非零**,可做交付卡口。

### 场景三:增量更新(有新导出/新视频时——每次这么走)

```bash
# 把新的 LS 导出 JSON 放进 raw/exports/
.venv/bin/python 00_status.py          # 看差异:未下载/未质检/未归属/遗失/孤儿
.venv/bin/python 01_pull_data.py       # 补下"未下载"
# 人工质检合格的,追加到 config.yaml 的 only_videos
.venv/bin/python 00_status.py --assign # 回填"未归属"(已有视频 split 不变 -> 天然增量)
.venv/bin/python 02_build_dataset.py   # 重建数据集
.venv/bin/python 04_validate.py        # (如重训了)重新验收
```

`00_status.py` 的分类与动作:

| 分类 | 含义 | 该做什么 |
|------|------|---------|
| 未下载 | 导出引用了但磁盘没有 | 跑 `01_pull_data.py` |
| 未质检 | 已下载但不在 `only_videos` | 人工质检后追加到 `config.only_videos` |
| 未归属 | 已质检但 `splits.yaml` 无 split | 跑 `00_status.py --assign` |
| 遗失 | `splits.yaml` 有但磁盘没有 | 重下,或从 `splits.yaml` 删(不自动删) |
| 孤儿 | 磁盘有但导出没引用 | 陈旧下载,可清理 |

**增量之所以安全**:`splits.yaml` 里已有视频的 split 永不被自动改动,`--assign` 只回填新视频 → 天然增量,重建数据集不会打乱既有 train/val 划分。

---

## 额外考量

以下是维护时需要理解的设计契约与规范,平时按上面流程走即可,改动前请先读。

### 1. 数据来源与关联

- **视频**:存 LS 服务器,`01_pull_data.py` 下到 `raw/videos/`;**身份 = 文件名 stem**(如 `687e3c78-clip_<起>_<止>`)。
- **标注**:LS 导出 JSON 放 `raw/exports/`,脚本取**文件名排序最后一份**。同一份导出里 bbox(`videorectangle`)与时序(`timelinelabels`)聚合,本流程**只消费 bbox**。
- 视频与标注靠 `task.data.video` 的文件名关联;`00_status.py` 就是对齐"导出 / 磁盘 / splits / 白名单"四方。

### 2. 关键帧对齐(不做就框漂移 + 尾部丢标注)

- LS 的 `sequence` 只存**关键帧**,中间帧**线性插值**得框;`enabled=False` = 目标离场那段不出框。
- LS 帧号按**标注端 fps**(`ls_fps = framesCount/duration`,常 ~24)计,真实视频 fps 往往不同 → 用 `scale = ls_fps/real_fps`、`ls_frame = real_frame × scale` 把真实解码帧号映射回 LS 帧号。**绝不能拿真实帧号直接查框**。
- 逻辑集中在 `utils/lsexport.py`,勿各写一套。自查:每视频"尾部覆盖 ≈ 100%"(`02_build` 会打印,<80% 告警)。

### 3. 采样帧率

- `config.stride = 12`:每隔 12 个真实帧抽 1 张 → 30fps 视频 ≈ **2.5 张/秒**。调 `stride` 改抽帧密度(越小越密、图越多)。
- **只有"含分组内目标框"的帧才落盘**,空帧丢弃,避免大量负样本稀释。

### 4. 稳定切分契约(重点)

- **`splits.yaml` 是唯一真源**:`视频stem -> train/val/test/e2e_test`。人工可改,永不被自动重排。
- 未归属视频由 `--assign` 按 `hash(seed:stem)` **确定性**落到 train/val(比例 `val_ratio`),并写回清单。
- **同一视频永远同一 split**、新增视频不打乱已有分配、**一个视频的所有帧只进一个 split**——杜绝时间相邻泄漏,指标可信可复现。
- `test` / `e2e_test` 的视频**不进** YOLO 数据集,预留给端到端评测,可与时序模型共用同一份清单。
- 回填是**显式步骤**(`--assign`),不在 `build` 里静默改动;`02_build` 遇未归属视频会报错提示(或用 `--auto-assign` 当场回填)。

### 5. 数据集格式规范

`02_build_dataset.py` 产出标准 ultralytics YOLO 结构,每组一套:

```
datasets/<组>/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
  data.yaml
```

- **label 文件**:每行一个框 `class_id cx cy w h`,均为**归一化 [0,1]**;`class_id` 各组从 0 起(顺序即 `config.groups` 里的列表顺序)。
- **图片命名**:`{task:02d}_{stem12}_{frame:06d}.jpg`(task 序号 _ 视频名前 12 位 _ 真实帧号)。
- **data.yaml**:`path / train / val / nc / names`,`names` 为 `id: 显示名`。
- **坐标约定**:LS 左上角百分比 → YOLO 归一化中心点 `(cx,cy,w,h)`,裁剪到 [0,1]。
- **类别**:严格等于 LS 训练类别名(与 `config.groups` 里逐字一致);未列入任何组的类别(`short_brush`、`brush_tip_out`)自动忽略。
- 转换后务必核对:每视频"尾部覆盖"接近 100%;抽一帧反归一化画框确认落在目标上。

### 6. 模型验收标准

`04_validate.py` 在**验证集**上跑 `ultralytics.val`,取逐类 P/R/mAP 对照 `config.acceptance` 判 PASS/FAIL,写 `runs/<组>/acceptance_report.md`。默认门槛(在 `config.yaml` 调整):

| 项 | 默认门槛 | 说明 |
|----|---------|------|
| 整体 mAP@0.5 | ≥ 0.5 | 总体检出质量 |
| 整体 mAP@0.5:0.95 | ≥ 0.3 | 定位精度 |
| 逐类 recall | ≥ 0.7 | 漏检代价高,重点卡 |
| 逐类 precision | ≥ 0.5 | 误检 |
| 每个类别都有验证样本 | 必须 | 验证集缺某类 → 无法评估,判 FAIL |

> ⚠️ 当前数据集很小(单视频起步),大概率 FAIL,**属实**——先把流程和门槛立起来,数据变多、按视频切分足够后指标才有意义,再逐步收紧门槛。

### 7. 关键约定速查(改脚本别破坏)

- 所有路径都以 `yolo_pipeline/` 为根(`utils/common.py` 的 `ROOT`),别再往上级目录写,破坏自包含。
- fps 对齐、关键帧插值、坐标转换集中在 `utils/lsexport.py`,各脚本共用一份,别再各写一套。
- 类别只能**追加到 `config.groups` 列表末尾**,插中间会打乱已训权重的 class id 映射。
- 样本分布**不放在 `00_status`**(它是无解码、秒级的对账工具),而由 `02_build` 在数据集生成后从落盘 label 统计。

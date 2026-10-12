---
name: training-capacity-preflight
description: Use before launching any automated training run in this repo — sweep presets/grids, run_*_matrix.py, or a direct model.train — and whenever a run dies before epoch 0 or its ETA exceeds the available window. Gates batch / workers / imgsz / epochs against this machine's actual RAM, VRAM and wall-clock budget, and scales down in a fixed order so a mis-sized run degrades into a contained, resumable failure instead of a killed WSL VM.
metadata:
  short-description: 训练前先过本机容量闸门，规模不够就缩
---

# Training Capacity Preflight

## Scope

在**本机（12GB WSL + RTX 4060 Laptop 8GB + 32GB 焊死宿主）**上启动任何自动训练之前，
先过下面四道闸门。闸门不过就**先缩小规模**，不要"先跑起来看会不会炸"。

**首要目标是把训练跑完，不是把规模开到最大。** 一个被 OOM 杀掉的 run 产出为零；
一个缩小但跑完的 run 产出可用的读数。

本 skill 只管**闸门与缩小动作**。硬件/预算的正文与数字来源见文末 §8，不要在本文件里重复。

## 1. 四道闸门

按成本从低到高，**任何一道不过就缩小**：

| # | 闸门 | 不过的后果 | 缩小对象 |
|---|---|---|---|
| 1 | **系统内存**（guest RAM + host 提交） | `global_oom` → 杀进程 → 整台 WSL VM 被拖垮重启 | `workers` → `batch` → `imgsz` |
| 2 | **显存**（8GB） | CUDA OOM，epoch 中途崩 | `batch`（按分辨率等比） |
| 3 | **墙钟预算**（ETA vs 可用时间窗） | 跑不完，等于零产出 | `epochs` → `fraction` → 换小模型 |
| 4 | **隔离**（cgroup 包装） | 尺寸估错时陪上整台 VM 与正在编辑的 VS Code | 不是缩小，是**必做** |

## 2. 闸门 1：系统内存（最常炸的一道）

### 本机客机内存构成（2026-10-12 实测，无训练时）

`MemTotal = 11.68 GiB`（`.wslconfig` 里 `memory=12GB`）：

| 项目 | 占用 |
|---|---|
| VS Code Remote Server（44 个扩展） | ≈ 5.34 GB |
| DSH harness | ≈ 0.94 GB |
| mysqld / ollama / postgres / snapd | ≈ 0.60 GB |
| 内核 slab + 系统 | ≈ 1.4 GB |
| **基线合计** | **≈ 8.0–8.3 GB** |
| **真正剩给训练** | **≈ 3.4 GB** |

`MemAvailable` 看起来"还有 6 GB"是**假象**——那 8 GB 基线里的 slab/anon 不可回收。

### 已发生的事故（不是假设）

- `imgsz=960 + batch=12 + workers=8 + copy_paste` 连续两次在 **epoch 0 之前**被杀；
- 累计 **710 次** `killed by the OOM killer`（`grep -c … /var/log/syslog`）、**6 处** `global_oom`
  现场（`/var/log/kern.log`）；WSL **每 35 秒**重启一次（`systemd[1]: Startup finished` 共 353 次），
  `init.scope` 里的 VS Code Server 一起被杀掉（`MainThread` 是 Node 的线程名，不是某个训练进程）。

### 判据

```
估计峰值 ≈ 基线（≈8 GB）+ 训练自身
训练自身 ≈ 模型权重 + batch 张量 + workers × 每 worker 缓冲
```

**能在本机安全跑的配置，`workers` 一律 ≤ 2。** 960px 及以上分辨率下，
**worker 进程是内存峰值主因**，不是 batch。所以缩小顺序永远是 **先砍 workers，再砍 batch**。

### 启动前必测

```bash
free -m                                   # guest 总量/可用
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.current   # 会话已占多少
ps -eo pid,rss,comm --sort=-rss | head -12   # 谁在占（沙箱里看不到时见 §7）
```

**若"基线 + 训练估计"超过 `MemTotal` 的 75%（本机约 8.8 GB），必须缩小后再启动。**

### 宿主机侧也要看一眼（它会让客人 OOM）

WSL 是**两级**内存：客机 12GB + 宿主提交上限。宿主吃紧时 Hyper-V 会通过
`hv_balloon`（Dynamic Memory）**主动从客人抽回内存** → 客人 OOM。

```powershell
$m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
'commit 余量 = {0:N2} GB   availRAM = {1:N2} GB   nonpaged = {2:N0} MB' -f `
  (($m.CommitLimit-$m.CommittedBytes)/1GB), ($m.AvailableBytes/1GB), ($m.PoolNonpagedBytes/1MB)
```

**宿主提交余量 < 5 GB 时不要启动训练。**

## 3. 闸门 2：显存（8GB）

本机 RTX 4060 Laptop 8GB（`total 8.0 GiB / available 7.8 GiB`）。

**显存与分辨率成平方关系**，这是唯一已经在代码里固化的一条：

```python
# framework/cleansight_eval/detection/sweep.py
adjust_batch_for_imgsz(base_batch, base_imgsz, target_imgsz)
# = max(2, int(base_batch * (base_imgsz / target_imgsz) ** 2))
```

| imgsz | 相对 640 的 batch 系数 | 预设现值（yolo11s） |
|---:|---:|---:|
| 640 | ×1 | 16 |
| 960 | ×0.44 | 12（copy_paste 组压到 8） |
| 1280 | ×0.25 | 8（yolo11m 取 6） |

换模型时再打折：yolo11m 的权重与激活更重，960px 取 4。

⚠️ **自动化/沙箱 shell 没有 GPU 透传**（`/dev/nvidia*` 不存在、`nvidia-smi` 报
`GPU access blocked`、`torch.cuda.is_available()=False`）——这**不代表本机不支持 GPU**。
**显存闸门只能在用户交互终端验证**，自动化会话里请直接沿用预设里已按 8GB 收缩过的值，
不要自行放大。

## 4. 闸门 3：墙钟预算（最容易被忽略的一道）

内存过了不等于能跑完。本机已经因为**时长**废掉过一个正式配方。

### 公式

```
每轮分钟 = iter_per_epoch / (it/s) / 60
ETA 小时 = 每轮分钟 × epochs / 60
```

`iter_per_epoch`：group1_large ≈ 5382，group2_small ≈ 1865（`fraction=1.0`）。
首次 epoch 含一次性 setup，**不要**把它当成每轮成本。

### 本机实测基线（用于外推，引用前注意日期）

| 配方 | 模型 / 分辨率 / batch / workers | it/s | 每轮 | epochs | ETA | 判定 |
|---|---|---:|---:|---:|---:|---|
| `large_baseline_lowmem` | yolo11n / 640 / 16 / 2 | — | 4.0 min | 150 | ≈ 10.0 h | 可跑 |
| `small_s_copy_paste_lowmem` | yolo11s / 960 / 8 / 2 | 3.6 | ≈ 10 min（含 val） | 200 | ≈ 33 h | 可跑但长 |
| `large_m_960_lowmem` | yolo11m / 960 / 4 / 2 | **1.83** | **49 min** | 200 | **≈ 163 h** | **本机不可行** |

### 判据

**ETA 超过可用的连续时间窗，就先缩小再启动。** 本机建议上限 **≤ 12 h/run**
（超过这个量级时，中途被 VS Code / 宿主内存波动打断的概率显著上升）。
不可行时的动作顺序：`epochs` ↓ → `fraction` ↓ → **换小模型**（这条最有效）。

新配方第一次跑，**先用 `--smoke` 探针**量一次 it/s，再据 §4 公式推算正式 ETA，
不要凭经验拍。

## 5. 闸门 4：cgroup 隔离（必做，不是可选项）

**尺寸估错时要让"训练自己死"，而不是"整台 VM 死"。** 已实测该机制有效：

| 测试 | 结果 |
|---|---|
| 无限制，逐页触碰 1000 MB | RSS 涨到 **1009.9 MB**（确实占用） |
| `-p MemoryMax=200M`，同样逐页触碰 | RSS **卡在 ≈205 MB**，继续写也不涨 |

行为是**先强制换出、实在不行才杀 cgroup** → 训练变慢或被单独杀掉，**VM 存活**。

```bash
systemd-run --user --unit=yolo-<预设>-$(date +%H%M%S) --pty \
  -p MemoryHigh=6G -p MemoryMax=8G -p MemorySwapMax=4G \
  -- bash -lc 'cd /home/caizh/programming/python_code/Cleansight_models && \
       source /home/caizh/programming/python_code/CleanSightBackend/.venv/bin/activate && \
       export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 && \
       python -m framework.cleansight_eval.cli.sweep \
         --group <group> --preset <preset> --device 0'
```

`MemoryMax` 与 `MemoryHigh` 按 §2 的剩余量给：留 ≥2 GB 给基线，别把上限设到 `MemTotal`。

## 6. 启动后 1 分钟内必须确认的三件事

不确认就"提交了命令"不等于训练在跑（`verification-first`）：

1. **进程真的起来了**：`systemctl --user status yolo-<单元名>` 或 `systemd-cgtop -m -n 1`
   里能看到该 scope，且 RSS 在涨。
2. **`args.yaml` 里是你要的参数**：ultralytics 会把**实际生效**的参数写进
   `runs/cleansight-yolo/<name>/args.yaml`，核对 `batch` / `workers` / `imgsz`。
   这是"缩小是否真的生效"的唯一硬证据 —— 预设里写了 `workers` 但代码不透传时，
   `args.yaml` 会显示 `workers: 8`。
3. **没有新的 OOM**：

```bash
dmesg -T | grep -iE 'out of memory|oom-kill' -A 5   # 有新记录 = 闸门估错了，立刻停
grep -c 'killed by the OOM killer' /var/log/syslog  # 计数应与启动前一致
```

## 7. 陷阱清单（每条都是踩过的）

| 陷阱 | 正确做法 |
|---|---|
| `results.csv` 的 `time` 列是**累计秒数**，不是每轮秒数 | 用**相邻两行相减**；并与进程 `etime` 交叉核对。曾因此高估 2.2 倍（21.7 h vs 实际 10.0 h），且连错三次 |
| `bytearray(n)` 只做惰性映射，不消耗物理内存 | 测内存占用必须**逐页触碰**，如 `b[::4096] = b"\xff" * (len(b)//4096)` |
| 沙箱/自动化 shell 有独立 PID namespace，`ps` 看不到其他进程 | 用 `wsl.exe -d Ubuntu -- ps -eo pid,ppid,rss,comm,args --sort=-rss` 从 Windows 侧起进程，或从 cgroup 侧看（`systemd-cgtop`） |
| 只看 `MemAvailable` 就认为"内存够" | 基线 8 GB 里的 slab/anon 不可回收，看**剩余量**与 §2 判据 |
| 只看客人不看宿主 | WSL 两级内存 + `hv_balloon`；宿主提交余量 <5 GB 不要启动 |
| 用 `nvidia-smi` 在自动化 shell 里判断显存 | 会得到 `GPU access blocked`；**这不代表无 GPU**，显存闸门只在交互终端验证 |
| 预设里写了 `workers` 就以为生效 | 必须查 `args.yaml`；`sweep.py` 的 `train_kwargs` 是**白名单式**构造，未登记的键不会转发 |
| 从日志统计"发生了几次"时凭印象填数 | 必须写明**哪个文件的哪个 pattern**。`killed by the OOM killer`（syslog）= 710，与 `Startup finished`（= 启动次数）353 **不是一回事**；本文件初稿就把这两个数字记串过 |

## 8. 记录要求（否则缩小过的配方会被当成原配方）

- **任何缩小都要留痕**：写进 run 说明与周简报（`docs/weeks/<周六>_<周五>/README.md`），
  写清"相对哪个配方、改了哪个键、为什么"。
- **不得把缩小后的 run 当作原配方的读数**（同 `modelset-quality` 的 smoke 规则）。
  例如 group2 那批 `fraction=0.2` 的读数必须先显式标注 smoke 口径再引用。
- 新增低内存配方时，**不要改已有预设**——另开 `<原名>_lowmem`，保证历史 run 的
  `args.yaml` 仍能复现（`large_*_lowmem` / `small_s_copy_paste_lowmem` 都是这么加的）。
- 注释里写清**否决理由**，避免下次重复踩：如 `large_m_960_lowmem` 的 49 min/epoch
  已写进 `sweep.py` 注释。

## 9. 数字的来源与归属（不要在本文件重复正文）

| 内容 | 归属位置 |
|---|---|
| 硬件与预算建议正文 | `docs/YOLO_OPTIMIZATION.md` §6 |
| 最新实测读数与复现命令 | `docs/weeks/2026-10-10_2026-10-16/`（周简报 + `YOLO_GROUP1_FULL_20261012.md`） |
| 预设表与 `workers` 透传 | `framework/cleansight_eval/detection/sweep.py` |
| 模型/基准质量规则 | `modelset-quality` skill |

⚠️ **引用 `§6` 前先量一遍**：它写于本轮实测之前，其"RTX 4060 完整流水线 2~3 周"与
"高分辨率预设 batch 调小（640→8、960→4、1280→2）"两处**已被当前实测取代**
（现预设为 640→16、960→12/8、1280→8/6，且 §6 完全没提 12GB 系统内存 / `workers` 这两个真正的杀手）。
按 `modelset-quality` 的规则：旧文档里的绝对断言引用前必须复测，不符就地更正并保留错处。

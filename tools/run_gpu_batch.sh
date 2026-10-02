#!/usr/bin/env bash
# GPU 批处理：在**交互终端**一次跑完若干实验批次（训练 + 官方评测 + 汇总）。
#
# 为什么需要这个脚本：DSH 沙盒（bwrap `--dev /dev`）看不到 NVIDIA 设备，
# `torch.cuda.is_available()` 恒为 False；但项目 venv 的 torch 是 CUDA 构建，
# **在交互终端执行时 `pick_device()` 会自动选 CUDA**，命令本身零改动。
# 详见 docs/experiments/EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md 的沙盒说明。
#
# 用法：
#     bash tools/run_gpu_batch.sh                # 跑全部批次（默认 8 seed）
#     bash tools/run_gpu_batch.sh recheck-recipe # 只跑指定批次
#     SEEDS=42,7,2026 bash tools/run_gpu_batch.sh   # 覆盖 seed 列表
#     RUNS_DIR=runs/gpu-batch bash tools/run_gpu_batch.sh
#
# 为什么先跑这三个批次：前几轮的结论都建立在 4~5 seed 上，而项目标配是 8 seed。
# 用 8 seed 复核「旗舰 vs 推荐配方 vs asformer」是当前性价比最高的一步——
# 尤其 asformer 的退化解判定是从 cw 扫描中得出的，值得在更多 seed 上确认。

set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-../CleanSightBackend/.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export MPLBACKEND=Agg

SEEDS="${SEEDS:-42,7,2026,1,2,3,4,5}"
RUNS_DIR="${RUNS_DIR:-runs/gpu-batch}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-0.0005}"

echo "=== 环境自检 ==="
python - <<'PY'
import sys
import torch
print(f"  python : {sys.version.split()[0]}")
print(f"  torch  : {torch.__version__}")
if not torch.cuda.is_available():
    print("  ❌ CUDA 不可用。")
    print("     若你是在 DSH 沙盒里执行，这是预期行为（沙盒不透传 GPU）——请在交互终端重跑本脚本。")
    print("     若在交互终端仍不可用，先检查 `nvidia-smi` 是否正常。")
    raise SystemExit(1)
print(f"  ✅ GPU   : {torch.cuda.get_device_name(0)}（{torch.cuda.device_count()} 张）")
PY

# 每个批次一行：<名称>|<容量点>|<额外 --set 参数>
#
# ⚠ 容量点必须走 `--hidden`，**不能**写成 `--set model.hidden=N`：
# run_capacity_matrix.py 会自己追加 `-S model.hidden=<hidden>`，且它排在 `--set` 之后，
# 会把 `--set model.hidden` 覆盖掉；而省略 `--hidden` 时它默认跑 16,32,64,128,256 五个点。
# （本脚本初版就踩了这个坑：本以为跑 1 个点，实际跑了 5 个。）
BATCHES=(
  "recheck-flagship|128|--set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10"
  "recheck-recipe|32|--set model.type=mstcn2 --set model.num_stages=2 --set model.num_layers=5 --set train.class_weight_clip=0.03,5.0"
  "recheck-asformer|32|--set model.type=asformer --set model.heads=4 --set model.num_encoders=3 --set model.num_decoders=2 --set train.class_weight_clip=0.03,5.0"
)

wanted=("$@")
run_one() {
  local name="$1" hidden="$2" extra="$3"
  if [ ${#wanted[@]} -gt 0 ]; then
    local hit=0
    for w in "${wanted[@]}"; do [ "$w" = "$name" ] && hit=1; done
    [ "$hit" = 1 ] || return 0
  fi
  echo
  echo "=== 批次 $name（hidden=$hidden）→ $RUNS_DIR/$name ==="
  local start
  start=$(date +%s)
  # shellcheck disable=SC2086
  python tools/run_capacity_matrix.py \
    --runs-dir "$RUNS_DIR/$name" \
    --hidden "$hidden" \
    --seeds "$SEEDS" --best-metric val_edit --epochs "$EPOCHS" \
    --set train.lr="$LR" --no-eval-last --no-probe-baseline \
    $extra
  echo "--- $name 耗时 $(( $(date +%s) - start )) 秒 ---"
}

for entry in "${BATCHES[@]}"; do
  IFS='|' read -r b_name b_hidden b_extra <<< "$entry"
  run_one "$b_name" "$b_hidden" "$b_extra"
done

echo
echo "=== 全部分批完成 ==="
echo "结果目录： $RUNS_DIR/<批次名>/CAPACITY_SUMMARY.md"
echo
echo "接下来（在同一个终端或回到 agent 会话）："
echo "  # 验收口径总表 + 退化解告警（含逐视频配对检验）"
echo "  python tools/probe_segment_coverage.py \\"
echo "      --run \"$RUNS_DIR/recheck-flagship/h128/mstcn2-*\" --label 旗舰 \\"
echo "      --run \"$RUNS_DIR/recheck-recipe/h32/mstcn2-*\"     --label 推荐配方"
echo "  # 叠加 S-NCM（训练无关）"
echo "  python tools/probe_segment_ncm.py --run \"$RUNS_DIR/recheck-recipe/h32/mstcn2-*\""

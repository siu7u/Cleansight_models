#!/usr/bin/env bash
# 一键执行完整时序训练链路：视频 → YOLO 标注 → convert → 训练 → 评测。
#
# 用法（一句命令）：
#     bash tools/run_auto_train.sh --videos <视频或目录> [选项]
#
# 选项：
#     --videos <路径>        视频文件或目录（必填）
#     --config <yaml>        训练配置（默认 framework/experiments/mstcn-autoannotate-smoke.yaml）
#     --epochs <N>           训练轮数（默认 2，冒烟）
#     --out <目录>           中间产物根（默认 tmp/one-shot，git 忽略）
#     --labels-export <json> 人工动作标签导出（默认 project-10 最新导出）
#     --frame-stride <N>     YOLO 帧采样（默认 4，等效 7.5fps）
#     --resume               跳过已存在标注 JSON（配合已有 outputs/annotations 增量使用）
#
# 流程：annotate run → convert(split=train) → cli.train(-S 覆盖数据根) → 评测(eval)
# 说明：一键链路面向冒烟/增量验证；正式多 split 训练请按 docs/AUTO_ANNOTATION.md
#       分步执行（convert 分 train/val/test + testsets.yaml 登记）。

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

VIDEOS=""
TRAIN_CFG="framework/experiments/mstcn-autoannotate-smoke.yaml"
EPOCHS=2
OUT="tmp/one-shot"
EXPORT="legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json"
FRAME_STRIDE=4
RESUME=""

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --videos) VIDEOS="$2"; shift 2 ;;
    --config) TRAIN_CFG="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --labels-export) EXPORT="$2"; shift 2 ;;
    --frame-stride) FRAME_STRIDE="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    *) echo "未知参数: $1"; usage ;;
  esac
done
[ -n "$VIDEOS" ] || { echo "缺少 --videos"; usage; }

if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$ROOT/../CleanSightBackend/.venv/bin/python" ]; then PY="$ROOT/../CleanSightBackend/.venv/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY="python"; fi
echo "[one-shot] 解释器: $PY"
echo "[one-shot] 视频: $VIDEOS | 配置: $TRAIN_CFG | epochs: $EPOCHS | 产物: $OUT"

mkdir -p "$OUT"

echo; echo "===== ① YOLO 自动标注 ====="
"$PY" -m framework.cleansight_eval.cli.annotate run \
  --videos "$VIDEOS" \
  --config framework/experiments/auto-annotate.yaml \
  --frame-stride "$FRAME_STRIDE" \
  --out "$OUT/annotations" $RESUME || { echo "[one-shot] ① 失败"; exit 1; }

echo; echo "===== ② convert（标注 JSON + 人工动作标签 → 训练数据）====="
"$PY" -m framework.cleansight_eval.cli.annotate convert \
  --annotations "$OUT/annotations" \
  --labels-export "$EXPORT" \
  --out "$OUT/data" --split train || { echo "[one-shot] ② 失败"; exit 1; }

echo; echo "===== ③ 训练（使用②产出的标注数据）====="
"$PY" -m framework.cleansight_eval.cli.train \
  --config "$TRAIN_CFG" \
  -S "data.root=$PWD/$OUT/data" \
  -S data.split_val=train \
  -S "train.epochs=$EPOCHS" \
  --runs-dir "$OUT/runs" || { echo "[one-shot] ③ 失败"; exit 1; }
BEST="$(ls -d "$OUT"/runs/*/checkpoints/best.pt 2>/dev/null | head -1)"
[ -n "$BEST" ] || { echo "[one-shot] ③ 未找到 best.pt"; exit 1; }

echo; echo "===== ④ 评测 ====="
"$PY" - "$TRAIN_CFG" "$PWD/$OUT/data" "$OUT/eval.yaml" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['data'] = {'root': sys.argv[2], 'split_train': 'train', 'split_val': 'train', 'split_eval': 'train'}
cfg['evaluation'] = {'mode': 'exploratory', 'visualize': False}
with open(sys.argv[3], 'w') as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True)
PYEOF
"$PY" -m benchmark.cli.eval --config "$OUT/eval.yaml" --ckpt "$BEST" || { echo "[one-shot] ④ 失败"; exit 1; }

echo; echo "===== 汇总 ====="
EVAL_JSON="$(ls "$OUT"/runs/*/evals/*.evaluation.json 2>/dev/null | tail -1)"
"$PY" - "$EVAL_JSON" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d['metrics']['summary']
print(f"评测 acc: {s['acc']['value']}  edit: {s['edit']['value']}")
PYEOF
echo "checkpoint: $BEST"
echo "评测报告: $EVAL_JSON"
echo "训练历史: $OUT/runs/*/history.csv"
echo "一键链路完成 ✅（中间产物在 $OUT/，可删除）"

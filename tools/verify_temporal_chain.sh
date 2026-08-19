#!/usr/bin/env bash
# 完整时序训练链路手动验证脚本。
#
# 覆盖：环境/数据就绪 → 自动标注(JSON) → convert(训练数据) → 数据加载(40维特征)
#       → 自动通道训练 → 手动通道训练 → 正式评测。每步输出 [PASS]/[FAIL]，
#       末尾汇总；任一失败以非零退出码结束（可接 CI 或手动查看）。
#
# 前置：
#   在仓库根目录执行：bash tools/verify_temporal_chain.sh
#   解释器自动探测：环境变量 PYTHON > ../CleanSightBackend/.venv > 仓库 .venv > PATH
#
# 说明：所有中间产物写入 tmp/verify-chain/（git 忽略），验证完可整目录删除。

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/../CleanSightBackend/.venv/bin/python" ]; then
  PY="$ROOT/../CleanSightBackend/.venv/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python"
fi
echo "使用解释器: $PY"

WORK="tmp/verify-chain"
ANN_DIR="$WORK/annotations"
DATA_DIR="$WORK/data"
RUNS_AUTO="$WORK/runs-auto"
RUNS_MANUAL="$WORK/runs-manual"
VIDEO="legacy/yolo-detection/pipeline/raw/videos/7e8f5b4f-clip_1781584064111_1781584068667.mp4"
EXPORT="legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json"

PASS=0
FAIL=0

step() { echo; echo "========== [$1] $2 =========="; }

check() { # check <名字> <exit_code> [额外说明]
  if [ "$2" -eq 0 ]; then PASS=$((PASS+1)); echo "[PASS] $1"
  else FAIL=$((FAIL+1)); echo "[FAIL] $1${3:+ —— $3}"; fi
}

mkdir -p "$WORK"

# ---------- Phase 0：环境与数据就绪 ----------
step "0/8" "环境与数据就绪"
"$PY" -c "import torch, yaml; print('torch', torch.__version__)" >/dev/null 2>&1
check "依赖 torch/yaml 可用" $? "先激活 venv：source ../CleanSightBackend/.venv/bin/activate"

"$PY" -m framework.cleansight_eval.cli.dataset --check >/dev/null 2>&1
check "数据集就绪（yolo + actionmixed 手动）" $?

"$PY" tools/validate_testsets.py >/dev/null 2>&1
check "testsets 登记校验（13 个 testset）" $?

# ---------- Phase 1：自动通道数据链路 ----------
step "1/8" "YOLO 自动标注（冒烟：单小视频前 60 帧）"
rm -rf "$ANN_DIR"; mkdir -p "$ANN_DIR"
"$PY" -m framework.cleansight_eval.cli.annotate run \
  --videos "$VIDEO" \
  --config framework/experiments/auto-annotate.yaml \
  --max-frames 60 --out "$ANN_DIR" >"$WORK/step1.log" 2>&1
rc=$?
[ $rc -eq 0 ] && ls "$ANN_DIR"/*.json >/dev/null 2>&1 && rc=$?
check "annotate run → 检测 JSON" $rc "见 $WORK/step1.log"

step "2/8" "convert（JSON + 人工动作标签 → 训练数据布局）"
rm -rf "$DATA_DIR"
"$PY" -m framework.cleansight_eval.cli.annotate convert \
  --annotations "$ANN_DIR" \
  --labels-export "$EXPORT" \
  --out "$DATA_DIR" --split train >"$WORK/step2.log" 2>&1
rc=$?
[ $rc -eq 0 ] && ls "$DATA_DIR"/labels/train/*.txt >/dev/null 2>&1 && \
  ls "$DATA_DIR"/frames/train/*.txt >/dev/null 2>&1 && rc=$?
check "convert → labels/ + frames/" $rc "见 $WORK/step2.log"

step "3/8" "数据加载（自动通道，[T,40] 特征 + 6 类标签）"
"$PY" - "$DATA_DIR" >"$WORK/step3.log" 2>&1 <<'EOF'
import sys
sys.path.insert(0, '.')
from framework.cleansight_eval.temporal.data import load_split
root = sys.argv[1]
feats, truths, id2name = load_split(
    {'root': root, 'labels_dir': 'labels', 'frames_dir': 'frames'}, 'train',
    feature_schema={'dim': 40, 'version': 'actionmixed-bbox-8cls-v1'})
assert feats[0].shape[1] == 40, feats[0].shape
assert len(id2name) == 6, id2name
assert len(feats) == len(truths)
print(f'OK: {len(feats)} 视频, {feats[0].shape}, 6 类')
EOF
check "自动通道 load_split → 40 维" $? "见 $WORK/step3.log"

step "4/8" "数据加载（手动通道 actionmixed-v2，能力保留确认）"
"$PY" >"$WORK/step4.log" 2>&1 <<'EOF'
import sys
sys.path.insert(0, '.')
from framework.cleansight_eval.temporal.data import load_split
feats, truths, id2name = load_split(
    {'root': 'datasets/cleansight-ActionMixed', 'labels_dir': 'labels',
     'frames_dir': 'frames'}, 'train',
    feature_schema={'dim': 40, 'version': 'actionmixed-bbox-8cls-v1'})
assert feats[0].shape[1] == 40, feats[0].shape
assert len(feats) == len(truths) and len(feats) > 0
print(f'OK: {len(feats)} 视频, {feats[0].shape}, 6 类')
EOF
check "手动通道 load_split → 40 维" $? "见 $WORK/step4.log"

# ---------- Phase 2：训练 ----------
step "5/8" "自动通道训练（smoke 配置，2 epoch）"
rm -rf "$RUNS_AUTO"
"$PY" -m framework.cleansight_eval.cli.train \
  --config framework/experiments/mstcn-autoannotate-smoke.yaml \
  -S train.epochs=2 --runs-dir "$RUNS_AUTO" >"$WORK/step5.log" 2>&1
rc=$?
BEST_AUTO="$(ls -d "$RUNS_AUTO"/mstcn-*/checkpoints/best.pt 2>/dev/null | head -1)"
[ $rc -eq 0 ] && [ -n "$BEST_AUTO" ] && rc=$?
check "自动通道训练 → best.pt" $rc "见 $WORK/step5.log（$BEST_AUTO）"

step "6/8" "手动通道训练（mstcn-actionmixed，2 epoch，能力保留确认）"
rm -rf "$RUNS_MANUAL"
"$PY" -m framework.cleansight_eval.cli.train \
  --config framework/experiments/mstcn-actionmixed.yaml \
  -S train.epochs=2 --runs-dir "$RUNS_MANUAL" >"$WORK/step6.log" 2>&1
rc=$?
BEST_MANUAL="$(ls -d "$RUNS_MANUAL"/mstcn-*/checkpoints/best.pt 2>/dev/null | head -1)"
[ $rc -eq 0 ] && [ -n "$BEST_MANUAL" ] && rc=$?
check "手动通道训练 → best.pt" $rc "见 $WORK/step6.log（$BEST_MANUAL）"

# ---------- Phase 3：评测 ----------
step "7/8" "正式评测（自动通道 best.pt，formal/exploratory 均可）"
[ -n "${BEST_AUTO:-}" ] && "$PY" -m benchmark.cli.eval \
  --config framework/experiments/mstcn-autoannotate-smoke.yaml \
  --ckpt "$BEST_AUTO" >"$WORK/step7.log" 2>&1
rc=$?
[ $rc -eq 0 ] && ls "$WORK"/runs-auto/mstcn-*/evals/*.evaluation.json >/dev/null 2>&1 && rc=$?
check "评测 → evaluation.json" $rc "见 $WORK/step7.log"

step "8/8" "运行状态（训练 run 的 status.json）"
"$PY" >"$WORK/step8.log" 2>&1 <<'EOF'
import glob, json, sys
ok = True
for run in glob.glob('tmp/verify-chain/runs-*/*/status.json'):
    s = json.load(open(run))
    state = s.get('state')
    print(run, '→', state)
    ok = ok and state == 'succeeded'
sys.exit(0 if ok else 1)
EOF
check "训练 run 全部 succeeded" $? "见 $WORK/step8.log"

# ---------- 汇总 ----------
echo
echo "=============================="
echo "验证结果：$PASS PASS / $FAIL FAIL"
echo "中间产物：$WORK/（验证完可删除）"
[ "$FAIL" -eq 0 ] && echo "链路验证通过 ✅" || echo "存在失败步骤，见上方 [FAIL] 与对应 log"
exit $((FAIL > 0 ? 1 : 0))

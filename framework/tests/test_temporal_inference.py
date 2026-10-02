"""`temporal/inference.py` 的单元测试（不加载 checkpoint，纯数值与契约校验）。

覆盖三类不变量：

1. **S-NCM 的语义**：只改段内标签、不动段边界；当"投票"与"边界"同源时逐帧恒等
   （该恒等性正是 `tools/probe_segment_ncm.py --self-check` 的判据）。
2. **类均值的契约**：按类求均值；某类无样本时**报错**而不是静默产生 NaN 中心。
3. **挂载点登记**：所有已注册的时序模型都必须能在 `HIDDEN_HOOKS` 里找到 penultimate
   表示的挂载路径——新增架构时忘记登记会被这条测试拦住。
"""

import numpy as np
import pytest

from framework.cleansight_eval.temporal.inference import (
    HIDDEN_HOOKS,
    class_means,
    forward_with_hidden,
    ncm_predict,
    sncm_predict,
)
from framework.cleansight_eval.temporal.models import _MODELS, build_model


def test_sncm_keeps_boundaries_and_takes_segment_mode():
    """段边界必须原样保留；每段标签变成该段票数的众数。"""

    boundaries = np.array([0, 0, 0, 3, 3, 3, 3, 0])
    votes = np.array([2, 2, 2, 5, 4, 5, 5, 1])
    out = sncm_predict([boundaries], [votes])[0]
    assert np.array_equal(np.where(np.diff(out) != 0)[0], np.where(np.diff(boundaries) != 0)[0])
    assert list(out[:3]) == [2, 2, 2]      # 段内众数 = 2
    assert list(out[3:7]) == [5, 5, 5, 5]  # 段内众数 = 5（5 出现 3 次 > 4 的 1 次）
    assert out[7] == 1


def test_sncm_is_identity_when_votes_share_the_boundaries():
    """自检路径：votes 与 boundaries 同源时，S-NCM 必须逐帧恒等（否则自检失去意义）。"""

    boundaries = np.array([0, 0, 1, 1, 1, 1, 0, 2, 2])
    out = sncm_predict([boundaries], [boundaries.copy()])[0]
    assert np.array_equal(out, boundaries)


def test_sncm_rejects_length_mismatch():
    """长度不一致必须报错，而不是静默截断。"""

    with pytest.raises(ValueError, match="长度不一致"):
        sncm_predict([np.zeros(4, dtype=np.int64)], [np.zeros(5, dtype=np.int64)])


def test_class_means_averages_per_class():
    """类均值 = 该类各帧表示的平均。"""

    hiddens = [np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])]
    labels = [np.array([0, 0, 1])]
    centers = class_means(hiddens, labels, num_classes=2)
    assert np.allclose(centers[0], [1.0, 1.0])
    assert np.allclose(centers[1], [10.0, 10.0])


def test_class_means_rejects_class_without_samples():
    """缺类必须报错——静默产生 NaN 中心会让下游 NCM 全部退化且难以察觉。"""

    hiddens = [np.zeros((3, 2))]
    labels = [np.array([0, 0, 0])]
    with pytest.raises(ValueError, match="没有样本"):
        class_means(hiddens, labels, num_classes=2)


def test_ncm_predict_picks_nearest_center():
    """逐帧 NCM 取欧氏距离最近的类中心。"""

    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    hiddens = [np.array([[0.5, 0.5], [9.0, 9.0], [6.0, 6.0]])]
    assert list(ncm_predict(hiddens, centers)[0]) == [0, 1, 1]


@pytest.mark.parametrize("model_type,model_cfg", [
    ("mstcn", {"hidden": 8}),
    ("mstcn2", {"hidden": 8, "num_stages": 2, "num_layers": 3}),
    ("asformer", {"hidden": 8, "heads": 2, "num_encoders": 2, "num_decoders": 1}),
    ("actionness_tcn", {"hidden": 8, "num_stages": 2, "num_layers": 3}),
    ("transformer", {"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16}),
    ("gru", {"hidden": 8, "num_layers": 1}),
    ("fact", {"d_model": 8, "nhead": 2, "num_blocks": 1, "num_tokens": 4, "frame_layers": 1}),
])
def test_hidden_hook_yields_time_major_hidden(model_type, model_cfg):
    """每个挂载点都必须给出 ``[T, H]`` 的表示，且与 logits 的时间轴长度一致。

    这条测试锁定 `HIDDEN_HOOKS` 里的 **布局声明**：卷积头给 ``[B,H,T]``、线性头给
    ``[B,T,H]``，声明写反时 `forward_with_hidden` 不会报错、只会静默产出错误的类均值，
    因此必须用形状断言把它钉住。
    """

    length = 7
    cfg = {"type": model_type, "input_dim": 12, "num_classes": 3, **model_cfg}
    model = build_model(cfg)
    features = [np.random.default_rng(0).normal(size=(length, 12)).astype(np.float32)]
    logits, hiddens = forward_with_hidden(model, features, model_type)
    assert logits[0].shape == (length, 3)
    assert hiddens[0].shape[0] == length, (
        f"{model_type}: 表示的时间轴应为 {length}，实际 {hiddens[0].shape}——"
        f"检查 HIDDEN_HOOKS 的 is_batch_time_hidden 声明是否写反"
    )
    assert hiddens[0].ndim == 2


def test_every_registered_temporal_model_has_a_hidden_hook():
    """挂载点登记的强制契约：新增时序模型必须同时补 `HIDDEN_HOOKS`。

    例外是仅供外部 checkpoint 兼容、不参与本轮诊断的 `clean_*` / `legacy_*` 家族；
    它们若将来需要 S-NCM，必须在此处显式登记而不是让测试放行。
    """

    exempt = {name for name in _MODELS if name.startswith(("clean_", "legacy_"))}
    missing = sorted(set(_MODELS) - set(HIDDEN_HOOKS) - exempt)
    assert missing == [], (
        f"以下已注册时序模型缺少 penultimate 表示挂载点: {missing}；"
        f"请在 framework/cleansight_eval/temporal/inference.py 的 HIDDEN_HOOKS 补登记"
    )

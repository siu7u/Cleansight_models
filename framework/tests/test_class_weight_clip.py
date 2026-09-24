"""类别权重截断区间（train.class_weight_clip）：活动量旋钮的口径与校验。

背景：idle 的归一化权重被下限截断抬到 0.1（原始倒数权重 ≈0.031），这个"抬升"决定了模型
有多"敢说"非 idle。把它开成配置项后，下限就成了与特征侧"删通道"等价的对照旋钮，因此
解析与校验必须有测试兜住（非法值不能静默退回默认）。
"""

from __future__ import annotations

import pytest

from framework.cleansight_eval.temporal.util import CLASS_WEIGHT_CLIP, compute_class_weights, resolve_class_weight_clip


class _FakeLoader:
    """最小 DataLoader 替身：只提供 compute_class_weights 需要的迭代协议。"""

    def __init__(self, labels):
        import torch

        class _DS:
            def __init__(self, labels):
                self.x = torch.zeros(len(labels), 1)
                self.y = torch.tensor(labels)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, index):
                return self.x[index], self.y[index]

        from torch.utils.data import DataLoader

        self._loader = DataLoader(_DS(labels), batch_size=4, shuffle=False)

    def __iter__(self):
        return iter(self._loader)


def test_resolve_clip_defaults_and_parses_forms():
    assert resolve_class_weight_clip(None) == CLASS_WEIGHT_CLIP
    assert resolve_class_weight_clip("") == CLASS_WEIGHT_CLIP
    assert resolve_class_weight_clip("0.03,5.0") == (0.03, 5.0)
    assert resolve_class_weight_clip([0.2, 4.0]) == (0.2, 4.0)


@pytest.mark.parametrize("bad", ["0.03", "0,5", "x,5", "5,0.1", 0.1, ["a", "b"], "-1,5"])
def test_resolve_clip_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        resolve_class_weight_clip(bad)


def test_clip_floor_sets_idle_weight():
    """下限就是 idle 的权重：0.03 → 还原倒数频率；0.2 → 比默认更强地强调多数类。"""

    labels = [0] * 6274 + [1] * 194 + [2] * 893 + [3] * 1289 + [4] * 437 + [5] * 488
    loader = _FakeLoader(labels)
    default = compute_class_weights(loader, num_classes=6)
    low = compute_class_weights(loader, num_classes=6, clip=(0.03, 5.0))
    high = compute_class_weights(loader, num_classes=6, clip=(0.2, 5.0))
    assert default[0] == pytest.approx(0.1)          # 默认下限把 idle 抬到 0.1
    assert low[0] < default[0] == 0.1 < high[0]      # 下限单调控制 idle 权重
    assert low[0] == pytest.approx(194 / 6274, rel=1e-3)  # 0.03 ≈ 原始倒数频率
    # 非 idle 类不受下限影响（它们远高于下限）
    for cls in (1, 2, 3, 4, 5):
        assert default[cls] == pytest.approx(low[cls])

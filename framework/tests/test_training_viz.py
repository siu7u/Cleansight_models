"""训练 history.csv 的无界面可视化测试。"""

from pathlib import Path

import pytest

from cleansight_eval.core.history import HistoryWriter, plot_training_history, try_plot_training_history


def test_plot_training_history_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    history = tmp_path / "history.csv"
    writer = HistoryWriter(
        history,
        ["epoch", "train_loss", "val_loss", "val_acc", "val_edit", "val_f1_0.5", "lr", "epoch_sec"],
    )
    writer.append(
        {
            "epoch": 1,
            "train_loss": 1.2,
            "val_loss": 1.4,
            "val_acc": 60.0,
            "val_edit": 55.0,
            "val_f1_0.5": 40.0,
            "lr": 0.001,
            "epoch_sec": 2.5,
        }
    )
    writer.append(
        {
            "epoch": 2,
            "train_loss": 0.9,
            "val_loss": 1.1,
            "val_acc": 70.0,
            "val_edit": 62.0,
            "val_f1_0.5": 52.0,
            "lr": 0.001,
            "epoch_sec": 2.2,
        }
    )

    output = plot_training_history(history, tmp_path / "training_curves.png")

    assert output == tmp_path / "training_curves.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_try_plot_training_history_does_not_raise_for_empty_history(tmp_path):
    history = tmp_path / "history.csv"
    HistoryWriter(history, ["epoch", "train_loss"])

    output, error = try_plot_training_history(history, tmp_path / "training_curves.png")

    assert output is None
    assert error is not None and "没有 epoch 记录" in error
    assert not Path(tmp_path / "training_curves.png").exists()

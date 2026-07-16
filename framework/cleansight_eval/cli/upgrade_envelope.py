"""把历史 schema v1 envelope 转换为非覆盖式 schema v2 文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.envelope import EvalEnvelope
from ..core.integrity import check_envelope_complete
from ..core.provenance import build_checkpoint_info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="升级 framework eval envelope 到 schema v2")
    parser.add_argument("--input", required=True, help="历史 *.envelope.json")
    parser.add_argument("--out", help="输出路径；默认在原文件名后增加 .v2.json，不覆盖历史文件")
    return parser.parse_args(argv)


def main(argv=None) -> str:
    """读取 v1/v2 envelope，补充能从本地恢复的字段并写新 v2 文件。"""

    args = parse_args(argv)
    source = Path(args.input)
    raw = json.loads(source.read_text(encoding="utf-8"))
    envelope = EvalEnvelope.from_dict(raw)
    if not envelope.run:
        envelope.run = {
            "id": f"legacy-import-{source.stem}",
            "created_at": envelope.timestamp,
            "device": "unknown",
            "source_envelope": str(source),
        }
    if not envelope.testset:
        envelope.testset = {
            "id": f"legacy:{envelope.dataset}",
            "registered": False,
            "dataset_version": envelope.dataset,
            "split": "unknown",
            "purpose": "legacy_import",
            "validation_errors": ["历史 envelope 缺少钉定 testset，不能补造 fingerprint"],
        }
    checkpoint = Path(envelope.checkpoint)
    if checkpoint.is_file() and not envelope.checkpoint_info:
        envelope.checkpoint_info = build_checkpoint_info(checkpoint)
    envelope.integrity = check_envelope_complete(envelope)

    output = Path(args.out) if args.out else source.with_name(source.stem + ".v2.json")
    if output.resolve() == source.resolve():
        raise ValueError("升级默认不允许覆盖历史 envelope，请指定新的 --out")
    envelope.write(output)
    print(f"[upgrade-envelope] {output}")
    return str(output)


if __name__ == "__main__":
    main()

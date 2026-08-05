import json

import pytest

from benchmark.core.delivery import (
    build_delivery_manifest,
    validate_delivery_manifest,
    write_delivery_manifest,
)


def test_delivery_manifest_hashes_required_files_and_roundtrips(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    evaluation = tmp_path / "result.evaluation.json"
    evaluation.write_text('{"schema_version": 2}\n', encoding="utf-8")

    manifest = build_delivery_manifest(
        run_id="run-1",
        model_id="gru-1k",
        base_dir=tmp_path,
        files=[
            ("checkpoint", checkpoint, True),
            ("evaluation", evaluation, True),
            ("card", tmp_path / "CARD.md", False),
        ],
    )
    assert {item["role"] for item in manifest["files"]} == {"checkpoint", "evaluation"}
    assert all(item["portable"] for item in manifest["files"])
    assert next(item for item in manifest["files"] if item["role"] == "evaluation")[
        "content_schema_version"
    ] == 2

    path = write_delivery_manifest(tmp_path / "delivery.manifest.json", manifest)
    validate_delivery_manifest(json.loads(path.read_text(encoding="utf-8")))


def test_delivery_manifest_rejects_missing_required_file(tmp_path):
    evaluation = tmp_path / "result.json"
    evaluation.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_delivery_manifest(
            run_id="run-1",
            model_id="gru",
            base_dir=tmp_path,
            files=[
                ("checkpoint", tmp_path / "missing.pt", True),
                ("evaluation", evaluation, True),
            ],
        )


def test_published_json_schemas_are_standalone_files():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "evaluation-result-v2.schema.json",
        "prediction-artifact-v1.schema.json",
        "delivery-manifest-v1.schema.json",
    ):
        payload = json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
        assert payload["$schema"].endswith("2020-12/schema")
        assert payload["type"] == "object"

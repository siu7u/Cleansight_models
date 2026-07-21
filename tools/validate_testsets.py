#!/usr/bin/env python3
"""验证统一 testset manifest、样本文件和跨 split 数据泄漏。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.core.testsets import (  # noqa: E402
    DEFAULT_CATALOG,
    get_testset,
    load_testsets,
    manifest_fingerprint,
    validate_catalog,
)


def parse_args() -> argparse.Namespace:
    """解析可选 testset 过滤和 JSON 输出参数。"""

    parser = argparse.ArgumentParser(description="验证 CleanSight benchmark testset manifest")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="testsets.yaml 路径")
    parser.add_argument("--testset", help="只报告一个 testset id；temporal 会同时检查同版本 train/test 互斥")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser.parse_args()


def _error_payload(message: str) -> dict:
    """构造清单无法读取时的统一失败结果。"""

    return {"ok": False, "catalog": None, "testsets": {}, "errors": [message]}


def main() -> int:
    """运行验证；任一错误返回退出码 2。"""

    args = parse_args()
    catalog_path = Path(args.catalog).expanduser().resolve()
    try:
        catalog = load_testsets(catalog_path)
        selected = get_testset(args.testset, catalog) if args.testset else None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        payload = _error_payload(str(exc))
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"[FAIL] {exc}")
        return 2

    validation_catalog = catalog
    if selected is not None:
        if selected.family == "temporal":
            validation_catalog = {
                testset_id: spec
                for testset_id, spec in catalog.items()
                if spec.family == "temporal" and spec.dataset_version == selected.dataset_version
            }
        else:
            validation_catalog = {selected.id: selected}
    validation = validate_catalog(validation_catalog)
    visible_ids = [selected.id] if selected is not None else sorted(catalog)

    output: dict[str, dict] = {}
    for testset_id in visible_ids:
        spec = catalog[testset_id]
        errors = validation.get(testset_id, [])
        fingerprint = None
        try:
            fingerprint = manifest_fingerprint(spec)
        except (OSError, TypeError, ValueError):
            pass
        output[testset_id] = {
            "ok": not errors,
            "dataset": spec.dataset,
            "family": spec.family,
            "dataset_version": spec.dataset_version,
            "split": spec.split,
            "purpose": spec.purpose,
            "split_overlap_policy": spec.split_overlap_policy,
            "fingerprint": fingerprint,
            "errors": errors,
        }

    ok = all(item["ok"] for item in output.values())
    payload = {
        "ok": ok,
        "catalog": str(catalog_path),
        "testsets": output,
        "errors": [],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for testset_id, item in output.items():
            print(f"[{'OK' if item['ok'] else 'FAIL'}] {testset_id}")
            print(f"  split_overlap_policy: {item['split_overlap_policy']}")
            if item["fingerprint"]:
                print(f"  fingerprint: {item['fingerprint']}")
            for error in item["errors"]:
                print(f"  - {error}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

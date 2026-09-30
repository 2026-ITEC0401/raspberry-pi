#!/usr/bin/env python3
"""Validate real Hearo v3.2 deployment artifacts without audio hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from hearo_hybrid_classifier import (  # noqa: E402
    HybridDecisionEngine,
    load_hybrid_policy,
)
from hearo_model_runtime import (  # noqa: E402
    HearoModelRuntime,
    ModelContractError,
    validate_repeatability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="마이크 없이 YAMNet v3.2 모델·메타데이터 계약과 추론을 검사합니다."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_DIR / "model",
        help="배포 모델 디렉터리 (기본값: 프로젝트/model)",
    )
    parser.add_argument(
        "--profile",
        default="v3_2",
        choices=("v3_2", "v3.2"),
        help="검증할 모델 프로필",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=3,
        help="동일 embedding 반복 추론 횟수 (2 이상)",
    )
    return parser.parse_args()


def validate_unknown_suppression(runtime: HearoModelRuntime) -> None:
    contract = runtime.contract
    policy_path = contract.model_dir / "hybrid_policy_v3.json"
    if not policy_path.is_file():
        raise ModelContractError(f"하이브리드 정책 파일이 없습니다: {policy_path}")
    policy = load_hybrid_policy(policy_path)
    engine = HybridDecisionEngine(
        policy=policy,
        categories=contract.categories,
        unknown_label=contract.unknown_label,
        class_thresholds=contract.class_thresholds,
        class_mapping=contract.class_mapping,
        delivery_policy=contract.delivery_policy,
        base_decision_source="hearo_v3_2",
        threshold_clip_range=(0.0, 1.0),
    )
    scores = np.zeros((1, 521), dtype=np.float32)
    probabilities = np.zeros(len(contract.categories), dtype=np.float32)
    probabilities[contract.unknown_index] = 1.0
    decision = engine.decide_from_outputs(scores, probabilities, "validation-device")
    if decision.sound is not None or decision.raw_label != contract.unknown_label:
        raise ModelContractError("unknown_label이 경고 없이 억제되지 않았습니다.")


def main() -> int:
    args = parse_args()
    if args.repeat_count < 2:
        raise ValueError("--repeat-count는 2 이상이어야 합니다.")

    runtime = HearoModelRuntime.load(args.model_dir, args.profile)
    contract = runtime.contract

    # Deterministic, hardware-free YAMNet inference at the metadata sample rate.
    time_axis = np.arange(contract.sample_rate, dtype=np.float32) / contract.sample_rate
    waveform = (0.01 * np.sin(2.0 * np.pi * 440.0 * time_axis)).astype(np.float32)
    scores, embeddings = runtime.run_yamnet(waveform)
    if scores.ndim != 2 or scores.shape[1] != 521:
        raise ModelContractError(f"YAMNet score shape 오류: {scores.shape}")
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ModelContractError(f"YAMNet embedding shape 오류: {embeddings.shape}")

    probabilities = validate_repeatability(
        runtime,
        embeddings,
        repeats=args.repeat_count,
    )
    if probabilities.shape[-1] != len(contract.categories):
        raise ModelContractError(
            "분류기 출력 클래스 수가 categories와 다릅니다: "
            f"{probabilities.shape[-1]} != {len(contract.categories)}"
        )
    if not np.all(np.isfinite(scores)):
        raise ModelContractError("YAMNet score에 NaN 또는 infinity가 있습니다.")
    if not np.all(np.isfinite(embeddings)):
        raise ModelContractError("YAMNet embedding에 NaN 또는 infinity가 있습니다.")
    if not np.all(np.isfinite(probabilities)):
        raise ModelContractError("분류기 출력에 NaN 또는 infinity가 있습니다.")

    validate_unknown_suppression(runtime)

    summary = {
        "status": "ok",
        "profile": contract.profile,
        "model_name": contract.model_name,
        "selected_tflite_variant": contract.selected_tflite_variant,
        "temperature_embedded_in_tflite": contract.temperature_embedded_in_tflite,
        "interpreter": runtime.interpreter_runtime_name,
        "yamnet_score_shape": list(scores.shape),
        "yamnet_embedding_shape": list(embeddings.shape),
        "classifier_output_shape": list(probabilities.shape),
        "category_count": len(contract.categories),
        "unknown_label": contract.unknown_label,
        "repeat_count": args.repeat_count,
        "unknown_alert_suppressed": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("HEARO V3.2 MODEL VALIDATION OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ModelContractError, OSError, ValueError) as exc:
        print(f"HEARO V3.2 MODEL VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

from __future__ import annotations

import json

import numpy as np
import pytest

from hearo_hybrid_classifier import (
    HybridDecisionEngine,
    HybridPolicyError,
    fallback_policy,
    load_hybrid_policy,
    validate_hybrid_policy,
)


CATEGORIES = [
    "노크_목재",
    "노크_철재문",
    "도어락_개방음",
    "도어락_입력음",
    "사이렌_삐뽀삐뽀",
    "사이렌_안내음",
    "사이렌_애애애애앵",
    "사이렌_철철철",
    "아기 울음",
    "비표적음",
]
MAPPING = {
    "노크_목재": "노크소리",
    "노크_철재문": "노크소리",
    "도어락_개방음": "도어락소리",
    "도어락_입력음": "도어락소리",
    "사이렌_삐뽀삐뽀": "비상벨소리",
    "사이렌_안내음": "비상벨소리",
    "사이렌_애애애애앵": "비상벨소리",
    "사이렌_철철철": "비상벨소리",
    "아기 울음": "아기울음소리",
}
THRESHOLDS = {label: 0.6 for label in CATEGORIES}


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 1.0) -> None:
        self.value += seconds


def enabled_policy(*, shadow_mode: bool = False):
    return {
        "schema_version": 3,
        "policy_version": "test-hybrid-v3",
        "enabled": True,
        "shadow_mode": shadow_mode,
        "fallback": "hearo_v2",
        "runtime": {
            "history_expiry_seconds": 4.0,
            "max_window_age_seconds": 2.0,
            "stream_gap_reset_ms": 2000,
        },
        "families": {
            "critical_siren": {
                "indices": [316, 317, 318, 319, 390, 391, 393, 394],
                "frame_pooling": "max_class_then_max_time",
                "threshold": 0.8,
                "required_hits": 2,
                "history_size": 3,
            },
            "baby": {
                "core_indices": [20],
                "support_indices": [19, 21, 22],
                "frame_pooling": "max_class_then_max_time",
                "threshold": 0.8,
                "combined_core_threshold": 0.5,
                "support_threshold": 0.7,
                "required_hits": 2,
                "history_size": 3,
            },
            "knock": {
                "core_indices": [353],
                "support_indices": [354],
                "frame_pooling": "max_class_then_max_time",
                "yamnet_threshold": 0.7,
                "hearo_floor": 0.4,
            },
            "door_soft_support": {
                "indices": [348, 351, 352, 386, 475, 476, 477],
                "frame_pooling": "max_class_then_max_time",
                "max_threshold_bonus": 0.2,
                "minimum_hearo_floor": 0.35,
            },
        },
    }


def engine(policy=None, clock=None):
    return HybridDecisionEngine(
        policy=policy or enabled_policy(),
        categories=CATEGORIES,
        unknown_label="비표적음",
        class_thresholds=THRESHOLDS,
        class_mapping=MAPPING,
        clock=clock or Clock(),
    )


def scores(**values):
    result = np.zeros((4, 521), dtype=np.float32)
    for index, value in values.items():
        result[:, int(index)] = value
    return result


def probabilities(**values):
    result = np.zeros(len(CATEGORIES), dtype=np.float32)
    for label, value in values.items():
        result[CATEGORIES.index(label)] = value
    return result


def unknown_probabilities():
    return probabilities(비표적음=0.9)


def test_disabled_policy_is_exact_v2_fallback():
    decision = engine(fallback_policy("test disabled")).decide_from_outputs(
        scores(**{"390": 0.99}),
        probabilities(**{"노크_목재": 0.7, "비표적음": 0.3}),
        "rpi-001",
    )
    assert decision.sound == "노크소리"
    assert decision.decision_source == "hearo_v2"
    assert decision.yamnet_score is None


def test_enabled_policy_with_unselected_values_fails_closed():
    policy = enabled_policy()
    policy["families"]["critical_siren"]["threshold"] = None
    with pytest.raises(HybridPolicyError, match="threshold"):
        validate_hybrid_policy(policy)


def test_missing_policy_file_falls_back_but_invalid_enabled_policy_stops(tmp_path):
    missing = load_hybrid_policy(tmp_path / "missing.json")
    assert missing["enabled"] is False
    assert "FileNotFoundError" in missing["fallback_reason"]

    invalid = enabled_policy()
    invalid["families"]["knock"]["hearo_floor"] = None
    path = tmp_path / "invalid-enabled.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(HybridPolicyError, match="hearo_floor"):
        load_hybrid_policy(path)


def test_siren_requires_persistence_and_histories_are_source_local():
    clock = Clock()
    classifier = engine(clock=clock)
    high = scores(**{"390": 0.9})
    first = classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_1")
    assert first.sound is None

    clock.advance()
    other = classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_2")
    assert other.sound is None

    clock.advance()
    second = classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_1")
    assert second.sound == "비상벨소리"
    assert second.decision_source == "yamnet_safety_override"


def test_baby_support_only_never_alerts():
    clock = Clock()
    classifier = engine(clock=clock)
    support_only = scores(**{"19": 0.99, "21": 0.99, "22": 0.99})
    for _ in range(3):
        decision = classifier.decide_from_outputs(
            support_only, unknown_probabilities(), "esp32_3"
        )
        assert decision.sound is None
        clock.advance()


def test_knock_requires_hearo_floor_and_yamnet_together():
    classifier = engine()
    yamnet_knock = scores(**{"353": 0.9})
    assert classifier.decide_from_outputs(
        yamnet_knock, unknown_probabilities(), "esp32_1"
    ).sound is None

    fused = classifier.decide_from_outputs(
        yamnet_knock,
        probabilities(**{"노크_목재": 0.45, "비표적음": 0.55}),
        "esp32_1",
    )
    assert fused.sound == "노크소리"
    assert fused.decision_source == "hearo_yamnet_fusion"


def test_door_yamnet_alone_is_blocked_and_bonus_is_capped():
    classifier = engine()
    door_support = scores(**{"475": 1.0})
    assert classifier.decide_from_outputs(
        door_support, unknown_probabilities(), "esp32_2"
    ).sound is None

    supported = classifier.decide_from_outputs(
        door_support,
        probabilities(**{"도어락_입력음": 0.45, "비표적음": 0.55}),
        "esp32_2",
    )
    assert supported.sound == "도어락소리"
    assert supported.applied_threshold == pytest.approx(0.4)
    assert THRESHOLDS["도어락_입력음"] - supported.applied_threshold <= 0.2


def test_siren_priority_wins_when_baby_also_passes():
    clock = Clock()
    classifier = engine(clock=clock)
    both = scores(**{"390": 0.95, "20": 0.95})
    classifier.decide_from_outputs(both, unknown_probabilities(), "rpi-001")
    clock.advance()
    decision = classifier.decide_from_outputs(both, unknown_probabilities(), "rpi-001")
    assert decision.sound == "비상벨소리"


def test_reset_and_stale_window_cannot_complete_old_history():
    clock = Clock()
    classifier = engine(clock=clock)
    high = scores(**{"390": 0.9})
    classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_3")
    classifier.reset_source("esp32_3")
    clock.advance()
    assert classifier.decide_from_outputs(
        high, unknown_probabilities(), "esp32_3"
    ).sound is None

    stale = classifier.decide_from_outputs(
        high,
        unknown_probabilities(),
        "esp32_3",
        observed_at=clock.value - 3.0,
    )
    assert stale.sound is None
    assert stale.decision_source == "hybrid_stale_rejected"


def test_waveform_path_invokes_yamnet_once():
    calls = {"yamnet": 0, "classifier": 0}

    def run_yamnet(waveform):
        calls["yamnet"] += 1
        return scores(), np.zeros((4, 1024), dtype=np.float32)

    def run_classifier(embeddings):
        calls["classifier"] += 1
        return np.tile(unknown_probabilities(), (4, 1))

    classifier = HybridDecisionEngine(
        policy=fallback_policy("test"),
        categories=CATEGORIES,
        unknown_label="비표적음",
        class_thresholds=THRESHOLDS,
        class_mapping=MAPPING,
        run_yamnet=run_yamnet,
        run_classifier=run_classifier,
        aggregate_context=lambda values: values.mean(axis=0),
    )
    classifier.classify(np.zeros(32_000, dtype=np.float32), "rpi-001")
    assert calls == {"yamnet": 1, "classifier": 1}


def test_shadow_mode_keeps_v2_delivery_and_only_records_v3_candidate():
    clock = Clock()
    classifier = engine(enabled_policy(shadow_mode=True), clock)
    high = scores(**{"390": 0.9})
    classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_1")
    clock.advance()
    decision = classifier.decide_from_outputs(high, unknown_probabilities(), "esp32_1")
    assert decision.sound is None
    assert decision.decision_source == "hearo_v2_shadow_rejected"
    assert decision.shadow_sound == "비상벨소리"
    assert decision.shadow_decision_source == "yamnet_safety_override"

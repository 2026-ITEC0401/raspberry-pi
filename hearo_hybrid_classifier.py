"""Pure hybrid decision layer for Hearo v3.

The module has no GPIO, MQTT, socket, or TensorFlow imports.  Production code
injects the existing YAMNet/Hearo v2 inference functions, while unit tests can
exercise the policy with synthetic score matrices.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


YAMNET_CLASS_COUNT = 521
SUPPORTED_POOLING = {
    "max_class_then_max_time",
    "max_class_then_mean_time",
    "max_class_then_topk_time",
}
SOUND_PRIORITY = {
    "비상벨소리": 4,
    "아기울음소리": 3,
    "노크소리": 2,
    "도어락소리": 2,
}


class HybridPolicyError(ValueError):
    """Raised when an enabled policy is unsafe or incomplete."""


@dataclass(frozen=True)
class ClassificationDecision:
    sound: str | None
    raw_label: str
    confidence: float
    decision_source: str
    confidence_kind: str
    yamnet_family: str | None
    yamnet_score: float | None
    hearo_confidence: float | None
    applied_threshold: float | None
    policy_version: str
    shadow_sound: str | None = None
    shadow_decision_source: str | None = None

    def diagnostic_fields(self) -> dict[str, Any]:
        """Return optional MQTT/API fields without changing legacy fields."""
        return {
            "decision_source": self.decision_source,
            "confidence_kind": self.confidence_kind,
            "yamnet_family": self.yamnet_family,
            "yamnet_score": _rounded(self.yamnet_score),
            "hearo_confidence": _rounded(self.hearo_confidence),
            "applied_threshold": _rounded(self.applied_threshold),
            "policy_version": self.policy_version,
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def fallback_policy(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "policy_version": "hearo-hybrid-v3-fallback",
        "enabled": False,
        "shadow_mode": False,
        "fallback": "hearo_v2",
        "fallback_reason": reason,
        "families": {},
    }


def load_hybrid_policy(path: Path) -> dict[str, Any]:
    """Load a policy, falling back only when it is absent/unreadable/disabled-invalid.

    A syntactically valid policy that explicitly enables hybrid decisions must
    fail closed when required experiment values are absent or invalid.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            policy = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return fallback_policy(f"{type(exc).__name__}: {exc}")

    try:
        validate_hybrid_policy(policy)
    except HybridPolicyError as exc:
        if isinstance(policy, dict) and policy.get("enabled") is True:
            raise
        return fallback_policy(f"invalid disabled policy: {exc}")
    return policy


def _require_number(
    mapping: Mapping[str, Any], key: str, *, minimum: float = 0.0, maximum: float = 1.0
) -> float:
    value = mapping.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridPolicyError(f"{key}는 실험으로 선택한 숫자여야 합니다.")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise HybridPolicyError(f"{key}는 {minimum}~{maximum} 범위여야 합니다.")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, minimum: int = 1) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HybridPolicyError(f"{key}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _validate_indices(family: Mapping[str, Any], key: str) -> list[int]:
    values = family.get(key)
    if not isinstance(values, list) or not values:
        raise HybridPolicyError(f"{key}는 비어 있지 않은 YAMNet 인덱스 배열이어야 합니다.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise HybridPolicyError(f"{key}에는 정수 인덱스만 사용할 수 있습니다.")
    if any(value < 0 or value >= YAMNET_CLASS_COUNT for value in values):
        raise HybridPolicyError(f"{key}에 0~520 범위를 벗어난 인덱스가 있습니다.")
    if len(values) != len(set(values)):
        raise HybridPolicyError(f"{key}에 중복 인덱스가 있습니다.")
    return values


def _validate_pooling(family: Mapping[str, Any]) -> None:
    pooling = family.get("frame_pooling")
    if pooling not in SUPPORTED_POOLING:
        raise HybridPolicyError(f"지원하지 않는 frame_pooling입니다: {pooling}")
    if pooling == "max_class_then_topk_time":
        _require_int(family, "top_k")


def validate_hybrid_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping):
        raise HybridPolicyError("정책 최상위 값은 JSON object여야 합니다.")
    if policy.get("schema_version") != 3:
        raise HybridPolicyError("schema_version은 3이어야 합니다.")
    if not isinstance(policy.get("policy_version"), str) or not policy["policy_version"].strip():
        raise HybridPolicyError("policy_version이 필요합니다.")
    if not isinstance(policy.get("enabled"), bool):
        raise HybridPolicyError("enabled는 boolean이어야 합니다.")
    if policy.get("fallback") != "hearo_v2":
        raise HybridPolicyError("fallback은 hearo_v2여야 합니다.")
    if "shadow_mode" in policy and not isinstance(policy["shadow_mode"], bool):
        raise HybridPolicyError("shadow_mode는 boolean이어야 합니다.")
    if not policy["enabled"]:
        return

    runtime = policy.get("runtime")
    if not isinstance(runtime, Mapping):
        raise HybridPolicyError("enabled 정책에는 runtime 설정이 필요합니다.")
    _require_number(runtime, "history_expiry_seconds", minimum=0.1, maximum=60.0)
    _require_number(runtime, "max_window_age_seconds", minimum=0.1, maximum=30.0)
    _require_int(runtime, "stream_gap_reset_ms", minimum=1)

    families = policy.get("families")
    required = {"critical_siren", "baby", "knock", "door_soft_support"}
    if not isinstance(families, Mapping) or not required.issubset(families):
        raise HybridPolicyError(f"families에는 {sorted(required)}가 모두 필요합니다.")
    for family_name in required:
        if not isinstance(families[family_name], Mapping):
            raise HybridPolicyError(f"{family_name} family는 JSON object여야 합니다.")

    siren = families["critical_siren"]
    _validate_indices(siren, "indices")
    _validate_pooling(siren)
    _require_number(siren, "threshold")
    hits = _require_int(siren, "required_hits")
    size = _require_int(siren, "history_size")
    if hits > size:
        raise HybridPolicyError("critical_siren required_hits는 history_size 이하여야 합니다.")

    baby = families["baby"]
    _validate_indices(baby, "core_indices")
    _validate_indices(baby, "support_indices")
    _validate_pooling(baby)
    _require_number(baby, "threshold")
    _require_number(baby, "combined_core_threshold")
    _require_number(baby, "support_threshold")
    hits = _require_int(baby, "required_hits")
    size = _require_int(baby, "history_size")
    if hits > size:
        raise HybridPolicyError("baby required_hits는 history_size 이하여야 합니다.")

    knock = families["knock"]
    _validate_indices(knock, "core_indices")
    _validate_indices(knock, "support_indices")
    _validate_pooling(knock)
    _require_number(knock, "yamnet_threshold")
    _require_number(knock, "hearo_floor")

    door = families["door_soft_support"]
    _validate_indices(door, "indices")
    _validate_pooling(door)
    _require_number(door, "max_threshold_bonus", maximum=0.5)
    _require_number(door, "minimum_hearo_floor")


EXPECTED_YAMNET_NAMES: dict[int, tuple[str, ...]] = {
    19: ("crying", "sobbing"),
    20: ("baby cry", "infant cry"),
    21: ("whimper",),
    22: ("wail", "moan"),
    316: ("emergency vehicle",),
    317: ("police car", "siren"),
    318: ("ambulance", "siren"),
    319: ("fire engine", "fire truck", "siren"),
    348: ("door",),
    351: ("sliding door",),
    352: ("slam",),
    353: ("knock",),
    354: ("tap",),
    386: ("dtmf",),
    390: ("siren",),
    391: ("civil defense siren",),
    393: ("smoke detector", "smoke alarm"),
    394: ("fire alarm",),
    475: ("beep", "bleep"),
    476: ("ping",),
    477: ("ding",),
}


def validate_yamnet_class_names(policy: Mapping[str, Any], class_names: Sequence[str]) -> None:
    if not policy.get("enabled") or not class_names:
        return
    if len(class_names) != YAMNET_CLASS_COUNT:
        raise HybridPolicyError(
            f"yamnet_classes.txt는 {YAMNET_CLASS_COUNT}개여야 합니다: {len(class_names)}"
        )
    families = policy["families"]
    indices: set[int] = set()
    for family in families.values():
        for key in ("indices", "core_indices", "support_indices"):
            indices.update(family.get(key, []))
    for index in indices:
        expected = EXPECTED_YAMNET_NAMES.get(index)
        if expected and not any(fragment in class_names[index].casefold() for fragment in expected):
            raise HybridPolicyError(
                f"YAMNet class map 불일치: index={index}, actual={class_names[index]!r}, "
                f"expected one of {expected}"
            )


class HybridDecisionEngine:
    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        categories: Sequence[str],
        unknown_label: str,
        class_thresholds: Mapping[str, float],
        class_mapping: Mapping[str, str],
        delivery_policy: Mapping[str, Mapping[str, Any]] | None = None,
        sensitivity_offset: Callable[[], float] | None = None,
        run_yamnet: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        run_classifier: Callable[[np.ndarray], np.ndarray] | None = None,
        aggregate_context: Callable[[np.ndarray], np.ndarray] | None = None,
        yamnet_gate_allows: Callable[[np.ndarray], bool] | None = None,
        inference_lock: threading.RLock | threading.Lock | None = None,
        yamnet_class_names: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ):
        validate_hybrid_policy(policy)
        validate_yamnet_class_names(policy, yamnet_class_names)
        self.policy = dict(policy)
        self.categories = list(categories)
        self.unknown_label = unknown_label
        self.class_thresholds = {key: float(value) for key, value in class_thresholds.items()}
        self.class_mapping = dict(class_mapping)
        self.delivery_policy = dict(delivery_policy or {})
        self.sensitivity_offset = sensitivity_offset or (lambda: 0.0)
        self.run_yamnet = run_yamnet
        self.run_classifier = run_classifier
        self.aggregate_context = aggregate_context
        self.yamnet_gate_allows = yamnet_gate_allows or (lambda scores: True)
        self.inference_lock = inference_lock or threading.RLock()
        self.clock = clock
        self._history: dict[str, dict[str, deque[tuple[float, float]]]] = defaultdict(dict)
        self._last_capture_ms: dict[str, int] = {}
        self._state_lock = threading.RLock()

        if unknown_label not in self.categories:
            raise HybridPolicyError("unknown_label이 categories에 없습니다.")
        missing = [label for label in self.categories if label not in self.class_thresholds]
        if missing:
            raise HybridPolicyError(f"class threshold가 없는 클래스가 있습니다: {missing}")

    @property
    def enabled(self) -> bool:
        return bool(self.policy.get("enabled"))

    @property
    def policy_version(self) -> str:
        return str(self.policy.get("policy_version", "hearo-hybrid-v3-fallback"))

    def reset_source(self, source_id: str) -> None:
        with self._state_lock:
            self._history.pop(source_id, None)
            self._last_capture_ms.pop(source_id, None)

    def classify(
        self,
        waveform: np.ndarray,
        source_id: str,
        *,
        capture_ms: int | None = None,
        observed_at: float | None = None,
    ) -> ClassificationDecision:
        if self.run_yamnet is None or self.run_classifier is None or self.aggregate_context is None:
            raise RuntimeError("실제 waveform 분류에는 주입된 YAMNet/Hearo 함수가 필요합니다.")
        with self.inference_lock:
            scores, embeddings = self.run_yamnet(waveform)
            frame_probabilities = self.run_classifier(embeddings)
            probabilities = self.aggregate_context(frame_probabilities)
        return self.decide_from_outputs(
            scores,
            probabilities,
            source_id,
            capture_ms=capture_ms,
            observed_at=observed_at,
            gate_allowed=self.yamnet_gate_allows(scores),
        )

    def decide_from_outputs(
        self,
        yamnet_scores: np.ndarray,
        hearo_probabilities: np.ndarray,
        source_id: str,
        *,
        capture_ms: int | None = None,
        observed_at: float | None = None,
        gate_allowed: bool = True,
    ) -> ClassificationDecision:
        scores = np.asarray(yamnet_scores, dtype=np.float64)
        probabilities = np.asarray(hearo_probabilities, dtype=np.float64).reshape(-1)
        if scores.ndim != 2 or scores.shape[1] != YAMNET_CLASS_COUNT:
            raise ValueError(f"YAMNet scores shape은 [frames, 521]이어야 합니다: {scores.shape}")
        if probabilities.shape != (len(self.categories),):
            raise ValueError(
                f"Hearo probabilities shape은 [{len(self.categories)}]여야 합니다: {probabilities.shape}"
            )
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(probabilities)):
            raise ValueError("분류 점수에 NaN 또는 infinity가 있습니다.")
        if np.any(scores < -1e-6) or np.any(scores > 1.0 + 1e-6):
            raise ValueError("YAMNet scores는 0~1 범위여야 합니다.")
        if np.any(probabilities < -1e-6) or np.any(probabilities > 1.0 + 1e-6):
            raise ValueError("Hearo probabilities는 0~1 범위여야 합니다.")

        base = self._hearo_v2_decision(probabilities, gate_allowed)
        if not self.enabled:
            return base

        now = self.clock()
        window_time = now if observed_at is None else float(observed_at)
        runtime = self.policy["runtime"]
        if now - window_time > float(runtime["max_window_age_seconds"]):
            self.reset_source(source_id)
            return ClassificationDecision(
                sound=None,
                raw_label="stale_window_rejected",
                confidence=0.0,
                decision_source="hybrid_stale_rejected",
                confidence_kind="none",
                yamnet_family=None,
                yamnet_score=None,
                hearo_confidence=base.hearo_confidence,
                applied_threshold=None,
                policy_version=self.policy_version,
            )

        self._reset_on_capture_discontinuity(source_id, capture_ms)
        hybrid = self._hybrid_decision(scores, probabilities, source_id, window_time, base)
        if self.policy.get("shadow_mode", False):
            return replace(
                base,
                decision_source=("hearo_v2_shadow" if base.sound else "hearo_v2_shadow_rejected"),
                shadow_sound=hybrid.sound,
                shadow_decision_source=hybrid.decision_source,
            )
        return hybrid

    def _reset_on_capture_discontinuity(self, source_id: str, capture_ms: int | None) -> None:
        if capture_ms is None:
            return
        capture_ms = int(capture_ms)
        with self._state_lock:
            previous = self._last_capture_ms.get(source_id)
            if previous is not None:
                delta = capture_ms - previous
                if delta <= 0 or delta > int(self.policy["runtime"]["stream_gap_reset_ms"]):
                    self._history.pop(source_id, None)
            self._last_capture_ms[source_id] = capture_ms

    def _hearo_v2_decision(
        self, probabilities: np.ndarray, gate_allowed: bool
    ) -> ClassificationDecision:
        best_index = int(np.argmax(probabilities))
        raw_label = self.categories[best_index]
        confidence = float(probabilities[best_index])
        threshold = self._threshold(raw_label)
        common = {
            "confidence": confidence,
            "confidence_kind": "hearo_probability",
            "yamnet_family": None,
            "yamnet_score": None,
            "hearo_confidence": confidence,
            "applied_threshold": threshold,
            "policy_version": self.policy_version,
        }
        if not gate_allowed:
            return ClassificationDecision(
                sound=None,
                raw_label="yamnet_gate_rejected",
                decision_source="hearo_v2",
                **common,
            )
        if raw_label == self.unknown_label or confidence < threshold:
            return ClassificationDecision(
                sound=None,
                raw_label=raw_label,
                decision_source="hearo_v2",
                **common,
            )
        if self.delivery_policy.get(raw_label, {}).get("publish_enabled") is False:
            return ClassificationDecision(
                sound=None,
                raw_label=raw_label,
                decision_source="hearo_v2_local_only",
                **common,
            )
        mapped = self.class_mapping.get(raw_label)
        if mapped not in SOUND_PRIORITY:
            return ClassificationDecision(
                sound=None,
                raw_label=raw_label,
                decision_source="hearo_v2_unmapped",
                **common,
            )
        return ClassificationDecision(
            sound=mapped,
            raw_label=raw_label,
            decision_source="hearo_v2",
            **common,
        )

    def _threshold(self, raw_label: str) -> float:
        base = self.class_thresholds[raw_label] + float(self.sensitivity_offset())
        return float(np.clip(base, 0.05, 0.99))

    def _best_candidate(self, probabilities: np.ndarray, sound: str) -> tuple[str, float]:
        candidates = [
            (label, float(probabilities[index]))
            for index, label in enumerate(self.categories)
            if self.class_mapping.get(label) == sound
            and self.delivery_policy.get(label, {}).get("publish_enabled") is not False
        ]
        if not candidates:
            return self.unknown_label, 0.0
        return max(candidates, key=lambda item: item[1])

    def _family_score(
        self, scores: np.ndarray, family: Mapping[str, Any], index_key: str
    ) -> float:
        indices = np.asarray(family[index_key], dtype=int)
        per_frame = np.max(scores[:, indices], axis=1)
        pooling = family["frame_pooling"]
        if pooling == "max_class_then_max_time":
            return float(np.max(per_frame))
        if pooling == "max_class_then_mean_time":
            return float(np.mean(per_frame))
        top_k = min(len(per_frame), int(family["top_k"]))
        return float(np.partition(per_frame, len(per_frame) - top_k)[-top_k:].mean())

    def _record_and_count_hits(
        self,
        source_id: str,
        family_name: str,
        value: float,
        threshold: float,
        history_size: int,
        timestamp: float,
    ) -> int:
        expiry = float(self.policy["runtime"]["history_expiry_seconds"])
        with self._state_lock:
            source = self._history[source_id]
            history = source.get(family_name)
            if history is None or history.maxlen != history_size:
                history = deque(maxlen=history_size)
                source[family_name] = history
            while history and timestamp - history[0][0] > expiry:
                history.popleft()
            history.append((timestamp, value))
            return sum(score >= threshold for _, score in history)

    def _hybrid_decision(
        self,
        scores: np.ndarray,
        probabilities: np.ndarray,
        source_id: str,
        timestamp: float,
        base: ClassificationDecision,
    ) -> ClassificationDecision:
        families = self.policy["families"]
        candidates: list[ClassificationDecision] = []
        if base.sound is not None:
            candidates.append(base)

        siren = families["critical_siren"]
        siren_score = self._family_score(scores, siren, "indices")
        siren_hits = self._record_and_count_hits(
            source_id,
            "critical_siren",
            siren_score,
            float(siren["threshold"]),
            int(siren["history_size"]),
            timestamp,
        )
        if siren_hits >= int(siren["required_hits"]):
            source = "both" if base.sound == "비상벨소리" else "yamnet_safety_override"
            candidates.append(self._yamnet_decision(
                "비상벨소리", "yamnet_critical_siren", source,
                "critical_siren", siren_score, base.hearo_confidence,
                float(siren["threshold"]),
            ))

        baby = families["baby"]
        baby_core = self._family_score(scores, baby, "core_indices")
        baby_support = self._family_score(scores, baby, "support_indices")
        core_hits = self._record_and_count_hits(
            source_id,
            "baby_core",
            baby_core,
            float(baby["threshold"]),
            int(baby["history_size"]),
            timestamp,
        )
        combined_value = min(
            baby_core / max(float(baby["combined_core_threshold"]), 1e-9),
            baby_support / max(float(baby["support_threshold"]), 1e-9),
        )
        combined_hits = self._record_and_count_hits(
            source_id,
            "baby_combined",
            combined_value,
            1.0,
            int(baby["history_size"]),
            timestamp,
        )
        if max(core_hits, combined_hits) >= int(baby["required_hits"]):
            score = baby_core if core_hits >= int(baby["required_hits"]) else min(baby_core, baby_support)
            source = "both" if base.sound == "아기울음소리" else "yamnet_baby_override"
            candidates.append(self._yamnet_decision(
                "아기울음소리", "yamnet_baby", source, "baby", score,
                base.hearo_confidence, float(baby["threshold"]),
            ))

        knock = families["knock"]
        knock_core = self._family_score(scores, knock, "core_indices")
        knock_support = self._family_score(scores, knock, "support_indices")
        knock_score = max(knock_core, knock_support)
        knock_label, knock_hearo = self._best_candidate(probabilities, "노크소리")
        if (
            knock_hearo >= float(knock["hearo_floor"])
            and knock_score >= float(knock["yamnet_threshold"])
        ):
            source = "both" if base.sound == "노크소리" else "hearo_yamnet_fusion"
            candidates.append(ClassificationDecision(
                sound="노크소리",
                raw_label=knock_label,
                confidence=knock_hearo,
                decision_source=source,
                confidence_kind="hearo_probability_with_yamnet_support",
                yamnet_family="knock",
                yamnet_score=knock_score,
                hearo_confidence=knock_hearo,
                applied_threshold=float(knock["hearo_floor"]),
                policy_version=self.policy_version,
            ))

        door = families["door_soft_support"]
        door_score = self._family_score(scores, door, "indices")
        door_label, door_hearo = self._best_candidate(probabilities, "도어락소리")
        original_threshold = self._threshold(door_label)
        bonus = min(
            float(door["max_threshold_bonus"]),
            float(door["max_threshold_bonus"]) * max(0.0, min(1.0, door_score)),
        )
        adjusted_threshold = max(
            float(door["minimum_hearo_floor"]), original_threshold - bonus
        )
        if door_hearo >= adjusted_threshold and door_hearo >= float(door["minimum_hearo_floor"]):
            source = "both" if base.sound == "도어락소리" else "hearo_yamnet_soft_support"
            candidates.append(ClassificationDecision(
                sound="도어락소리",
                raw_label=door_label,
                confidence=door_hearo,
                decision_source=source,
                confidence_kind="hearo_probability_with_yamnet_soft_support",
                yamnet_family="door_soft_support",
                yamnet_score=door_score,
                hearo_confidence=door_hearo,
                applied_threshold=adjusted_threshold,
                policy_version=self.policy_version,
            ))

        if not candidates:
            return replace(base, decision_source="hybrid_rejected")
        return max(
            candidates,
            key=lambda item: (SOUND_PRIORITY.get(item.sound or "", 0), item.confidence),
        )

    def _yamnet_decision(
        self,
        sound: str,
        raw_label: str,
        decision_source: str,
        family: str,
        score: float,
        hearo_confidence: float | None,
        threshold: float,
    ) -> ClassificationDecision:
        return ClassificationDecision(
            sound=sound,
            raw_label=raw_label,
            confidence=score,
            decision_source=decision_source,
            confidence_kind="yamnet_family_score",
            yamnet_family=family,
            yamnet_score=score,
            hearo_confidence=hearo_confidence,
            applied_threshold=threshold,
            policy_version=self.policy_version,
        )

"""Hearo v3 inference hub for the Pi microphone and three ESP32 streams.

YAMNet still runs once per window. The optional hybrid policy reuses its 521
scores beside the existing Hearo v2 classifier; an absent/disabled policy falls
back to the original v2 decision without changing the TFLite models.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import appAWS_v2 as base
from hearo_audio_receiver import AudioUdpReceiver, AudioWindow
from hearo_hybrid_classifier import (
    ClassificationDecision,
    HybridDecisionEngine,
    load_hybrid_policy,
)


REMOTE_COOLDOWN_SECONDS = float(
    os.getenv("HEARO_REMOTE_COOLDOWN_SECONDS", str(base.COOLDOWN_SECONDS))
)
POLICY_PATH = Path(
    os.getenv("HEARO_HYBRID_POLICY_PATH", str(base.MODEL_DIR / "hybrid_policy_v3.json"))
)

_cooldown_lock = threading.Lock()
_last_alert: dict[tuple[str, str], float] = {}


def _sensitivity_offset() -> float:
    return float(base.SENSITIVITY_THRESHOLD_OFFSETS[base.cloud_runtime.sensitivity])


HYBRID_POLICY = load_hybrid_policy(POLICY_PATH)
decision_engine = HybridDecisionEngine(
    policy=HYBRID_POLICY,
    categories=base.CATEGORIES,
    unknown_label=base.UNKNOWN_LABEL,
    class_thresholds=base.CLASS_THRESHOLDS,
    class_mapping=base.CLASS_MAPPING,
    delivery_policy=base.DELIVERY_POLICY,
    sensitivity_offset=_sensitivity_offset,
    run_yamnet=base.run_yamnet,
    run_classifier=base.run_classifier,
    aggregate_context=base.aggregate_context,
    yamnet_gate_allows=base.yamnet_gate_allows,
    inference_lock=base.INFERENCE_LOCK,
    yamnet_class_names=base.YAMNET_CLASSES,
)


def _log_decision(source_id: str, location: str, decision: ClassificationDecision) -> None:
    if decision.shadow_decision_source is not None:
        print(
            f"[하이브리드 shadow] {location}/{source_id}: "
            f"v2={decision.sound or '거부'}, "
            f"v3={decision.shadow_sound or '거부'}({decision.shadow_decision_source})"
        )
    if decision.sound is None:
        return
    print(
        f"[판정] {location}/{source_id}: {decision.sound} "
        f"({decision.raw_label}, {decision.confidence * 100:.1f}%, "
        f"source={decision.decision_source}, policy={decision.policy_version})"
    )


def classify_source(
    waveform: np.ndarray,
    *,
    source_id: str,
    location: str,
    capture_ms: int | None = None,
    observed_at: float | None = None,
) -> ClassificationDecision:
    decision = decision_engine.classify(
        waveform,
        source_id,
        capture_ms=capture_ms,
        observed_at=observed_at,
    )
    _log_decision(source_id, location, decision)
    return decision


def _cooldown_allows(source_id: str, sound: str) -> bool:
    now = time.monotonic()
    key = (source_id, sound)
    with _cooldown_lock:
        if now - _last_alert.get(key, 0.0) < REMOTE_COOLDOWN_SECONDS:
            return False
        _last_alert[key] = now
    return True


def deliver_decision(
    decision: ClassificationDecision,
    *,
    source_id: str,
    location: str,
) -> None:
    if decision.sound is None or not _cooldown_allows(source_id, decision.sound):
        return

    base.trigger_led(decision.sound)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sound": decision.sound,
        "raw_label": decision.raw_label,
        "type": base.SOUND_TYPE_MAP.get(decision.sound, "Urgent"),
        "confidence": round(float(decision.confidence), 4),
        "model_version": base.MODEL_METADATA.get("model_name", "hearo_classifier_v2"),
        **decision.diagnostic_fields(),
    }
    if base.cloud_runtime.publish_alert(
        payload,
        publisher_device_id=base.DEVICE_ID,
        capture_device_id=source_id,
        location=location,
    ):
        print(f"  [MQTT] {decision.sound} / {decision.confidence * 100:.1f}% / {location}")
    else:
        print("  [로컬 전용] Raspberry Pi MQTT OFF/연결 끊김")


def handle_remote_window(window: AudioWindow) -> None:
    rms = float(np.sqrt(np.mean(window.waveform ** 2) + 1e-12))
    if rms < base.SILENCE_RMS_THRESHOLD:
        return
    decision = classify_source(
        window.waveform,
        source_id=window.device_id,
        location=window.location,
        capture_ms=window.capture_ms,
        observed_at=window.observed_at,
    )
    deliver_decision(decision, source_id=window.device_id, location=window.location)


def classify_local(waveform: np.ndarray) -> ClassificationDecision:
    return classify_source(
        waveform,
        source_id=base.DEVICE_ID,
        location=base.LOCATION,
        observed_at=time.monotonic(),
    )


def deliver_local(decision: ClassificationDecision) -> None:
    deliver_decision(decision, source_id=base.DEVICE_ID, location=base.LOCATION)


def main() -> None:
    pre_shared_key = os.getenv("HEARO_AUDIO_PSK", "")
    if len(pre_shared_key.encode("utf-8")) < 16:
        raise RuntimeError("HEARO_AUDIO_PSK는 ESP32와 동일한 16바이트 이상 값이어야 합니다.")
    bind_host = os.getenv("HEARO_AUDIO_BIND_HOST", "0.0.0.0")
    bind_port = int(os.getenv("HEARO_AUDIO_PORT", "41000"))
    receiver = AudioUdpReceiver(
        pre_shared_key=pre_shared_key,
        bind_host=bind_host,
        bind_port=bind_port,
        on_window=handle_remote_window,
        on_stream_reset=decision_engine.reset_source,
    )
    fallback_reason = HYBRID_POLICY.get("fallback_reason")
    mode = "enabled" if decision_engine.enabled else "v2 fallback"
    shadow_active = decision_engine.enabled and bool(HYBRID_POLICY.get("shadow_mode", False))
    print(
        f"[하이브리드] policy={decision_engine.policy_version}, mode={mode}, "
        f"shadow={shadow_active}"
    )
    if fallback_reason:
        print(f"[하이브리드] 정책 load fallback: {fallback_reason}")
    print(
        f"[원격 오디오] UDP {bind_host}:{bind_port} / "
        "esp32_1=안방, esp32_2=현관, esp32_3=화장실"
    )
    base.main(
        on_started=receiver.start,
        on_stopping=receiver.stop,
        classify_fn=classify_local,
        send_alert_fn=deliver_local,
    )


if __name__ == "__main__":
    main()

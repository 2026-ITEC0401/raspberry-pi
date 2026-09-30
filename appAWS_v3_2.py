"""Hearo Raspberry Pi hub using the versioned v3.2 deployment artifacts.

The Colab notebook is not executed here.  This process loads the exported
YAMNet TFLite model, the selected final Hearo classifier, categories, and
schema-v4 metadata.  Local Pi audio, authenticated ESP32 UDP audio, hybrid
decisions, GPIO LEDs, MQTT, and device-control APIs remain in one hub process.

Set HEARO_MODEL_PROFILE=v2 only for a pre-deployment compatibility check.  The
existing appAWS_v2.py and appAWS_v3.py remain the authoritative v2 rollback.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import RPi.GPIO as GPIO
import sounddevice as sd
from scipy.signal import resample_poly

from hearo_audio_receiver import AudioUdpReceiver, AudioWindow
from hearo_device_runtime import DeviceCloudRuntime
from hearo_hybrid_classifier import (
    ClassificationDecision,
    HybridDecisionEngine,
    load_hybrid_policy,
)
from hearo_model_runtime import HearoModelRuntime


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("HEARO_MODEL_DIR", str(BASE_DIR / "model"))).expanduser()
MODEL_PROFILE = os.getenv("HEARO_MODEL_PROFILE", "v3_2")
MODEL_RUNTIME = HearoModelRuntime.load(MODEL_DIR, MODEL_PROFILE)
MODEL_CONTRACT = MODEL_RUNTIME.contract

LOCATION = os.getenv("HEARO_LOCATION", "거실")
DEVICE_ID = os.getenv("HEARO_DEVICE_ID", "rpi-001")
HOUSEHOLD_ID = os.getenv("HEARO_HOUSEHOLD_ID", "")
MIC_SAMPLE_RATE = int(os.getenv("HEARO_MIC_SAMPLE_RATE", "48000"))
SILENCE_RMS_THRESHOLD = float(os.getenv("HEARO_SILENCE_RMS_THRESHOLD", "0.003"))
COOLDOWN_SECONDS = float(os.getenv("HEARO_COOLDOWN_SECONDS", "5.0"))
REMOTE_COOLDOWN_SECONDS = float(
    os.getenv("HEARO_REMOTE_COOLDOWN_SECONDS", str(COOLDOWN_SECONDS))
)
POLICY_PATH = Path(
    os.getenv("HEARO_HYBRID_POLICY_PATH", str(MODEL_DIR / "hybrid_policy_v3.json"))
)
RECORD_STEP_SECONDS = MODEL_CONTRACT.record_step_seconds
ROLLING_BUFFER_SECONDS = MODEL_CONTRACT.rolling_buffer_seconds
BUFFER_CHUNKS = MODEL_CONTRACT.buffer_chunks

if MIC_SAMPLE_RATE <= 0:
    raise RuntimeError("HEARO_MIC_SAMPLE_RATE는 양수여야 합니다.")
if SILENCE_RMS_THRESHOLD < 0:
    raise RuntimeError("HEARO_SILENCE_RMS_THRESHOLD는 0 이상이어야 합니다.")

LED_PINS = {
    "비상벨소리": 17,
    "도어락소리": 22,
    "노크소리": 22,
    "아기울음소리": 27,
}
BLINK_INTERVAL = {
    "비상벨소리": 0.2,
    "도어락소리": 0.5,
    "노크소리": 0.5,
    "아기울음소리": 0.5,
}
BLINK_DURATION = 5.0
SOUND_TYPE_MAP = {
    "도어락소리": "Visitor",
    "노크소리": "Visitor",
    "비상벨소리": "Urgent",
    "아기울음소리": "Noise",
}
SENSITIVITY_THRESHOLD_OFFSETS = {"low": 0.10, "default": 0.0, "high": -0.05}

INFERENCE_LOCK = threading.RLock()
_cooldown_lock = threading.Lock()
_last_alert: dict[tuple[str, str], float] = {}


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in set(LED_PINS.values()):
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
_led_blinking = {pin: False for pin in set(LED_PINS.values())}


cloud_runtime = DeviceCloudRuntime(
    household_id=HOUSEHOLD_ID,
    device_id=DEVICE_ID,
    location=LOCATION,
    firmware_version=(
        "rpi-inference-v3.2" if MODEL_CONTRACT.profile == "v3_2" else "rpi-inference-v2.1"
    ),
)


def _sensitivity_offset() -> float:
    return float(SENSITIVITY_THRESHOLD_OFFSETS[cloud_runtime.sensitivity])


HYBRID_POLICY = load_hybrid_policy(POLICY_PATH)
decision_engine = HybridDecisionEngine(
    policy=HYBRID_POLICY,
    categories=MODEL_CONTRACT.categories,
    unknown_label=MODEL_CONTRACT.unknown_label,
    class_thresholds=MODEL_CONTRACT.class_thresholds,
    class_mapping=MODEL_CONTRACT.class_mapping,
    delivery_policy=MODEL_CONTRACT.delivery_policy,
    sensitivity_offset=_sensitivity_offset,
    run_yamnet=MODEL_RUNTIME.run_yamnet,
    run_classifier=MODEL_RUNTIME.run_classifier,
    aggregate_context=MODEL_RUNTIME.aggregate_context,
    yamnet_gate_allows=MODEL_RUNTIME.yamnet_gate_allows,
    inference_lock=INFERENCE_LOCK,
    yamnet_class_names=MODEL_CONTRACT.yamnet_classes,
    base_decision_source=(
        "hearo_v3_2" if MODEL_CONTRACT.profile == "v3_2" else "hearo_v2"
    ),
    # schema-v4 thresholds are already selected by the training notebook.  With
    # default sensitivity this preserves even boundary values such as 0.0/1.0.
    threshold_clip_range=(0.0, 1.0),
)


def select_microphone_device() -> int:
    devices = list(sd.query_devices())
    inputs = [
        (index, device)
        for index, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]
    requested_index = os.getenv("HEARO_MIC_DEVICE_INDEX", "").strip()
    requested_name = os.getenv("HEARO_MIC_DEVICE_NAME", "").strip().casefold()
    if requested_index:
        try:
            index = int(requested_index)
            device = devices[index]
        except (ValueError, IndexError) as exc:
            raise RuntimeError("HEARO_MIC_DEVICE_INDEX가 유효한 장치 번호가 아닙니다.") from exc
        if int(device.get("max_input_channels", 0)) <= 0:
            raise RuntimeError("HEARO_MIC_DEVICE_INDEX가 입력 가능한 마이크가 아닙니다.")
        return index
    if requested_name:
        for index, device in inputs:
            if requested_name in str(device.get("name", "")).casefold():
                return index
        raise RuntimeError(
            f"HEARO_MIC_DEVICE_NAME과 일치하는 입력 장치가 없습니다: {requested_name}"
        )
    for index, device in inputs:
        name = str(device.get("name", "")).casefold()
        if "voicehat" in name or "i2s" in name:
            return index
    available = ", ".join(f"{index}:{device.get('name', '')}" for index, device in inputs)
    raise RuntimeError(
        "INMP441/I2S 입력 장치를 자동으로 찾지 못했습니다. "
        "HEARO_MIC_DEVICE_INDEX 또는 HEARO_MIC_DEVICE_NAME을 설정하십시오. "
        f"입력 장치={available or '없음'}"
    )


MIC_DEVICE_INDEX = select_microphone_device()


def record_audio_chunk() -> np.ndarray:
    frame_count = int(MIC_SAMPLE_RATE * RECORD_STEP_SECONDS)
    audio = sd.rec(
        frame_count,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE_INDEX,
    )
    sd.wait()
    values = np.asarray(audio, np.float32).reshape(-1)
    divisor = math.gcd(MIC_SAMPLE_RATE, MODEL_CONTRACT.sample_rate)
    downsampled = resample_poly(
        values,
        MODEL_CONTRACT.sample_rate // divisor,
        MIC_SAMPLE_RATE // divisor,
    )
    return np.clip(downsampled, -1.0, 1.0).astype(np.float32)


def _blink_led(pin: int, interval: float, duration: float) -> None:
    _led_blinking[pin] = True
    started = time.time()
    try:
        while time.time() - started < duration:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(interval)
    finally:
        GPIO.output(pin, GPIO.LOW)
        _led_blinking[pin] = False


def trigger_led(sound: str) -> None:
    if not cloud_runtime.led_alert_enabled:
        return
    pin = LED_PINS.get(sound)
    if pin is None or _led_blinking.get(pin, False):
        return
    threading.Thread(
        target=_blink_led,
        args=(pin, BLINK_INTERVAL.get(sound, 0.5), BLINK_DURATION),
        daemon=True,
    ).start()
    print(f"  [LED] {sound}: GPIO {pin}")


def _log_decision(source_id: str, location: str, decision: ClassificationDecision) -> None:
    if decision.shadow_decision_source is not None:
        print(
            f"[하이브리드 shadow] {location}/{source_id}: "
            f"baseline={decision.sound or '거부'}, "
            f"hybrid={decision.shadow_sound or '거부'}({decision.shadow_decision_source})"
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
    trigger_led(decision.sound)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sound": decision.sound,
        "raw_label": decision.raw_label,
        "type": SOUND_TYPE_MAP.get(decision.sound, "Urgent"),
        "confidence": round(float(decision.confidence), 4),
        "model_version": MODEL_CONTRACT.model_name,
        # The deployed backend uses a strict InternalAlertRequest schema.
        # Profile/variant remain available in local startup logs and metadata;
        # model_version is the supported wire-level model identifier.
        **decision.diagnostic_fields(),
    }
    if cloud_runtime.publish_alert(
        payload,
        publisher_device_id=DEVICE_ID,
        capture_device_id=source_id,
        location=location,
    ):
        print(f"  [MQTT] {decision.sound} / {decision.confidence * 100:.1f}% / {location}")
    else:
        print("  [로컬 전용] Raspberry Pi MQTT OFF/연결 끊김")


def handle_remote_window(window: AudioWindow) -> None:
    rms = float(np.sqrt(np.mean(window.waveform**2) + 1e-12))
    if rms < SILENCE_RMS_THRESHOLD:
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
        source_id=DEVICE_ID,
        location=LOCATION,
        observed_at=time.monotonic(),
    )


def print_startup(bind_host: str, bind_port: int) -> None:
    fallback_reason = HYBRID_POLICY.get("fallback_reason")
    hybrid_mode = "enabled" if decision_engine.enabled else "baseline fallback"
    shadow_active = decision_engine.enabled and bool(HYBRID_POLICY.get("shadow_mode", False))
    print(f"[초기화] {MODEL_RUNTIME.interpreter_runtime_name} 사용")
    print(
        f"[하이브리드] policy={decision_engine.policy_version}, "
        f"mode={hybrid_mode}, shadow={shadow_active}"
    )
    if fallback_reason:
        print(f"[하이브리드] 정책 load fallback: {fallback_reason}")
    print(
        f"[원격 오디오] UDP {bind_host}:{bind_port} / "
        "esp32_1=안방, esp32_2=현관, esp32_3=화장실"
    )
    print("\n" + "=" * 60)
    print(f"Hearo 환경음 인식 {MODEL_CONTRACT.profile} 시작")
    print(f"모델: {MODEL_CONTRACT.model_name}")
    print(
        f"TFLite: {MODEL_CONTRACT.classifier_model_path.name} "
        f"({MODEL_CONTRACT.selected_tflite_variant or 'legacy'})"
    )
    print(
        f"클래스: {len(MODEL_CONTRACT.categories)}개 "
        f"(unknown={MODEL_CONTRACT.unknown_label})"
    )
    print(
        f"표현: {MODEL_CONTRACT.representation}, "
        f"pooling={MODEL_CONTRACT.frame_pooling}, "
        f"context={MODEL_CONTRACT.context_frames}"
    )
    print(
        f"temperature embedded={MODEL_CONTRACT.temperature_embedded_in_tflite}, "
        f"sample_rate={MODEL_CONTRACT.sample_rate}"
    )
    print(
        f"녹음 step={RECORD_STEP_SECONDS:.1f}s, "
        f"rolling={ROLLING_BUFFER_SECONDS:.1f}s"
    )
    print(f"위치={LOCATION}, device={DEVICE_ID}, cooldown={COOLDOWN_SECONDS:.1f}s")
    print("=" * 60)


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
    print_startup(bind_host, bind_port)

    rolling_chunks: deque[np.ndarray] = deque(maxlen=BUFFER_CHUNKS)
    cloud_started = False
    receiver_started = False
    try:
        cloud_runtime.start()
        cloud_started = True
        receiver.start()
        receiver_started = True
        while True:
            chunk = record_audio_chunk()
            rolling_chunks.append(chunk)
            waveform = np.concatenate(tuple(rolling_chunks)).astype(np.float32)
            rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
            if rms < SILENCE_RMS_THRESHOLD:
                continue
            decision = classify_local(waveform)
            if decision.sound is None:
                continue
            print(
                f"[감지] {decision.sound} "
                f"({decision.raw_label}, {decision.confidence * 100:.1f}%)"
            )
            deliver_decision(decision, source_id=DEVICE_ID, location=LOCATION)
    except KeyboardInterrupt:
        print("\n종료 요청을 받았습니다.")
    finally:
        if receiver_started:
            receiver.stop()
        GPIO.cleanup()
        if cloud_started:
            cloud_runtime.stop()
        print("GPIO와 MQTT를 정리했습니다.")


if __name__ == "__main__":
    main()

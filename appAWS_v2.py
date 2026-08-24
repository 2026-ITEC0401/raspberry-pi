"""
Hearo Raspberry Pi inference v2.

Required files in ./model:
  - yamnet.tflite
  - hearo_classifier_v2.tflite
  - categories_v2.txt
  - model_metadata_v2.json
  - yamnet_classes.txt (diagnostic names only)

The classifier consumes N YAMNet embeddings with shape [N, 1024]. This app
records one second at a time, keeps a rolling two-second buffer, applies the
pooling and thresholds selected by yamnet_fine_tuning_v2.ipynb, and suppresses
the 비표적음 class without publishing an alert.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import RPi.GPIO as GPIO
import sounddevice as sd
from scipy.signal import resample_poly

from hearo_device_runtime import DeviceCloudRuntime


# -----------------------------------------------------------------------------
# Site configuration
# -----------------------------------------------------------------------------
LOCATION = "거실"
DEVICE_ID = "rpi-001"
HOUSEHOLD_ID = os.getenv("HEARO_HOUSEHOLD_ID", "")

MIC_SAMPLE_RATE = 48_000
YAMNET_SAMPLE_RATE = 16_000
RECORD_STEP_SECONDS = 1.0
ROLLING_BUFFER_SECONDS = 2.0
SILENCE_RMS_THRESHOLD = float(
    os.getenv("HEARO_SILENCE_RMS_THRESHOLD", "0.003")
)
COOLDOWN_SECONDS = 5.0

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

# Used only when model_metadata_v2.json explicitly enables the first-stage gate.
DEFAULT_YAMNET_TARGET_INDICES = {
    19, 20,
    316, 317, 318, 319,
    348, 349, 351, 352, 353, 354, 355, 373,
    382, 389, 390, 391, 392, 393, 394,
    475, 476, 477, 485,
}


# -----------------------------------------------------------------------------
# Model contract
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
YAMNET_MODEL_PATH = MODEL_DIR / "yamnet.tflite"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "hearo_classifier_v2.tflite"
CATEGORIES_PATH = MODEL_DIR / "categories_v2.txt"
METADATA_PATH = MODEL_DIR / "model_metadata_v2.json"
YAMNET_CLASSES_PATH = MODEL_DIR / "yamnet_classes.txt"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


MODEL_METADATA = load_json(METADATA_PATH)
CATEGORIES = load_lines(CATEGORIES_PATH)
YAMNET_CLASSES = load_lines(YAMNET_CLASSES_PATH) if YAMNET_CLASSES_PATH.exists() else []

if MODEL_METADATA.get("schema_version") != 2:
    raise RuntimeError("지원하지 않는 model_metadata schema입니다.")
if MODEL_METADATA.get("categories") != CATEGORIES:
    raise RuntimeError("categories_v2.txt와 model_metadata_v2.json의 클래스 순서가 다릅니다.")
if int(MODEL_METADATA.get("sample_rate", -1)) != YAMNET_SAMPLE_RATE:
    raise RuntimeError("모델 metadata의 sample_rate가 YAMNet 16 kHz 계약과 다릅니다.")

UNKNOWN_LABEL = MODEL_METADATA["unknown_label"]
if UNKNOWN_LABEL not in CATEGORIES:
    raise RuntimeError("unknown_label이 categories에 없습니다.")
UNKNOWN_INDEX = CATEGORIES.index(UNKNOWN_LABEL)
REPRESENTATION = MODEL_METADATA.get("representation", "frame")
FRAME_POOLING = MODEL_METADATA.get("frame_pooling", "mean_probability")
CONTEXT_FRAMES = MODEL_METADATA.get("context_frames", "full")
CLASS_THRESHOLDS = {
    label: float(MODEL_METADATA["class_thresholds"][label]) for label in CATEGORIES
}
CLASS_MAPPING = MODEL_METADATA.get("class_mapping", {})
DELIVERY_POLICY = MODEL_METADATA.get("delivery_policy", {})
SENSITIVITY_THRESHOLD_OFFSETS = {"low": 0.10, "default": 0.0, "high": -0.05}
INFERENCE_LOCK = threading.RLock()

# Notebook metadata is authoritative, but keep constants readable for operators.
RECORD_STEP_SECONDS = float(MODEL_METADATA.get("record_step_seconds", RECORD_STEP_SECONDS))
ROLLING_BUFFER_SECONDS = float(MODEL_METADATA.get("rolling_buffer_seconds", ROLLING_BUFFER_SECONDS))
BUFFER_CHUNKS = max(1, int(math.ceil(ROLLING_BUFFER_SECONDS / RECORD_STEP_SECONDS)))


try:
    from ai_edge_litert.interpreter import Interpreter
    print("[초기화] ai-edge-litert 사용")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("[초기화] tflite_runtime 사용")
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
        print("[초기화] tensorflow.lite 사용")


yamnet_interpreter = Interpreter(model_path=str(YAMNET_MODEL_PATH))
yamnet_interpreter.allocate_tensors()
classifier_interpreter = Interpreter(model_path=str(CLASSIFIER_MODEL_PATH))
classifier_interpreter.allocate_tensors()


def validate_interpreter_contracts() -> None:
    classifier_input = classifier_interpreter.get_input_details()[0]
    classifier_output = classifier_interpreter.get_output_details()[0]
    if int(classifier_input["shape"][-1]) != 1024:
        raise RuntimeError(f"분류기 입력 마지막 차원은 1024여야 합니다: {classifier_input['shape']}")
    if int(classifier_output["shape"][-1]) != len(CATEGORIES):
        raise RuntimeError(
            f"분류기 출력 클래스 수({classifier_output['shape'][-1]})와 categories({len(CATEGORIES)})가 다릅니다."
        )


validate_interpreter_contracts()


# -----------------------------------------------------------------------------
# GPIO and cloud runtime
# -----------------------------------------------------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in set(LED_PINS.values()):
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
led_blinking = {pin: False for pin in set(LED_PINS.values())}


cloud_runtime = DeviceCloudRuntime(
    household_id=HOUSEHOLD_ID,
    device_id=DEVICE_ID,
    location=LOCATION,
    firmware_version="rpi-inference-v2.1",
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
        raise RuntimeError(f"HEARO_MIC_DEVICE_NAME과 일치하는 입력 장치가 없습니다: {requested_name}")

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


def blink_led(pin: int, interval: float, duration: float) -> None:
    led_blinking[pin] = True
    started = time.time()
    try:
        while time.time() - started < duration:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(interval)
    finally:
        GPIO.output(pin, GPIO.LOW)
        led_blinking[pin] = False


def trigger_led(sound: str) -> None:
    if not cloud_runtime.led_alert_enabled:
        return
    pin = LED_PINS.get(sound)
    if pin is None or led_blinking.get(pin, False):
        return
    interval = BLINK_INTERVAL.get(sound, 0.5)
    threading.Thread(
        target=blink_led,
        args=(pin, interval, BLINK_DURATION),
        daemon=True,
    ).start()
    print(f"  [LED] {sound}: GPIO {pin}")


# -----------------------------------------------------------------------------
# Audio and TFLite inference helpers
# -----------------------------------------------------------------------------
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
    audio = np.asarray(audio, np.float32).reshape(-1)
    divisor = math.gcd(MIC_SAMPLE_RATE, YAMNET_SAMPLE_RATE)
    downsampled = resample_poly(
        audio,
        YAMNET_SAMPLE_RATE // divisor,
        MIC_SAMPLE_RATE // divisor,
    )
    return np.clip(downsampled, -1.0, 1.0).astype(np.float32)


def quantize_for_input(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail["quantization"]
        if scale <= 0:
            raise RuntimeError("정수 TFLite 입력의 quantization scale이 유효하지 않습니다.")
        info = np.iinfo(dtype)
        values = np.clip(np.round(values / scale + zero_point), info.min, info.max)
    return values.astype(dtype)


def dequantize_output(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero_point = detail["quantization"]
        return (values.astype(np.float32) - zero_point) * scale
    return values.astype(np.float32)


def run_yamnet(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    input_detail = yamnet_interpreter.get_input_details()[0]
    yamnet_interpreter.resize_tensor_input(input_detail["index"], [len(waveform)], strict=False)
    yamnet_interpreter.allocate_tensors()
    input_detail = yamnet_interpreter.get_input_details()[0]
    yamnet_interpreter.set_tensor(input_detail["index"], quantize_for_input(waveform, input_detail))
    yamnet_interpreter.invoke()

    scores = None
    embeddings = None
    for detail in yamnet_interpreter.get_output_details():
        result = dequantize_output(yamnet_interpreter.get_tensor(detail["index"]), detail)
        if result.shape[-1] == 521:
            scores = result
        elif result.shape[-1] == 1024:
            embeddings = result
    if scores is None or embeddings is None:
        raise RuntimeError("YAMNet 출력에서 scores/embeddings를 찾지 못했습니다.")
    return np.atleast_2d(scores), np.atleast_2d(embeddings)


def yamnet_gate_allows(scores: np.ndarray) -> bool:
    gate = MODEL_METADATA.get("yamnet_gate", {})
    if not bool(gate.get("enabled", False)):
        return True
    indices = gate.get("target_indices", sorted(DEFAULT_YAMNET_TARGET_INDICES))
    threshold = float(gate.get("threshold", 0.0))
    score = float(np.max(scores[:, np.asarray(indices, dtype=int)]))
    if score < threshold:
        return False
    return True


def run_classifier(embeddings: np.ndarray) -> np.ndarray:
    model_input = embeddings.mean(axis=0, keepdims=True) if REPRESENTATION == "clip_mean" else embeddings
    input_detail = classifier_interpreter.get_input_details()[0]
    requested_shape = [len(model_input), 1024]
    if list(input_detail["shape"]) != requested_shape:
        classifier_interpreter.resize_tensor_input(input_detail["index"], requested_shape, strict=False)
        classifier_interpreter.allocate_tensors()
        input_detail = classifier_interpreter.get_input_details()[0]
    classifier_interpreter.set_tensor(
        input_detail["index"], quantize_for_input(model_input.astype(np.float32), input_detail)
    )
    classifier_interpreter.invoke()
    output_detail = classifier_interpreter.get_output_details()[0]
    probabilities = dequantize_output(
        classifier_interpreter.get_tensor(output_detail["index"]), output_detail
    )
    probabilities = np.atleast_2d(probabilities)

    output_type = MODEL_METADATA.get("classifier_output", {}).get("type", "probabilities")
    if output_type == "probabilities":
        probabilities = np.clip(probabilities, 1e-9, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        shifted = probabilities - probabilities.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.clip(probabilities, 1e-9, 1.0)


def pool_probabilities(probabilities: np.ndarray, pooling: str) -> np.ndarray:
    probabilities = np.asarray(probabilities, np.float64)
    if len(probabilities) == 1:
        pooled = np.clip(probabilities[0], 1e-12, None)
        return pooled / pooled.sum()
    if pooling == "mean_probability":
        pooled = probabilities.mean(axis=0)
    elif pooling == "topk_probability":
        count = max(1, int(math.ceil(len(probabilities) * 0.5)))
        pooled = np.sort(probabilities, axis=0)[-count:].mean(axis=0)
    elif pooling == "logit_logmeanexp":
        log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
        maximum = log_probabilities.max(axis=0, keepdims=True)
        pooled_logits = maximum[0] + np.log(
            np.exp(log_probabilities - maximum).mean(axis=0) + 1e-12
        )
        pooled_logits -= pooled_logits.max()
        pooled = np.exp(pooled_logits)
    else:
        raise RuntimeError(f"지원하지 않는 frame_pooling: {pooling}")
    pooled = np.clip(pooled, 1e-12, None)
    return pooled / pooled.sum()


def aggregate_context(probabilities: np.ndarray) -> np.ndarray:
    if REPRESENTATION == "clip_mean" or CONTEXT_FRAMES == "full":
        return pool_probabilities(probabilities, FRAME_POOLING)
    width = max(1, int(CONTEXT_FRAMES))
    if len(probabilities) <= width:
        windows = [probabilities]
    else:
        windows = [
            probabilities[start:start + width]
            for start in range(len(probabilities) - width + 1)
        ]
    pooled_windows = np.stack([pool_probabilities(window, FRAME_POOLING) for window in windows])
    target_confidence = pooled_windows[:, :UNKNOWN_INDEX].max(axis=1)
    return pooled_windows[int(np.argmax(target_confidence))]


def classify_embeddings(embeddings: np.ndarray) -> tuple[str | None, float, str]:
    frame_probabilities = run_classifier(embeddings)
    probabilities = aggregate_context(frame_probabilities)
    order = np.argsort(probabilities)[::-1][:3]
    print("  [분류기] Top3:", [(CATEGORIES[i], f"{probabilities[i] * 100:.1f}%") for i in order])

    best_index = int(np.argmax(probabilities))
    raw_label = CATEGORIES[best_index]
    confidence = float(probabilities[best_index])
    if raw_label == UNKNOWN_LABEL:
        return None, confidence, raw_label
    offset = SENSITIVITY_THRESHOLD_OFFSETS[cloud_runtime.sensitivity]
    threshold = float(np.clip(CLASS_THRESHOLDS[raw_label] + offset, 0.05, 0.99))
    if confidence < threshold:
        print(f"  [거부] {raw_label} {confidence:.3f} < threshold {threshold:.3f}")
        return None, confidence, raw_label

    policy = DELIVERY_POLICY.get(raw_label, {})
    if policy.get("publish_enabled") is False:
        print(f"  [로컬 분류] {raw_label} {confidence:.3f} — 외부 알림 정책 OFF")
        return None, confidence, raw_label

    mapped = CLASS_MAPPING.get(raw_label)
    if mapped not in LED_PINS:
        raise RuntimeError(f"사용자 알림 매핑이 없거나 유효하지 않습니다: {raw_label} -> {mapped}")
    return mapped, confidence, raw_label


def classify_waveform(waveform: np.ndarray) -> tuple[str | None, float, str]:
    """Run both TFLite models under one lock so local and remote streams are safe."""
    with INFERENCE_LOCK:
        scores, embeddings = run_yamnet(waveform)
        if not yamnet_gate_allows(scores):
            return None, 0.0, "yamnet_gate_rejected"
        return classify_embeddings(embeddings)


# -----------------------------------------------------------------------------
# Alert publication
# -----------------------------------------------------------------------------
def send_alert(sound: str, confidence: float, raw_label: str) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sound": sound,
        "raw_label": raw_label,
        "type": SOUND_TYPE_MAP.get(sound, "Urgent"),
        "location": LOCATION,
        "confidence": round(float(confidence), 4),
        "model_version": MODEL_METADATA.get("model_name", "hearo_classifier_v2"),
    }
    if cloud_runtime.publish_alert(payload):
        print(f"  [MQTT] {sound} / {confidence * 100:.1f}% / {LOCATION}")
    else:
        print("  [로컬 전용] MQTT OFF/연결 끊김 — 원격 알림을 보내지 않았습니다.")


def print_startup() -> None:
    print("\n" + "=" * 60)
    print("Hearo 환경음 인식 v2 시작")
    print(f"모델: {MODEL_METADATA.get('model_name')}")
    print(f"클래스: {len(CATEGORIES)}개 (unknown={UNKNOWN_LABEL})")
    print(f"표현: {REPRESENTATION}, pooling={FRAME_POOLING}, context={CONTEXT_FRAMES}")
    print(f"녹음 step={RECORD_STEP_SECONDS:.1f}s, rolling={ROLLING_BUFFER_SECONDS:.1f}s")
    print(f"위치={LOCATION}, device={DEVICE_ID}, cooldown={COOLDOWN_SECONDS:.1f}s")
    print("=" * 60)


def _classification_parts(result: Any) -> tuple[str | None, float, str]:
    """Accept the legacy tuple or a v3 decision object without importing v3."""
    if isinstance(result, tuple) and len(result) == 3:
        sound, confidence, raw_label = result
        return sound, float(confidence), str(raw_label)
    try:
        return result.sound, float(result.confidence), str(result.raw_label)
    except AttributeError as exc:
        raise TypeError("분류 callback은 3-tuple 또는 decision object를 반환해야 합니다.") from exc


def main(
    *,
    on_started=None,
    on_stopping=None,
    classify_fn=None,
    send_alert_fn=None,
) -> None:
    """Run local inference, optionally delegating v3 decision/delivery callbacks.

    With no callbacks this is the existing v2 decision path. A custom
    send_alert_fn owns cooldown and LED/MQTT delivery so v3 can share one
    source-aware implementation with remote ESP32 windows.
    """
    print_startup()
    rolling_chunks: deque[np.ndarray] = deque(maxlen=BUFFER_CHUNKS)
    last_alert_time = 0.0
    last_alert_sound = ""
    cloud_started = False
    extension_requested = False

    try:
        cloud_runtime.start()
        cloud_started = True
        if on_started is not None:
            extension_requested = True
            on_started()

        while True:
            chunk = record_audio_chunk()
            rolling_chunks.append(chunk)
            waveform = np.concatenate(list(rolling_chunks)).astype(np.float32)
            rms = float(np.sqrt(np.mean(waveform ** 2) + 1e-12))
            if rms < SILENCE_RMS_THRESHOLD:
                continue

            classification = (
                classify_fn(waveform) if classify_fn is not None else classify_waveform(waveform)
            )
            sound, confidence, raw_label = _classification_parts(classification)
            if sound is None:
                continue

            if send_alert_fn is not None:
                print(f"[감지] {sound} ({raw_label}, {confidence * 100:.1f}%)")
                send_alert_fn(classification)
                continue

            now = time.time()
            if sound == last_alert_sound and now - last_alert_time < COOLDOWN_SECONDS:
                continue

            print(f"[감지] {sound} ({raw_label}, {confidence * 100:.1f}%)")
            trigger_led(sound)
            send_alert(sound, confidence, raw_label)
            last_alert_time = now
            last_alert_sound = sound
    except KeyboardInterrupt:
        print("\n종료 요청을 받았습니다.")
    finally:
        if extension_requested and on_stopping is not None:
            on_stopping()
        GPIO.cleanup()
        if cloud_started:
            cloud_runtime.stop()
        print("GPIO와 MQTT를 정리했습니다.")


if __name__ == "__main__":
    main()

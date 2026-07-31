"""
=============================================================
Hearo - 라즈베리파이 환경음 인식 + LED 알림 시스템 (AWS 버전)
=============================================================
(기능)
- INMP441 마이크로 환경음 수집
- YAMNet + Hearo 분류기로 4개 카테고리 분류
- 로컬 LED 즉시 알림  → 인터넷이 끊겨도 항상 동작
- AWS EC2의 Mosquitto 브로커로 MQTT 발행
    · hearo/alert : 소리 이름만 (ESP32 원격 노드 → LED 점등)
    · hearo/log   : 상세 JSON (EC2 로거 → DynamoDB 저장)

(LED 매핑)
    · 비상벨      : 빨간 LED  빠른 깜빡임   (0.2초 간격, GPIO 17)
    · 도어락/노크 : 파란 LED  천천히 깜빡임 (0.5초 간격, GPIO 22)
    · 아기울음    : 노란 LED  천천히 깜빡임 (0.5초 간격, GPIO 27)

※ Firebase는 사용하지 않습니다. 알림 저장은 EC2가 담당합니다.
=============================================================
"""

import numpy as np
import time
import os
import json                                   # ★ JSON 페이로드 생성용
from datetime import datetime, timezone       # ★ 타임스탬프 직접 생성용
import sounddevice as sd
from scipy.signal import resample
import threading

import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt

# ============================================================
# 설정값
# ============================================================
LOCATION = "거실"
DEVICE_ID = "rpi-001"
CLASSIFIER_THRESHOLD = 0.7

MIC_SAMPLE_RATE = 48000
YAMNET_SAMPLE_RATE = 16000
RECORD_DURATION = 1.0
COOLDOWN_SECONDS = 5

# ============================================================
# ★ MQTT 설정 (AWS EC2의 Mosquitto 브로커)
# ============================================================
MQTT_BROKER = "EC2_퍼블릭_IP_입력"      # ⚠️ 본인 EC2 퍼블릭 IP로 변경!
MQTT_PORT = 1883
MQTT_USER = "hearo"                      # Mosquitto 계정
MQTT_PASS = "브로커_비밀번호_입력"       # ⚠️ mosquitto_passwd로 설정한 비밀번호
MQTT_CLIENT_ID = "hearo-rpi"             # 다른 기기와 겹치면 안 됨

TOPIC_ALERT = "hearo/alert"              # ESP32 노드가 구독 (LED용)
TOPIC_LOG = "hearo/log"                  # EC2 로거가 구독 (DynamoDB 저장용)

# ============================================================
# LED GPIO 핀 설정
# ============================================================
LED_PINS = {
    "비상벨소리": 17,    # 빨간 LED
    "도어락소리": 22,    # 파란 LED (노크와 공용)
    "노크소리": 22,      # 파란 LED (도어락과 공용)
    "아기울음소리": 27,  # 노란 LED
}

# 깜빡임 간격 (초) — 소리별로 다름
BLINK_INTERVAL = {
    "비상벨소리": 0.2,   # 빠른 깜빡임 (긴급)
    "도어락소리": 0.5,
    "노크소리": 0.5,
    "아기울음소리": 0.5,
}

# 깜빡임 지속 시간 (초)
BLINK_DURATION = 5.0

# GPIO 초기화
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in set(LED_PINS.values()):  # 중복 제거 (파란 LED는 한 번만)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# LED 깜빡임 제어 플래그 (현재 깜빡이는 중인지)
led_blinking = {pin: False for pin in set(LED_PINS.values())}

# ============================================================
# YAMNet → 분류기 매핑 (기존과 동일)
# ============================================================
YAMNET_TO_CLASSIFIER = {
    353, 354,                              # 노크
    19, 20,                                # 아기울음
    316, 317, 318, 319,                    # 비상차량
    382, 389, 390, 391, 392, 393, 394,     # 사이렌/알람
    348, 349, 351, 352, 355, 373,          # 도어/도어락
    475, 476, 477, 485,                    # 알림음
}

# ============================================================
# 모델 파일 경로 (기존과 동일)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
YAMNET_MODEL_PATH = os.path.join(MODEL_DIR, "yamnet.tflite")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "hearo_classifier.tflite")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.txt")
YAMNET_CLASSES_PATH = os.path.join(MODEL_DIR, "yamnet_classes.txt")

# 카테고리 로드
with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    CATEGORIES = [line.strip() for line in f if line.strip()]
with open(YAMNET_CLASSES_PATH, "r", encoding="utf-8") as f:
    YAMNET_CLASSES = [line.strip() for line in f if line.strip()]

# ============================================================
# TFLite 모델 로드 (기존과 동일)
# ============================================================
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

yamnet_interpreter = Interpreter(model_path=YAMNET_MODEL_PATH)
yamnet_interpreter.allocate_tensors()
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()

classifier_interpreter = Interpreter(model_path=CLASSIFIER_MODEL_PATH)
classifier_interpreter.allocate_tensors()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()


# ============================================================
# ★ MQTT 클라이언트 초기화 (원격 브로커 + 자동 재연결)
# ============================================================
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    """브로커 연결 성공/실패 시 호출됨."""
    if rc == 0:
        print(f"[MQTT] ✅ 브로커 연결 완료: {MQTT_BROKER}:{MQTT_PORT}")
    else:
        # rc=4 : 아이디/비밀번호 오류, rc=5 : 인증 거부
        print(f"[MQTT] ⚠️ 연결 실패 (코드 {rc}) — 계정/비밀번호를 확인하세요")


def on_mqtt_disconnect(client, userdata, rc, properties=None):
    """연결이 끊겼을 때 호출됨 (백그라운드에서 자동 재연결 시도)."""
    if rc != 0:
        print("[MQTT] ⚠️ 연결이 끊겼습니다 — 자동 재연결 시도 중...")


print("\n[초기화] MQTT 브로커 연결 준비 중...")

# paho-mqtt 1.x / 2.x 둘 다 호환되도록 처리
try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                              client_id=MQTT_CLIENT_ID)   # paho-mqtt 2.x
except AttributeError:
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)   # paho-mqtt 1.x

mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_disconnect = on_mqtt_disconnect

# connect_async + loop_start : 브로커가 잠시 꺼져 있어도 프로그램이 멈추지 않고
# 백그라운드에서 계속 재연결을 시도합니다 (인터넷이 늦게 붙는 상황 대비).
mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# ============================================================
# 마이크 장치 자동 검색 (기존과 동일)
# ============================================================
MIC_DEVICE_INDEX = None
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0 and \
       ('voicehat' in dev['name'].lower() or 'i2s' in dev['name'].lower()):
        MIC_DEVICE_INDEX = i

if MIC_DEVICE_INDEX is None:
    raise RuntimeError("INMP441 마이크를 찾을 수 없습니다!")


# ============================================================
# LED 제어 함수 (기존과 동일)
# ============================================================
def blink_led(pin, interval, duration):
    """
    별도 스레드에서 실행되어 LED를 깜빡임.
    메인 루프(소리 인식)를 막지 않음.
    """
    led_blinking[pin] = True
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(interval)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(interval)
    finally:
        GPIO.output(pin, GPIO.LOW)
        led_blinking[pin] = False


def trigger_led(sound):
    """
    소리 종류에 맞는 LED를 깜빡임 (백그라운드 스레드).
    이미 깜빡이는 중이면 무시 (중복 실행 방지).
    """
    pin = LED_PINS.get(sound)
    if pin is None:
        return

    interval = BLINK_INTERVAL.get(sound, 0.5)

    if led_blinking.get(pin, False):
        print(f"  [LED] 이미 깜빡이는 중 (GPIO {pin}) - 건너뜀")
        return

    thread = threading.Thread(
        target=blink_led,
        args=(pin, interval, BLINK_DURATION),
        daemon=True
    )
    thread.start()
    print(f"  [LED] 💡 {sound} 알림 시작 (GPIO {pin}, {interval}초 간격, {BLINK_DURATION}초간)")


# ============================================================
# 핵심 함수들 (기존과 동일)
# ============================================================
def record_audio():
    audio = sd.rec(
        int(MIC_SAMPLE_RATE * RECORD_DURATION),
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=MIC_DEVICE_INDEX
    )
    sd.wait()
    audio = audio.flatten()
    target_length = int(len(audio) * YAMNET_SAMPLE_RATE / MIC_SAMPLE_RATE)
    audio_16k = resample(audio, target_length).astype(np.float32)
    return audio_16k


def run_yamnet(waveform):
    input_data = waveform.astype(np.float32)
    yamnet_interpreter.resize_tensor_input(
        yamnet_input_details[0]['index'], [len(input_data)]
    )
    yamnet_interpreter.allocate_tensors()
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], input_data)
    yamnet_interpreter.invoke()

    scores = None
    embeddings = None
    for output in yamnet_output_details:
        result = yamnet_interpreter.get_tensor(output['index'])
        if result.shape[-1] == 521:
            scores = result
        elif result.shape[-1] == 1024:
            embeddings = result

    if scores is not None:
        scores = scores.mean(axis=0) if len(scores.shape) > 1 else scores.flatten()
    if embeddings is not None:
        embeddings = embeddings.mean(axis=0) if len(embeddings.shape) > 1 else embeddings.flatten()

    return scores, embeddings.astype(np.float32)


def classify_with_yamnet(scores):
    top_idx = np.argmax(scores)
    top_confidence = float(scores[top_idx])

    top3_idx = np.argsort(scores)[::-1][:3]
    top3_info = [(YAMNET_CLASSES[i], f"{scores[i]:.3f}") for i in top3_idx]
    print(f"  [YAMNet] Top3: {top3_info}")

    if top_idx in YAMNET_TO_CLASSIFIER:
        print()
        print(f"\033[96m  [YAMNet] 2차 분류기로 넘김: {YAMNET_CLASSES[top_idx]} ({top_confidence:.3f})\033[0m")
        return "classifier", None, top_confidence

    return "ignore", None, top_confidence


def classify_sound(embedding):
    input_data = embedding.reshape(1, -1).astype(np.float32)
    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], input_data)
    classifier_interpreter.invoke()
    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])[0]

    top3_idx = np.argsort(prediction)[::-1][:3]
    top3_info = [(CATEGORIES[i], f"{prediction[i]*100:.1f}%") for i in top3_idx]
    print(f"  [분류기] Top3: {top3_info}")

    best_idx = np.argmax(prediction)
    raw_category = CATEGORIES[best_idx]
    best_confidence = float(prediction[best_idx])

    if raw_category in ["노크_목재", "노크_철재문"]:
        best_category = "노크소리"
    elif raw_category in ["도어락_개방음", "도어락_입력음"]:
        best_category = "도어락소리"
    elif raw_category in ["사이렌_삐뽀삐뽀", "사이렌_안내음", "사이렌_애애애애앵", "사이렌_철철철"]:
        best_category = "비상벨소리"
    elif raw_category in ["아기 울음", "아기울음"]:
        best_category = "아기울음소리"
    else:
        if "노크" in raw_category:
            best_category = "노크소리"
        elif "도어락" in raw_category:
            best_category = "도어락소리"
        elif "사이렌" in raw_category or "비상벨" in raw_category:
            best_category = "비상벨소리"
        elif "아기 울음" in raw_category or "아기울음" in raw_category:
            best_category = "아기울음소리"
        else:
            return None, 0.0

    return best_category, best_confidence


# 알림 분류용 매핑 (웹 대시보드에서 사용)
SOUND_TYPE_MAP = {
    "도어락소리": "Visitor",
    "노크소리": "Visitor",
    "비상벨소리": "Urgent",
    "아기울음소리": "Noise"
}

SOUND_EMOJI_MAP = {
    "도어락소리": "🔐",
    "노크소리": "✊",
    "비상벨소리": "🚨",
    "아기울음소리": "👶"
}


# ============================================================
# ★ 알림 발행 (Firebase → MQTT 2개 토픽으로 변경)
# ============================================================
def send_alert(sound, confidence):
    """
    감지된 소리를 AWS EC2의 MQTT 브로커로 발행합니다.

    ① hearo/alert : 소리 이름만 → ESP32 원격 노드가 구독해서 LED 점등
    ② hearo/log   : 상세 JSON  → EC2 로거가 구독해서 DynamoDB에 저장

    ※ publish()는 백그라운드 큐에 넣기만 하므로 메인 루프를 막지 않습니다.
       네트워크가 끊겨 있어도 이 함수 때문에 프로그램이 멈추지는 않습니다.
    """
    sound_type = SOUND_TYPE_MAP.get(sound, "Urgent")
    sound_emoji = SOUND_EMOJI_MAP.get(sound, "🔔")

    # ① ESP32 LED 알림용 (기존 형식 그대로 — 소리 이름 문자열)
    try:
        mqtt_client.publish(TOPIC_ALERT, sound)
        print(f"  → 📡 알림 발행 [{TOPIC_ALERT}]: {sound_emoji} {sound}")
    except Exception as e:
        print(f"  → [오류] 알림 발행 실패: {e}")

    # ② DynamoDB 저장용 상세 로그 (JSON)
    #    키 이름은 EC2의 mqtt_to_dynamo.py 및 hearo-db 테이블 스키마와 일치해야 함
    try:
        log_payload = {
            "device_id": DEVICE_ID,                                   # 파티션 키
            "timestamp": datetime.now(timezone.utc).isoformat(),      # 정렬 키
            "sound": sound,
            "type": sound_type,
            "location": LOCATION,
            "confidence": round(float(confidence), 4),
        }
        mqtt_client.publish(TOPIC_LOG, json.dumps(log_payload, ensure_ascii=False))
        print(f"  → 🗄️ 로그 발행 [{TOPIC_LOG}]: {confidence*100:.1f}% ({LOCATION})")
    except Exception as e:
        print(f"  → [오류] 로그 발행 실패: {e}")

    print()


# ============================================================
# 메인 루프
# ============================================================
def main():
    print()
    print("=" * 55)
    print("  Hearo 환경음 인식 + LED 알림 시작 (AWS 연동)")
    print("=" * 55)
    print(f"  설치 위치:    {LOCATION}")
    print(f"  기기 ID:      {DEVICE_ID}")
    print(f"  분류기 임계값: {CLASSIFIER_THRESHOLD * 100:.0f}%")
    print(f"  녹음 길이:    {RECORD_DURATION}초")
    print(f"  쿨다운:       {COOLDOWN_SECONDS}초")
    print(f"  MQTT 브로커:  {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  LED 알림:     비상벨🔴 / 도어락+노크🔵 / 아기울음🟡")
    print("=" * 55)
    print()
    print("소리를 듣고 있습니다... (종료: Ctrl+C)")
    print()

    last_alert_time = 0
    last_alert_sound = ""

    while True:
        try:
            waveform = record_audio()

            volume = np.abs(waveform).mean()
            if volume < 0.003:
                continue

            scores, embedding = run_yamnet(waveform)
            action, category, yamnet_confidence = classify_with_yamnet(scores)

            if action == "classifier":
                sound, confidence = classify_sound(embedding)

                if sound and confidence >= CLASSIFIER_THRESHOLD:
                    now = time.time()
                    if sound == last_alert_sound and (now - last_alert_time) < COOLDOWN_SECONDS:
                        continue

                    print(f"[감지] {sound} ({confidence*100:.1f}%) - {LOCATION} [분류기]")

                    # ★ 로컬 LED를 먼저 켠 뒤 네트워크 발행
                    #    (네트워크 상태와 무관하게 집 안 알림이 항상 먼저 뜨도록)
                    trigger_led(sound)
                    send_alert(sound, confidence)

                    last_alert_time = now
                    last_alert_sound = sound

            else:
                pass

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"[오류] {e}")
            time.sleep(1)

    # 종료 시 정리
    GPIO.cleanup()
    print("GPIO 정리 완료")

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("MQTT 연결 종료")


if __name__ == "__main__":
    main()

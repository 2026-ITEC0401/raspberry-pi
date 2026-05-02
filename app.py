"""
=============================================================
Hearo - 라즈베리파이 환경음 인식 스크립트
=============================================================

[이 파일이 하는 일]
INMP441 마이크로 주변 소리를 계속 듣고 있다가,
1차로 YAMNet(521개 카테고리)으로 소리 종류를 판별하고,
2차로 Hearo 분류기(도어락/인터폰/가전)로 세부 분류합니다.

[전체 흐름]
마이크 녹음 (1초 단위)
    → 1차: YAMNet 모델로 521개 카테고리 중 분류
        → 대화, 음악 등 무관한 소리 → 무시
        → 개 짖음, 사이렌 등 8개 → YAMNet 결과로 바로 확정
        → 도어락/인터폰/가전 관련 → 2차 분류기로 넘김
    → 2차: Hearo 분류기로 도어락/인터폰/가전 판별
    → confidence(확신도)가 threshold 이상이면
    → 백엔드 서버(FastAPI)로 POST 전송

[실행 전 준비사항]
1. 필요한 패키지 설치:
   pip install -r requirements.txt

2. INMP441 마이크가 라즈베리파이에 연결되어 있어야 합니다.
   확인 명령어: arecord -l

3. 백엔드 서버가 실행 중이어야 합니다.
   서버 주소를 아래 SERVER_URL에 맞게 수정하세요.

[실행 방법]
   python app.py

[종료 방법]
   Ctrl + C
=============================================================
"""

import numpy as np
import time
import requests
import os
import sounddevice as sd
from scipy.signal import resample

# ============================================================
# 설정값 (필요에 따라 수정하세요)
# ============================================================

# 백엔드 서버 주소 (서버를 실행하는 PC의 IP 주소로 변경하세요)
SERVER_URL = "http://192.168.137.1:8000"

# 이 라즈베리파이의 설치 위치 (프론트엔드에 표시됨)
LOCATION = "현관"

# 이 라즈베리파이의 기기 식별자 (여러 대일 때 구분용)
DEVICE_ID = "rpi-001"

# threshold: 이 값 이상일 때만 서버로 전송 (0.0 ~ 1.0)
# YAMNet 직접 매핑: YAMNet의 confidence를 사용
# 2차 분류기: 분류기의 confidence를 사용
CONFIDENCE_THRESHOLD = 0.2

# 녹음 설정
MIC_SAMPLE_RATE = 48000  # INMP441 마이크의 샘플레이트 (48kHz)
YAMNET_SAMPLE_RATE = 16000  # YAMNet이 요구하는 샘플레이트 (16kHz)
RECORD_DURATION = 1.0    # 한 번에 녹음할 길이 (초)

# heartbeat 주기 (초)
HEARTBEAT_INTERVAL = 30

# 같은 소리가 반복 전송되는 것을 방지하는 쿨다운 (초)
COOLDOWN_SECONDS = 5

# ============================================================
# YAMNet → Hearo 카테고리 매핑
# ============================================================
# YAMNet의 521개 카테고리 중 우리 11개와 관련된 것을 매핑
#
# 1. 직접 매핑: YAMNet 결과를 바로 우리 카테고리로 변환 (분류기 안 거침)
# 2. 2차 분류기: YAMNet이 관련 소리로 판단하면 분류기로 넘김
# 3. 나머지: 무시 (대화, 음악, 동물 등)

# --- YAMNet 인덱스 → Hearo 카테고리 직접 매핑 ---
# 이 소리들은 YAMNet이 충분히 정확하므로 분류기 없이 바로 확정
YAMNET_DIRECT_MAP = {
    # 개 짖음
    69: "개 짖음",      # Dog
    70: "개 짖음",      # Bark
    71: "개 짖음",      # Yip
    73: "개 짖음",      # Bow-wow

    # 물 소리
    282: "물 소리",     # Water

    # 초인종 → 인터폰으로 통합
    349: "인터폰",      # Doorbell
    350: "인터폰",      # Ding-dong

    # 노크 소리
    353: "노크 소리",   # Knock
    354: "노크 소리",   # Tap

    # 창문 깨지는 소리
    435: "창문 깨지는 소리",  # Glass
    437: "창문 깨지는 소리",  # Shatter
    463: "창문 깨지는 소리",  # Smash, crash
    464: "창문 깨지는 소리",  # Breaking
}

# --- 2차 분류기로 넘길 YAMNet 인덱스 ---
# 이 소리들은 한국 가정음 특화이거나 세부 판단이 필요해서
# 우리 분류기가 판별해야 함
YAMNET_TO_CLASSIFIER = {
    # 아기 울음 관련 (사이렌과 혼동 방지를 위해 2차 분류기가 판단)
    19,    # Crying, sobbing
    20,    # Baby cry, infant cry

    # 사이렌 관련 (한국 사이렌은 나라마다 패턴이 달라 분류기가 판단)
    382,   # Alarm
    389,   # Alarm clock
    390,   # Siren
    391,   # Civil defense siren
    393,   # Smoke detector, smoke alarm
    394,   # Fire alarm

    # 도어락 관련
    348,   # Door
    351,   # Sliding door
    352,   # Slam
    355,   # Squeak
    373,   # Keys jangling
    475,   # Beep, bleep
    476,   # Ping

    # 인터폰 관련
    383,   # Telephone
    386,   # Telephone dialing, DTMF
    387,   # Dial tone
    388,   # Busy signal
    392,   # Buzzer

    # 가전제품 관련
    356,   # Cupboard open or close
    357,   # Drawer open or close
    358,   # Dishes, pots, and pans
    359,   # Cutlery, silverware
    360,   # Chopping (food)
    361,   # Frying (food)
    362,   # Microwave oven
    363,   # Blender
    364,   # Water tap, faucet
    365,   # Sink (filling or washing)
    366,   # Bathtub (filling or washing)
    367,   # Hair dryer
    368,   # Toilet flush
    369,   # Toothbrush
    370,   # Electric toothbrush
    371,   # Vacuum cleaner
    376,   # Electric shaver, electric razor
    406,   # Mechanical fan
    407,   # Air conditioning
    478,   # Clang
    479,   # Squeal
    484,   # Sizzle
    490,   # Hum

    # 기타 관련 가능성 있는 소리
    395,   # Foghorn
    396,   # Whistle
    397,   # Steam whistle
    420,   # Explosion
    428,   # Burst, pop
    430,   # Boom
    436,   # Chink, clink
    477,   # Ding
    489,   # Jingle, tinkle
}

# ============================================================
# 모델 파일 경로
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

YAMNET_MODEL_PATH = os.path.join(MODEL_DIR, "yamnet.tflite")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "hearo_classifier.tflite")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.txt")
YAMNET_CLASSES_PATH = os.path.join(MODEL_DIR, "yamnet_classes.txt")

# ============================================================
# 카테고리 목록 로드
# ============================================================

with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    CATEGORIES = [line.strip() for line in f if line.strip()]

with open(YAMNET_CLASSES_PATH, "r", encoding="utf-8") as f:
    YAMNET_CLASSES = [line.strip() for line in f if line.strip()]

print(f"[초기화] 카테고리 {len(CATEGORIES)}개 로드: {CATEGORIES}")
print(f"[초기화] YAMNet 클래스 {len(YAMNET_CLASSES)}개 로드")

# ============================================================
# TFLite 모델 로드
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

# --- YAMNet 모델 로드 ---
yamnet_interpreter = Interpreter(model_path=YAMNET_MODEL_PATH)
yamnet_interpreter.allocate_tensors()

yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()

print(f"[초기화] YAMNet 모델 로드 완료")
print(f"  - 입력: {yamnet_input_details[0]['shape']} {yamnet_input_details[0]['dtype']}")
for i, out in enumerate(yamnet_output_details):
    print(f"  - 출력[{i}]: {out['shape']} {out['dtype']} (index={out['index']})")

# --- Hearo 분류기 모델 로드 ---
classifier_interpreter = Interpreter(model_path=CLASSIFIER_MODEL_PATH)
classifier_interpreter.allocate_tensors()

classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

print(f"[초기화] Hearo 분류기 로드 완료")
print(f"  - 입력: {classifier_input_details[0]['shape']} (1024차원 임베딩)")
print(f"  - 출력: {len(CATEGORIES)}개 카테고리 확률")
print(f"  - 직접 매핑: {len(YAMNET_DIRECT_MAP)}개 YAMNet 카테고리")
print(f"  - 2차 분류기: {len(YAMNET_TO_CLASSIFIER)}개 YAMNet 카테고리")

# ============================================================
# 핵심 함수들
# ============================================================

def record_audio():
    """
    INMP441 마이크에서 소리를 녹음합니다.
    48kHz로 녹음한 뒤 YAMNet이 요구하는 16kHz로 변환합니다.
    """
    audio = sd.rec(
        int(MIC_SAMPLE_RATE * RECORD_DURATION),
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=2  # INMP441 마이크 (arecord -l에서 card 2번)
    )
    sd.wait()

    audio = audio.flatten()

    # 48kHz → 16kHz 리샘플링 (GAIN 증폭 없음)
    target_length = int(len(audio) * YAMNET_SAMPLE_RATE / MIC_SAMPLE_RATE)
    audio_16k = resample(audio, target_length).astype(np.float32)

    return audio_16k


def run_yamnet(waveform):
    """
    YAMNet을 실행하여 521개 카테고리 점수와 1024차원 임베딩을 반환합니다.

    Returns:
        tuple: (scores, embedding)
            - scores: 521개 카테고리별 점수 (여러 프레임의 평균)
            - embedding: 1024차원 임베딩 벡터 (여러 프레임의 평균)
    """
    input_data = waveform.astype(np.float32)

    # 입력 텐서 크기를 실제 오디오 길이로 재설정
    yamnet_interpreter.resize_tensor_input(
        yamnet_input_details[0]['index'], [len(input_data)]
    )
    yamnet_interpreter.allocate_tensors()

    # YAMNet 실행
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], input_data)
    yamnet_interpreter.invoke()

    # 출력 추출: scores(521개)와 embeddings(1024차원)
    scores = None
    embeddings = None

    for output in yamnet_output_details:
        result = yamnet_interpreter.get_tensor(output['index'])
        if result.shape[-1] == 521:
            scores = result
        elif result.shape[-1] == 1024:
            embeddings = result

    # 여러 프레임의 평균
    if scores is not None:
        scores = scores.mean(axis=0) if len(scores.shape) > 1 else scores.flatten()
    if embeddings is not None:
        embeddings = embeddings.mean(axis=0) if len(embeddings.shape) > 1 else embeddings.flatten()

    return scores, embeddings.astype(np.float32)


def classify_with_yamnet(scores):
    """
    YAMNet의 521개 카테고리 점수를 분석하여:
    1. 직접 매핑 가능한 소리인지 확인
    2. 2차 분류기로 넘길 소리인지 확인
    3. 무시할 소리인지 확인

    Returns:
        tuple: (action, category, confidence)
            - action: "direct" / "classifier" / "ignore"
            - category: Hearo 카테고리 이름 (direct일 때만 유효)
            - confidence: 해당 카테고리의 확신도
    """
    # 상위 카테고리 찾기
    top_idx = np.argmax(scores)
    top_confidence = float(scores[top_idx])

    # 상위 3개 디버그 출력
    top3_idx = np.argsort(scores)[::-1][:3]
    top3_info = [(YAMNET_CLASSES[i], f"{scores[i]:.3f}") for i in top3_idx]
    print(f"  [YAMNet] Top3: {top3_info}")

    # 1. 직접 매핑 확인
    if top_idx in YAMNET_DIRECT_MAP:
        category = YAMNET_DIRECT_MAP[top_idx]
        print(f"  [YAMNet] 직접 매핑: {YAMNET_CLASSES[top_idx]} → {category} ({top_confidence:.3f})")
        return "direct", category, top_confidence

    # 2. 2차 분류기로 넘길지 확인
    if top_idx in YAMNET_TO_CLASSIFIER:
        print(f"  [YAMNet] 2차 분류기로 넘김: {YAMNET_CLASSES[top_idx]} ({top_confidence:.3f})")
        return "classifier", None, top_confidence

    # 3. 그 외는 무시
    return "ignore", None, top_confidence


def classify_sound(embedding):
    """
    Hearo 분류기로 임베딩을 11개 카테고리로 분류합니다.
    (도어락, 인터폰, 가전제품에서만 사용)
    """
    input_data = embedding.reshape(1, -1).astype(np.float32)

    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], input_data)
    classifier_interpreter.invoke()

    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])[0]

    # 디버그: 상위 3개
    top3_idx = np.argsort(prediction)[::-1][:3]
    top3_info = [(CATEGORIES[i], f"{prediction[i]*100:.1f}%") for i in top3_idx]
    print(f"  [분류기] Top3: {top3_info}")

    best_idx = np.argmax(prediction)
    best_category = CATEGORIES[best_idx]
    best_confidence = float(prediction[best_idx])

    return best_category, best_confidence


def send_alert(sound, confidence):
    """감지된 소리 정보를 백엔드 서버로 전송합니다."""
    try:
        response = requests.post(
            f"{SERVER_URL}/api/alerts",
            json={
                "sound": sound,
                "confidence": confidence,
                "location": LOCATION,
                "device_id": DEVICE_ID
            },
            timeout=5
        )
        result = response.json()
        print(f"  → 서버 전송 완료: {result}")
    except requests.exceptions.ConnectionError:
        print(f"  → [오류] 서버 연결 실패! 서버가 켜져있는지 확인하세요: {SERVER_URL}")
    except Exception as e:
        print(f"  → [오류] 서버 전송 실패: {e}")


def send_heartbeat():
    """서버에 연결 상태를 알립니다."""
    try:
        requests.post(
            f"{SERVER_URL}/api/heartbeat",
            json={
                "device_id": DEVICE_ID,
                "location": LOCATION
            },
            timeout=5
        )
    except Exception:
        pass


# ============================================================
# 메인 루프
# ============================================================

def main():
    print()
    print("=" * 55)
    print("  Hearo 환경음 인식 시작")
    print("=" * 55)
    print(f"  서버 주소:    {SERVER_URL}")
    print(f"  설치 위치:    {LOCATION}")
    print(f"  기기 ID:      {DEVICE_ID}")
    print(f"  threshold:    {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"  녹음 길이:    {RECORD_DURATION}초")
    print(f"  쿨다운:       {COOLDOWN_SECONDS}초")
    print(f"  구조:         YAMNet(1차) → 분류기(2차, 도어락/인터폰/가전/사이렌/아기울음)")
    print("=" * 55)
    print()
    print("소리를 듣고 있습니다... (종료: Ctrl+C)")
    print()

    last_alert_time = 0
    last_alert_sound = ""
    last_heartbeat_time = 0

    while True:
        try:
            # --- heartbeat ---
            now = time.time()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_heartbeat()
                last_heartbeat_time = now

            # --- 1. 마이크에서 소리 녹음 ---
            waveform = record_audio()

            # 무음 체크
            volume = np.abs(waveform).mean()
            if volume < 0.003:
                continue

            print(f"  [녹음] 볼륨: 평균={volume:.6f}, 최대={np.abs(waveform).max():.6f}")

            # --- 2. YAMNet 실행 (521개 분류 + 임베딩 추출) ---
            scores, embedding = run_yamnet(waveform)

            # --- 3. YAMNet 결과 분석 ---
            action, category, yamnet_confidence = classify_with_yamnet(scores)

            if action == "direct":
                # YAMNet 직접 매핑: 바로 확정
                if yamnet_confidence >= CONFIDENCE_THRESHOLD:
                    # 쿨다운 체크
                    now = time.time()
                    if category == last_alert_sound and (now - last_alert_time) < COOLDOWN_SECONDS:
                        continue

                    print(f"[감지] {category} ({yamnet_confidence*100:.1f}%) - {LOCATION} [YAMNet]")
                    send_alert(category, yamnet_confidence)

                    last_alert_time = now
                    last_alert_sound = category

            elif action == "classifier":
                # 2차 분류기로 세부 판별 (도어락/인터폰/가전/사이렌/아기울음)
                sound, confidence = classify_sound(embedding)

                # 초인종 → 인터폰으로 통합
                if sound == "초인종":
                    sound = "인터폰"

                # 벨소리는 무시 (음악과 구분 불가)
                if sound == "휴대폰 벨소리":
                    continue

                if confidence >= CONFIDENCE_THRESHOLD:
                    now = time.time()
                    if sound == last_alert_sound and (now - last_alert_time) < COOLDOWN_SECONDS:
                        continue

                    print(f"[감지] {sound} ({confidence*100:.1f}%) - {LOCATION} [분류기]")
                    send_alert(sound, confidence)

                    last_alert_time = now
                    last_alert_sound = sound

            else:
                # 무시 (대화, 음악 등)
                pass

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"[오류] {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()

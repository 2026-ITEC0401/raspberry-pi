"""
=============================================================
Hearo - 라즈베리파이 환경음 인식 스크립트
=============================================================

[이 파일이 하는 일]
USB 마이크로 주변 소리를 계속 듣고 있다가,
학습된 AI 모델로 소리를 분류하고,
threshold(기준점)를 넘기면 백엔드 서버로 결과를 전송합니다.

[전체 흐름]
마이크 녹음 (1초 단위)
    → YAMNet 모델: 소리를 1024차원 숫자(임베딩)로 변환
    → hearo_classifier 모델: 숫자를 11개 카테고리로 분류
    → confidence(확신도)가 threshold 이상이면
    → 백엔드 서버(FastAPI)로 POST 전송

[실행 전 준비사항]
1. 필요한 패키지 설치:
   pip install -r requirements.txt

2. USB 마이크가 라즈베리파이에 연결되어 있어야 합니다.
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
# 예: 같은 Wi-Fi에서 서버 PC의 IP가 192.168.0.10이면
#     "http://192.168.0.10:8000"으로 변경
SERVER_URL = "http://192.168.137.1:8000"

# 이 라즈베리파이의 설치 위치 (프론트엔드에 표시됨)
LOCATION = "현관"

# 이 라즈베리파이의 기기 식별자 (여러 대일 때 구분용)
DEVICE_ID = "rpi-001"

# threshold: 이 값 이상일 때만 서버로 전송 (0.0 ~ 1.0)
# 0.9 = 90% 이상 확신할 때만 알림
CONFIDENCE_THRESHOLD = 0.9

# 녹음 설정
MIC_SAMPLE_RATE = 48000  # INMP441 마이크의 샘플레이트 (48kHz)
YAMNET_SAMPLE_RATE = 16000  # YAMNet이 요구하는 샘플레이트 (16kHz)
RECORD_DURATION = 1.0    # 한 번에 녹음할 길이 (초)

# heartbeat 주기 (초): 서버에 "나 연결돼있어"라고 알려주는 간격
HEARTBEAT_INTERVAL = 30

# 같은 소리가 반복 전송되는 것을 방지하는 쿨다운 (초)
# 예: 5초 안에 같은 "초인종" 소리가 연속 감지되면 첫 번째만 전송
COOLDOWN_SECONDS = 5

# ============================================================
# 모델 파일 경로 (model/ 폴더에 있는 파일들)
# ============================================================

# 이 스크립트 파일이 있는 폴더 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

YAMNET_MODEL_PATH = os.path.join(MODEL_DIR, "yamnet.tflite")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "hearo_classifier.tflite")
CATEGORIES_PATH = os.path.join(MODEL_DIR, "categories.txt")

# ============================================================
# 카테고리 목록 로드
# ============================================================

# categories.txt에서 11개 카테고리 이름을 읽어옴
# ["가전제품", "개 짖음", "노크 소리", ... , "휴대폰 벨소리"]
with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
    CATEGORIES = [line.strip() for line in f if line.strip()]

print(f"[초기화] 카테고리 {len(CATEGORIES)}개 로드: {CATEGORIES}")

# ============================================================
# TFLite 모델 로드
# ============================================================

# tflite_runtime은 tensorflow 전체를 설치하지 않아도 되는 경량 패키지
# 라즈베리파이에서는 이걸 써야 메모리/성능 문제가 없음
try:
    # Python 3.13 이상: ai-edge-litert 사용
    from ai_edge_litert.interpreter import Interpreter
    print("[초기화] ai-edge-litert 사용")
except ImportError:
    try:
        # 기존 tflite_runtime
        from tflite_runtime.interpreter import Interpreter
        print("[초기화] tflite_runtime 사용")
    except ImportError:
        # PC에서 테스트할 때는 tensorflow에 포함된 것 사용
        from tensorflow.lite.python.interpreter import Interpreter
        print("[초기화] tensorflow.lite 사용")

# --- 1단계 모델: YAMNet (소리 → 1024차원 임베딩 변환) ---
# 소리 파형(waveform)을 받아서 의미 있는 숫자 배열로 바꿔주는 역할
# 이 모델 자체는 분류를 하지 않고, 특징만 추출함
yamnet_interpreter = Interpreter(model_path=YAMNET_MODEL_PATH)
yamnet_interpreter.allocate_tensors()

# 모델의 입력/출력 정보 확인
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()

print(f"[초기화] YAMNet 모델 로드 완료")
print(f"  - 입력: {yamnet_input_details[0]['shape']} {yamnet_input_details[0]['dtype']} (소리 파형)")
for i, out in enumerate(yamnet_output_details):
    print(f"  - 출력[{i}]: {out['shape']} {out['dtype']} (index={out['index']})")

# --- 2단계 모델: Hearo 분류기 (1024차원 임베딩 → 11개 카테고리) ---
# YAMNet이 뽑아준 숫자 배열을 보고 "이건 도어락 소리다"라고 판단하는 역할
classifier_interpreter = Interpreter(model_path=CLASSIFIER_MODEL_PATH)
classifier_interpreter.allocate_tensors()

classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

print(f"[초기화] Hearo 분류기 로드 완료")
print(f"  - 입력: {classifier_input_details[0]['shape']} (1024차원 임베딩)")
print(f"  - 출력: {len(CATEGORIES)}개 카테고리 확률")

# ============================================================
# 핵심 함수들
# ============================================================

def record_audio():
    """
    INMP441 마이크에서 소리를 녹음합니다.
    48kHz로 녹음한 뒤 YAMNet이 요구하는 16kHz로 변환합니다.

    Returns:
        numpy array: 녹음된 소리 데이터 (1차원, float32, 16kHz)
    """
    # 48kHz로 녹음 (INMP441 기본 샘플레이트)
    audio = sd.rec(
        int(MIC_SAMPLE_RATE * RECORD_DURATION),
        samplerate=MIC_SAMPLE_RATE,
        channels=1,          # 모노 (마이크 1개)
        dtype='float32'
    )
    sd.wait()  # 녹음이 끝날 때까지 대기

    # (N, 1) 형태를 (N,) 1차원으로 변환
    audio = audio.flatten()

    # 48kHz → 16kHz로 리샘플링 (YAMNet 입력 요구사항)
    # 주의: GAIN 증폭을 하지 않음!
    # 학습 데이터가 원본 볼륨 그대로 학습되었기 때문에,
    # 여기서도 원본 볼륨을 유지해야 일관성이 맞음
    target_length = int(len(audio) * YAMNET_SAMPLE_RATE / MIC_SAMPLE_RATE)
    audio_16k = resample(audio, target_length).astype(np.float32)

    return audio_16k


def extract_embedding(waveform):
    """
    YAMNet 모델로 소리에서 임베딩(특징 벡터)을 추출합니다.

    쉽게 말하면: 소리를 AI가 이해할 수 있는 1024개의 숫자로 바꿔주는 함수

    Args:
        waveform: 녹음된 소리 데이터 (1차원 numpy array)

    Returns:
        numpy array: 1024차원 임베딩 벡터
    """
    # YAMNet TFLite는 동적 입력 크기를 사용함 (shape=[1])
    # 실제 waveform 길이에 맞게 입력 텐서 크기를 조정해야 함
    input_data = waveform.astype(np.float32)

    # 입력 텐서 크기를 실제 오디오 길이로 재설정
    yamnet_interpreter.resize_tensor_input(
        yamnet_input_details[0]['index'], [len(input_data)]
    )
    yamnet_interpreter.allocate_tensors()

    # YAMNet 실행: 소리 → 임베딩
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], input_data)
    yamnet_interpreter.invoke()

    # YAMNet 출력은 여러 개인데, 임베딩은 보통 두 번째(index 1)
    # 출력 shape을 확인해서 1024차원인 것을 찾음
    embeddings = None
    for output in yamnet_output_details:
        result = yamnet_interpreter.get_tensor(output['index'])
        # 1024차원인 출력이 임베딩
        if result.shape[-1] == 1024:
            embeddings = result
            break

    if embeddings is None:
        # 못 찾으면 첫 번째 출력 사용
        embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[0]['index'])

    # 여러 프레임의 임베딩을 평균내서 하나로 만듦
    # (Colab 학습할 때도 같은 방식: embeddings.numpy().mean(axis=0))
    return embeddings.mean(axis=0).astype(np.float32)


def classify_sound(embedding):
    """
    Hearo 분류기로 임베딩을 11개 카테고리로 분류합니다.

    쉽게 말하면: 1024개의 숫자를 보고 "이건 도어락 소리야"라고 판단하는 함수

    Args:
        embedding: 1024차원 임베딩 벡터

    Returns:
        tuple: (카테고리 이름, 확신도)
               예: ("도어락", 0.95)
    """
    # 모델 입력 형태에 맞게 변환: (1024,) → (1, 1024)
    input_data = embedding.reshape(1, -1).astype(np.float32)

    # 분류기 실행: 임베딩 → 11개 카테고리별 확률
    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], input_data)
    classifier_interpreter.invoke()

    # 결과: [0.01, 0.02, 0.01, 0.95, ...] 같은 확률 배열
    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])[0]

    # 디버그: 상위 3개 카테고리 확률 출력
    top3_idx = np.argsort(prediction)[::-1][:3]
    top3_info = [(CATEGORIES[i], f"{prediction[i]*100:.1f}%") for i in top3_idx]
    print(f"  [디버그] Top3: {top3_info}")

    # 가장 높은 확률의 카테고리 찾기
    best_idx = np.argmax(prediction)
    best_category = CATEGORIES[best_idx]
    best_confidence = float(prediction[best_idx])

    return best_category, best_confidence


def send_alert(sound, confidence):
    """
    감지된 소리 정보를 백엔드 서버로 전송합니다.

    서버에서는 이 데이터를 DB에 저장하고,
    WebSocket으로 프론트엔드에 실시간 알림을 보냅니다.

    Args:
        sound: 감지된 소리 카테고리 (예: "도어락")
        confidence: 확신도 (예: 0.95)
    """
    try:
        response = requests.post(
            f"{SERVER_URL}/api/alerts",
            json={
                "sound": sound,
                "confidence": confidence,
                "location": LOCATION,
                "device_id": DEVICE_ID
            },
            timeout=5  # 5초 안에 응답 없으면 포기
        )
        result = response.json()
        print(f"  → 서버 전송 완료: {result}")
    except requests.exceptions.ConnectionError:
        print(f"  → [오류] 서버 연결 실패! 서버가 켜져있는지 확인하세요: {SERVER_URL}")
    except Exception as e:
        print(f"  → [오류] 서버 전송 실패: {e}")


def send_heartbeat():
    """
    서버에 "이 기기가 연결돼있다"고 알려줍니다.

    서버는 이 신호가 일정 시간 안 오면 기기가 꺼진 것으로 판단하고,
    프론트엔드에 isConnected: false로 표시합니다.
    """
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
        pass  # heartbeat 실패는 무시 (다음에 다시 보내면 됨)


# ============================================================
# 메인 루프: 계속 소리를 듣고 분류하고 전송
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
    print("=" * 55)
    print()
    print("소리를 듣고 있습니다... (종료: Ctrl+C)")
    print()

    # 마지막으로 알림을 보낸 시각과 카테고리 (중복 전송 방지용)
    last_alert_time = 0
    last_alert_sound = ""

    # 마지막 heartbeat 전송 시각
    last_heartbeat_time = 0

    while True:
        try:
            # --- heartbeat 전송 (일정 간격마다) ---
            now = time.time()
            if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                send_heartbeat()
                last_heartbeat_time = now

            # --- 1. 마이크에서 소리 녹음 ---
            waveform = record_audio()

            # 무음 체크: 너무 조용하면 건너뜀 (불필요한 추론 방지)
            # 무음 체크: 배경 소음 수준이면 추론을 건너뜀
            # INMP441 기준 실제 소리가 있을 때 평균 볼륨 약 0.003 이상
            # 이 기준이 없으면 무음도 11개 카테고리 중 하나로 강제 분류됨
            volume = np.abs(waveform).mean()
            if volume < 0.003:
                continue

            # --- 디버그: 입력 오디오 상태 확인 ---
            print(f"  [디버그] 볼륨: 평균={volume:.6f}, 최대={np.abs(waveform).max():.6f}")

            # --- 2. YAMNet으로 임베딩 추출 ---
            embedding = extract_embedding(waveform)

            # --- 3. Hearo 분류기로 카테고리 판별 ---
            sound, confidence = classify_sound(embedding)

            # --- 디버그: 분류 결과 상세 출력 ---
            print(f"  [디버그] 결과: {sound} ({confidence*100:.1f}%)")

            # --- 4. threshold 체크 ---
            if confidence >= CONFIDENCE_THRESHOLD:
                # 쿨다운 체크: 같은 소리가 연속으로 감지되면 무시
                now = time.time()
                if sound == last_alert_sound and (now - last_alert_time) < COOLDOWN_SECONDS:
                    continue

                print(f"[감지] {sound} ({confidence * 100:.1f}%) - {LOCATION}")

                # --- 5. 서버로 전송 ---
                send_alert(sound, confidence)

                last_alert_time = now
                last_alert_sound = sound

        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"[오류] {e}")
            time.sleep(1)  # 오류 발생 시 1초 대기 후 재시도


# 이 파일을 직접 실행했을 때만 main() 실행
# (다른 파일에서 import할 때는 실행 안 됨)
if __name__ == "__main__":
    main()

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
        → 노크/도어락/비상벨/아기울음 관련 → 2차 분류기로 넘김
    → 2차: Hearo 분류기로 4개 카테고리 판별
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
import os
import sounddevice as sd
from scipy.signal import resample

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# ============================================================
# 설정값 (필요에 따라 수정하세요)
# ============================================================

# 이 라즈베리파이의 설치 위치 (프론트엔드에 표시됨)
LOCATION = "거실"

# 이 라즈베리파이의 기기 식별자 (여러 대일 때 구분용)
DEVICE_ID = "rpi-001"

# threshold: 이 값 이상일 때만 서버로 전송 (0.0 ~ 1.0)
# 2차 분류기: 분류기의 confidence를 사용 (0.7)
CLASSIFIER_THRESHOLD = 0.7

# 녹음 설정
MIC_SAMPLE_RATE = 48000  # INMP441 마이크의 샘플레이트 (48kHz)
YAMNET_SAMPLE_RATE = 16000  # YAMNet이 요구하는 샘플레이트 (16kHz)
RECORD_DURATION = 1.0    # 한 번에 녹음할 길이 (초)

# 같은 소리가 반복 전송되는 것을 방지하는 쿨다운 (초)
COOLDOWN_SECONDS = 5

# ============================================================
# YAMNet → Hearo 카테고리 매핑 (v3)
# ============================================================
# YAMNet의 521개 카테고리 중 관련된 것을 매핑
#
# 2차 분류기 (4개): 노크소리, 도어락소리, 비상벨소리, 아기울음소리
# 나머지: 무시 (대화, 음악, 동물 등)

# --- 2차 분류기로 넘길 YAMNet 인덱스 ---
# 이 소리들은 한국 가정음 특화이거나 세부 판단이 필요해서
# 우리 분류기가 판별해야 함
YAMNET_TO_CLASSIFIER = {
    # 노크 소리 관련 (YAMNet 단독으로 정확도 부족, 2차 분류기가 판단)
    353,   # Knock
    354,   # Tap

    # 아기 울음 관련 (사이렌과 혼동 방지를 위해 2차 분류기가 판단)
    19,    # Crying, sobbing
    20,    # Baby cry, infant cry

    # 사이렌 관련 (한국 사이렌 패턴 특화)
    304,   # Car alarm
    316,   # Emergency vehicle
    317,   # Police car (siren)
    318,   # Ambulance (siren)
    319,   # Fire engine, fire truck (siren)
    382,   # Alarm
    389,   # Alarm clock
    390,   # Siren
    391,   # Civil defense siren
    392,   # Buzzer
    393,   # Smoke detector, smoke alarm
    394,   # Fire alarm
    494,   # Sine wave (전자음/비프음 — 사이렌, 가전, 벨소리 등이 이것으로 잡힘)

    # 도어락 관련
    348,   # Door
    349,   # Doorbell
    351,   # Sliding door
    352,   # Slam
    355,   # Squeak
    373,   # Keys jangling
    475,   # Beep, bleep
    476,   # Ping
    477,   # Ding
    485,   # Clicking

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

# print(f"[초기화] 카테고리 {len(CATEGORIES)}개 로드: {CATEGORIES}")
# print(f"[초기화] YAMNet 클래스 {len(YAMNET_CLASSES)}개 로드")

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

# print(f"[초기화] YAMNet 모델 로드 완료")
# print(f"  - 입력: {yamnet_input_details[0]['shape']} {yamnet_input_details[0]['dtype']}")
# for i, out in enumerate(yamnet_output_details):
#     print(f"  - 출력[{i}]: {out['shape']} {out['dtype']} (index={out['index']})")

# --- Hearo 분류기 모델 로드 ---
classifier_interpreter = Interpreter(model_path=CLASSIFIER_MODEL_PATH)
classifier_interpreter.allocate_tensors()

classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# print(f"[초기화] Hearo 분류기 로드 완료")
# print(f"  - 입력: {classifier_input_details[0]['shape']} (1024차원 임베딩)")
# print(f"  - 출력: {len(CATEGORIES)}개 카테고리 확률")
# print(f"  - 2차 분류기: {len(YAMNET_TO_CLASSIFIER)}개 YAMNet 카테고리")

# ============================================================
# 마이크 장치 자동 검색 (시작 시 한 번)
# ============================================================
MIC_DEVICE_INDEX = None
# print("\n[초기화] 마이크 장치 검색 중...")
for i, dev in enumerate(sd.query_devices()):
    # print(f"  [{i}] {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
    # 입력 채널이 있고, voicehat 또는 i2s가 이름에 포함된 장치 찾기
    if dev['max_input_channels'] > 0 and ('voicehat' in dev['name'].lower() or 'i2s' in dev['name'].lower()):
        MIC_DEVICE_INDEX = i

if MIC_DEVICE_INDEX is None:
    raise RuntimeError("INMP441 마이크를 찾을 수 없습니다! arecord -l 명령으로 확인하세요.")

# print(f"[초기화] ✅ 마이크 장치 선택: [{MIC_DEVICE_INDEX}]\n")

# ============================================================
# 핵심 함수들
# ============================================================

def record_audio():
    audio = sd.rec(
        int(MIC_SAMPLE_RATE * RECORD_DURATION),
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=MIC_DEVICE_INDEX  # ← 자동 검색된 번호 사용
    )
    sd.wait()

    audio = audio.flatten()
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

    # 1. 2차 분류기로 넘길지 확인
    if top_idx in YAMNET_TO_CLASSIFIER:
        print()
        print(f"\033[96m  [YAMNet] 2차 분류기로 넘김: {YAMNET_CLASSES[top_idx]} ({top_confidence:.3f})\033[0m")
        return "classifier", None, top_confidence

    # 2. 그 외는 무시
    return "ignore", None, top_confidence


def classify_sound(embedding):
    """
    Hearo 분류기로 임베딩을 4개 상위 카테고리로 분류합니다.
    (노크소리, 도어락소리, 비상벨소리, 아기울음소리)

    세부 카테고리(예: 사이렌_삐뽀삐뽀, 사이렌_안내음 등)의 확률을
    상위 카테고리 그룹별로 합산하여 반환합니다.
    """
    input_data = embedding.reshape(1, -1).astype(np.float32)

    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], input_data)
    classifier_interpreter.invoke()

    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])[0]

    # 디버그: 세부 카테고리별 확률
    top3_idx = np.argsort(prediction)[::-1][:3]
    top3_info = [(CATEGORIES[i], f"{prediction[i]*100:.1f}%") for i in top3_idx]
    print(f"  [분류기] 세부 Top3: {top3_info}")

    # 세부 카테고리 → 상위 카테고리 그룹 매핑
    GROUP_MAP = {
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

    # 상위 카테고리 그룹별 확률 합산
    group_scores = {}
    for i, category in enumerate(CATEGORIES):
        group = GROUP_MAP.get(category)
        if group is None:
            continue
        group_scores[group] = group_scores.get(group, 0.0) + float(prediction[i])

    # 디버그: 그룹별 합산 결과
    print(f"  [분류기] 그룹 합산: {{{', '.join(f'{k}: {v*100:.1f}%' for k, v in group_scores.items())}}}")

    if not group_scores:
        return None, 0.0

    best_category = max(group_scores, key=group_scores.get)
    best_confidence = group_scores[best_category]

    return best_category, best_confidence
    

# 분류기 결과를 프론트엔드 LED 색상 타입으로 매핑
SOUND_TYPE_MAP = {
    "도어락소리": "Visitor",
    "노크소리": "Visitor",
    "비상벨소리": "Urgent",
    "아기울음소리": "Noise"
}

# 소리 카테고리별 이모티콘 매핑
SOUND_EMOJI_MAP = {
    "도어락소리": "🔐",
    "노크소리": "✊",
    "비상벨소리": "🚨",
    "아기울음소리": "👶"
}

def send_alert(sound, confidence):
    """감지된 소리 정보를 Firebase(Firestore)로 전송합니다."""
    try:
        # 1. 프론트엔드 명세에 맞게 type 필드 추가
        sound_type = SOUND_TYPE_MAP.get(sound, "Urgent-red") 
        sound_emoji = SOUND_EMOJI_MAP.get(sound, "🔔") # 기본 이모티콘 설정
        
        alert_data = {
            "sound": sound,
            "type": sound_type,              # Urgent, Visitor, Appliance
            "confidence": float(confidence), 
            "location": LOCATION,
            "device_id": DEVICE_ID,
            # 2. timestamp -> time으로 필드명 변경, 3. 타임스탬프 형식은 기존대로 유지
            "time": firestore.SERVER_TIMESTAMP  
        }
        
        # Firestore의 'alarms' 컬렉션에 데이터 추가
        db.collection('alarms').add(alert_data)
        print(f"  → Firebase 전송 완료: {sound_emoji} {sound} ({confidence*100:.1f}%) [type: {sound_type}]")
        print()
        
    except Exception as e:
        print(f"  → [오류] Firebase 전송 실패: {e}")


# ============================================================
# 메인 루프
# ============================================================

def main():
    print()
    print("=" * 55)
    print("  Hearo 환경음 인식 시작")
    print("=" * 55)
    print(f"  설치 위치:    {LOCATION}")
    print(f"  기기 ID:      {DEVICE_ID}")
    print(f"  분류기 임계값: {CLASSIFIER_THRESHOLD * 100:.0f}%")
    print(f"  녹음 길이:    {RECORD_DURATION}초")
    print(f"  쿨다운:       {COOLDOWN_SECONDS}초")
    print(f"  구조:         YAMNet(1차) → 분류기(2차, 노크/도어락/비상벨/아기울음)")
    print("=" * 55)
    print()
    print("소리를 듣고 있습니다... (종료: Ctrl+C)")
    print()

    last_alert_time = 0
    last_alert_sound = ""

    while True:
        try:
            # --- 1. 마이크에서 소리 녹음 ---
            waveform = record_audio()

            # 무음 체크
            volume = np.abs(waveform).mean()
            if volume < 0.003:
                continue

            # print(f"  [녹음] 볼륨: 평균={volume:.6f}, 최대={np.abs(waveform).max():.6f}")

            # --- 2. YAMNet 실행 (521개 분류 + 임베딩 추출) ---
            scores, embedding = run_yamnet(waveform)

            # --- 3. YAMNet 결과 분석 ---
            action, category, yamnet_confidence = classify_with_yamnet(scores)

            if action == "classifier":
                # 2차 분류기로 세부 판별 (노크/도어락/비상벨/아기울음)
                sound, confidence = classify_sound(embedding)

                if sound and confidence >= CLASSIFIER_THRESHOLD:
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

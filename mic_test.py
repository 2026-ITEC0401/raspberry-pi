"""
=============================================================
INMP441 마이크 테스트 코드
=============================================================

[이 파일이 하는 일]
INMP441 마이크가 제대로 동작하는지 확인하는 테스트 코드입니다.
3초 동안 녹음한 뒤 test.wav 파일로 저장합니다.

[배선 확인]
INMP441 핀    →    라즈베리파이 핀 번호
  VDD         →    1번  (3.3V)
  GND         →    6번  (GND)
  L/R         →    6번  (GND) ※ GND와 같은 핀에 연결
  SCK         →    12번 (GPIO 18)
  WS          →    35번 (GPIO 19)
  SD          →    38번 (GPIO 20)

[실행 전 준비]
1. I2S 드라이버 활성화 (최초 1회만):
   echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a /boot/config.txt
   sudo reboot

2. 마이크 인식 확인:
   arecord -l
   → card 번호가 나오면 인식된 것

3. 패키지 설치:
   sudo apt-get install libportaudio2
   pip install sounddevice numpy scipy

[실행 방법]
   python mic_test.py

[결과 확인]
- "마이크 정상 동작!" → 성공
- "소리가 감지되지 않습니다" → 배선 또는 드라이버 확인 필요
- 생성된 test.wav 파일을 재생해서 소리가 들리는지 확인
=============================================================
"""

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# ============================================================
# 설정
# ============================================================

SAMPLE_RATE = 16000    # 샘플레이트 (16kHz)
DURATION = 3           # 녹음 시간 (초)
OUTPUT_FILE = "test.wav"

# ============================================================
# 1단계: 오디오 장치 확인
# ============================================================

print("=" * 50)
print("  INMP441 마이크 테스트")
print("=" * 50)
print()

# 현재 연결된 오디오 장치 목록 출력
print("[1단계] 오디오 장치 목록:")
print("-" * 50)
devices = sd.query_devices()
print(devices)
print("-" * 50)
print()

# 기본 입력 장치 확인
try:
    default_input = sd.query_devices(kind='input')
    print(f"기본 입력 장치: {default_input['name']}")
    print(f"  - 채널 수: {default_input['max_input_channels']}")
    print(f"  - 기본 샘플레이트: {default_input['default_samplerate']}")
    print()
except Exception as e:
    print(f"[오류] 입력 장치를 찾을 수 없습니다: {e}")
    print("  → 배선과 I2S 드라이버 설정을 확인하세요.")
    print("  → arecord -l 명령어로 마이크가 인식되는지 확인하세요.")
    exit(1)

# ============================================================
# 2단계: 녹음
# ============================================================

print(f"[2단계] {DURATION}초 동안 녹음합니다... 소리를 내보세요!")
print()

try:
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # 녹음 끝날 때까지 대기
except Exception as e:
    print(f"[오류] 녹음 실패: {e}")
    print("  → 마이크 배선을 다시 확인하세요.")
    print("  → I2S 드라이버가 활성화되었는지 확인하세요.")
    exit(1)

# ============================================================
# 3단계: 결과 분석
# ============================================================

print("[3단계] 녹음 결과 분석:")
print(f"  - 샘플 수: {len(audio)}")
print(f"  - 샘플레이트: {SAMPLE_RATE}Hz")
print(f"  - 녹음 길이: {len(audio) / SAMPLE_RATE:.1f}초")

# 볼륨 분석
max_volume = np.abs(audio).max()
avg_volume = np.abs(audio).mean()
print(f"  - 최대 볼륨: {max_volume:.4f}")
print(f"  - 평균 볼륨: {avg_volume:.4f}")
print()

# ============================================================
# 4단계: wav 파일 저장
# ============================================================

write(OUTPUT_FILE, SAMPLE_RATE, audio)
print(f"[4단계] {OUTPUT_FILE} 저장 완료!")
print()

# ============================================================
# 5단계: 판정
# ============================================================

print("=" * 50)
if max_volume > 0.01:
    print("  마이크 정상 동작!")
    print(f"  녹음된 파일: {OUTPUT_FILE}")
    print("  이 파일을 재생해서 소리가 들리는지 확인하세요.")
elif max_volume > 0.001:
    print("  소리가 매우 작게 감지됩니다.")
    print("  마이크 가까이에서 큰 소리를 내보세요.")
    print("  그래도 작으면 배선을 확인하세요.")
else:
    print("  소리가 감지되지 않습니다.")
    print("  확인할 것:")
    print("  1. 배선이 올바른지 (특히 SD→38번, SCK→12번, WS→35번)")
    print("  2. L/R 핀이 GND(9번)에 연결되었는지")
    print("  3. I2S 드라이버가 활성화되었는지")
    print("     → arecord -l 로 확인")
print("=" * 50)

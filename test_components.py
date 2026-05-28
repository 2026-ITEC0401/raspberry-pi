"""
=============================================================
Hearo 하드웨어 연결 테스트 (LED 4개 + 진동 모터)
=============================================================
각 부품을 하나씩 순서대로 켜고 끄며 연결을 확인합니다.
실행: python test_components.py
종료: Ctrl + C
=============================================================
"""
import RPi.GPIO as GPIO
import time

# ============================================================
# 핀 설정 (실제 연결한 GPIO 번호로 수정하세요!)
# ============================================================
LED_PINS = {
    "빨강 LED (화재경보)": 17,
    "노랑 LED (초인종)": 27,
    "파랑 LED (아기울음)": 22,
    "초록 LED (개짖음)": 23,
}
MOTOR_PIN = 24  # 진동 모터 (트랜지스터 베이스)

# ============================================================
# GPIO 초기화
# ============================================================
GPIO.setmode(GPIO.BCM)   # BCM 번호 체계 사용 (GPIO 번호)
GPIO.setwarnings(False)

# 모든 핀을 출력으로 설정
for name, pin in LED_PINS.items():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(MOTOR_PIN, GPIO.OUT)
GPIO.output(MOTOR_PIN, GPIO.LOW)

print("=" * 55)
print("  Hearo 하드웨어 연결 테스트 시작")
print("=" * 55)
print()

try:
    # --- 1. LED 순차 테스트 ---
    print("[1단계] LED 4개 순차 점등 테스트")
    print("-" * 55)
    for name, pin in LED_PINS.items():
        print(f"  → {name} (GPIO {pin}) 켜짐... ", end="", flush=True)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(1.5)
        GPIO.output(pin, GPIO.LOW)
        print("꺼짐 ✓")
        time.sleep(0.5)

    print()

    # --- 2. LED 전체 동시 점등 ---
    print("[2단계] LED 전체 동시 점등 (2초)")
    print("-" * 55)
    for name, pin in LED_PINS.items():
        GPIO.output(pin, GPIO.HIGH)
    print("  → 모든 LED 켜짐!")
    time.sleep(2)
    for name, pin in LED_PINS.items():
        GPIO.output(pin, GPIO.LOW)
    print("  → 모든 LED 꺼짐 ✓")
    print()

    # --- 3. 진동 모터 테스트 ---
    print("[3단계] 진동 모터 테스트")
    print("-" * 55)
    print(f"  → 진동 모터 (GPIO {MOTOR_PIN}) 작동... ", end="", flush=True)
    GPIO.output(MOTOR_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(MOTOR_PIN, GPIO.LOW)
    print("정지 ✓")
    print()

    # --- 4. 종합 알림 시뮬레이션 ---
    print("[4단계] 실제 알림 시뮬레이션 (LED + 진동 동시)")
    print("-" * 55)
    print("  → '화재경보 감지!' 시뮬레이션")
    for _ in range(3):
        GPIO.output(LED_PINS["빨강 LED (화재경보)"], GPIO.HIGH)
        GPIO.output(MOTOR_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED_PINS["빨강 LED (화재경보)"], GPIO.LOW)
        GPIO.output(MOTOR_PIN, GPIO.LOW)
        time.sleep(0.3)
    print("  → 시뮬레이션 완료 ✓")
    print()

    print("=" * 55)
    print("  ✅ 모든 테스트 완료!")
    print("=" * 55)

except KeyboardInterrupt:
    print("\n\n테스트를 중단합니다.")

finally:
    GPIO.cleanup()  # 핀 정리 (중요!)
    print("GPIO 정리 완료.")

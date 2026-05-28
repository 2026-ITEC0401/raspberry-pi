"""
=============================================================
Hearo 하드웨어 대화형 테스트
=============================================================
원하는 부품을 골라서 켜고 끌 수 있습니다.
실행: python test_interactive.py
=============================================================
"""
import RPi.GPIO as GPIO

# 핀 설정 (실제 연결한 GPIO로 수정하세요!)
COMPONENTS = {
    "1": ("빨강 LED", 17),
    "2": ("노랑 LED", 27),
    "3": ("파랑 LED", 22),
    "4": ("초록 LED", 23),
    "5": ("진동 모터", 24),
}

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for key, (name, pin) in COMPONENTS.items():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

def show_menu():
    print("\n" + "=" * 45)
    print("  테스트할 부품을 선택하세요")
    print("=" * 45)
    for key, (name, pin) in COMPONENTS.items():
        state = "🟢 켜짐" if GPIO.input(pin) else "⚫ 꺼짐"
        print(f"  {key}. {name} (GPIO {pin}) - {state}")
    print("  a. 전체 켜기")
    print("  z. 전체 끄기")
    print("  q. 종료")
    print("=" * 45)

try:
    while True:
        show_menu()
        choice = input("선택: ").strip().lower()

        if choice == "q":
            break
        elif choice == "a":
            for key, (name, pin) in COMPONENTS.items():
                GPIO.output(pin, GPIO.HIGH)
            print("✅ 전체 켜짐")
        elif choice == "z":
            for key, (name, pin) in COMPONENTS.items():
                GPIO.output(pin, GPIO.LOW)
            print("✅ 전체 꺼짐")
        elif choice in COMPONENTS:
            name, pin = COMPONENTS[choice]
            # 현재 상태 반전
            current = GPIO.input(pin)
            GPIO.output(pin, not current)
            new_state = "켜짐" if not current else "꺼짐"
            print(f"✅ {name} → {new_state}")
        else:
            print("❌ 잘못된 입력입니다.")

except KeyboardInterrupt:
    print("\n중단합니다.")

finally:
    GPIO.cleanup()
    print("GPIO 정리 완료.")

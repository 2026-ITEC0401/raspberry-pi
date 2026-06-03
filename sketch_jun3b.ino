// LED 핀
const int LED_RED    = 25;  // 비상벨 - 빠른 깜빡임
const int LED_YELLOW = 26;  // 도어락/노크 - 천천히 깜빡임
const int LED_BLUE   = 27;  // 아기울음 - 천천히 깜빡임
const int LED_GREEN  = 14;  // 예비

void setup() {
  Serial.begin(115200);
  
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  
  // 초기 모두 끄기
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_GREEN, LOW);
  
  Serial.println("ESP32 LED 패턴 테스트 시작!");
}

// LED 깜빡임 함수
// pin: GPIO 핀 번호
// interval: 깜빡임 간격 (밀리초)
// duration: 총 지속 시간 (밀리초)
void blinkLED(int pin, int interval, int duration) {
  int elapsed = 0;
  while (elapsed < duration) {
    digitalWrite(pin, HIGH);
    delay(interval);
    digitalWrite(pin, LOW);
    delay(interval);
    elapsed += interval * 2;
  }
  digitalWrite(pin, LOW);  // 확실히 끄기
}

void loop() {
  // 비상벨 시뮬레이션 (빠른 깜빡임)
  Serial.println("🚨 비상벨 알림!");
  blinkLED(LED_RED, 200, 3000);  // 0.2초 간격, 3초간
  
  delay(2000);
  
  // 도어락 시뮬레이션 (천천히)
  Serial.println("🔐 도어락 알림!");
  blinkLED(LED_YELLOW, 500, 3000);  // 0.5초 간격, 3초간
  
  delay(2000);
  
  // 아기울음 시뮬레이션
  Serial.println("👶 아기울음 알림!");
  blinkLED(LED_BLUE, 500, 3000);
  
  delay(3000);
}
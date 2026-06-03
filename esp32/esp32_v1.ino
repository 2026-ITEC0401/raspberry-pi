/*
=============================================================
Hearo ESP32 — Wi-Fi MQTT 수신 + LED 알림
=============================================================
라파이에서 보낸 소리 분류 결과를 받아 LED 깜빡임
=============================================================
*/

#include <WiFi.h>
#include <PubSubClient.h>

// ============================================================
// Wi-Fi 설정 (본인 핫스팟 정보로 변경!)
// ============================================================
const char* WIFI_SSID = "wonseok";       // ⚠️ 본인 핫스팟 이름
const char* WIFI_PASSWORD = "dlwlrma0";  // ⚠️ 본인 비밀번호

// ============================================================
// MQTT 설정
// ============================================================
const char* MQTT_BROKER = "172.20.10.11";  // ⚠️ 라파이 IP 주소!
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "hearo/alert";
const char* MQTT_CLIENT_ID = "hearo-esp32";

// ============================================================
// LED 핀 설정
// ============================================================
const int LED_RED = 25;     // 비상벨
const int LED_YELLOW = 26;  // 도어락/노크
const int LED_BLUE = 27;    // 아기울음
const int LED_GREEN = 14;   // 예비

// 깜빡임 설정 (밀리초)
const int BLINK_DURATION = 5000;  // 5초간

// ============================================================
// Wi-Fi 및 MQTT 클라이언트 객체
// ============================================================
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ============================================================
// Wi-Fi 연결 함수
// ============================================================
void connectWiFi() {
  Serial.println();
  Serial.print("[Wi-Fi] 연결 중: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("[Wi-Fi] ✅ 연결 완료!");
  Serial.print("[Wi-Fi] IP 주소: ");
  Serial.println(WiFi.localIP());
}

// ============================================================
// LED 깜빡임 함수
// ============================================================
void blinkLED(int pin, int interval, int duration) {
  int elapsed = 0;
  while (elapsed < duration) {
    digitalWrite(pin, HIGH);
    delay(interval);
    digitalWrite(pin, LOW);
    delay(interval);
    elapsed += interval * 2;
  }
  digitalWrite(pin, LOW);
}

// ============================================================
// MQTT 메시지 수신 콜백 함수 (★ 핵심!)
// ============================================================
void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  // 메시지를 문자열로 변환
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.println();
  Serial.print("[MQTT] 수신: ");
  Serial.println(message);

  // 소리 종류에 따라 LED 깜빡임
  if (message == "비상벨소리") {
    Serial.println("🚨 비상벨 알림! (빨간 LED 빠르게)");
    blinkLED(LED_RED, 200, BLINK_DURATION);
  } else if (message == "도어락소리" || message == "노크소리") {
    Serial.println("🔐 방문자 알림! (노란 LED 천천히)");
    blinkLED(LED_YELLOW, 500, BLINK_DURATION);
  } else if (message == "아기울음소리") {
    Serial.println("👶 아기 알림! (파란 LED 천천히)");
    blinkLED(LED_BLUE, 500, BLINK_DURATION);
  } else {
    Serial.print("⚠️ 알 수 없는 메시지: ");
    Serial.println(message);
  }
}

// ============================================================
// MQTT 연결 함수
// ============================================================
void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] 연결 시도... ");

    if (mqttClient.connect(MQTT_CLIENT_ID)) {
      Serial.println("✅ 연결 성공!");

      // 토픽 구독
      mqttClient.subscribe(MQTT_TOPIC);
      Serial.print("[MQTT] 구독 중: ");
      Serial.println(MQTT_TOPIC);
    } else {
      Serial.print("❌ 실패, 상태 코드: ");
      Serial.println(mqttClient.state());
      Serial.println("5초 후 재시도...");
      delay(5000);
    }
  }
}

// ============================================================
// 초기 설정
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=================================");
  Serial.println("  Hearo ESP32 시작!");
  Serial.println("=================================");

  // LED 핀 초기화
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);

  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_GREEN, LOW);

  // Wi-Fi 연결
  connectWiFi();

  // MQTT 설정
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);

  // MQTT 연결
  connectMQTT();

  Serial.println();
  Serial.println("✅ 모든 준비 완료! 알림 대기 중...");
}

// ============================================================
// 메인 루프
// ============================================================
void loop() {
  // MQTT 연결 유지
  if (!mqttClient.connected()) {
    Serial.println("[MQTT] 재연결 시도...");
    connectMQTT();
  }

  mqttClient.loop();  // MQTT 메시지 처리
  delay(10);
}
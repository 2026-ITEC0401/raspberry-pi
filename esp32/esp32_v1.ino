/*
=============================================================
Hearo ESP32 — Wi-Fi MQTT 수신 + LED 알림
=============================================================
업데이트:
- LED 매핑 변경: 도어락/노크 → 파랑, 아기울음 → 노랑
- Client ID 고유화 (충돌 방지)
=============================================================
*/

#include <WiFi.h>
#include <PubSubClient.h>

// ============================================================
// Wi-Fi 설정 (본인 정보)
// ============================================================
const char* WIFI_SSID = "원슥";
const char* WIFI_PASSWORD = "본인비밀번호";

// ============================================================
// MQTT 설정
// ============================================================
const char* MQTT_BROKER = "172.20.10.11";  // 라파이 IP
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "hearo/alert";
const char* MQTT_CLIENT_ID = "hearo-esp32-1";  // ★ 각 ESP32마다 1, 2, 3 다르게!

// ============================================================
// LED 핀 설정
// ============================================================
const int LED_RED    = 25;  // 비상벨
const int LED_YELLOW = 26;  // ★ 아기울음 (변경됨)
const int LED_BLUE   = 27;  // ★ 도어락/노크 (변경됨)
const int LED_GREEN  = 14;  // 예비

const int BLINK_DURATION = 5000;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ============================================================
// Wi-Fi 연결
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
// LED 깜빡임
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
// MQTT 메시지 수신 콜백
// ============================================================
void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.println();
  Serial.print("[MQTT] 수신: ");
  Serial.println(message);
  
  if (message == "비상벨소리") {
    Serial.println("🚨 비상벨 알림! (빨간 LED 빠르게)");
    blinkLED(LED_RED, 200, BLINK_DURATION);
  } 
  else if (message == "도어락소리" || message == "노크소리") {
    Serial.println("🔐 방문자 알림! (파란 LED 천천히)");  // ★ 메시지도 변경
    blinkLED(LED_BLUE, 500, BLINK_DURATION);  // ★ 파란 LED로 변경
  } 
  else if (message == "아기울음소리") {
    Serial.println("👶 아기 알림! (노란 LED 천천히)");  // ★ 메시지도 변경
    blinkLED(LED_YELLOW, 500, BLINK_DURATION);  // ★ 노란 LED로 변경
  } 
  else {
    Serial.print("⚠️ 알 수 없는 메시지: ");
    Serial.println(message);
  }
}

// ============================================================
// MQTT 연결
// ============================================================
void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("[MQTT] 연결 시도 (ID: ");
    Serial.print(MQTT_CLIENT_ID);
    Serial.print(")... ");
    
    if (mqttClient.connect(MQTT_CLIENT_ID)) {
      Serial.println("✅ 연결 성공!");
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
  Serial.print("  Client ID: ");
  Serial.println(MQTT_CLIENT_ID);
  Serial.println("=================================");
  
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_BLUE, LOW);
  digitalWrite(LED_GREEN, LOW);
  
  connectWiFi();
  
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(onMqttMessage);
  
  connectMQTT();
  
  Serial.println();
  Serial.println("✅ 모든 준비 완료! 알림 대기 중...");
}

// ============================================================
// 메인 루프
// ============================================================
void loop() {
  if (!mqttClient.connected()) {
    Serial.println("[MQTT] 재연결 시도...");
    connectMQTT();
  }
  
  mqttClient.loop();
  delay(10);
}

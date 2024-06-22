#define PIR_SENSOR_PIN 7
#define LED_PIN_13 13
#define LED_PIN_12 12
#define BUZZER_PIN 8

bool motionDetected = false;
unsigned long motionDetectedTime = 0;

void setup() {
  pinMode(PIR_SENSOR_PIN, INPUT);
  pinMode(LED_PIN_13, OUTPUT);
  pinMode(LED_PIN_12, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  Serial.begin(9600); // 시리얼 통신 시작
}

void loop() {
  if (!motionDetected && digitalRead(PIR_SENSOR_PIN) == HIGH) {
    // Motion detected for the first time
    motionDetected = true;
    motionDetectedTime = millis();
    blinkLED(LED_PIN_13);
    blinkLED(LED_PIN_12);
    playSound();
  }

  // Check if 5 seconds have passed since motion detection
  if (motionDetected && millis() - motionDetectedTime >= 5000) {
    // Send signal to Python to open webcam
    Serial.println("Open Webcam");
    motionDetected = false; // Reset motion detection flag
    delay(10000); // 10초 동안 재감지 방지
  }
}

void blinkLED(int pin) {
  // LED를 2초 동안 깜빡이는 함수
  for (int i = 0; i < 2; i++) {
    digitalWrite(pin, HIGH);
    delay(500);
    digitalWrite(pin, LOW);
    delay(500);
  }
}

void playSound() {
  // 도레미파 솔라시도 음성 출력
  int frequencies[] = {261, 293, 329, 349, 392, 440, 493}; // 도레미파솔라시도에 해당하는 주파수 배열
  int duration = 500; // 음성 출력 지속 시간 (밀리초)

  for (int i = 0; i < 7; i++) {
    tone(BUZZER_PIN, frequencies[i], duration); // 부저 모듈에서 주파수 출력
    delay(duration); // 음성 출력 지속 시간만큼 대기
  }
}

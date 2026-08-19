# Hearo Raspberry Pi

Hearo는 청각장애인이 가정 안의 비상벨, 도어락, 노크, 아기 울음 같은 중요한 소리를 시각 알림으로 확인할 수 있도록 돕는 프로젝트입니다. 이 저장소는 Raspberry Pi의 YAMNet 기반 추론, ESP32 로컬 오디오 수신, LED 알림, AWS MQTT 연동 및 모델 학습 자료를 관리합니다.

## 현재 구성

```text
Raspberry Pi INMP441 ─┐
ESP32 3대 INMP441 ────┼─(로컬 UDP PCM16)─> Raspberry Pi
                      │                       │
                      └───────────────────────┤ YAMNet 임베딩
                                              ├ Hearo v2 10-class 분류
                                              ├ 로컬 LED 알림
                                              └ MQTT TLS -> AWS -> DynamoDB/API
```

| 기기 ID | 설치 위치 | 주요 역할 |
|---|---|---|
| `rpi-001` | 거실 | 마이크 수집, YAMNet/v2 추론, LED, 클라우드 발행 |
| `esp32_1` | 안방 | INMP441 수집, Pi로 로컬 오디오 전송, LED |
| `esp32_2` | 현관 | INMP441 수집, Pi로 로컬 오디오 전송, LED |
| `esp32_3` | 화장실 | INMP441 수집, Pi로 로컬 오디오 전송, LED |

클라우드 MQTT 연결을 꺼도 Raspberry Pi의 로컬 추론과 거실 LED는 계속 동작합니다. ESP32 오디오는 클라우드가 아니라 같은 LAN의 Raspberry Pi로만 전송되며 파일로 저장하지 않습니다.

## 저장소 구조

```text
.
├── appAWS_v2.py                 # 배포 기준: Pi 마이크 v2 추론과 LED/MQTT
├── appAWS_v3.py                 # Pi + ESP32 오디오 허브, 선택적 하이브리드 정책
├── hearo_device_runtime.py      # HTTPS config polling, heartbeat, MQTT TLS 상태 동기화
├── hearo_audio_protocol.py      # ESP32-Pi PCM16 UDP 패킷/HMAC 계약
├── hearo_audio_receiver.py      # 기기별 2초 rolling buffer와 추론 큐
├── hearo_hybrid_classifier.py   # v2 fallback을 보존하는 선택적 YAMNet 규칙 결합
├── model/                       # YAMNet 및 Hearo TFLite 배포 파일
├── esp32/                       # ESP32 펌웨어
├── tests/                       # 로컬 오디오 및 하이브리드 의사결정 테스트
└── yamnet/
    ├── yamnet_fine_tuning_v2.ipynb
    ├── manifests/metadata.csv
    ├── results/v2/              # 검증된 수치와 재생성 가능한 그래프
    └── docs/
```

## YAMNet v2 모델

YAMNet 자체를 다시 학습하는 대신, 16 kHz 오디오에서 0.96초 창·0.48초 간격의 1024차원 임베딩을 추출하고 Hearo 전용 10-class 분류기를 학습합니다. 9개 표적 클래스와 `비표적음`을 함께 학습해 일상 배경음이 알림으로 전달되는 비율을 평가합니다.

사용자에게 전달되는 네 종류의 알림은 다음과 같습니다.

| 사용자 알림 | 세부 학습 클래스 |
|---|---|
| 비상벨소리 | 사이렌 4종 |
| 도어락소리 | 도어락 개방음·입력음 |
| 노크소리 | 목재·철재문 노크 |
| 아기울음소리 | 아기 울음 |
| 알림 없음 | 비표적음 또는 정책상 거부된 결과 |

최종 격리 test 결과는 표적 9개 macro-F1 **78.77%**, 비표적 false-alert rate **0.80%**, 전체 accuracy **97.37%**입니다. 전체 accuracy는 비표적음 377개가 포함된 불균형 자료의 영향을 크게 받으므로 표적 macro-F1과 함께 해석해야 합니다. 클래스별 원본 그룹 수가 적어 macro-F1의 95% 그룹 bootstrap 구간도 넓습니다. 자세한 결과는 [v2 평가 보고서](yamnet/docs/v2-evaluation.md)를 참고하십시오.

배포 파일은 dynamic-range quantized TFLite이며 304,784 bytes입니다. float32 모델과 같은 parity-set macro-F1을 유지하면서 약 74.4% 작고, Colab 측정 추론 시간도 약 0.061 ms로 더 짧았습니다. 이 시간은 Raspberry Pi 실측값이 아닙니다.

## Raspberry Pi 설치

권장 환경은 Raspberry Pi OS 64-bit와 Python 3.11 이상입니다.

```bash
git clone https://github.com/2026-ITEC0401/raspberry-pi.git
cd raspberry-pi
python3 -m venv hearo-env
source hearo-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rpi-v2.txt
```

TFLite 런타임은 Pi와 Python 버전에 맞는 `ai-edge-litert`, `tflite-runtime` 또는 TensorFlow 중 하나가 추가로 필요합니다. 설치 후 아래 파일이 존재하는지 확인합니다.

```text
model/yamnet.tflite
model/yamnet_classes.txt
model/hearo_classifier_v2.tflite
model/categories_v2.txt
model/model_metadata_v2.json
```

`.env.example`을 참고해 실제 값은 Git에 올리지 않는 별도 환경 파일에 저장합니다. 현재 프로그램은 환경 파일을 자동으로 읽지 않으므로 실행 전 셸 또는 systemd `EnvironmentFile`을 통해 변수를 주입해야 합니다.

```bash
cp .env.example .env
# .env의 placeholder를 실제 발급 값으로 수정
set -a
source .env
set +a
python appAWS_v2.py
```

마이크 자동 탐색이 실패하면 장치 목록을 확인하고 `HEARO_MIC_DEVICE_INDEX` 또는 `HEARO_MIC_DEVICE_NAME`을 지정합니다.

```bash
python -m sounddevice
export HEARO_MIC_DEVICE_INDEX=1
python appAWS_v2.py
```

`appAWS_v3.py`는 세 ESP32가 보내는 인증된 UDP 오디오를 함께 처리합니다. `HEARO_AUDIO_PSK`를 모든 노드에 동일하게 설정한 뒤 실행합니다.

```bash
python appAWS_v3.py
```

현재 `model/hybrid_policy_v3.json`은 임계값이 아직 선정되지 않아 `enabled=false`입니다. 따라서 v3를 실행해도 분류 결정은 안전하게 검증된 Hearo v2로 fallback합니다. 정책을 활성화하려면 별도 validation 자료로 모든 `null` 파라미터를 선택하고 회귀 테스트를 통과해야 합니다.

## 모델 재학습

Colab에서 [yamnet_fine_tuning_v2.ipynb](yamnet/yamnet_fine_tuning_v2.ipynb)를 열고 Google Drive의 `소리 정리` 폴더와 `metadata.csv`를 준비한 후 위에서 아래로 전체 실행합니다. 동일 원본의 잘라낸 파일·증강본은 같은 `group_id`를 사용해야 하며, 원본 그룹이 train/validation/test에 겹치면 안 됩니다. 자세한 절차와 산출물 계약은 [v2 학습 가이드](yamnet/README_V2.md)에 있습니다.

## 테스트

하드웨어를 사용하지 않는 모듈 테스트는 다음과 같이 실행합니다.

```bash
python -m pytest tests/test_audio_protocol.py \
  tests/test_audio_receiver.py \
  tests/test_hybrid_classifier.py
```

실기 배포 전에는 Pi와 ESP32가 같은 LAN에서 UDP 41000으로 통신하는지, 클래스 순서가 metadata와 같은지, LED GPIO와 MQTT/HTTPS 자격 증명이 올바른지 별도로 확인해야 합니다.

## 보안 주의사항

- Wi-Fi, MQTT, API 비밀번호와 `HEARO_AUDIO_PSK`를 소스 코드에 넣지 않습니다.
- 운영 MQTT는 평문 1883 대신 TLS 8883과 기기별 ACL을 사용합니다.
- UDP 41000은 EC2 보안 그룹에 열지 않고 Raspberry Pi의 로컬 LAN에서만 허용합니다.
- 이미 공개 저장소에 노출된 자격 증명은 파일에서 지우는 것만으로 충분하지 않으며 즉시 폐기·재발급해야 합니다.

## 문서

- [YAMNet v2 학습·실행 가이드](yamnet/README_V2.md)
- [YAMNet v2 평가 결과](yamnet/docs/v2-evaluation.md)
- [기존 YAMNet 개요](yamnet/docs/overview.md)
- [기존 파인튜닝 가이드](yamnet/docs/fine-tuning-guide.md)

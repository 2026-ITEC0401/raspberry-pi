# Hearo YAMNet v3.2 newdata Raspberry Pi 배포 안내서

## 1. 배포 원칙

- Raspberry Pi에서는 `yamnet_fine_tuning_colab_paper_v3_2.ipynb`를 실행하지 않는다.
- 코랩이 평가를 마친 뒤 생성한 최종 파일 `hearo_classifier_v3_2.tflite`만 배포한다.
- `_float32.tflite`, `_dynamic.tflite` 중 하나를 사람이 다시 선택하거나 최종 파일명으로 바꾸지 않는다.
- 기존 `appAWS_v2.py`, `appAWS_v3.py`와 v2 모델은 삭제하거나 이름을 바꾸지 않는다.
- v3.2는 별도 실행 파일과 별도 systemd 서비스로 전환한다.
- 실제 모델 검증이 성공하기 전에는 운영 서비스를 바꾸지 않는다.

## 2. 최종 파일 구조

다음 구조에서 파일명이 정확히 일치해야 한다.

```text
/home/team7/raspberry-pi/
├── appAWS_v2.py                         # 기존 v2, 보존
├── appAWS_v3.py                         # 기존 v2+ESP32 허브, 보존
├── appAWS_v3_2.py                       # 새 v3.2 실행 파일
├── hearo_model_runtime.py               # 새 모델 계약·추론 로더
├── hearo_hybrid_classifier.py           # 하위 호환 확장
├── hearo_audio_protocol.py              # 기존 파일, 보존
├── hearo_audio_receiver.py              # 기존 파일, 보존
├── hearo_device_runtime.py              # 기존 파일, 보존
├── requirements-rpi-v2.txt              # 기존 파일, 보존
├── requirements-rpi-v3-2.txt            # 새 의존성 목록
├── scripts/
│   └── validate_model_v3_2.py           # 마이크 없는 실모델 검사
└── model/
    ├── yamnet.tflite                    # 기존 YAMNet
    ├── yamnet_classes.txt               # 기존 521개 클래스
    ├── hearo_classifier_v3_2.tflite     # 코랩이 선택한 최종 파일
    ├── categories_v3_2.txt              # 코랩 산출물
    ├── model_metadata_v3_2.json          # schema_version=4 코랩 산출물
    └── hybrid_policy_v3.json             # 기존 하이브리드 정책
```

## 3. 배포 전에 Mac에서 확인

이번 변경에서 Pi로 복사하는 정확한 파일 목록은 다음 12개이다.

```text
appAWS_v3_2.py
hearo_model_runtime.py
hearo_hybrid_classifier.py
requirements-rpi-v3-2.txt
scripts/validate_model_v3_2.py
infra/systemd/hearo-rpi-v3-2.service
model/yamnet.tflite
model/yamnet_classes.txt
model/hearo_classifier_v3_2.tflite
model/categories_v3_2.txt
model/model_metadata_v3_2.json
model/hybrid_policy_v3.json
```

`hearo_audio_protocol.py`, `hearo_audio_receiver.py`, `hearo_device_runtime.py`, `requirements-rpi-v2.txt`, `appAWS_v2.py`, `appAWS_v3.py`는 기존 Pi 파일을 그대로 사용하며 복사하지 않는다. 단, 2절의 경로에 존재하는지 확인한다.

프로젝트 디렉터리로 이동한 뒤 필수 파일이 비어 있지 않은지 검사한다.

```bash
cd /Users/jang-wonseog/Desktop/Project

for file in \
  appAWS_v3_2.py \
  hearo_model_runtime.py \
  hearo_hybrid_classifier.py \
  requirements-rpi-v3-2.txt \
  scripts/validate_model_v3_2.py \
  infra/systemd/hearo-rpi-v3-2.service \
  model/yamnet.tflite \
  model/yamnet_classes.txt \
  model/hearo_classifier_v3_2.tflite \
  model/categories_v3_2.txt \
  model/model_metadata_v3_2.json \
  model/hybrid_policy_v3.json
do
  test -s "$file" || { echo "누락 또는 빈 파일: $file"; exit 1; }
done

echo "전송 전 필수 파일 확인 완료"
```

현재 작업본에는 v3.2 코랩 산출물과 기존 YAMNet 파일이 없으므로, 코랩 다운로드 파일을 `model/`에 넣기 전에는 이 검사가 실패하는 것이 정상이다.

선택된 모델 파일명과 schema를 별도로 확인한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path

metadata = json.loads(
    Path("model/model_metadata_v3_2.json").read_text(encoding="utf-8")
)
assert metadata["schema_version"] == 4
assert metadata["model_name"] == "hearo_classifier_v3_2"
assert metadata["selected_tflite_variant"] in {"float32", "dynamic_range"}
assert metadata["temperature_embedded_in_tflite"] is True
assert Path("model/hearo_classifier_v3_2.tflite").is_file()
print("selected_tflite_variant:", metadata["selected_tflite_variant"])
print("v3.2 metadata 기본 확인 완료")
PY
```

## 4. Pi에 임시 업로드

Pi 주소가 `team7.local`인 경우 다음과 같이 전송한다. 주소가 다르면 `team7.local`만 실제 주소로 바꾼다.

```bash
ssh team7@team7.local 'install -d -m 700 /home/team7/hearo-v3-2-upload/model /home/team7/hearo-v3-2-upload/scripts /home/team7/hearo-v3-2-upload/infra/systemd'

scp \
  appAWS_v3_2.py \
  hearo_model_runtime.py \
  hearo_hybrid_classifier.py \
  requirements-rpi-v3-2.txt \
  team7@team7.local:/home/team7/hearo-v3-2-upload/

scp scripts/validate_model_v3_2.py \
  team7@team7.local:/home/team7/hearo-v3-2-upload/scripts/

scp \
  model/yamnet.tflite \
  model/yamnet_classes.txt \
  model/hearo_classifier_v3_2.tflite \
  model/categories_v3_2.txt \
  model/model_metadata_v3_2.json \
  model/hybrid_policy_v3.json \
  team7@team7.local:/home/team7/hearo-v3-2-upload/model/

scp infra/systemd/hearo-rpi-v3-2.service \
  team7@team7.local:/home/team7/hearo-v3-2-upload/infra/systemd/
```

## 5. 기존 운영 파일 백업

다음 명령은 Raspberry Pi에서 실행한다. 아직 서비스를 중지하지 않는다.

```bash
cd /home/team7/raspberry-pi

DEPLOY_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="/home/team7/hearo-backups/${DEPLOY_STAMP}-before-v3-2"

install -d -m 700 "$BACKUP_DIR/project" "$BACKUP_DIR/systemd"

for file in \
  appAWS_v2.py \
  appAWS_v3.py \
  appAWS_v3_2.py \
  hearo_model_runtime.py \
  hearo_hybrid_classifier.py \
  requirements-rpi-v3-2.txt \
  scripts/validate_model_v3_2.py \
  infra/systemd/hearo-rpi-v3-2.service \
  model/yamnet.tflite \
  model/yamnet_classes.txt \
  model/hearo_classifier_v3_2.tflite \
  model/categories_v3_2.txt \
  model/model_metadata_v3_2.json \
  model/hybrid_policy_v3.json
do
  test -e "$file" && cp -a --parents "$file" "$BACKUP_DIR/project/"
done

sudo cp -a /etc/systemd/system/hearo-rpi.service \
  "$BACKUP_DIR/systemd/hearo-rpi.service"
if sudo test -e /etc/systemd/system/hearo-rpi-v3-2.service; then
  sudo cp -a /etc/systemd/system/hearo-rpi-v3-2.service \
    "$BACKUP_DIR/systemd/hearo-rpi-v3-2.service"
fi
sudo chown -R team7:team7 "$BACKUP_DIR"

printf '백업 위치: %s\n' "$BACKUP_DIR"
find "$BACKUP_DIR" -maxdepth 6 -type f -print
```

`백업 위치` 출력은 롤백 때 필요하므로 기록한다. `cp --parents`를 지원하지 않는 환경이면 파일별로 일반 `cp -a`를 사용하되 원본 경로가 구분되도록 하위 디렉터리를 만든다.

## 6. 파일 설치와 권한 설정

Pi에서 계속 실행한다.

```bash
cd /home/team7/raspberry-pi
UPLOAD_DIR=/home/team7/hearo-v3-2-upload

install -d -m 755 scripts model infra/systemd

install -m 644 "$UPLOAD_DIR/appAWS_v3_2.py" appAWS_v3_2.py
install -m 644 "$UPLOAD_DIR/hearo_model_runtime.py" hearo_model_runtime.py
install -m 644 "$UPLOAD_DIR/hearo_hybrid_classifier.py" hearo_hybrid_classifier.py
install -m 644 "$UPLOAD_DIR/requirements-rpi-v3-2.txt" requirements-rpi-v3-2.txt
install -m 755 "$UPLOAD_DIR/scripts/validate_model_v3_2.py" scripts/validate_model_v3_2.py

install -m 644 "$UPLOAD_DIR/model/yamnet.tflite" model/yamnet.tflite
install -m 644 "$UPLOAD_DIR/model/yamnet_classes.txt" model/yamnet_classes.txt
install -m 644 "$UPLOAD_DIR/model/hearo_classifier_v3_2.tflite" model/hearo_classifier_v3_2.tflite
install -m 644 "$UPLOAD_DIR/model/categories_v3_2.txt" model/categories_v3_2.txt
install -m 644 "$UPLOAD_DIR/model/model_metadata_v3_2.json" model/model_metadata_v3_2.json
install -m 644 "$UPLOAD_DIR/model/hybrid_policy_v3.json" model/hybrid_policy_v3.json

install -m 644 "$UPLOAD_DIR/infra/systemd/hearo-rpi-v3-2.service" \
  infra/systemd/hearo-rpi-v3-2.service

chown -R team7:team7 \
  appAWS_v3_2.py \
  hearo_model_runtime.py \
  hearo_hybrid_classifier.py \
  requirements-rpi-v3-2.txt \
  scripts/validate_model_v3_2.py \
  model \
  infra/systemd/hearo-rpi-v3-2.service
```

기기 인증정보가 있는 `/home/team7/.config/hearo/device-runtime.env`는 복사하거나 권한을 바꾸지 않는다.

## 7. 패키지와 코드 검사

기존 가상환경에 이미 `ai_edge_litert`가 있으면 재설치하지 않는다.

```bash
cd /home/team7/raspberry-pi
source /home/team7/hearo-env/bin/activate

python - <<'PY'
from ai_edge_litert.interpreter import Interpreter
import numpy
print("ai-edge-litert import OK")
print("numpy:", numpy.__version__)
PY
```

위 import가 실패할 때만 설치한다.

```bash
python -m pip install -r requirements-rpi-v3-2.txt
```

운영 서비스를 건드리기 전에 문법과 실모델을 검사한다.

```bash
python -m py_compile \
  appAWS_v3_2.py \
  hearo_model_runtime.py \
  hearo_hybrid_classifier.py \
  hearo_audio_protocol.py \
  hearo_audio_receiver.py \
  hearo_device_runtime.py \
  scripts/validate_model_v3_2.py

python scripts/validate_model_v3_2.py \
  --model-dir /home/team7/raspberry-pi/model \
  --profile v3_2 \
  --repeat-count 3
```

마지막 줄이 `HEARO V3.2 MODEL VALIDATION OK`가 아니면 서비스를 전환하지 않는다. 이 검사는 다음을 포함한다.

- 필수 파일과 UTF-8 한글 읽기
- metadata schema 4와 클래스 순서
- YAMNet `[frames, 521]` score 및 `[frames, 1024]` embedding 식별
- 분류기 `float32 [N, 1024]` 입력과 출력 클래스 수
- 최종 확률의 NaN·infinity·범위·합계
- 동일 embedding 반복 추론 일관성
- `unknown_label` 경고 억제
- 하이브리드 정책 로딩

## 8. 짧은 수동 실행 검사

마이크와 UDP 41000 포트는 기존 서비스와 동시에 사용할 수 없으므로 이 단계에서만 기존 서비스를 잠시 중지한다.

```bash
sudo systemctl stop hearo-rpi

cd /home/team7/raspberry-pi
set -a
source /home/team7/.config/hearo/device-runtime.env
set +a

export HEARO_MODEL_PROFILE=v3_2
/home/team7/hearo-env/bin/python -u appAWS_v3_2.py
```

정상 시작 로그에는 다음 내용이 포함되어야 한다.

```text
Hearo 환경음 인식 v3_2 시작
모델: hearo_classifier_v3_2
TFLite: hearo_classifier_v3_2.tflite (...)
temperature embedded=True
[MQTT] TLS 연결 완료
```

소리 입력과 ESP32 오디오까지 확인한 뒤 `Ctrl+C`로 종료한다. systemd 전환을 바로 하지 않을 경우 기존 서비스를 복구한다.

```bash
sudo systemctl start hearo-rpi
sudo systemctl is-active hearo-rpi
```

## 9. systemd 등록 및 전환

별도 서비스로 등록하므로 기존 unit은 그대로 남는다.

```bash
sudo install \
  -o root \
  -g root \
  -m 644 \
  /home/team7/raspberry-pi/infra/systemd/hearo-rpi-v3-2.service \
  /etc/systemd/system/hearo-rpi-v3-2.service

sudo systemd-analyze verify /etc/systemd/system/hearo-rpi-v3-2.service
sudo systemctl daemon-reload

sudo systemctl disable --now hearo-rpi
sudo systemctl enable --now hearo-rpi-v3-2

sudo systemctl is-enabled hearo-rpi-v3-2
sudo systemctl is-active hearo-rpi-v3-2
sudo systemctl status hearo-rpi-v3-2 --no-pager -l
```

새 서비스가 `active`가 아니면 즉시 아래 롤백 절차를 실행한다.

## 10. 로그와 실제 연동 확인

실시간 Pi 로그:

```bash
sudo journalctl -u hearo-rpi-v3-2 -f -o cat
```

최근 시작·오류 로그:

```bash
sudo journalctl \
  -u hearo-rpi-v3-2 \
  --since "10 minutes ago" \
  --no-pager \
  | grep -E '초기화|모델:|TFLite:|temperature|MQTT|판정|감지|실패|오류|Traceback'
```

실행 경로와 환경 프로필:

```bash
sudo systemctl show hearo-rpi-v3-2 \
  -p MainPID \
  -p WorkingDirectory \
  -p Environment \
  -p ExecStart \
  --no-pager
```

Pi 로그의 `[MQTT] ...`만으로 서버 저장 성공을 단정하지 않는다. EC2에서 MQTT Bridge의 `/internal/mqtt/alert` 응답이 `200`인지와 DynamoDB의 최신 알림 `model_version=hearo_classifier_v3_2`를 함께 확인한다.

## 11. v2 롤백

별도 서비스 방식의 기본 롤백:

```bash
sudo systemctl disable --now hearo-rpi-v3-2
sudo systemctl enable --now hearo-rpi

sudo systemctl is-active hearo-rpi
sudo systemctl status hearo-rpi --no-pager -l
sudo journalctl -u hearo-rpi -n 100 --no-pager
```

기존 `hearo-rpi.service`의 `ExecStart`가 `/home/team7/raspberry-pi/appAWS_v3.py`인지 확인한다.

```bash
sudo systemctl show hearo-rpi -p ExecStart -p Environment --no-pager
```

기본 롤백 후에도 문제가 있을 때만 5단계에서 기록한 백업 위치의 `hearo_hybrid_classifier.py`와 기존 systemd unit을 복원한다. 백업 경로를 눈으로 확인한 뒤 다음의 `/정확한/백업/경로`를 실제 값으로 바꾼다.

```bash
sudo systemctl stop hearo-rpi hearo-rpi-v3-2

cp -a \
  /정확한/백업/경로/project/hearo_hybrid_classifier.py \
  /home/team7/raspberry-pi/hearo_hybrid_classifier.py

sudo cp -a \
  /정확한/백업/경로/systemd/hearo-rpi.service \
  /etc/systemd/system/hearo-rpi.service

sudo systemctl daemon-reload
sudo systemctl enable --now hearo-rpi
```

v3.2 파일과 v2 파일은 롤백 과정에서도 삭제하지 않는다.

## 12. 배포 후 남는 검증 항목

다음 항목은 개발 Mac의 모의 interpreter 테스트만으로 성공을 단정할 수 없다.

- 코랩에서 실제 생성된 세 v3.2 산출물의 TFLite 입출력 및 metadata 일치
- Raspberry Pi OS와 현재 Python 버전에서 `ai-edge-litert` wheel 호환성
- Pi 4에서 v3.2 지연시간, CPU, 메모리 및 발열
- 실제 INMP441 입력과 세 ESP32 동시 오디오의 장시간 안정성
- v3.2 클래스별 임계값의 실제 생활 오탐·미탐 성능
- MQTT, API, DynamoDB, WebSocket, 프론트까지의 종단 연동
- 현재 `hybrid_policy_v3.json`은 정책값이 미선정된 비활성 상태이므로, v3.2 분류기 기본 판정은 동작하지만 YAMNet 기반 hybrid override는 활성화되지 않음

이 항목들은 실제 파일을 배치한 뒤 7~10단계 결과로 별도 확인해야 한다.

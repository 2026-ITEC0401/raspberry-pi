# Hearo YAMNet fine-tuning v2

`yamnet_fine_tuning_v2.ipynb`는 한국 가정환경음 9개 표적 클래스와 `비표적음`을 분류하는 TFLite 모델을 한 번의 Colab 전체 실행으로 학습·선택·검증합니다.

## v1과 달라진 핵심

- 단순 파일 80:20 분할 대신 원본 단위 grouped hold-out와 `StratifiedGroupKFold`를 사용합니다.
- 같은 원본에서 잘라낸 조각, YAMNet 프레임, 증강본은 같은 `group_id`를 상속합니다.
- 9개 표적 macro-F1과 비표적 false-alert rate를 분리해 평가합니다.
- 파일 평균 MLP, 프레임 선형 head, compact MLP와 여러 pooling을 같은 split에서 비교합니다.
- 후보가 incumbent보다 macro-F1 0.005 이상 높고 false-alert rate 5% 이하일 때만 승격합니다.
- 최종 test는 모델과 threshold 선택이 끝난 뒤 한 번만 사용합니다.
- float32와 dynamic-range TFLite의 정확도·크기·추론 시간을 비교합니다.

## 필요한 Drive 구조

기본 경로는 `/content/drive/MyDrive/소리 정리`입니다.

```text
소리 정리/
├── metadata.csv
├── 노크_목재/
├── 노크_철재문/
├── 도어락_개방음/
├── 도어락_입력음/
├── 사이렌_삐뽀삐뽀/
├── 사이렌_안내음/
├── 사이렌_애애애애앵/
├── 사이렌_철철철/
├── 아기 울음/
└── 비표적음/
```

`metadata.csv`의 필수 열은 다음과 같습니다.

| 열 | 의미 | 예시 |
|---|---|---|
| `relative_path` | `소리 정리` 기준 상대 경로 | `노크_목재/wood_01.wav` |
| `label` | 폴더와 일치하는 10개 클래스 중 하나 | `노크_목재` |
| `group_id` | 동일 원본을 묶는 ID | `wood-original-01` |

저장소의 [metadata.csv](manifests/metadata.csv)는 실제 실행에 사용한 manifest의 스냅샷입니다. 오디오 원본은 용량과 라이선스 때문에 저장소에 포함하지 않습니다.

## 실행 순서

1. Google Colab에서 노트북을 엽니다.
2. 런타임을 GPU로 설정하고 Google Drive를 마운트합니다.
3. `DATA_DIR`, `OUTPUT_DIR`, seed를 확인합니다.
4. `런타임 > 모두 실행`으로 데이터 검증부터 TFLite parity 검사까지 실행합니다.
5. 오류가 발생하면 해당 셀만 건너뛰지 말고 metadata 또는 음원 문제를 수정한 후 처음부터 다시 실행합니다.
6. `/content/drive/MyDrive/Hearo_model_v2` 산출물을 검토합니다.

노트북은 누락·미등록 파일, 손상 음원, 잘못된 label, 중복 경로, 동일 음원의 클래스 간 중복을 초기에 검사합니다. 클래스별 독립 `group_id`가 너무 적어 최소 3-fold를 만들 수 없으면 의도적으로 중단합니다.

## 주요 산출물

| 파일 | 용도 |
|---|---|
| `hearo_classifier_v2.tflite` | Pi가 YAMNet 임베딩 `[N,1024]`을 분류 |
| `categories_v2.txt` | 출력 10-class 순서의 단일 기준 |
| `model_metadata_v2.json` | pooling, context, threshold, class mapping 계약 |
| `experiment_results.csv` | round/candidate별 development 결과 |
| `metrics.json` | 최종 development/test/quantization 지표 |
| `test_per_class_metrics.csv` | 격리 test 클래스별 precision/recall/F1 |

배포할 때는 classifier만 교체하지 말고 `categories_v2.txt`와 `model_metadata_v2.json`을 항상 같은 실행 결과에서 함께 복사해야 합니다.

## 현재 결과 해석

현재 실행은 1라운드의 파일 평균 임베딩 MLP를 최종 구성으로 유지했습니다. 프레임 head 후보들이 promotion 기준을 넘지 못해 설계된 중단 조건에 따라 2·3라운드는 실행하지 않았습니다. 이는 코드 실패가 아니라 과적합 가능성이 있는 후보를 자동으로 승격하지 않은 결과입니다.

test 표적 macro-F1은 0.7877이고 비표적 false-alert rate는 0.0080입니다. 클래스별 표적 support가 1~16개로 작기 때문에 단일 실행 수치를 확정적 일반화 성능으로 보지 말고, 실제 집 환경과 새 원본 그룹을 추가한 뒤 같은 grouped 평가를 반복해야 합니다.

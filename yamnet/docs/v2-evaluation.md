# Hearo YAMNet v2 평가 보고서

## 평가 원칙

모델 선택은 development 데이터의 grouped out-of-fold 예측만 사용했고, 약 20%의 test 그룹은 선택이 끝날 때까지 격리했습니다. 평가는 사용자 알림 대상인 9개 클래스의 macro-F1과 `비표적음`이 표적 알림으로 통과한 false-alert rate를 핵심 지표로 사용합니다.

## 최종 결과

| 지표 | Development OOF | 격리 test |
|---|---:|---:|
| 표적 9개 macro-F1 | 0.7348 | 0.7877 |
| 전체 10개 macro-F1 | 0.7596 | 0.8080 |
| Accuracy | 0.9601 | 0.9737 |
| 비표적 false-alert rate | 0.0067 | 0.0080 |
| Expected calibration error | 0.0128 | 0.0168 |

격리 test의 그룹 bootstrap 95% 신뢰구간은 표적 macro-F1 0.3334~0.8763, false-alert rate 0~0.0215, accuracy 0.9523~0.9909입니다. 표적 원본 그룹 수가 적어 macro-F1 구간이 넓으므로 더 다양한 집·거리·마이크 조건의 독립 원본이 필요합니다.

## 후보 비교와 자동 중단

| 후보 | 가장 좋은 pooling | Development target macro-F1 | 평균 grouped-CV target macro-F1 | 승격 |
|---|---|---:|---:|---|
| 파일 평균 임베딩 + 256/128 MLP | mean probability | 0.7348 | 0.7136 | 기준선 유지 |
| 프레임별 linear head | top-k probability | 0.6677 | 0.6364 | 아니요 |
| 프레임별 compact MLP | mean/log-mean-exp | 0.6938 | 0.6743 | 아니요 |

프레임 후보는 false-alert rate를 더 낮췄지만 표적 macro-F1이 기준선보다 낮았습니다. 최고 후보가 incumbent보다 0.005 이상 좋아야 한다는 규칙을 충족하지 못해 1라운드에서 종료했고, 증강·class weight·운영 threshold를 탐색하는 후속 라운드는 실행하지 않았습니다.

따라서 이번 개선의 핵심은 무조건 복잡한 모델로 교체한 것이 아니라 다음과 같습니다.

- 동일 원본 누수를 막는 grouped 평가로 성능 추정의 신뢰성을 높였습니다.
- `비표적음`을 명시적으로 학습하고 false-alert 5% 제한을 검증했습니다.
- 후보가 실제로 좋아지지 않으면 기존 표현을 유지하는 자동 승격·중단 규칙을 적용했습니다.
- 배포 계약, 클래스 순서, Keras/TFLite 오차 및 quantization을 함께 검증했습니다.

## 클래스별 test 결과

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 노크_목재 | 0.333 | 0.333 | 0.333 | 3 |
| 노크_철재문 | 0.500 | 1.000 | 0.667 | 1 |
| 도어락_개방음 | 1.000 | 0.667 | 0.800 | 3 |
| 도어락_입력음 | 1.000 | 0.667 | 0.800 | 3 |
| 사이렌_삐뽀삐뽀 | 1.000 | 0.667 | 0.800 | 9 |
| 사이렌_안내음 | 1.000 | 1.000 | 1.000 | 1 |
| 사이렌_애애애애앵 | 0.800 | 1.000 | 0.889 | 16 |
| 사이렌_철철철 | 1.000 | 1.000 | 1.000 | 2 |
| 아기 울음 | 1.000 | 0.667 | 0.800 | 3 |
| 비표적음 | 0.989 | 0.992 | 0.991 | 377 |

가장 먼저 보강할 대상은 F1 0.333인 `노크_목재`입니다. F1이 1.0이어도 support가 1~2개인 `사이렌_안내음`, `사이렌_철철철`은 충분히 검증됐다고 볼 수 없습니다. 클래스별 최소 독립 원본 그룹을 늘린 뒤 같은 hold-out/CV seed 정책으로 다시 측정해야 합니다.

## TFLite 배포 검증

| 항목 | Float32 | Dynamic-range |
|---|---:|---:|
| 파일 크기 | 1,188,568 bytes | 304,784 bytes |
| Parity-set target macro-F1 | 0.9683 | 0.9683 |
| Parity-set false-alert rate | 0 | 0 |
| Colab classifier latency | 0.118 ms | 0.061 ms |

dynamic-range 모델은 float32 대비 약 74.4% 작고, Keras와의 최대 확률 절대오차는 0.00623이었습니다. 정확도 저하 기준 0.005 이내를 만족해 `hearo_classifier_v2.tflite` 배포본으로 선택했습니다. latency는 classifier 1-frame에 대한 Colab 측정이므로 Raspberry Pi의 전체 YAMNet + classifier 종단 지연과는 다릅니다.

## 그래프 재생성

`yamnet/results/v2/render_figures.py`는 저장된 JSON/CSV만 읽어 한글 폰트를 적용한 결과 그림을 재생성합니다.

```bash
python yamnet/results/v2/render_figures.py
```

원본 수치는 [metrics.json](../results/v2/metrics.json), [experiment_results.csv](../results/v2/experiment_results.csv), [test_per_class_metrics.csv](../results/v2/test_per_class_metrics.csv)에 있습니다.

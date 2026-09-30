"""Versioned Hearo/YAMNet TFLite loading and inference contracts.

This module deliberately has no microphone, GPIO, MQTT, or Raspberry Pi
imports.  It can therefore validate deployment artifacts on a development
machine and can be unit-tested with fake TFLite interpreters.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


YAMNET_CLASS_COUNT = 521
YAMNET_EMBEDDING_SIZE = 1024
YAMNET_SAMPLE_RATE = 16_000

SUPPORTED_REPRESENTATIONS = {"frame", "clip_mean"}
SUPPORTED_FRAME_POOLING = {
    "mean_probability",
    "topk_probability",
    "logit_logmeanexp",
}
SUPPORTED_TFLITE_VARIANTS = {"float32", "dynamic_range"}
ALERT_SOUNDS = {"노크소리", "도어락소리", "비상벨소리", "아기울음소리"}


class ModelContractError(RuntimeError):
    """Raised when deployment files do not satisfy the runtime contract."""


@dataclass(frozen=True)
class ModelFileSet:
    profile: str
    schema_version: int
    classifier_filename: str
    categories_filename: str
    metadata_filename: str


MODEL_FILE_SETS: dict[str, ModelFileSet] = {
    "v2": ModelFileSet(
        profile="v2",
        schema_version=2,
        classifier_filename="hearo_classifier_v2.tflite",
        categories_filename="categories_v2.txt",
        metadata_filename="model_metadata_v2.json",
    ),
    "v3_2": ModelFileSet(
        profile="v3_2",
        schema_version=4,
        classifier_filename="hearo_classifier_v3_2.tflite",
        categories_filename="categories_v3_2.txt",
        metadata_filename="model_metadata_v3_2.json",
    ),
}
MODEL_PROFILE_ALIASES = {
    "v2": "v2",
    "2": "v2",
    "v3_2": "v3_2",
    "v3.2": "v3_2",
    "3_2": "v3_2",
    "3.2": "v3_2",
}


@dataclass(frozen=True)
class ModelContract:
    profile: str
    model_dir: Path
    yamnet_model_path: Path
    classifier_model_path: Path
    categories_path: Path
    metadata_path: Path
    yamnet_classes_path: Path
    metadata: Mapping[str, Any]
    categories: tuple[str, ...]
    yamnet_classes: tuple[str, ...]
    unknown_label: str
    representation: str
    frame_pooling: str
    context_frames: str | int
    class_thresholds: Mapping[str, float]
    class_mapping: Mapping[str, str]
    delivery_policy: Mapping[str, Mapping[str, Any]]
    record_step_seconds: float
    rolling_buffer_seconds: float
    sample_rate: int
    temperature_embedded_in_tflite: bool
    selected_tflite_variant: str | None

    @property
    def unknown_index(self) -> int:
        return self.categories.index(self.unknown_label)

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    @property
    def buffer_chunks(self) -> int:
        return max(1, int(math.ceil(self.rolling_buffer_seconds / self.record_step_seconds)))


def normalize_model_profile(profile: str | None) -> str:
    raw = (profile or os.getenv("HEARO_MODEL_PROFILE", "v2")).strip().casefold()
    normalized = MODEL_PROFILE_ALIASES.get(raw)
    if normalized is None:
        raise ModelContractError(
            f"지원하지 않는 HEARO_MODEL_PROFILE={profile!r}; "
            f"허용값={sorted(MODEL_FILE_SETS)}"
        )
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"UTF-8 JSON을 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelContractError(f"metadata 최상위 값은 JSON object여야 합니다: {path}")
    return value


def _read_lines(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ModelContractError(f"UTF-8 텍스트를 읽을 수 없습니다: {path}: {exc}") from exc
    values = [line.strip() for line in raw.splitlines() if line.strip()]
    if not values:
        raise ModelContractError(f"파일에 값이 없습니다: {path}")
    return values


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ModelContractError("필수 모델 파일이 없습니다:\n- " + "\n- ".join(missing))
    empty = [str(path) for path in paths if path.stat().st_size <= 0]
    if empty:
        raise ModelContractError("비어 있는 모델 파일이 있습니다:\n- " + "\n- ".join(empty))


def _require_number(
    metadata: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelContractError(f"metadata.{key}는 숫자여야 합니다.")
    value = float(value)
    if not math.isfinite(value):
        raise ModelContractError(f"metadata.{key}는 유한한 숫자여야 합니다.")
    if minimum is not None and value < minimum:
        raise ModelContractError(f"metadata.{key}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and value > maximum:
        raise ModelContractError(f"metadata.{key}는 {maximum} 이하여야 합니다.")
    return value


def _validate_shape_metadata(
    metadata: Mapping[str, Any],
    key: str,
    expected_last_dimension: int,
) -> None:
    contract = metadata.get(key)
    if not isinstance(contract, Mapping):
        raise ModelContractError(f"metadata.{key} object가 필요합니다.")
    if contract.get("dtype") != "float32":
        raise ModelContractError(f"metadata.{key}.dtype은 float32여야 합니다.")
    shape = contract.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] is not None
        or shape[1] != expected_last_dimension
    ):
        raise ModelContractError(
            f"metadata.{key}.shape은 [null, {expected_last_dimension}]이어야 합니다: {shape}"
        )


def _validate_metadata(
    metadata: Mapping[str, Any],
    categories: Sequence[str],
    files: ModelFileSet,
) -> None:
    if metadata.get("schema_version") != files.schema_version:
        raise ModelContractError(
            f"{files.profile} metadata schema_version은 {files.schema_version}이어야 합니다: "
            f"{metadata.get('schema_version')!r}"
        )
    model_name = metadata.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelContractError("metadata.model_name이 필요합니다.")
    if files.profile == "v3_2" and model_name != "hearo_classifier_v3_2":
        raise ModelContractError(
            "v3.2 metadata.model_name은 hearo_classifier_v3_2여야 합니다."
        )

    metadata_categories = metadata.get("categories")
    if metadata_categories != list(categories):
        raise ModelContractError(
            f"{files.categories_filename}와 {files.metadata_filename}의 클래스 수 또는 순서가 다릅니다."
        )
    if len(categories) != len(set(categories)):
        raise ModelContractError("categories에 중복 클래스가 있습니다.")
    if any(not isinstance(label, str) or not label for label in categories):
        raise ModelContractError("categories에는 비어 있지 않은 문자열만 사용할 수 있습니다.")

    unknown_label = metadata.get("unknown_label")
    if unknown_label not in categories:
        raise ModelContractError("metadata.unknown_label이 categories에 없습니다.")

    sample_rate = metadata.get("sample_rate")
    if isinstance(sample_rate, bool) or sample_rate != YAMNET_SAMPLE_RATE:
        raise ModelContractError(
            f"metadata.sample_rate는 YAMNet 계약값 {YAMNET_SAMPLE_RATE}이어야 합니다."
        )
    representation = metadata.get("representation")
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ModelContractError(f"지원하지 않는 representation입니다: {representation!r}")
    pooling = metadata.get("frame_pooling")
    if pooling not in SUPPORTED_FRAME_POOLING:
        raise ModelContractError(f"지원하지 않는 frame_pooling입니다: {pooling!r}")
    context_frames = metadata.get("context_frames")
    if context_frames != "full" and (
        isinstance(context_frames, bool)
        or not isinstance(context_frames, int)
        or context_frames < 1
    ):
        raise ModelContractError("context_frames는 'full' 또는 1 이상의 정수여야 합니다.")

    record_step = _require_number(metadata, "record_step_seconds", minimum=0.1)
    rolling = _require_number(metadata, "rolling_buffer_seconds", minimum=record_step)
    if rolling < record_step:
        raise ModelContractError("rolling_buffer_seconds는 record_step_seconds 이상이어야 합니다.")

    thresholds = metadata.get("class_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ModelContractError("metadata.class_thresholds object가 필요합니다.")
    missing_thresholds = [label for label in categories if label not in thresholds]
    extra_thresholds = [label for label in thresholds if label not in categories]
    if missing_thresholds or extra_thresholds:
        raise ModelContractError(
            "class_thresholds 키가 categories와 일치하지 않습니다: "
            f"missing={missing_thresholds}, extra={extra_thresholds}"
        )
    for label in categories:
        value = thresholds[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelContractError(f"{label!r} class threshold는 숫자여야 합니다.")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ModelContractError(f"{label!r} class threshold는 0~1 범위여야 합니다.")

    class_mapping = metadata.get("class_mapping")
    if not isinstance(class_mapping, Mapping):
        raise ModelContractError("metadata.class_mapping object가 필요합니다.")
    expected_mapping_labels = set(categories) - {str(unknown_label)}
    if set(class_mapping) != expected_mapping_labels:
        raise ModelContractError(
            "class_mapping은 unknown을 제외한 모든 categories와 정확히 일치해야 합니다: "
            f"expected={sorted(expected_mapping_labels)}, actual={sorted(class_mapping)}"
        )
    invalid_sounds = {
        label: sound for label, sound in class_mapping.items() if sound not in ALERT_SOUNDS
    }
    if invalid_sounds:
        raise ModelContractError(f"지원하지 않는 사용자 알림 매핑이 있습니다: {invalid_sounds}")

    if files.profile == "v3_2":
        _validate_shape_metadata(metadata, "classifier_input", YAMNET_EMBEDDING_SIZE)
        _validate_shape_metadata(metadata, "classifier_output", len(categories))
        output = metadata["classifier_output"]
        if output.get("type") != "probabilities":
            raise ModelContractError(
                "v3.2 classifier_output.type은 probabilities여야 합니다."
            )
        if metadata.get("temperature_embedded_in_tflite") is not True:
            raise ModelContractError(
                "v3.2는 temperature_embedded_in_tflite=true여야 합니다."
            )
        temperature = _require_number(metadata, "temperature", minimum=1e-6)
        if temperature <= 0:
            raise ModelContractError("metadata.temperature는 양수여야 합니다.")
        variant = metadata.get("selected_tflite_variant")
        if variant not in SUPPORTED_TFLITE_VARIANTS:
            raise ModelContractError(
                "selected_tflite_variant는 float32 또는 dynamic_range여야 합니다."
            )


def load_model_contract(model_dir: Path | str, profile: str | None = None) -> ModelContract:
    normalized = normalize_model_profile(profile)
    files = MODEL_FILE_SETS[normalized]
    directory = Path(model_dir).expanduser().resolve()
    yamnet_path = directory / "yamnet.tflite"
    classifier_path = directory / files.classifier_filename
    categories_path = directory / files.categories_filename
    metadata_path = directory / files.metadata_filename
    yamnet_classes_path = directory / "yamnet_classes.txt"
    _require_files(
        [
            yamnet_path,
            classifier_path,
            categories_path,
            metadata_path,
            yamnet_classes_path,
        ]
    )

    categories = _read_lines(categories_path)
    yamnet_classes = _read_lines(yamnet_classes_path)
    metadata = _read_json(metadata_path)
    _validate_metadata(metadata, categories, files)
    if len(yamnet_classes) != YAMNET_CLASS_COUNT:
        raise ModelContractError(
            f"yamnet_classes.txt는 {YAMNET_CLASS_COUNT}개여야 합니다: {len(yamnet_classes)}"
        )

    thresholds = {
        label: float(metadata["class_thresholds"][label]) for label in categories
    }
    delivery_policy = metadata.get("delivery_policy", {})
    if not isinstance(delivery_policy, Mapping):
        raise ModelContractError("metadata.delivery_policy는 JSON object여야 합니다.")

    return ModelContract(
        profile=normalized,
        model_dir=directory,
        yamnet_model_path=yamnet_path,
        classifier_model_path=classifier_path,
        categories_path=categories_path,
        metadata_path=metadata_path,
        yamnet_classes_path=yamnet_classes_path,
        metadata=dict(metadata),
        categories=tuple(categories),
        yamnet_classes=tuple(yamnet_classes),
        unknown_label=str(metadata["unknown_label"]),
        representation=str(metadata["representation"]),
        frame_pooling=str(metadata["frame_pooling"]),
        context_frames=metadata["context_frames"],
        class_thresholds=thresholds,
        class_mapping=dict(metadata["class_mapping"]),
        delivery_policy=dict(delivery_policy),
        record_step_seconds=float(metadata["record_step_seconds"]),
        rolling_buffer_seconds=float(metadata["rolling_buffer_seconds"]),
        sample_rate=int(metadata["sample_rate"]),
        temperature_embedded_in_tflite=bool(
            metadata.get("temperature_embedded_in_tflite", False)
        ),
        selected_tflite_variant=metadata.get("selected_tflite_variant"),
    )


def resolve_interpreter() -> tuple[type[Any], str]:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter, "ai-edge-litert"
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter, "tflite-runtime"
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter

                return Interpreter, "tensorflow-lite"
            except ImportError as exc:
                raise ModelContractError(
                    "TFLite Interpreter가 없습니다. ai-edge-litert를 설치하십시오."
                ) from exc


def _shape(detail: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(detail["shape"]).tolist())


def _shape_signature(detail: Mapping[str, Any]) -> tuple[int, ...]:
    value = detail.get("shape_signature", detail["shape"])
    return tuple(int(item) for item in np.asarray(value).tolist())


def _is_matrix_with_width(detail: Mapping[str, Any], width: int) -> bool:
    shape = _shape_signature(detail)
    return len(shape) == 2 and shape[-1] == width


def _quantize_for_input(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale <= 0:
            raise ModelContractError("정수 TFLite 입력 quantization scale이 유효하지 않습니다.")
        info = np.iinfo(dtype)
        values = np.clip(np.round(values / scale + zero_point), info.min, info.max)
    return values.astype(dtype)


def _dequantize_output(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale <= 0:
            raise ModelContractError("정수 TFLite 출력 quantization scale이 유효하지 않습니다.")
        return (values.astype(np.float32) - zero_point) * scale
    return values.astype(np.float32)


class HearoModelRuntime:
    """YAMNet plus a versioned Hearo classifier."""

    def __init__(
        self,
        contract: ModelContract,
        *,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.contract = contract
        if interpreter_factory is None:
            interpreter_factory, runtime_name = resolve_interpreter()
        else:
            runtime_name = getattr(interpreter_factory, "__name__", "injected-interpreter")
        self.interpreter_runtime_name = runtime_name
        try:
            self.yamnet_interpreter = interpreter_factory(
                model_path=str(contract.yamnet_model_path)
            )
            self.yamnet_interpreter.allocate_tensors()
            self.classifier_interpreter = interpreter_factory(
                model_path=str(contract.classifier_model_path)
            )
            self.classifier_interpreter.allocate_tensors()
        except Exception as exc:
            raise ModelContractError(f"TFLite 모델 초기화 실패: {exc}") from exc
        self._validate_interpreter_contracts()

    @classmethod
    def load(
        cls,
        model_dir: Path | str,
        profile: str | None = None,
        *,
        interpreter_factory: Callable[..., Any] | None = None,
    ) -> "HearoModelRuntime":
        return cls(
            load_model_contract(model_dir, profile),
            interpreter_factory=interpreter_factory,
        )

    def _validate_interpreter_contracts(self) -> None:
        yamnet_inputs = self.yamnet_interpreter.get_input_details()
        yamnet_outputs = self.yamnet_interpreter.get_output_details()
        if len(yamnet_inputs) != 1:
            raise ModelContractError(
                f"YAMNet 입력 tensor는 1개여야 합니다: {len(yamnet_inputs)}"
            )
        yamnet_input = yamnet_inputs[0]
        if len(_shape_signature(yamnet_input)) != 1:
            raise ModelContractError(
                f"YAMNet 입력은 원시 파형 [samples]이어야 합니다: {_shape_signature(yamnet_input)}"
            )
        if np.dtype(yamnet_input["dtype"]) != np.dtype(np.float32):
            raise ModelContractError("YAMNet 입력 dtype은 float32여야 합니다.")

        score_outputs = [
            detail
            for detail in yamnet_outputs
            if _is_matrix_with_width(detail, YAMNET_CLASS_COUNT)
        ]
        embedding_outputs = [
            detail
            for detail in yamnet_outputs
            if _is_matrix_with_width(detail, YAMNET_EMBEDDING_SIZE)
        ]
        if len(score_outputs) != 1 or len(embedding_outputs) != 1:
            raise ModelContractError(
                "YAMNet 출력에서 [frames, 521] score와 [frames, 1024] embedding을 "
                f"각각 정확히 하나씩 찾아야 합니다: {[ _shape_signature(item) for item in yamnet_outputs ]}"
            )

        classifier_inputs = self.classifier_interpreter.get_input_details()
        classifier_outputs = self.classifier_interpreter.get_output_details()
        if len(classifier_inputs) != 1 or len(classifier_outputs) != 1:
            raise ModelContractError("Hearo 분류기의 입력과 출력 tensor는 각각 1개여야 합니다.")
        classifier_input = classifier_inputs[0]
        classifier_output = classifier_outputs[0]
        input_shape = _shape_signature(classifier_input)
        output_shape = _shape_signature(classifier_output)
        if len(input_shape) != 2 or input_shape[-1] != YAMNET_EMBEDDING_SIZE:
            raise ModelContractError(
                f"Hearo 분류기 입력은 [N, {YAMNET_EMBEDDING_SIZE}]여야 합니다: {input_shape}"
            )
        if len(output_shape) != 2 or output_shape[-1] != len(self.contract.categories):
            raise ModelContractError(
                "Hearo 분류기 출력 클래스 수가 categories와 다릅니다: "
                f"output={output_shape}, categories={len(self.contract.categories)}"
            )
        if self.contract.profile == "v3_2":
            if np.dtype(classifier_input["dtype"]) != np.dtype(np.float32):
                raise ModelContractError("v3.2 분류기 입력 dtype은 float32여야 합니다.")
            if np.dtype(classifier_output["dtype"]) != np.dtype(np.float32):
                raise ModelContractError("v3.2 분류기 출력 dtype은 float32여야 합니다.")

    def run_yamnet(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("YAMNet 입력 파형이 비어 있습니다.")
        if not np.all(np.isfinite(values)):
            raise ValueError("YAMNet 입력 파형에 NaN 또는 infinity가 있습니다.")

        detail = self.yamnet_interpreter.get_input_details()[0]
        self.yamnet_interpreter.resize_tensor_input(
            detail["index"], [len(values)], strict=False
        )
        self.yamnet_interpreter.allocate_tensors()
        detail = self.yamnet_interpreter.get_input_details()[0]
        self.yamnet_interpreter.set_tensor(
            detail["index"], _quantize_for_input(values, detail)
        )
        self.yamnet_interpreter.invoke()

        scores = None
        embeddings = None
        for output_detail in self.yamnet_interpreter.get_output_details():
            result = _dequantize_output(
                self.yamnet_interpreter.get_tensor(output_detail["index"]),
                output_detail,
            )
            if result.ndim == 2 and result.shape[-1] == YAMNET_CLASS_COUNT:
                if scores is not None:
                    raise ModelContractError("YAMNet score 출력이 둘 이상입니다.")
                scores = np.atleast_2d(result)
            elif result.ndim == 2 and result.shape[-1] == YAMNET_EMBEDDING_SIZE:
                if embeddings is not None:
                    raise ModelContractError("YAMNet embedding 출력이 둘 이상입니다.")
                embeddings = np.atleast_2d(result)
        if scores is None or embeddings is None:
            raise ModelContractError(
                "YAMNet 실행 결과에서 [frames, 521] score와 [frames, 1024] embedding을 찾지 못했습니다."
            )
        if scores.shape[0] != embeddings.shape[0]:
            raise ModelContractError(
                f"YAMNet score/embedding frame 수가 다릅니다: {scores.shape}, {embeddings.shape}"
            )
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(embeddings)):
            raise ValueError("YAMNet 출력에 NaN 또는 infinity가 있습니다.")
        if np.any(scores < -1e-5) or np.any(scores > 1.0 + 1e-5):
            raise ValueError("YAMNet score가 0~1 확률 범위를 벗어났습니다.")
        return scores.astype(np.float32), embeddings.astype(np.float32)

    def run_classifier(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != YAMNET_EMBEDDING_SIZE:
            raise ValueError(
                f"Hearo 분류기 입력은 [N, {YAMNET_EMBEDDING_SIZE}]이어야 합니다: {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Hearo 분류기 입력에 NaN 또는 infinity가 있습니다.")
        model_input = (
            values.mean(axis=0, keepdims=True)
            if self.contract.representation == "clip_mean"
            else values
        )

        detail = self.classifier_interpreter.get_input_details()[0]
        requested_shape = [len(model_input), YAMNET_EMBEDDING_SIZE]
        if list(_shape(detail)) != requested_shape:
            self.classifier_interpreter.resize_tensor_input(
                detail["index"], requested_shape, strict=False
            )
            self.classifier_interpreter.allocate_tensors()
            detail = self.classifier_interpreter.get_input_details()[0]
        self.classifier_interpreter.set_tensor(
            detail["index"], _quantize_for_input(model_input, detail)
        )
        self.classifier_interpreter.invoke()
        output_detail = self.classifier_interpreter.get_output_details()[0]
        output = _dequantize_output(
            self.classifier_interpreter.get_tensor(output_detail["index"]), output_detail
        )
        output = np.atleast_2d(output)
        expected_shape = (len(model_input), len(self.contract.categories))
        if output.shape != expected_shape:
            raise ModelContractError(
                f"Hearo 분류기 실행 출력 shape 오류: expected={expected_shape}, actual={output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("Hearo 분류기 출력에 NaN 또는 infinity가 있습니다.")

        output_type = self.contract.metadata.get("classifier_output", {}).get(
            "type", "probabilities"
        )
        if self.contract.temperature_embedded_in_tflite:
            if output_type != "probabilities":
                raise ModelContractError(
                    "temperature가 포함된 TFLite 출력은 probabilities여야 합니다."
                )
            if np.any(output < -1e-5) or np.any(output > 1.0 + 1e-5):
                raise ValueError("v3.2 분류기 확률이 0~1 범위를 벗어났습니다.")
            row_sums = output.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=1e-3):
                raise ValueError(
                    "v3.2 분류기 확률 합이 1이 아닙니다. "
                    "temperature scaling 또는 softmax가 TFLite에 포함됐는지 확인하십시오: "
                    f"sums={row_sums.tolist()}"
                )
            # Do not apply temperature scaling, softmax, or renormalization here.
            return np.clip(output, 0.0, 1.0).astype(np.float32)

        if output_type == "probabilities":
            probabilities = np.clip(output, 1e-9, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        else:
            shifted = output - output.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return np.clip(probabilities, 1e-9, 1.0).astype(np.float32)

    def aggregate_context(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.contract.categories):
            raise ValueError(
                "확률 입력 shape이 categories와 다릅니다: "
                f"{values.shape}, categories={len(self.contract.categories)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("집계할 확률에 NaN 또는 infinity가 있습니다.")
        if self.contract.representation == "clip_mean" or self.contract.context_frames == "full":
            return self._pool_probabilities(values)
        width = int(self.contract.context_frames)
        if len(values) <= width:
            windows = [values]
        else:
            windows = [values[start : start + width] for start in range(len(values) - width + 1)]
        pooled_windows = np.stack([self._pool_probabilities(window) for window in windows])
        target_indices = [
            index for index in range(len(self.contract.categories))
            if index != self.contract.unknown_index
        ]
        target_confidence = pooled_windows[:, target_indices].max(axis=1)
        return pooled_windows[int(np.argmax(target_confidence))]

    def _pool_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        if len(probabilities) == 1:
            pooled = probabilities[0]
        elif self.contract.frame_pooling == "mean_probability":
            pooled = probabilities.mean(axis=0)
        elif self.contract.frame_pooling == "topk_probability":
            count = max(1, int(math.ceil(len(probabilities) * 0.5)))
            pooled = np.sort(probabilities, axis=0)[-count:].mean(axis=0)
        elif self.contract.frame_pooling == "logit_logmeanexp":
            log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
            maximum = log_probabilities.max(axis=0, keepdims=True)
            pooled_logits = maximum[0] + np.log(
                np.exp(log_probabilities - maximum).mean(axis=0) + 1e-12
            )
            pooled_logits -= pooled_logits.max()
            pooled = np.exp(pooled_logits)
        else:  # Protected by metadata validation.
            raise ModelContractError(
                f"지원하지 않는 frame_pooling: {self.contract.frame_pooling}"
            )
        pooled = np.clip(pooled, 1e-12, None)
        return (pooled / pooled.sum()).astype(np.float32)

    def yamnet_gate_allows(self, scores: np.ndarray) -> bool:
        gate = self.contract.metadata.get("yamnet_gate", {})
        if not isinstance(gate, Mapping):
            raise ModelContractError("metadata.yamnet_gate는 JSON object여야 합니다.")
        if not bool(gate.get("enabled", False)):
            return True
        indices = gate.get("target_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or any(index < 0 or index >= YAMNET_CLASS_COUNT for index in indices)
        ):
            raise ModelContractError("yamnet_gate.target_indices가 유효하지 않습니다.")
        threshold = gate.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ModelContractError("yamnet_gate.threshold가 유효하지 않습니다.")
        return float(np.max(np.asarray(scores)[:, np.asarray(indices, dtype=int)])) >= float(
            threshold
        )


def validate_repeatability(
    runtime: HearoModelRuntime,
    embeddings: np.ndarray,
    *,
    repeats: int = 2,
    atol: float = 1e-7,
) -> np.ndarray:
    if repeats < 2:
        raise ValueError("반복 일관성 검사는 최소 2회여야 합니다.")
    outputs = [runtime.run_classifier(embeddings) for _ in range(repeats)]
    reference = outputs[0]
    for index, output in enumerate(outputs[1:], start=2):
        if not np.allclose(reference, output, atol=atol, rtol=0.0):
            difference = float(np.max(np.abs(reference - output)))
            raise ModelContractError(
                f"동일 입력의 {index}번째 추론 결과가 다릅니다: max_abs_diff={difference}"
            )
    return reference

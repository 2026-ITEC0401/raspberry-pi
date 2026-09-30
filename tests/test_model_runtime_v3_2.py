from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from hearo_hybrid_classifier import HybridDecisionEngine, fallback_policy
from hearo_model_runtime import (
    HearoModelRuntime,
    ModelContractError,
    load_model_contract,
    validate_repeatability,
)


CATEGORIES = [
    "노크_목재",
    "노크_철재문",
    "도어락_개방음",
    "도어락_입력음",
    "사이렌_고저교번형",
    "사이렌_완만변조형",
    "사이렌_급속변조형",
    "아기 울음",
    "비표적음",
]
CLASS_MAPPING = {
    "노크_목재": "노크소리",
    "노크_철재문": "노크소리",
    "도어락_개방음": "도어락소리",
    "도어락_입력음": "도어락소리",
    "사이렌_고저교번형": "비상벨소리",
    "사이렌_완만변조형": "비상벨소리",
    "사이렌_급속변조형": "비상벨소리",
    "아기 울음": "아기울음소리",
}


def metadata() -> dict:
    return {
        "schema_version": 4,
        "model_name": "hearo_classifier_v3_2",
        "sample_rate": 16_000,
        "record_step_seconds": 1.0,
        "rolling_buffer_seconds": 2.0,
        "classifier_input": {"dtype": "float32", "shape": [None, 1024]},
        "classifier_output": {
            "dtype": "float32",
            "shape": [None, len(CATEGORIES)],
            "type": "probabilities",
        },
        "categories": CATEGORIES,
        "unknown_label": "비표적음",
        "representation": "clip_mean",
        "frame_pooling": "mean_probability",
        "context_frames": "full",
        "temperature": 1.25,
        "temperature_embedded_in_tflite": True,
        "class_thresholds": {label: 0.6 for label in CATEGORIES},
        "selected_tflite_variant": "dynamic_range",
        "class_mapping": CLASS_MAPPING,
        "yamnet_gate": {"enabled": False},
    }


def write_artifacts(path: Path, *, metadata_value: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "yamnet.tflite").write_bytes(b"fake-yamnet")
    (path / "hearo_classifier_v3_2.tflite").write_bytes(b"fake-classifier")
    (path / "categories_v3_2.txt").write_text(
        "\n".join(CATEGORIES) + "\n", encoding="utf-8"
    )
    (path / "model_metadata_v3_2.json").write_text(
        json.dumps(metadata_value or metadata(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path / "yamnet_classes.txt").write_text(
        "\n".join(f"yamnet-{index}" for index in range(521)) + "\n",
        encoding="utf-8",
    )


class FakeInterpreter:
    def __init__(self, *, model_path: str, class_count: int = len(CATEGORIES), nan=False):
        self.is_yamnet = Path(model_path).name == "yamnet.tflite"
        self.class_count = class_count
        self.nan = nan
        self.input_shape = [1] if self.is_yamnet else [1, 1024]
        self.input_value = None

    def allocate_tensors(self):
        return None

    def get_input_details(self):
        signature = [-1] if self.is_yamnet else [-1, 1024]
        return [
            {
                "index": 0,
                "shape": np.asarray(self.input_shape, dtype=np.int32),
                "shape_signature": np.asarray(signature, dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
            }
        ]

    def get_output_details(self):
        if self.is_yamnet:
            return [
                {
                    "index": 1,
                    "shape": np.asarray([2, 521], dtype=np.int32),
                    "shape_signature": np.asarray([-1, 521], dtype=np.int32),
                    "dtype": np.float32,
                    "quantization": (0.0, 0),
                },
                {
                    "index": 2,
                    "shape": np.asarray([2, 1024], dtype=np.int32),
                    "shape_signature": np.asarray([-1, 1024], dtype=np.int32),
                    "dtype": np.float32,
                    "quantization": (0.0, 0),
                },
                {
                    "index": 3,
                    "shape": np.asarray([2, 64], dtype=np.int32),
                    "shape_signature": np.asarray([-1, 64], dtype=np.int32),
                    "dtype": np.float32,
                    "quantization": (0.0, 0),
                },
            ]
        return [
            {
                "index": 1,
                "shape": np.asarray([self.input_shape[0], self.class_count], dtype=np.int32),
                "shape_signature": np.asarray([-1, self.class_count], dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
            }
        ]

    def resize_tensor_input(self, index, shape, strict=False):
        assert index == 0
        self.input_shape = list(shape)

    def set_tensor(self, index, value):
        assert index == 0
        self.input_value = np.asarray(value)

    def invoke(self):
        assert self.input_value is not None

    def get_tensor(self, index):
        if self.is_yamnet:
            if index == 1:
                result = np.full((2, 521), 0.1, dtype=np.float32)
                return result
            if index == 2:
                return np.arange(2 * 1024, dtype=np.float32).reshape(2, 1024) / 2048.0
            if index == 3:
                return np.zeros((2, 64), dtype=np.float32)
            raise AssertionError(index)
        probabilities = np.asarray(
            [0.40, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.20],
            dtype=np.float32,
        )[: self.class_count]
        if self.class_count != len(CATEGORIES):
            probabilities = np.full(self.class_count, 1.0 / self.class_count, dtype=np.float32)
        result = np.tile(probabilities, (self.input_shape[0], 1))
        if self.nan:
            result[0, 0] = np.nan
        return result


def fake_factory(**kwargs):
    return FakeInterpreter(**kwargs)


def test_schema4_utf8_categories_and_selected_final_filename(tmp_path):
    write_artifacts(tmp_path)
    contract = load_model_contract(tmp_path, "v3.2")
    assert contract.profile == "v3_2"
    assert contract.categories == tuple(CATEGORIES)
    assert contract.classifier_model_path.name == "hearo_classifier_v3_2.tflite"
    assert contract.selected_tflite_variant == "dynamic_range"
    assert contract.temperature_embedded_in_tflite is True
    assert contract.buffer_chunks == 2


def test_missing_model_file_has_actionable_error(tmp_path):
    write_artifacts(tmp_path)
    (tmp_path / "hearo_classifier_v3_2.tflite").unlink()
    with pytest.raises(ModelContractError, match="hearo_classifier_v3_2.tflite"):
        load_model_contract(tmp_path, "v3_2")


def test_categories_order_must_match_metadata(tmp_path):
    value = metadata()
    value["categories"] = list(reversed(CATEGORIES))
    write_artifacts(tmp_path, metadata_value=value)
    with pytest.raises(ModelContractError, match="클래스 수 또는 순서"):
        load_model_contract(tmp_path, "v3_2")


def test_every_category_requires_a_threshold(tmp_path):
    value = metadata()
    del value["class_thresholds"]["노크_목재"]
    write_artifacts(tmp_path, metadata_value=value)
    with pytest.raises(ModelContractError, match="missing=.*노크_목재"):
        load_model_contract(tmp_path, "v3_2")


def test_classifier_contract_is_float_embedding_to_matching_class_count(tmp_path):
    write_artifacts(tmp_path)
    runtime = HearoModelRuntime.load(tmp_path, "v3_2", interpreter_factory=fake_factory)
    assert runtime.classifier_interpreter.get_input_details()[0]["shape_signature"].tolist() == [
        -1,
        1024,
    ]
    assert runtime.classifier_interpreter.get_output_details()[0]["shape_signature"].tolist() == [
        -1,
        len(CATEGORIES),
    ]

    def wrong_output_factory(**kwargs):
        return FakeInterpreter(**kwargs, class_count=len(CATEGORIES) - 1)

    with pytest.raises(ModelContractError, match="출력 클래스 수"):
        HearoModelRuntime.load(
            tmp_path, "v3_2", interpreter_factory=wrong_output_factory
        )


def test_yamnet_outputs_are_identified_by_521_and_1024(tmp_path):
    write_artifacts(tmp_path)
    runtime = HearoModelRuntime.load(tmp_path, "v3_2", interpreter_factory=fake_factory)
    scores, embeddings = runtime.run_yamnet(np.zeros(16_000, dtype=np.float32))
    assert scores.shape == (2, 521)
    assert embeddings.shape == (2, 1024)


def test_yamnet_outputs_must_be_two_dimensional_matrices(tmp_path):
    write_artifacts(tmp_path)

    class BadYamnetShapeInterpreter(FakeInterpreter):
        def get_output_details(self):
            details = super().get_output_details()
            if self.is_yamnet:
                details[0]["shape"] = np.asarray([521], dtype=np.int32)
                details[0]["shape_signature"] = np.asarray([521], dtype=np.int32)
            return details

    def bad_yamnet_factory(**kwargs):
        return BadYamnetShapeInterpreter(**kwargs)

    with pytest.raises(ModelContractError, match="각각 정확히 하나씩"):
        HearoModelRuntime.load(
            tmp_path,
            "v3_2",
            interpreter_factory=bad_yamnet_factory,
        )


def test_embedded_temperature_output_is_not_softmaxed_again_and_is_repeatable(tmp_path):
    write_artifacts(tmp_path)
    runtime = HearoModelRuntime.load(tmp_path, "v3_2", interpreter_factory=fake_factory)
    embeddings = np.ones((3, 1024), dtype=np.float32)
    output = validate_repeatability(runtime, embeddings, repeats=3)
    expected = np.asarray(
        [[0.40, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.20]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(output, expected)
    assert np.all(np.isfinite(output))


def test_nan_classifier_output_is_rejected(tmp_path):
    write_artifacts(tmp_path)

    def nan_factory(**kwargs):
        return FakeInterpreter(**kwargs, nan=Path(kwargs["model_path"]).name != "yamnet.tflite")

    runtime = HearoModelRuntime.load(tmp_path, "v3_2", interpreter_factory=nan_factory)
    with pytest.raises(ValueError, match="NaN 또는 infinity"):
        runtime.run_classifier(np.ones((2, 1024), dtype=np.float32))


def test_unknown_is_suppressed_and_class_threshold_is_applied(tmp_path):
    write_artifacts(tmp_path)
    contract = load_model_contract(tmp_path, "v3_2")
    engine = HybridDecisionEngine(
        policy=fallback_policy("v3.2 unit test"),
        categories=contract.categories,
        unknown_label=contract.unknown_label,
        class_thresholds=contract.class_thresholds,
        class_mapping=contract.class_mapping,
        base_decision_source="hearo_v3_2",
    )
    scores = np.zeros((2, 521), dtype=np.float32)
    unknown = np.zeros(len(CATEGORIES), dtype=np.float32)
    unknown[-1] = 1.0
    decision = engine.decide_from_outputs(scores, unknown, "rpi-001")
    assert decision.sound is None
    assert decision.raw_label == "비표적음"
    assert decision.decision_source == "hearo_v3_2"

    below = np.zeros(len(CATEGORIES), dtype=np.float32)
    below[0] = 0.59
    below[-1] = 0.41
    assert engine.decide_from_outputs(scores, below, "rpi-001").sound is None

    above = np.zeros(len(CATEGORIES), dtype=np.float32)
    above[0] = 0.61
    above[-1] = 0.39
    assert engine.decide_from_outputs(scores, above, "rpi-001").sound == "노크소리"


def test_rpi_entrypoint_keeps_cloud_gpio_audio_and_hybrid_integrations():
    path = Path(__file__).resolve().parents[1] / "appAWS_v3_2.py"
    source = path.read_text(encoding="utf-8", errors="strict")
    ast.parse(source, filename=str(path))
    for required in (
        "DeviceCloudRuntime",
        "AudioUdpReceiver",
        "HybridDecisionEngine",
        "GPIO.cleanup",
        "publish_alert",
        "record_audio_chunk",
        "HearoModelRuntime",
        "HEARO_MODEL_PROFILE",
    ):
        assert required in source


def test_v3_2_mqtt_payload_only_uses_backend_supported_explicit_fields():
    path = Path(__file__).resolve().parents[1] / "appAWS_v3_2.py"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"))
    deliver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "deliver_decision"
    )
    payload = next(
        node.value
        for node in ast.walk(deliver)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "payload"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    explicit_keys = {
        key.value
        for key in payload.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    supported = {
        "timestamp",
        "sound",
        "raw_label",
        "type",
        "confidence",
        "model_version",
    }
    assert explicit_keys == supported
    assert "model_profile" not in explicit_keys
    assert "selected_tflite_variant" not in explicit_keys


def test_v3_2_engine_can_apply_metadata_boundary_threshold_without_legacy_clamp(
    tmp_path,
):
    value = metadata()
    value["class_thresholds"]["노크_목재"] = 0.0
    write_artifacts(tmp_path, metadata_value=value)
    contract = load_model_contract(tmp_path, "v3_2")
    engine = HybridDecisionEngine(
        policy=fallback_policy("v3.2 boundary threshold test"),
        categories=contract.categories,
        unknown_label=contract.unknown_label,
        class_thresholds=contract.class_thresholds,
        class_mapping=contract.class_mapping,
        base_decision_source="hearo_v3_2",
        threshold_clip_range=(0.0, 1.0),
    )
    scores = np.zeros((1, 521), dtype=np.float32)
    probabilities = np.zeros(len(CATEGORIES), dtype=np.float32)
    probabilities[0] = 0.01
    decision = engine.decide_from_outputs(scores, probabilities, "rpi-001")
    assert decision.sound == "노크소리"
    assert decision.applied_threshold == 0.0

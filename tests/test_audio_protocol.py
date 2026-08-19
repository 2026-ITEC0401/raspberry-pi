from __future__ import annotations

import struct

import pytest

from hearo_audio_protocol import (
    AudioProtocolError,
    FRAME_SAMPLES,
    HEADER_SIZE,
    SAMPLE_RATE,
    decode_audio_packet,
    encode_audio_packet,
)


PSK = "test-audio-key-at-least-16-bytes"


def pcm_frame() -> bytes:
    return struct.pack("<" + "h" * FRAME_SAMPLES, *range(FRAME_SAMPLES))


def test_audio_packet_round_trip():
    encoded = encode_audio_packet(
        device_id="esp32_2",
        sequence=42,
        capture_ms=123456,
        pcm16le=pcm_frame(),
        pre_shared_key=PSK,
    )
    assert len(encoded) == HEADER_SIZE + FRAME_SAMPLES * 2
    decoded = decode_audio_packet(encoded, PSK)
    assert decoded.device_id == "esp32_2"
    assert decoded.sequence == 42
    assert decoded.capture_ms == 123456
    assert decoded.sample_rate == SAMPLE_RATE
    assert decoded.sample_count == FRAME_SAMPLES
    assert decoded.pcm16le == pcm_frame()


def test_audio_packet_rejects_tampering_and_wrong_key():
    encoded = bytearray(
        encode_audio_packet(
            device_id="esp32_1",
            sequence=1,
            capture_ms=10,
            pcm16le=pcm_frame(),
            pre_shared_key=PSK,
        )
    )
    encoded[-1] ^= 0x01
    with pytest.raises(AudioProtocolError):
        decode_audio_packet(bytes(encoded), PSK)

    valid = encode_audio_packet(
        device_id="esp32_1",
        sequence=1,
        capture_ms=10,
        pcm16le=pcm_frame(),
        pre_shared_key=PSK,
    )
    with pytest.raises(AudioProtocolError):
        decode_audio_packet(valid, "different-audio-key-123456")


def test_audio_packet_validates_contract():
    with pytest.raises(AudioProtocolError):
        encode_audio_packet(
            device_id="device-id-that-is-far-too-long",
            sequence=1,
            capture_ms=1,
            pcm16le=pcm_frame(),
            pre_shared_key=PSK,
        )
    with pytest.raises(AudioProtocolError):
        encode_audio_packet(
            device_id="esp32_3",
            sequence=1,
            capture_ms=1,
            pcm16le=b"\x00",
            pre_shared_key=PSK,
        )

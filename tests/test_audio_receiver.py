from __future__ import annotations

import struct

import numpy as np

from hearo_audio_protocol import FRAME_SAMPLES, encode_audio_packet
from hearo_audio_receiver import AudioUdpReceiver, DeviceStreamBuffer, WINDOW_FRAMES


PSK = "receiver-test-key-at-least-16-bytes"


def packet(sequence: int, value: int = 1000):
    payload = struct.pack("<" + "h" * FRAME_SAMPLES, *([value] * FRAME_SAMPLES))
    encoded = encode_audio_packet(
        device_id="esp32_3",
        sequence=sequence,
        capture_ms=sequence * 20,
        pcm16le=payload,
        pre_shared_key=PSK,
    )
    from hearo_audio_protocol import decode_audio_packet

    return decode_audio_packet(encoded, PSK)


def encoded_packet(sequence: int, value: int = 1000):
    payload = struct.pack("<" + "h" * FRAME_SAMPLES, *([value] * FRAME_SAMPLES))
    return encode_audio_packet(
        device_id="esp32_3",
        sequence=sequence,
        capture_ms=sequence * 20,
        pcm16le=payload,
        pre_shared_key=PSK,
    )


def test_stream_emits_two_second_window_and_then_every_second():
    stream = DeviceStreamBuffer("esp32_3", "화장실")
    emitted = []
    for sequence in range(WINDOW_FRAMES):
        window = stream.ingest(packet(sequence))
        if window is not None:
            emitted.append(window)
    assert len(emitted) == 1
    assert emitted[0].device_id == "esp32_3"
    assert emitted[0].location == "화장실"
    assert emitted[0].waveform.shape == (32_000,)
    assert np.isclose(emitted[0].waveform.mean(), 1000 / 32768.0)

    for sequence in range(WINDOW_FRAMES, WINDOW_FRAMES + 50):
        window = stream.ingest(packet(sequence))
        if window is not None:
            emitted.append(window)
    assert len(emitted) == 2


def test_stream_zero_fills_short_gap_and_drops_old_packet():
    stream = DeviceStreamBuffer("esp32_3", "화장실")
    stream.ingest(packet(10))
    stream.ingest(packet(12))
    assert stream.stats.packets_missing == 1
    assert len(stream.frames) == 3
    assert np.all(stream.frames[1] == 0)
    stream.ingest(packet(11))
    assert stream.stats.packets_duplicate_or_old == 1


def test_receiver_notifies_hybrid_history_when_stream_resets():
    resets = []
    receiver = AudioUdpReceiver(
        pre_shared_key=PSK,
        on_window=lambda window: None,
        on_stream_reset=resets.append,
    )
    assert receiver.ingest_datagram(encoded_packet(0))
    assert receiver.ingest_datagram(encoded_packet(30))
    assert resets == ["esp32_3"]

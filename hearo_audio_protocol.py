"""Binary LAN audio protocol shared by Hearo ESP32 nodes and Raspberry Pi.

Audio never passes through the cloud broker.  Each datagram contains one
20 ms PCM16 frame, a CRC for accidental corruption detection, and a truncated
HMAC-SHA256 tag for authenticating devices on the local network.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import zlib
from dataclasses import dataclass


MAGIC = b"HRA3"
PROTOCOL_VERSION = 3
DEVICE_ID_BYTES = 16
AUTH_TAG_BYTES = 16
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
FRAME_MILLISECONDS = 20

# magic, version, flags, header_size, sequence, capture_ms, sample_rate,
# sample_count, zero-padded device id, payload CRC32
_PREFIX = struct.Struct("!4sBBHIQHH16sI")
HEADER_SIZE = _PREFIX.size + AUTH_TAG_BYTES


class AudioProtocolError(ValueError):
    """Raised when a received audio datagram violates the wire contract."""


@dataclass(frozen=True)
class AudioPacket:
    device_id: str
    sequence: int
    capture_ms: int
    sample_rate: int
    sample_count: int
    flags: int
    pcm16le: bytes


def _key_bytes(pre_shared_key: str | bytes) -> bytes:
    value = pre_shared_key.encode("utf-8") if isinstance(pre_shared_key, str) else pre_shared_key
    if len(value) < 16:
        raise AudioProtocolError("오디오 PSK는 16바이트 이상이어야 합니다.")
    return value


def _device_bytes(device_id: str) -> bytes:
    encoded = device_id.encode("ascii")
    if not encoded or len(encoded) >= DEVICE_ID_BYTES:
        raise AudioProtocolError("device_id는 ASCII 1~15바이트여야 합니다.")
    return encoded.ljust(DEVICE_ID_BYTES, b"\0")


def encode_audio_packet(
    *,
    device_id: str,
    sequence: int,
    capture_ms: int,
    pcm16le: bytes,
    pre_shared_key: str | bytes,
    sample_rate: int = SAMPLE_RATE,
    flags: int = 0,
) -> bytes:
    if len(pcm16le) % 2:
        raise AudioProtocolError("PCM16 payload 길이는 짝수여야 합니다.")
    sample_count = len(pcm16le) // 2
    if sample_count <= 0 or sample_count > FRAME_SAMPLES:
        raise AudioProtocolError(f"sample_count는 1~{FRAME_SAMPLES} 범위여야 합니다.")
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise AudioProtocolError("sequence는 uint32 범위여야 합니다.")
    if not 0 <= capture_ms <= 0xFFFFFFFFFFFFFFFF:
        raise AudioProtocolError("capture_ms는 uint64 범위여야 합니다.")
    if not 0 <= flags <= 0xFF:
        raise AudioProtocolError("flags는 uint8 범위여야 합니다.")

    crc = zlib.crc32(pcm16le) & 0xFFFFFFFF
    prefix = _PREFIX.pack(
        MAGIC,
        PROTOCOL_VERSION,
        flags,
        HEADER_SIZE,
        sequence,
        capture_ms,
        sample_rate,
        sample_count,
        _device_bytes(device_id),
        crc,
    )
    tag = hmac.new(_key_bytes(pre_shared_key), prefix + pcm16le, hashlib.sha256).digest()[:AUTH_TAG_BYTES]
    return prefix + tag + pcm16le


def decode_audio_packet(datagram: bytes, pre_shared_key: str | bytes) -> AudioPacket:
    if len(datagram) < HEADER_SIZE + 2:
        raise AudioProtocolError("오디오 datagram이 너무 짧습니다.")
    prefix = datagram[: _PREFIX.size]
    tag = datagram[_PREFIX.size : HEADER_SIZE]
    payload = datagram[HEADER_SIZE:]
    (
        magic,
        version,
        flags,
        header_size,
        sequence,
        capture_ms,
        sample_rate,
        sample_count,
        encoded_device_id,
        expected_crc,
    ) = _PREFIX.unpack(prefix)

    if magic != MAGIC or version != PROTOCOL_VERSION:
        raise AudioProtocolError("지원하지 않는 오디오 protocol입니다.")
    if header_size != HEADER_SIZE:
        raise AudioProtocolError("오디오 header 크기가 일치하지 않습니다.")
    if sample_rate != SAMPLE_RATE:
        raise AudioProtocolError(f"지원하지 않는 sample rate입니다: {sample_rate}")
    if sample_count <= 0 or sample_count > FRAME_SAMPLES or len(payload) != sample_count * 2:
        raise AudioProtocolError("sample_count와 PCM payload 길이가 일치하지 않습니다.")
    if zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
        raise AudioProtocolError("PCM payload CRC가 일치하지 않습니다.")
    expected_tag = hmac.new(
        _key_bytes(pre_shared_key), prefix + payload, hashlib.sha256
    ).digest()[:AUTH_TAG_BYTES]
    if not hmac.compare_digest(tag, expected_tag):
        raise AudioProtocolError("오디오 HMAC 인증에 실패했습니다.")
    try:
        device_id = encoded_device_id.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise AudioProtocolError("device_id가 ASCII가 아닙니다.") from exc
    if not device_id:
        raise AudioProtocolError("device_id가 비어 있습니다.")

    return AudioPacket(
        device_id=device_id,
        sequence=sequence,
        capture_ms=capture_ms,
        sample_rate=sample_rate,
        sample_count=sample_count,
        flags=flags,
        pcm16le=payload,
    )

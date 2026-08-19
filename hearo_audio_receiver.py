"""Authenticated LAN audio receiver for Hearo Raspberry Pi inference.

The receiver keeps an independent two-second rolling window for every ESP32,
emits one inference job per second, and never writes PCM to disk.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from hearo_audio_protocol import (
    AudioPacket,
    AudioProtocolError,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    decode_audio_packet,
)


DEVICE_LOCATIONS = {
    "esp32_1": "안방",
    "esp32_2": "현관",
    "esp32_3": "화장실",
}
WINDOW_SECONDS = 2
EMIT_SECONDS = 1
WINDOW_FRAMES = WINDOW_SECONDS * SAMPLE_RATE // FRAME_SAMPLES
EMIT_FRAMES = EMIT_SECONDS * SAMPLE_RATE // FRAME_SAMPLES
MAX_ZERO_FILL_FRAMES = 25


@dataclass(frozen=True)
class AudioWindow:
    device_id: str
    location: str
    capture_ms: int
    observed_at: float
    waveform: np.ndarray


@dataclass
class StreamStats:
    packets_received: int = 0
    packets_missing: int = 0
    packets_duplicate_or_old: int = 0
    packets_invalid: int = 0
    windows_emitted: int = 0
    windows_dropped: int = 0
    stream_resets: int = 0


def _sequence_ahead(expected: int, actual: int) -> int | None:
    distance = (actual - expected) & 0xFFFFFFFF
    if distance == 0:
        return 0
    return distance if distance < 0x80000000 else None


class DeviceStreamBuffer:
    def __init__(self, device_id: str, location: str):
        self.device_id = device_id
        self.location = location
        self.frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)
        self.expected_sequence: int | None = None
        self.frames_since_emit = 0
        self.stats = StreamStats()

    def ingest(self, packet: AudioPacket) -> AudioWindow | None:
        if packet.device_id != self.device_id:
            raise ValueError("packet device_id가 stream buffer와 다릅니다.")
        samples = np.frombuffer(packet.pcm16le, dtype="<i2").astype(np.float32) / 32768.0
        if len(samples) != packet.sample_count:
            raise ValueError("PCM sample 수가 packet metadata와 다릅니다.")

        if self.expected_sequence is not None:
            gap = _sequence_ahead(self.expected_sequence, packet.sequence)
            if gap is None:
                self.stats.packets_duplicate_or_old += 1
                return None
            if gap > MAX_ZERO_FILL_FRAMES:
                self.frames.clear()
                self.frames_since_emit = 0
                self.stats.stream_resets += 1
            else:
                for _ in range(gap):
                    self.frames.append(np.zeros(FRAME_SAMPLES, dtype=np.float32))
                self.frames_since_emit += gap
                self.stats.packets_missing += gap

        self.expected_sequence = (packet.sequence + 1) & 0xFFFFFFFF
        self.frames.append(samples)
        self.frames_since_emit += 1
        self.stats.packets_received += 1

        if len(self.frames) < WINDOW_FRAMES or self.frames_since_emit < EMIT_FRAMES:
            return None
        self.frames_since_emit = 0
        self.stats.windows_emitted += 1
        waveform = np.concatenate(tuple(self.frames)).astype(np.float32, copy=False)
        return AudioWindow(
            device_id=self.device_id,
            location=self.location,
            capture_ms=packet.capture_ms,
            observed_at=time.monotonic(),
            waveform=waveform,
        )


class AudioUdpReceiver:
    def __init__(
        self,
        *,
        pre_shared_key: str,
        on_window: Callable[[AudioWindow], None],
        bind_host: str = "0.0.0.0",
        bind_port: int = 41000,
        device_locations: dict[str, str] | None = None,
        inference_queue_size: int = 6,
        on_stream_reset: Callable[[str], None] | None = None,
    ):
        self.pre_shared_key = pre_shared_key
        self.on_window = on_window
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.device_locations = dict(device_locations or DEVICE_LOCATIONS)
        self.on_stream_reset = on_stream_reset
        self.buffers = {
            device_id: DeviceStreamBuffer(device_id, location)
            for device_id, location in self.device_locations.items()
        }
        self.invalid_packets = 0
        self._queue: queue.Queue[AudioWindow] = queue.Queue(maxsize=inference_queue_size)
        self._stopping = threading.Event()
        self._socket: socket.socket | None = None
        self._receive_thread = threading.Thread(
            target=self._receive_loop, name="hearo-audio-udp", daemon=True
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="hearo-audio-inference", daemon=True
        )

    def ingest_datagram(self, datagram: bytes) -> bool:
        try:
            packet = decode_audio_packet(datagram, self.pre_shared_key)
        except AudioProtocolError:
            self.invalid_packets += 1
            return False
        stream = self.buffers.get(packet.device_id)
        if stream is None:
            self.invalid_packets += 1
            return False
        reset_count = stream.stats.stream_resets
        window = stream.ingest(packet)
        if stream.stats.stream_resets != reset_count:
            self._notify_stream_reset(packet.device_id)
        if window is None:
            return True
        try:
            self._queue.put_nowait(window)
        except queue.Full:
            try:
                stale = self._queue.get_nowait()
                self.buffers[stale.device_id].stats.windows_dropped += 1
                self._notify_stream_reset(stale.device_id)
            except queue.Empty:
                pass
            self._queue.put_nowait(window)
        return True

    def _notify_stream_reset(self, device_id: str) -> None:
        if self.on_stream_reset is None:
            return
        try:
            self.on_stream_reset(device_id)
        except Exception as exc:
            print(f"[원격 오디오] {device_id} history reset callback 실패: {type(exc).__name__}")

    def _receive_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_host, self.bind_port))
        sock.settimeout(1.0)
        self._socket = sock
        while not self._stopping.is_set():
            try:
                datagram, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            self.ingest_datagram(datagram)

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                window = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.on_window(window)
            except Exception as exc:  # Keep later audio windows alive after one inference failure.
                print(f"[원격 오디오] {window.device_id} 추론 실패: {type(exc).__name__}: {exc}")

    def start(self) -> None:
        if self._receive_thread.is_alive() or self._worker_thread.is_alive():
            raise RuntimeError("AudioUdpReceiver가 이미 실행 중입니다.")
        self._worker_thread.start()
        self._receive_thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._socket is not None:
            self._socket.close()
        for thread in (self._receive_thread, self._worker_thread):
            if thread.is_alive():
                thread.join(timeout=2)

    def stats(self) -> dict[str, object]:
        return {
            "invalid_packets": self.invalid_packets,
            "devices": {
                device_id: asdict(stream.stats) for device_id, stream in self.buffers.items()
            },
        }

"""Hearo Raspberry Pi/ESP-class Linux device cloud-control runtime.

The local inference loop remains independent from this class. Turning MQTT off
therefore stops cloud/remote notifications but never stops local AI or LEDs.
"""

from __future__ import annotations

import json
import os
import ssl
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class DeviceCloudRuntime:
    def __init__(
        self,
        *,
        household_id: str,
        device_id: str,
        location: str,
        firmware_version: str,
    ):
        import paho.mqtt.client as mqtt

        required = {
            "HEARO_HOUSEHOLD_ID": household_id,
            "HEARO_DEVICE_CREDENTIAL": os.getenv("HEARO_DEVICE_CREDENTIAL", ""),
            "HEARO_API_BASE_URL": os.getenv("HEARO_API_BASE_URL", ""),
            "HEARO_MQTT_HOST": os.getenv("HEARO_MQTT_HOST", ""),
            "HEARO_MQTT_USERNAME": os.getenv("HEARO_MQTT_USERNAME", ""),
            "HEARO_MQTT_PASSWORD": os.getenv("HEARO_MQTT_PASSWORD", ""),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"필수 기기 환경변수가 없습니다: {', '.join(missing)}")

        self.household_id = household_id
        self.device_id = device_id
        self.location = location
        self.firmware_version = firmware_version
        self.device_credential = required["HEARO_DEVICE_CREDENTIAL"]
        self.api_base_url = required["HEARO_API_BASE_URL"].rstrip("/")
        if not self.api_base_url.startswith("https://"):
            raise RuntimeError("HEARO_API_BASE_URL은 HTTPS 주소여야 합니다.")
        self.mqtt_host = required["HEARO_MQTT_HOST"]
        self.mqtt_port = int(os.getenv("HEARO_MQTT_PORT", "8883"))
        self.poll_seconds = max(5, int(os.getenv("HEARO_CONFIG_POLL_SECONDS", "15")))
        self.legacy_topics = os.getenv("HEARO_PUBLISH_LEGACY_TOPICS", "true").lower() == "true"
        self.state_path = Path(
            os.getenv("HEARO_DEVICE_STATE_PATH", f"/var/lib/hearo/{device_id}-state.json")
        )
        self.command_topic = f"hearo/{household_id}/devices/{device_id}/command"
        self.ack_topic = f"hearo/{household_id}/devices/{device_id}/ack"
        self.state_topic = f"hearo/{household_id}/devices/{device_id}/state"
        self.presence_topic = f"hearo/{household_id}/devices/{device_id}/presence"
        self.alerts_topic = f"hearo/{household_id}/alerts"

        persisted = self._load_state()
        self.desired_mqtt_connected = bool(persisted.get("desired_mqtt_connected", True))
        self.config_version = int(persisted.get("config_version", 0))
        self.led_alert_enabled = bool(persisted.get("led_alert_enabled", True))
        persisted_sensitivity = str(persisted.get("sensitivity", "default"))
        self.sensitivity = (
            persisted_sensitivity
            if persisted_sensitivity in {"low", "default", "high"}
            else "default"
        )
        self._connected = threading.Event()
        self._stopping = threading.Event()
        self._state_lock = threading.RLock()

        try:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"hearo-{device_id}",
                clean_session=True,
            )
        except (AttributeError, TypeError):
            self.client = mqtt.Client(client_id=f"hearo-{device_id}", clean_session=True)
        self.client.username_pw_set(
            required["HEARO_MQTT_USERNAME"], required["HEARO_MQTT_PASSWORD"]
        )
        ca_path = os.getenv("HEARO_MQTT_CA_PATH", "/etc/ssl/certs/ca-certificates.crt")
        self.client.tls_set(ca_certs=ca_path, cert_reqs=ssl.CERT_REQUIRED)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.will_set(
            self.presence_topic,
            json.dumps(
                {
                    "device_id": self.device_id,
                    "network_online": False,
                    "mqtt_connected": False,
                    "config_version": self.config_version,
                    "seen_at": _utc_now(),
                },
                ensure_ascii=False,
            ),
            qos=1,
            retain=True,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self._poll_thread = threading.Thread(
            target=self._control_loop, name="hearo-device-control", daemon=True
        )

    def _load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "desired_mqtt_connected": self.desired_mqtt_connected,
                "config_version": self.config_version,
                "led_alert_enabled": self.led_alert_enabled,
                "sensitivity": self.sensitivity,
            },
            ensure_ascii=False,
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", dir=str(self.state_path.parent)
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_name, self.state_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _success(reason_code: Any) -> bool:
        try:
            return int(reason_code) == 0
        except (TypeError, ValueError):
            return str(reason_code).casefold() in {"success", "0"}

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not self._success(reason_code):
            print(f"[MQTT] 연결 실패: {reason_code}")
            return
        if not self.desired_mqtt_connected:
            client.disconnect()
            return
        self._connected.set()
        client.subscribe(self.command_topic, qos=1)
        self._publish_presence(True)
        self._publish_state(True)
        print(f"[MQTT] TLS 연결 완료: {self.mqtt_host}:{self.mqtt_port}")

    def _on_disconnect(self, client, userdata, *args):
        self._connected.clear()
        if not self._stopping.is_set() and self.desired_mqtt_connected:
            print("[MQTT] 예기치 않은 연결 해제 — 자동 재연결 대기")

    def _on_message(self, client, userdata, message):
        if message.topic != self.command_topic:
            return
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            command_type = payload.get("type")
            if command_type not in {"mqtt.connection.set", "device.config.set"}:
                return
            version = int(payload["config_version"])
            enabled = bool(
                payload.get("enabled", payload.get("desired_mqtt_connected", self.desired_mqtt_connected))
            )
            led_enabled = payload.get("led_alert_enabled")
            sensitivity = payload.get("sensitivity")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            print("[MQTT] 잘못된 command를 무시했습니다.")
            return
        self._apply_config(
            enabled,
            version,
            acknowledge=True,
            led_alert_enabled=led_enabled,
            sensitivity=sensitivity,
        )

    def _publish_json(self, topic: str, payload: dict[str, Any], *, retain: bool = False):
        return self.client.publish(
            topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=retain
        )

    def _state_payload(self, mqtt_connected: bool) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "mqtt_connected": mqtt_connected,
            "network_online": True,
            "config_version": self.config_version,
            "firmware_version": self.firmware_version,
            "seen_at": _utc_now(),
        }

    def _publish_presence(self, online: bool) -> None:
        # Last Will/offline presence must never leave the device looking MQTT-connected.
        mqtt_connected = self._connected.is_set() if online else False
        payload = self._state_payload(mqtt_connected)
        payload["network_online"] = online
        self._publish_json(self.presence_topic, payload, retain=True)

    def _publish_state(self, mqtt_connected: bool):
        return self._publish_json(self.state_topic, self._state_payload(mqtt_connected))

    def _publish_ack(self, enabled: bool):
        payload = self._state_payload(enabled)
        payload["type"] = "mqtt.connection.ack"
        payload["accepted"] = True
        return self._publish_json(self.ack_topic, payload)

    def _apply_config(
        self,
        enabled: bool,
        version: int,
        *,
        acknowledge: bool,
        led_alert_enabled: bool | None = None,
        sensitivity: str | None = None,
    ) -> None:
        with self._state_lock:
            if version < self.config_version:
                return
            changed = version > self.config_version or enabled != self.desired_mqtt_connected
            self.config_version = version
            self.desired_mqtt_connected = enabled
            if led_alert_enabled is not None:
                self.led_alert_enabled = bool(led_alert_enabled)
            if sensitivity in {"low", "default", "high"}:
                self.sensitivity = sensitivity
            if changed:
                self._persist_state()

            if enabled:
                if acknowledge and self._connected.is_set():
                    self._publish_ack(True)
                if not self._connected.is_set() and not self._stopping.is_set():
                    try:
                        self.client.reconnect()
                    except (OSError, RuntimeError):
                        self.client.connect_async(self.mqtt_host, self.mqtt_port, 60)
            elif self._connected.is_set():
                self._publish_ack(False).wait_for_publish(timeout=2)
                self._publish_state(False).wait_for_publish(timeout=2)
                self.client.disconnect()

    def _api_json(self, method: str, path: str, payload: dict[str, Any] | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.api_base_url}{path}",
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "X-Device-Credential": self.device_credential,
            },
        )
        with urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))

    def _control_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                config = self._api_json("GET", "/device/v1/config")
                self._apply_config(
                    bool(config["desired_mqtt_connected"]),
                    int(config["config_version"]),
                    acknowledge=False,
                    led_alert_enabled=config.get("led_alert_enabled"),
                    sensitivity=config.get("sensitivity"),
                )
                self._api_json(
                    "POST",
                    "/device/v1/heartbeat",
                    {
                        "mqtt_connected": self._connected.is_set(),
                        "config_version": self.config_version,
                        "firmware_version": self.firmware_version,
                    },
                )
            except (HTTPError, URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
                print(f"[제어] 설정/heartbeat 실패: {type(exc).__name__}")
            self._stopping.wait(self.poll_seconds)

    def start(self) -> None:
        if self.desired_mqtt_connected:
            self.client.connect_async(self.mqtt_host, self.mqtt_port, 60)
        self.client.loop_start()
        self._poll_thread.start()

    def publish_alert(
        self,
        payload: dict[str, Any],
        *,
        source_device_id: str | None = None,
        publisher_device_id: str | None = None,
        capture_device_id: str | None = None,
        location: str | None = None,
    ) -> bool:
        if not self.desired_mqtt_connected or not self._connected.is_set():
            return False
        publisher = publisher_device_id or self.device_id
        capture = capture_device_id or source_device_id or publisher
        event = {
            **payload,
            "household_id": self.household_id,
            "event_id": payload.get("event_id") or uuid.uuid4().hex,
            "timestamp": payload.get("timestamp") or _utc_now(),
            # source_device_id stays the authenticated MQTT publisher for
            # backward-compatible authorization. capture_device_id identifies
            # the microphone and therefore the user-visible location.
            "source_device_id": publisher,
            "publisher_device_id": publisher,
            "capture_device_id": capture,
            "location": location or self.location,
        }
        self._publish_json(self.alerts_topic, event)
        if self.legacy_topics:
            self.client.publish("hearo/alert", event["sound"], qos=1)
            self._publish_json("hearo/log", {**event, "device_id": event["capture_device_id"]})
        return True

    def stop(self) -> None:
        self._stopping.set()
        if self._connected.is_set():
            self._publish_presence(False)
            self.client.disconnect()
        self.client.loop_stop()
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3)

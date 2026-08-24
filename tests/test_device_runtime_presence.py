import threading

from hearo_device_runtime import DeviceCloudRuntime


def test_offline_presence_never_reports_mqtt_connected():
    runtime = object.__new__(DeviceCloudRuntime)
    runtime.device_id = "rpi-001"
    runtime.config_version = 3
    runtime.firmware_version = "rpi-inference-v2.1"
    runtime._connected = threading.Event()
    runtime._connected.set()

    published = {}

    def capture(topic, payload, *, retain=False):
        published.update({"topic": topic, "payload": payload, "retain": retain})

    runtime.presence_topic = "hearo/home-test/devices/rpi-001/presence"
    runtime._publish_json = capture

    runtime._publish_presence(False)

    assert published["retain"] is True
    assert published["payload"]["network_online"] is False
    assert published["payload"]["mqtt_connected"] is False

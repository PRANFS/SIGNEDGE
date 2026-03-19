import json
import time
from threading import Lock

import paho.mqtt.client as mqtt


class MQTTPublisher:
    def __init__(self, host: str, port: int, base_topic: str, qos: int = 0, enabled: bool = True):
        self.enabled = enabled
        self.host = host
        self.port = port
        self.base_topic = base_topic.strip("/")
        self.qos = qos
        self.connected = False
        self._lock = Lock()

        self.client = None
        if not self.enabled:
            return

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.host, self.port, keepalive=30)
            self.client.loop_start()
        except Exception:
            self.connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        code = getattr(reason_code, "value", reason_code)
        self.connected = code == 0

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected = False

    def publish(self, topic_suffix: str, payload: dict):
        if not self.enabled or self.client is None:
            return

        topic = f"{self.base_topic}/{topic_suffix.strip('/')}"
        body = dict(payload)
        body.setdefault("ts", time.time())

        with self._lock:
            try:
                self.client.publish(topic, json.dumps(body), qos=self.qos)
            except Exception:
                self.connected = False

    def stop(self):
        if self.client is None:
            return
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

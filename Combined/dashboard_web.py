import json
import time
from collections import deque
from threading import Lock

import paho.mqtt.client as mqtt
from flask import Flask, jsonify, render_template_string

from core.config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    MQTT_BASE_TOPIC,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
)


class DashboardState:
    def __init__(self):
        self.lock = Lock()
        self.started_at = time.time()
        self.mqtt_connected = False
        self.last_seen_ts = None

        self.mode = "UNKNOWN"
        self.mode_message = "Waiting for telemetry"
        self.uptime_s = 0.0

        self.asl = {
            "label": None,
            "confidence": 0.0,
            "top3": [],
            "hold_progress": 0.0,
            "in_send_zone": False,
            "buffer": "",
            "sent_text": None,
        }
        self.eye = {
            "state": "UNKNOWN",
            "cursor_x": None,
            "cursor_y": None,
            "hovered_key": None,
            "blink_click_count": 0,
            "is_clicking": False,
            "text_length": 0,
        }
        self.stt = {"partial": "", "last_final": "", "status": ""}
        self.tts = {"status": ""}

        self.mode_switch_count = 0
        self.error_count = 0
        self.asl_commits = 0
        self.eye_click_events = 0

        self.confidence_window = deque(maxlen=120)
        self.loop_ms_window = deque(maxlen=120)
        self.raw_topics = {}

    def update_from_message(self, topic, payload):
        with self.lock:
            self.last_seen_ts = time.time()
            self.raw_topics[topic] = payload

            if topic.endswith("/system/heartbeat"):
                self.mode = payload.get("mode", self.mode)
                self.mode_message = payload.get("mode_message", self.mode_message)
                self.uptime_s = payload.get("uptime_s", self.uptime_s)
                loop_ms = payload.get("loop_ms")
                if isinstance(loop_ms, (int, float)):
                    self.loop_ms_window.append(float(loop_ms))
                if "mqtt_connected" in payload:
                    self.mqtt_connected = bool(payload.get("mqtt_connected"))

            elif topic.endswith("/mode/switch"):
                self.mode_switch_count += 1
                self.mode = payload.get("new_mode", self.mode)

            elif topic.endswith("/asl/metrics"):
                self.asl.update(payload)
                conf = payload.get("confidence")
                if isinstance(conf, (int, float)):
                    self.confidence_window.append(float(conf))
                if payload.get("sent_text"):
                    self.asl_commits += 1

            elif topic.endswith("/eye/metrics"):
                previous_count = self.eye.get("blink_click_count", 0)
                self.eye.update(payload)
                current_count = self.eye.get("blink_click_count", 0)
                if isinstance(previous_count, int) and isinstance(current_count, int):
                    if current_count > previous_count:
                        self.eye_click_events += (current_count - previous_count)

            elif topic.endswith("/stt/partial"):
                self.stt["partial"] = payload.get("text", "")

            elif topic.endswith("/stt/final"):
                self.stt["last_final"] = payload.get("text", "")

            elif topic.endswith("/stt/status"):
                self.stt["status"] = payload.get("message", "")
                event = str(payload.get("event", "")).lower()
                if event == "error":
                    self.error_count += 1

            elif topic.endswith("/tts/status"):
                self.tts["status"] = payload.get("message", "")
                event = str(payload.get("event", "")).lower()
                if event == "error":
                    self.error_count += 1

    def snapshot(self):
        with self.lock:
            avg_conf = (
                sum(self.confidence_window) / len(self.confidence_window)
                if self.confidence_window
                else 0.0
            )
            avg_loop_ms = (
                sum(self.loop_ms_window) / len(self.loop_ms_window)
                if self.loop_ms_window
                else 0.0
            )

            return {
                "generated_at": time.time(),
                "dashboard_uptime_s": time.time() - self.started_at,
                "mqtt_connected": self.mqtt_connected,
                "last_seen_ts": self.last_seen_ts,
                "mode": self.mode,
                "mode_message": self.mode_message,
                "app_uptime_s": self.uptime_s,
                "asl": dict(self.asl),
                "eye": dict(self.eye),
                "stt": dict(self.stt),
                "tts": dict(self.tts),
                "analytics": {
                    "mode_switch_count": self.mode_switch_count,
                    "error_count": self.error_count,
                    "asl_commits": self.asl_commits,
                    "eye_click_events": self.eye_click_events,
                    "avg_confidence": avg_conf,
                    "avg_loop_ms": avg_loop_ms,
                    "confidence_window": list(self.confidence_window),
                },
                "raw_topics": dict(self.raw_topics),
            }


state = DashboardState()
app = Flask(__name__)


def _on_connect(client, userdata, flags, reason_code, properties):
    code = getattr(reason_code, "value", reason_code)
    state.mqtt_connected = code == 0
    client.subscribe(f"{MQTT_BASE_TOPIC}/#", qos=0)


def _on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    state.mqtt_connected = False


def _on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        payload = {"raw": msg.payload.decode("utf-8", errors="replace")}
    state.update_from_message(msg.topic, payload)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = _on_connect
client.on_disconnect = _on_disconnect
client.on_message = _on_message


@app.route("/")
def index():
    return render_template_string(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SignEdge MQTT Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f1116; color: #e6e6e6; }
    .wrap { padding: 16px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .card { background: #171a22; border: 1px solid #2b3240; border-radius: 10px; padding: 12px; }
    .title { font-size: 14px; color: #9aa4b2; margin-bottom: 10px; }
    .big { font-size: 20px; font-weight: bold; }
    .ok { color: #35d07f; }
    .warn { color: #f8c14d; }
    .bad { color: #ff6b6b; }
    .mono { font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    .bar { height: 10px; border-radius: 5px; background: #293244; overflow: hidden; }
    .bar > div { height: 100%; background: #35d07f; }
    .span2 { grid-column: span 2; }
    .span3 { grid-column: span 3; }
    ul { margin: 6px 0; padding-left: 18px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>SignEdge Dashboard (MQTT)</h2>
    <div class="grid">
      <div class="card">
        <div class="title">System</div>
        <div id="mode" class="big">-</div>
        <div id="modeMessage"></div>
        <div>MQTT: <span id="mqttStatus">-</span></div>
        <div>App uptime: <span id="appUptime">0</span>s</div>
      </div>

      <div class="card">
        <div class="title">ASL Live</div>
        <div>Label: <b id="aslLabel">-</b></div>
        <div>Confidence: <span id="aslConfidence">0%</span></div>
        <div class="bar"><div id="aslConfBar" style="width:0%"></div></div>
        <div>Hold progress: <span id="aslHold">0%</span></div>
        <div>Top-3: <span id="aslTop3">-</span></div>
      </div>

      <div class="card">
        <div class="title">Eye Live</div>
        <div>State: <b id="eyeState">-</b></div>
        <div>Cursor: <span id="eyeCursor">-</span></div>
        <div>Hovered key: <span id="eyeHovered">-</span></div>
        <div>Blink/click count: <span id="eyeClicks">0</span></div>
      </div>

      <div class="card span2">
        <div class="title">Speech + Transcript Signals</div>
        <div>STT partial: <span id="sttPartial">-</span></div>
        <div>STT final: <span id="sttFinal">-</span></div>
        <div>STT status: <span id="sttStatus">-</span></div>
        <div>TTS status: <span id="ttsStatus">-</span></div>
      </div>

      <div class="card">
        <div class="title">Analytics</div>
        <ul>
          <li>Mode switches: <span id="modeSwitches">0</span></li>
          <li>ASL commits: <span id="aslCommits">0</span></li>
          <li>Eye click events: <span id="eyeEvents">0</span></li>
          <li>Errors: <span id="errorCount">0</span></li>
          <li>Avg confidence: <span id="avgConf">0%</span></li>
          <li>Avg loop time: <span id="avgLoop">0</span> ms</li>
        </ul>
      </div>

      <div class="card span3">
        <div class="title">Raw MQTT Payloads (latest by topic)</div>
        <div id="rawTopics" class="mono">waiting...</div>
      </div>
    </div>
  </div>

  <script>
    function pct(v) {
      return Math.max(0, Math.min(100, Math.round((Number(v) || 0) * 100)));
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    async function refresh() {
      const res = await fetch('/api/state');
      const data = await res.json();

      setText('mode', data.mode || '-');
      setText('modeMessage', data.mode_message || '-');
      setText('mqttStatus', data.mqtt_connected ? 'connected' : 'disconnected');
      setText('appUptime', Math.round(Number(data.app_uptime_s || 0)));

      const asl = data.asl || {};
      const confPct = pct(asl.confidence || 0);
      setText('aslLabel', asl.label || '-');
      setText('aslConfidence', confPct + '%');
      document.getElementById('aslConfBar').style.width = confPct + '%';
      setText('aslHold', pct(asl.hold_progress || 0) + '%');
      setText('aslTop3', JSON.stringify(asl.top3 || []));

      const eye = data.eye || {};
      setText('eyeState', eye.state || '-');
      setText('eyeCursor', `${Math.round(Number(eye.cursor_x || 0))}, ${Math.round(Number(eye.cursor_y || 0))}`);
      setText('eyeHovered', eye.hovered_key || '-');
      setText('eyeClicks', eye.blink_click_count || 0);

      const stt = data.stt || {};
      const tts = data.tts || {};
      setText('sttPartial', stt.partial || '-');
      setText('sttFinal', stt.last_final || '-');
      setText('sttStatus', stt.status || '-');
      setText('ttsStatus', tts.status || '-');

      const analytics = data.analytics || {};
      setText('modeSwitches', analytics.mode_switch_count || 0);
      setText('aslCommits', analytics.asl_commits || 0);
      setText('eyeEvents', analytics.eye_click_events || 0);
      setText('errorCount', analytics.error_count || 0);
      setText('avgConf', pct(analytics.avg_confidence || 0) + '%');
      setText('avgLoop', Math.round(Number(analytics.avg_loop_ms || 0)));

      const raw = data.raw_topics || {};
      document.getElementById('rawTopics').textContent = JSON.stringify(raw, null, 2);
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
        """
    )


@app.route("/api/state")
def api_state():
    return jsonify(state.snapshot())


def main():
    print(f"[Dashboard] MQTT broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    print(f"[Dashboard] Topic: {MQTT_BASE_TOPIC}/#")
    print(f"[Dashboard] URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=30)
        client.loop_start()
    except Exception as exc:
        print(f"[Dashboard] MQTT connection failed: {exc}")

    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()

# SignEdge

SignEdge is a privacy-first, fully offline communication aid designed for high-stakes environments such as hospitals and public services.

This project addresses communication barriers faced by individuals with hearing, speech, and motor impairments in high-stakes environments such as hospitals and public services. Existing assistive tools are often single-modality and rely on cloud processing, resulting in privacy risks, high latency, and limited reliability. To overcome these challenges, SignEdge proposes a unified, offline edge-based system for real-time, inclusive communication.

## Project Goal

Build a privacy-first, fully offline, low-latency edge device on Raspberry Pi 5 for real-time bidirectional communication in high-stakes environments.

Two core translation directions:

- Hearing-to-Deaf: spoken English to displayed text
- Deaf-to-Hearing: ASL fingerspelling (A-Z static letters) to spoken English

## Core Constraints and Design Principles

- End-to-end latency target: below 100 ms (measured loop around 51 ms)
- Strict privacy: raw video and raw audio never leave the device
- Offline-first design: no cloud dependency and no internet requirement for runtime
- Full on-device execution for reliability in network-dead zones
- Deployment must remain within Raspberry Pi 5 thermal and resource limits

## Hardware Platform

To meet the latency, privacy, and offline constraints, hardware options were evaluated before selecting a final platform.

### Evaluation Criteria

- Enough RAM for concurrent audio and vision pipelines
- Stable real-time inference under continuous load
- Software ecosystem maturity and tooling support
- Low integration risk within project timeline
- Full compatibility with offline deployment

### Evaluated Options

1. Raspberry Pi Zero 2 W

- 512 MB RAM is too constrained after OS/runtime overhead
- Insufficient headroom for model loading, frame buffering, and continuous inference
- Estimated 1 to 3 FPS under workload, not suitable for sub-100 ms target
- High risk of instability during prolonged operation

2. Milk-V Duo (with NPU)

- NPU is promising in theory but memory remains constrained in 512 MB configuration
- Requires conversion into proprietary SOPHON model format
- Immature tooling and limited operator support increase debugging complexity
- Unsupported layers can fall back to CPU, reducing real-world performance

### Final Decision: Raspberry Pi 5 (8 GB RAM)

- 8 GB RAM provides practical headroom for OS plus concurrent audio and vision pipelines
- Stable measured behavior aligns with project target (vision loop around 30 to 40 ms, overall loop around 51 ms)
- Mature ecosystem with strong support for MediaPipe and Python-based ML tooling
- Lower development risk and better maintainability
- Better scalability for future model, sensor, and feature upgrades

### Optional Hardware Enhancements

- Hailo AI accelerator (if available) for additional inference offload
- CSI ribbon camera module for lower latency and more stable frame timing than USB cameras

## What This App Does

The Combined app provides a single runtime with three coordinated modes:

- `BOOT_GATE`: startup gate that decides user path (ASL or Eye mode)
- `ASL`: hand-sign recognition and sentence composition for speech output
- `EYE`: gaze + blink keyboard for speech output with optional word completion

It also runs shared services:

- local speech-to-text (Moonshine)
- local text-to-speech (Piper, with espeak fallback)
- transcript panel in the UI
- MQTT telemetry publishing for monitoring

## High-Level Software Architecture

All processing is local on-device.

```text
[Webcam + Microphone]
          |
          v
 UnifiedSignEdgeApp (Combined/app.py)
          |
   +------+------------------------------+
   |                                     |
   v                                     v
StartupGate                      Shared Core Services
(Combined/modes/startup_gate.py) (Combined/core/*.py)
   |                             - Combined/core/audio_stt.py
   |                             - Combined/core/audio_tts.py
   +--> selects mode             - Combined/core/transcript_store.py
         (ASL / EYE)             - Combined/core/mqtt_publisher.py
          |
          v
   +------+-----------------------------+
   |                                    |
   v                                    v
ASL Mode                         Eye Mode
(Combined/modes/asl_mode.py)     (Combined/modes/eye_mode.py)
   |                                    |
MediaPipe Hands + TFLite MLP      FaceMesh + iris + blink dwell
   |                                    |
Stable text composer + send zone   Virtual keyboard + trie
   |                                + optional local LLM completion
   +---------------+----------------+
                   |
                   v
         OfflineSpeaker (TTS output)
                   |
                   v
                Speakers

Parallel observability path:
UnifiedSignEdgeApp -> MQTT topics -> Combined/dashboard_web.py (Flask dashboard)
```

### Implementation Notes

- Speech-to-text runs in a separate thread so it does not block the visual loop
- MediaPipe landmark extraction is around 15 ms
- MLP classification is around 10 ms
- Typical visual loop is around 30 to 40 ms
- Measured overall system loop is around 51 ms
- Target remains below 100 ms end-to-end latency

## Project Layout

```text
Combined/
  app.py
  dashboard_web.py
  requirements.txt
  assets/
    asl_mlp_int8.tflite
    label_map.json
    scaler_mean.npy
    scaler_scale.npy
    en_US-amy-medium.onnx
    en_US-amy-medium.onnx.json
    frequencies.txt
    moonshine-small-streaming/
      encoder.ort
      decoder_kv.ort
      cross_kv.ort
      adapter.ort
      frontend.ort
      streaming_config.json
      tokenizer.bin
  core/
    config.py
    audio_stt.py
    audio_tts.py
    mqtt_publisher.py
    transcript_store.py
  modes/
    startup_gate.py
    asl_mode.py
    eye_mode.py
  ui/
    chat_panel.py
  profiling_scripts/
    start_monitors.sh
    stop_and_summarize.sh
```

## Prerequisites

- Raspberry Pi 5 (8 GB recommended)
- Debian/Ubuntu-based Linux with Python 3.11+
- Webcam and microphone
- Speaker output
- Active cooling (fan/heatsink) for sustained inference

## Installation

Run from the `Combined` directory.

```bash
cd Combined
python3 -m venv .venv
source .venv/bin/activate

sudo apt update
sudo apt install -y build-essential cmake espeak
sudo apt install -y linux-perf sysstat moreutils jq gawk bc procps util-linux mosquitto-clients

python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional: local LLM support used by Eye mode completion
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python==0.3.16
```

### Dependency Note (important)

- Keep `numpy<2` when using `tflite-runtime==2.14.0`.
- If you manually reinstall packages, ensure NumPy remains below 2.

## Run

### 1) Main Unified App

```bash
cd Combined
python app.py
```

### 2) MQTT Dashboard (optional)

In a second terminal:

```bash
cd Combined
python dashboard_web.py
```

Open the dashboard URL printed at startup (defaults configured in `Combined/core/config.py`).

## Configuration Reference

All runtime settings are in `Combined/core/config.py`.

Key parameters:

- `APP_FPS = 30`
- `BOOT_GATE_SECONDS = 10.0`
- `ASL_SMOOTH_WINDOW = 7`
- `ASL_CONFIDENCE_THRESHOLD = 0.65`
- `ASL_HOLD_SECONDS = 0.85`
- `ASL_SEND_DWELL_SECONDS = 1.75`
- `EYE_EAR_THRESHOLD = 0.2`
- `EYE_DWELL_TIME = 0.6`
- `EYE_SMOOTHING_FRAMES = 20`
- `EYE_CURSOR_ALPHA = 0.022`
- `MQTT_ENABLED = True`
- `MQTT_BROKER_HOST = 192.168.137.197` (update for your network)
- `DASHBOARD_PORT = 5050`

## Model and Approach Justification

Vision path (ASL fingerspelling):

- Selected: MediaPipe Hands plus a tiny MLP on 21 landmarks only
- Why: low latency, low memory footprint, and better privacy posture because raw pixels can be discarded after landmark extraction
- Rejected alternatives for this use case: YOLOv8 and video-heavy models (for example 3D-CNN or ResNet-based video classification) due to higher latency and larger buffering/memory overhead

Audio path:

- Uses a lightweight offline speech-to-text approach suitable for on-device execution
- Keeps runtime independent from cloud APIs and external network availability

## Resource Budget

- Total estimated RAM usage: about 1.0 GB
- Multi-threading model:
  - 1 main thread
  - 1 TTS thread
  - 1 MQTT thread
  - 1+ STT internal thread(s)
  - 0 or 1 LLM thread (only if Eye LLM is enabled)
- Active cooling remains mandatory to avoid thermal throttling during sustained inference

## Risks and Mitigations

- Thermal throttling risk:
  - Mitigate with active cooling and bounded runtime workloads
- Privacy leakage risk:
  - Mitigate with strict local processing and no cloud path in runtime
- Detection quality risk under poor lighting/framing:
  - Mitigate with quality camera setup, user prompts, and frame skipping/recovery behavior

## Required Assets and Behavior

The app expects assets in `Combined/assets/`.

- ASL model assets missing (`Combined/assets/asl_mlp_int8.tflite`, scaler files, `Combined/assets/label_map.json`):
  - ASL mode starts with an on-screen model error.
- Moonshine files missing from `Combined/assets/moonshine-small-streaming/`:
  - STT is disabled and reports a status/error event.
- Piper ONNX voice missing (`Combined/assets/en_US-amy-medium.onnx`):
  - TTS falls back to `espeak`/`espeak-ng` if available.

## MQTT Telemetry Topics

Published under `signedge/local` by default:

- `system/start`, `system/stop`, `system/heartbeat`
- `mode/switch`
- `asl/metrics`, `eye/metrics`
- `stt/partial`, `stt/final`, `stt/status`
- `tts/status`

## Eye Tracking Tuning Notes

For steadier cursor behavior in Eye mode:

- Decrease `EYE_CURSOR_ALPHA` for heavier/slower cursor motion.
- Increase `EYE_SMOOTHING_FRAMES` to reduce jitter.
- Keep camera near face level and reduce tilt changes during calibration.

## Profiling Scripts

From the repository root:

```bash
bash Combined/profiling_scripts/start_monitors.sh --app-match "python.*app.py"
bash Combined/profiling_scripts/stop_and_summarize.sh
```

Outputs include:

- loop latency summary from MQTT heartbeat data
- CPU/power/thermal summaries (`vcgencmd`, `pidstat`, `mpstat`, `vmstat`)

## Troubleshooting

- App shows camera unavailable:
  - Check camera device index in `Combined/core/config.py` (`CAMERA_INDEX`).
- Dashboard does not update:
  - Verify `MQTT_BROKER_HOST` and network reachability.
- No STT activity:
  - Confirm all Moonshine model files are present in `Combined/assets/moonshine-small-streaming/`.
- No speech output:
  - Ensure Piper ONNX exists or install `espeak-ng`; also verify playback tools (`aplay` preferred).
- Eye mode has no LLM suggestions:
  - This is expected if `llama-cpp-python` or the configured model file is unavailable.

## Privacy and Offline Operation

- No cloud dependency in the runtime path.
- Audio and video are processed locally on-device.
- Telemetry is local MQTT within your own network setup.

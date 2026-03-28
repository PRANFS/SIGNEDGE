#!/usr/bin/env bash
set -euo pipefail

MQTT_HOST_DEFAULT="192.168.137.197"
MQTT_TOPIC_BASE_DEFAULT="signedge/local"
LOG_BASE_DEFAULT="${HOME}/profiling_logs"
APP_MATCH_DEFAULT="python.*app.py"
SAMPLE_SEC_DEFAULT="1"

RUN_ID=""
APP_PID=""
LOG_BASE="${LOG_BASE_DEFAULT}"
APP_MATCH="${APP_MATCH_DEFAULT}"
SAMPLE_SEC="${SAMPLE_SEC_DEFAULT}"
MQTT_HOST="${MQTT_HOST:-$MQTT_HOST_DEFAULT}"
MQTT_TOPIC_BASE="${MQTT_TOPIC_BASE:-$MQTT_TOPIC_BASE_DEFAULT}"
PERF_ENABLE="${PERF_ENABLE:-0}"

usage() {
  cat <<'EOF'
Usage: start_monitors.sh [options]

Options:
  --run-id <id>             Reuse a run id instead of creating a new one
  --app-pid <pid>           Attach monitors to this process id
  --app-match <regex>       Regex used by pgrep if --app-pid is omitted
  --log-base <dir>          Base log directory (default: ~/profiling_logs)
  --mqtt-host <host>        MQTT broker host
  --mqtt-topic-base <topic> MQTT base topic (default: signedge/local)
  --sample-sec <sec>        Sampling interval for vcgencmd loop (default: 1)
  --perf                    Enable live perf stat (requires sudo -n)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --app-pid)
      APP_PID="${2:-}"
      shift 2
      ;;
    --app-match)
      APP_MATCH="${2:-}"
      shift 2
      ;;
    --log-base)
      LOG_BASE="${2:-}"
      shift 2
      ;;
    --mqtt-host)
      MQTT_HOST="${2:-}"
      shift 2
      ;;
    --mqtt-topic-base)
      MQTT_TOPIC_BASE="${2:-}"
      shift 2
      ;;
    --sample-sec)
      SAMPLE_SEC="${2:-}"
      shift 2
      ;;
    --perf)
      PERF_ENABLE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

LOGDIR="${LOG_BASE}/${RUN_ID}"
mkdir -p "$LOGDIR"

if [[ -z "$APP_PID" ]]; then
  APP_PID="$(pgrep -n -f "$APP_MATCH" || true)"
fi

if [[ -z "$APP_PID" ]]; then
  echo "Could not find app process. Use --app-pid." >&2
  exit 1
fi

if ! kill -0 "$APP_PID" 2>/dev/null; then
  echo "App PID $APP_PID is not running." >&2
  exit 1
fi

PIDS_FILE="$LOGDIR/monitor.pids"
: > "$PIDS_FILE"

cat > "$LOGDIR/run.env" <<EOF
RUN_ID="$RUN_ID"
LOGDIR="$LOGDIR"
APP_PID="$APP_PID"
MQTT_HOST="$MQTT_HOST"
MQTT_TOPIC_BASE="$MQTT_TOPIC_BASE"
SAMPLE_SEC="$SAMPLE_SEC"
EOF

echo "ts,temp_c,volt_v,arm_hz,throttled_hex,cpu_pct,rss_kb,power_est_w" > "$LOGDIR/vcgencmd_power.csv"
(
  while kill -0 "$APP_PID" 2>/dev/null; do
    TS="$(date +%s.%N)"

    TEMP=""
    if command -v vcgencmd >/dev/null 2>&1; then
      TEMP="$(vcgencmd measure_temp | sed -E 's/temp=([0-9.]+).*/\1/' || true)"
      VOLT="$(vcgencmd measure_volts core | sed -E 's/volt=([0-9.]+).*/\1/' || true)"
      FREQ="$(vcgencmd measure_clock arm | cut -d= -f2 || true)"
      THR="$(vcgencmd get_throttled | cut -d= -f2 || true)"
    else
      TEMP=""
      VOLT=""
      FREQ=""
      THR=""
    fi

    CPU="$(ps -p "$APP_PID" -o %cpu= 2>/dev/null | tr -d ' ' || true)"
    RSS="$(ps -p "$APP_PID" -o rss= 2>/dev/null | tr -d ' ' || true)"

    if [[ -n "$CPU" && -n "$FREQ" && -n "$VOLT" ]]; then
      PEST="$(awk -v u="$CPU" -v f="$FREQ" -v v="$VOLT" 'BEGIN{printf "%.3f", 2.5 + 3.2*(u/100)*(f/1000000000)*(v*v)}')"
    else
      PEST=""
    fi

    echo "$TS,$TEMP,$VOLT,$FREQ,$THR,$CPU,$RSS,$PEST" >> "$LOGDIR/vcgencmd_power.csv"
    sleep "$SAMPLE_SEC"
  done
) > "$LOGDIR/vcgencmd_sampler.log" 2>&1 &
echo "vcgencmd_sampler:$!" >> "$PIDS_FILE"

(pidstat -h -rudw -p "$APP_PID" 1 > "$LOGDIR/pidstat.log" 2>&1) &
echo "pidstat:$!" >> "$PIDS_FILE"

(mpstat -P ALL 1 > "$LOGDIR/mpstat.log" 2>&1) &
echo "mpstat:$!" >> "$PIDS_FILE"

(vmstat 1 > "$LOGDIR/vmstat.log" 2>&1) &
echo "vmstat:$!" >> "$PIDS_FILE"

(mosquitto_sub -h "$MQTT_HOST" -t "$MQTT_TOPIC_BASE/#" -v | ts '%s.%N' > "$LOGDIR/mqtt.log") &
echo "mqtt:$!" >> "$PIDS_FILE"

if [[ "$PERF_ENABLE" == "1" ]]; then
  if command -v perf >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    (
      sudo -n perf stat -I 1000 -p "$APP_PID" \
      -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations,minor-faults,major-faults \
      -- sleep 31536000 2> "$LOGDIR/perf_stat_live.txt"
    ) &
    echo "perf_stat:$!" >> "$PIDS_FILE"
  else
    echo "Skipping perf live monitor: needs perf and passwordless sudo." | tee -a "$LOGDIR/warnings.log"
  fi
fi

echo "RUN_ID=$RUN_ID"
echo "LOGDIR=$LOGDIR"
echo "APP_PID=$APP_PID"
echo "Started monitors listed in: $PIDS_FILE"

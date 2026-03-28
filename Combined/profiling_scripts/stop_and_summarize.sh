#!/usr/bin/env bash
set -euo pipefail

LOG_BASE_DEFAULT="${HOME}/profiling_logs"
LOG_BASE="${LOG_BASE_DEFAULT}"
RUN_ID=""

usage() {
  cat <<'EOF'
Usage: stop_and_summarize.sh [options]

Options:
  --run-id <id>      Run id to stop/summarize. Default: latest under --log-base
  --log-base <dir>   Base log directory (default: ~/profiling_logs)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --log-base)
      LOG_BASE="${2:-}"
      shift 2
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
  latest_dir="$(ls -1dt "$LOG_BASE"/* 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest_dir" ]]; then
    echo "No runs found in $LOG_BASE" >&2
    exit 1
  fi
  RUN_ID="$(basename "$latest_dir")"
fi

LOGDIR="$LOG_BASE/$RUN_ID"
if [[ ! -d "$LOGDIR" ]]; then
  echo "Run directory does not exist: $LOGDIR" >&2
  exit 1
fi

RUN_ENV="$LOGDIR/run.env"
if [[ -f "$RUN_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RUN_ENV"
fi

MQTT_TOPIC_BASE="${MQTT_TOPIC_BASE:-signedge/local}"

PIDS_FILE="$LOGDIR/monitor.pids"
if [[ -f "$PIDS_FILE" ]]; then
  while IFS=: read -r _name pid; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done < "$PIDS_FILE"

  sleep 1

  while IFS=: read -r _name pid; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done < "$PIDS_FILE"
fi

LOOP_FILE="$LOGDIR/loop_ms.txt"
grep "$MQTT_TOPIC_BASE/system/heartbeat" "$LOGDIR/mqtt.log" 2>/dev/null \
  | sed -nE 's/.*"loop_ms":[[:space:]]*([0-9.]+).*/\1/p' > "$LOOP_FILE" || true

LAT_SUMMARY="$LOGDIR/latency_summary.txt"
if [[ -s "$LOOP_FILE" ]]; then
  python3 - "$LOOP_FILE" > "$LAT_SUMMARY" <<'PY'
import math
import statistics
import sys

vals = [float(x.strip()) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
vals.sort()

def pct(p):
    k = (len(vals) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)

deadline = 33.333
miss = sum(v > deadline for v in vals)
print(f"samples={len(vals)}")
print(f"mean_ms={statistics.fmean(vals):.3f}")
print(f"p50_ms={pct(50):.3f}")
print(f"p95_ms={pct(95):.3f}")
print(f"p99_ms={pct(99):.3f}")
print(f"max_ms={max(vals):.3f}")
print(f"deadline_miss_rate_pct={100 * miss / len(vals):.2f}")
PY
else
  echo "No loop_ms samples found in mqtt.log" > "$LAT_SUMMARY"
fi

POWER_SUMMARY="$LOGDIR/power_summary.txt"
if [[ -f "$LOGDIR/vcgencmd_power.csv" ]]; then
  awk -F, '
  NR==2 {tmin=$2; tmax=$2}
  NR>1 {
    if ($2 != "") {t += $2; if ($2 < tmin) tmin = $2; if ($2 > tmax) tmax = $2}
    if ($3 != "") v += $3
    if ($6 != "") c += $6
    if ($8 != "") p += $8
    if ($5 != "" && $5 != "0x0") bad++
    n++
  }
  END {
    if (n == 0) {
      print "No vcgencmd samples";
      exit
    }
    printf "samples=%d\n", n
    if (t > 0) printf "temp_avg_c=%.2f\n", t/n
    if (tmin != "") printf "temp_min_c=%.2f\n", tmin
    if (tmax != "") printf "temp_max_c=%.2f\n", tmax
    if (v > 0) printf "volt_avg_v=%.3f\n", v/n
    if (c > 0) printf "cpu_avg_pct=%.2f\n", c/n
    if (p > 0) printf "power_est_avg_w=%.3f\n", p/n
    printf "throttle_events=%d\n", bad+0
  }
  ' "$LOGDIR/vcgencmd_power.csv" > "$POWER_SUMMARY"
else
  echo "No vcgencmd_power.csv found" > "$POWER_SUMMARY"
fi

if [[ -f "$LOGDIR/perf_120s.data" && command -v perf >/dev/null 2>&1 ]]; then
  if [[ ! -f "$LOGDIR/perf_report_120s.txt" ]]; then
    perf report --stdio -i "$LOGDIR/perf_120s.data" > "$LOGDIR/perf_report_120s.txt" 2>/dev/null || true
  fi
fi

echo "Run summary complete"
echo "LOGDIR=$LOGDIR"
echo "Latency summary: $LAT_SUMMARY"
echo "Power summary:   $POWER_SUMMARY"

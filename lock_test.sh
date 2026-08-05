#!/usr/bin/env bash
# locktest.sh — spawn N concurrent flock workers on one test file.
# MODE: x = exclusive blocking (default), n = non-blocking, s = shared
set -euo pipefail

LOCKFILE="${1:-/tmp/locktest.lock}"
WORKERS="${2:-16}"
MODE="${3:-x}"
HOLD="${4:-0.2}"

: > "$LOCKFILE"
start=$(date +%s.%N)
now() { echo "$(date +%s.%N) - $start" | bc; }

worker() {
    local id="$1"
    exec 200>"$LOCKFILE"
    case "$MODE" in
        x) flock -x 200 ;;                                    # exclusive, waits
        s) flock -s 200 ;;                                    # shared, waits
        n) flock -nx 200 || { printf 'worker %2d: BUSY (skipped)\n' "$id"; exit 0; } ;;
        *) echo "bad mode: $MODE (use x|n|s)" >&2; exit 2 ;;
    esac
    printf '[%6.3fs] worker %2d: ACQUIRED (%s)\n' "$(now)" "$id" "$MODE"
    sleep "$HOLD"
    printf '[%6.3fs] worker %2d: releasing\n' "$(now)" "$id"
    flock -u 200
    exec 200>&-
}

echo "Spawning $WORKERS workers on $LOCKFILE  mode=$MODE  hold=${HOLD}s"
for i in $(seq 1 "$WORKERS"); do worker "$i" & done
wait
echo "Done."

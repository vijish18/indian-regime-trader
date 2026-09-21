#!/usr/bin/env bash
# Run from an immutable checkout under systemd. Never removes existing results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${IRT_DATA_ROOT:?Set the absolute immutable dataset path}"
: "${IRT_RUN_ROOT:?Set the absolute output directory}"
mkdir -p "$IRT_RUN_ROOT"
exec 9>"$IRT_RUN_ROOT/batch.lock"
flock -n 9 || { echo "Another batch owns this run directory"; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run_one() {
    local strategy="$1" attempt code
    local target="$IRT_RUN_ROOT/$strategy"
    mkdir -p "$target"
    for attempt in 1 2 3; do
        local resume=()
        if [[ -f "$target/series/run.manifest.json" ]]; then resume=(--resume); fi
        echo "$(date -u +%FT%TZ) $strategy attempt $attempt" >> "$target/run.log"
        if "$ROOT/.venv/bin/python" -u scripts/run_walk_forward.py \
            --from 2015-01-01 --to 2026-09-21 --initial-equity 100000 \
            --strategy "$strategy" --frame-cache 1000 \
            --state-dir "$target/state" --series-dir "$target/series" \
            --json-out "$target/reports.json" "${resume[@]}" >> "$target/run.log" 2>&1; then
            date -u +%FT%TZ > "$target/completed.txt"
            return 0
        else
            code=$?
            echo "$(date -u +%FT%TZ) failed with exit $code" >> "$target/run.log"
        fi
        if [[ "$attempt" -lt 3 ]]; then sleep 30; fi
    done
    return "$code"
}

# Four workers on the existing four-vCPU machine; the fifth queues behind HMM.
(run_one hmm; run_one shuffled_regime_control) & p1=$!
run_one buy_and_hold & p2=$!
run_one rolling_volatility & p3=$!
run_one moving_average_trend & p4=$!
result=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
    wait "$pid" || result=1
done
exit "$result"

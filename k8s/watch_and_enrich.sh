#!/usr/bin/env bash
# Continuous enrichment watcher for the streaming (self-looping) batch mode.
#
# Unlike run_loop.sh's lock-step "wait for all N slices, then enrich" pattern,
# this polls each slice's S3 output independently and re-runs enrichment as
# soon as ANY slice's output changes — a fast slice never waits on a slow one.
#
# Usage:
#   bash k8s/watch_and_enrich.sh \
#       --slices 3 --job-suffix -pmc-full \
#       --output-dir k8s/output/rev_pmc_streaming \
#       --seed-ontology data_gatherer/config/open_bio_data_repos.json \
#       [--poll-interval 60] [--plot]
#
# Exits once every slice has written its _DONE marker (k8s_processor.py writes
# this to S3 when a --loop-until-done run exhausts its assigned partition),
# after running one final enrichment pass on everything.

set -euo pipefail

if [[ -f ".env" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    export ANTHROPIC_API_KEY=$(grep -E "^ANTHROPIC_API_KEY=" .env | cut -d= -f2-)
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set — set it or add it to .env}"

N_SLICES=3
JOB_SUFFIX=""
OUTPUT_BASE="k8s/output/rev_pmc_streaming"
NAMESPACE="data-gatherer"
SEED_ONTOLOGY="data_gatherer/config/open_bio_data_repos_seed.json"
POLL_INTERVAL=60
PLOT_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --slices)         N_SLICES="$2";       shift 2 ;;
        --job-suffix)     JOB_SUFFIX="$2";      shift 2 ;;
        --output-dir)     OUTPUT_BASE="$2";     shift 2 ;;
        --seed-ontology)  SEED_ONTOLOGY="$2";   shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2";   shift 2 ;;
        --plot)           PLOT_FLAG="--plot";   shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE"
BUCKET=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
             -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)

CURRENT_ONTOLOGY="$SEED_ONTOLOGY"
SNAPSHOT=0
LAST_ETAGS=""

echo "[watch] Watching ${N_SLICES} slice(s) (suffix '${JOB_SUFFIX}') in s3://${BUCKET}, polling every ${POLL_INTERVAL}s"

slice_prefix() { echo "slice_${1}${JOB_SUFFIX}"; }

etag_of() {
    # Empty string if the object doesn't exist yet — treated as "no change" until it appears
    aws s3api head-object --bucket "$BUCKET" --key "$1" --query ETag --output text 2>/dev/null || echo ""
}

all_done() {
    for i in $(seq 1 "$N_SLICES"); do
        aws s3api head-object --bucket "$BUCKET" --key "$(slice_prefix "$i")/_DONE" &>/dev/null || return 1
    done
    return 0
}

run_enrichment_pass() {
    SNAPSHOT=$((SNAPSHOT + 1))
    local snap_dir="$OUTPUT_BASE/snapshot_${SNAPSHOT}"
    mkdir -p "$snap_dir"

    echo "[watch] Change detected — downloading + merging all ${N_SLICES} slice(s) for snapshot ${SNAPSHOT}..."
    local csv_list=""
    for i in $(seq 1 "$N_SLICES"); do
        local key="$(slice_prefix "$i")/dataset_citations.csv"
        local dest="$snap_dir/slice_${i}_citations.csv"
        if aws s3 cp "s3://${BUCKET}/${key}" "$dest" 2>/dev/null; then
            csv_list="$csv_list $dest"
        else
            echo "[watch]  [warn] slice_${i} has no output yet — skipping"
        fi
    done

    if [[ -z "$csv_list" ]]; then
        echo "[watch]  nothing to merge yet, skipping this pass"
        return
    fi

    python3 - <<PYEOF
import pandas as pd
files = "$csv_list".split()
dfs = [pd.read_csv(f) for f in files if f]
merged = pd.concat(dfs, ignore_index=True).drop_duplicates(
    subset=["dataset_identifier", "source_url"], keep="first"
)
merged.to_csv("$snap_dir/dataset_citations.csv", index=False)
print(f"  merged {len(merged)} rows from {len(files)} slice(s) -> $snap_dir/dataset_citations.csv")
PYEOF

    local new_ontology="$snap_dir/open_bio_data_repos_v${SNAPSHOT}.json"
    echo "[watch] Running enrichment (pipeline=group) for snapshot ${SNAPSHOT}..."
    python3 scripts/enrich_ontology.py \
        --citations "$snap_dir/dataset_citations.csv" \
        --current-ontology "$CURRENT_ONTOLOGY" \
        --output "$new_ontology" \
        --log "$snap_dir/enrich.log" \
        --pipeline group \
        $PLOT_FLAG

    echo "[watch] Uploading enriched ontology to S3..."
    aws s3 cp "$new_ontology" "s3://${BUCKET}/ontology/open_bio_data_repos.json"
    CURRENT_ONTOLOGY="$new_ontology"
    echo "[watch] Snapshot ${SNAPSHOT} done. Ontology: $CURRENT_ONTOLOGY"
}

# Upload the seed ontology once, so freshly-submitted slice jobs have something to start from
aws s3 cp "$SEED_ONTOLOGY" "s3://${BUCKET}/ontology/open_bio_data_repos.json"

while true; do
    CHANGED=0
    NEW_ETAGS=""
    for i in $(seq 1 "$N_SLICES"); do
        tag=$(etag_of "$(slice_prefix "$i")/dataset_citations.csv")
        NEW_ETAGS="${NEW_ETAGS}${i}:${tag};"
    done

    if [[ "$NEW_ETAGS" != "$LAST_ETAGS" ]]; then
        CHANGED=1
    fi
    LAST_ETAGS="$NEW_ETAGS"

    if [[ "$CHANGED" -eq 1 ]]; then
        run_enrichment_pass
    fi

    if all_done; then
        echo "[watch] All ${N_SLICES} slice(s) report _DONE. Running one final enrichment pass..."
        run_enrichment_pass
        echo "[watch] Done. Final ontology: $CURRENT_ONTOLOGY"
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done

#!/usr/bin/env bash
# Automated multi-GPU ontology enrichment loop.
#
# Each iteration:
#   1. Submit N parallel k8s jobs (one per GPU / input slice)
#   2. Wait for ALL jobs to complete
#   3. Copy + merge all slice output CSVs locally
#   4. Run enrichment script on merged data
#   5. Push updated ontology back to PVC
#   6. Repeat
#
# Usage:
#   bash k8s/run_loop.sh --iterations 2 --gpus 4 --max-articles-per-slice 2500 \
#                        --input article_ids_REV_pmc.csv --output-dir k8s/output \
#                        [--clean] [--cumulative] [--plot]
#
#   --cumulative  Pass all previous iterations' CSVs to enrich_ontology.py so groups
#                 are sized by cumulative evidence. Default: marginal (current iter only).
#
# Prerequisites:
#   - kubectl configured and pointing at the right cluster
#   - ANTHROPIC_API_KEY set
#   - Seed ontology already on PVC at /data/ontology/open_bio_data_repos.json

set -euo pipefail

# --- Defaults ---
ITERATIONS=2
N_GPUS=1
MAX_ARTICLES_PER_SLICE=250
INPUT_FILE="article_ids_test_500.csv"
OUTPUT_BASE="k8s/output"
NAMESPACE="data-gatherer"
JOB_TEMPLATE="k8s/deterministic-batch-job-template.yaml"
SEED_ONTOLOGY="data_gatherer/config/open_bio_data_repos_seed.json"
PVC_READER_POD="pvc-reader"
PVC_ONTOLOGY="/data/ontology/open_bio_data_repos.json"
JOB_TIMEOUT="7200s"
CLEAN=0
CUMULATIVE=0
PLOT_FLAG=""
NODE_NAME=""

# --- Args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)              ITERATIONS="$2";              shift 2 ;;
        --gpus)                    N_GPUS="$2";                  shift 2 ;;
        --max-articles-per-slice)  MAX_ARTICLES_PER_SLICE="$2";  shift 2 ;;
        --input)                   INPUT_FILE="$2";              shift 2 ;;
        --output-dir)              OUTPUT_BASE="$2";             shift 2 ;;
        --node)                    NODE_NAME="$2";               shift 2 ;;
        --clean)                   CLEAN=1;                      shift   ;;
        --cumulative)              CUMULATIVE=1;                 shift   ;;
        --plot)                    PLOT_FLAG="--plot";           shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export NODE_NAME  # may be empty; envsubst leaves the block if blank

# --- Helpers ---
pvc_reader_ready() {
    kubectl get pod "$PVC_READER_POD" -n "$NAMESPACE" \
        --no-headers -o custom-columns=":status.phase" 2>/dev/null | grep -q "Running"
}

ensure_pvc_reader() {
    if ! pvc_reader_ready; then
        echo "[loop] Launching pvc-reader pod..."
        kubectl run "$PVC_READER_POD" -n "$NAMESPACE" --image=busybox --restart=Never \
            --overrides='{
              "spec":{
                "volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-gatherer-pvc"}}],
                "containers":[{"name":"pvc-reader","image":"busybox","command":["sleep","3600"],
                  "volumeMounts":[{"mountPath":"/data","name":"data"}]}]
              }
            }' 2>/dev/null || true
        kubectl wait pod/"$PVC_READER_POD" -n "$NAMESPACE" \
            --for=condition=Ready --timeout=60s
    fi
}

split_and_upload_input() {
    echo "[loop] Splitting $INPUT_FILE into $N_GPUS slices and uploading to S3..."
    local bucket
    bucket=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
                 -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)

    python3 - <<PYEOF
import pandas as pd, math

df = pd.read_csv("k8s/input/$INPUT_FILE")

if "pmcid" not in df.columns and "citing_publication_link" in df.columns:
    df["pmcid"] = df["citing_publication_link"].str.extract(r"(PMC\d+)")

pmcids = df["pmcid"].dropna().reset_index(drop=True)
n = $N_GPUS
size = math.ceil(len(pmcids) / n)
for i in range(n):
    slc = pmcids.iloc[i*size:(i+1)*size]
    path = f"k8s/input/slice_{i+1}.csv"
    slc.to_csv(path, index=False)
    print(f"  slice_{i+1}.csv: {len(slc)} articles")
PYEOF

    for i in $(seq 1 "$N_GPUS"); do
        aws s3 cp "k8s/input/slice_${i}.csv" "s3://${bucket}/input/slice_${i}.csv"
        echo "  uploaded slice_${i}.csv to S3"
    done
}

release_pvc_reader() {
    if kubectl get pod "$PVC_READER_POD" -n "$NAMESPACE" &>/dev/null; then
        echo "[loop] Deleting pvc-reader to release PVC (ReadWriteOnce)..."
        kubectl delete pod "$PVC_READER_POD" -n "$NAMESPACE" --ignore-not-found
        kubectl wait pod/"$PVC_READER_POD" -n "$NAMESPACE" \
            --for=delete --timeout=60s 2>/dev/null || true
    fi
}

delete_slice_jobs() {
    for i in $(seq 1 "$N_GPUS"); do
        kubectl delete job "data-gatherer-batch-slice-${i}" \
            -n "$NAMESPACE" --ignore-not-found 2>/dev/null
    done
}

submit_slice_jobs() {
    release_pvc_reader  # must free PVC before pods can attach

    for i in $(seq 1 "$N_GPUS"); do
        SLICE_ID=$i MAX_ARTICLES=$MAX_ARTICLES_PER_SLICE \
            envsubst < "$JOB_TEMPLATE" | kubectl apply -f -
        echo "  submitted slice_${i} job"
    done
}

wait_for_all_jobs() {
    echo "[loop] Waiting for all $N_GPUS jobs (timeout: $JOB_TIMEOUT)..."
    local timeout_s="${JOB_TIMEOUT%s}"
    local deadline=$((SECONDS + timeout_s))
    local n_complete=0 n_failed=0 n_terminal=0

    while [[ $SECONDS -lt $deadline && $n_terminal -lt $N_GPUS ]]; do
        n_complete=0; n_failed=0
        for i in $(seq 1 "$N_GPUS"); do
            local succeeded cond_failed
            succeeded=$(kubectl get job "data-gatherer-batch-slice-${i}" -n "$NAMESPACE" \
                -o jsonpath='{.status.succeeded}' 2>/dev/null)
            cond_failed=$(kubectl get job "data-gatherer-batch-slice-${i}" -n "$NAMESPACE" \
                -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
            [[ "${succeeded:-0}" -ge 1 ]] && ((n_complete++)) || true
            [[ "$cond_failed" == "True" ]] && ((n_failed++)) || true
        done
        n_terminal=$((n_complete + n_failed))
        echo "[loop]  $n_terminal/$N_GPUS terminal — $n_complete complete, $n_failed failed"
        [[ $n_terminal -lt $N_GPUS ]] && sleep 30
    done

    if [[ $n_terminal -lt $N_GPUS ]]; then
        echo "[loop] WARNING: timeout — only $n_terminal/$N_GPUS reached terminal state"
    fi
    echo "[loop] $n_complete/$N_GPUS jobs succeeded."
    if [[ $n_complete -eq 0 ]]; then
        echo "[loop] ERROR: No slices completed. Aborting iteration."
        exit 1
    fi
}

copy_and_merge_outputs() {
    local iter_dir="$1"
    local bucket
    bucket=$(aws configure get aws_access_key_id &>/dev/null && \
             kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
                 -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d 2>/dev/null || echo "")

    echo "[loop] Downloading slice outputs from S3..."
    local csv_list=""
    for i in $(seq 1 "$N_GPUS"); do
        local dest="$iter_dir/slice_${i}_citations.csv"
        local ok=0
        for attempt in 1 2 3; do
            if aws s3 cp "s3://${bucket}/slice_${i}/dataset_citations.csv" "$dest"; then
                echo "  downloaded slice_${i}"
                csv_list="$csv_list $dest"
                ok=1
                break
            fi
            echo "  [warn] slice_${i} attempt ${attempt}/3 failed — retrying in 10s..."
            sleep 10
        done
        [[ $ok -eq 0 ]] && echo "  [error] slice_${i} could not be downloaded after 3 attempts — skipping"
    done

    echo "[loop] Merging outputs..."
    python3 - <<PYEOF
import pandas as pd, sys

files = "$csv_list".split()
if not files:
    print("  [error] no output files to merge")
    sys.exit(1)

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True).drop_duplicates(
    subset=["dataset_identifier", "source_url"], keep="first"
)
merged.to_csv("$iter_dir/dataset_citations.csv", index=False)
print(f"  merged {len(merged)} rows from {len(files)} slices → $iter_dir/dataset_citations.csv")
PYEOF
}

# ============================================================
# MAIN LOOP
# ============================================================
PREV_ONTOLOGY="$SEED_ONTOLOGY"

# Split input and upload slices once
split_and_upload_input

# Upload seed ontology to S3
BUCKET=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
             -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)
echo "[loop] Uploading seed ontology to S3..."
aws s3 cp "$SEED_ONTOLOGY" "s3://${BUCKET}/ontology/open_bio_data_repos.json"

# Clean S3 output before first iteration if requested
if [[ "$CLEAN" -eq 1 ]]; then
    echo "[loop] --clean: wiping s3://${BUCKET}/slice_*/..."
    for i in $(seq 1 "$N_GPUS"); do
        aws s3 rm "s3://${BUCKET}/slice_${i}/" --recursive 2>/dev/null || true
    done
    echo "[loop] S3 output cleared."
fi

for ITER in $(seq 1 "$ITERATIONS"); do
    echo ""
    echo "=========================================="
    echo "  ITERATION $ITER / $ITERATIONS  ($N_GPUS parallel jobs)"
    echo "=========================================="

    ITER_DIR="$OUTPUT_BASE/iter${ITER}"
    mkdir -p "$ITER_DIR"

    # 1. Delete previous jobs + submit all slices in parallel
    delete_slice_jobs
    submit_slice_jobs

    # 2. Wait for all jobs to finish
    wait_for_all_jobs

    # 3. Copy + merge all slice outputs
    copy_and_merge_outputs "$ITER_DIR"

    # 4. Run enrichment on merged data
    ENRICHED_ONTOLOGY="$ITER_DIR/open_bio_data_repos_v${ITER}.json"
    PLOT_ARG=""
    if [[ -n "$PLOT_FLAG" ]]; then
        PLOT_ARG="--plot $ITER_DIR/clusters_iter${ITER}.html"
    fi

    # Build citations list: marginal = current iter only; cumulative = all iters so far
    citations_args=("$ITER_DIR/dataset_citations.csv")
    if [[ "$CUMULATIVE" -eq 1 ]]; then
        for prev_iter in $(seq 1 $((ITER - 1))); do
            prev_csv="$OUTPUT_BASE/iter${prev_iter}/dataset_citations.csv"
            [[ -f "$prev_csv" ]] && citations_args+=("$prev_csv")
        done
        echo "[loop] Cumulative mode: ${#citations_args[@]} citation file(s)"
    fi

    echo "[loop] Running enrichment..."
    python3 scripts/enrich_ontology.py \
        --citations "${citations_args[@]}" \
        --current-ontology "$PREV_ONTOLOGY" \
        --output "$ENRICHED_ONTOLOGY" \
        --log "$ITER_DIR/enrich.log" \
        --pipeline group \
        $PLOT_ARG

    # 5. Push enriched ontology to S3
    BUCKET=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
                 -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)
    echo "[loop] Uploading enriched ontology to S3..."
    aws s3 cp "$ENRICHED_ONTOLOGY" "s3://${BUCKET}/ontology/open_bio_data_repos.json"

    PREV_ONTOLOGY="$ENRICHED_ONTOLOGY"
    echo "[loop] Iteration $ITER done. Ontology: $ENRICHED_ONTOLOGY"
done

echo ""
echo "=========================================="
echo "  ALL $ITERATIONS ITERATIONS COMPLETE"
echo "  Final ontology: $PREV_ONTOLOGY"
echo "=========================================="

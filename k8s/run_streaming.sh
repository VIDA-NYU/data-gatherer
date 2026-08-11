#!/usr/bin/env bash
# Streaming (self-looping) multi-GPU batch run, as an alternative to run_loop.sh's
# lock-step "wait for all N slices, then enrich" pattern.
#
# Each slice's k8s job is submitted ONCE and loops internally (--loop-until-done)
# until its whole partition is processed, uploading incrementally after every
# batch. A separate watcher (watch_and_enrich.sh) polls for changes and re-runs
# enrichment as soon as ANY slice's output updates — a fast slice never sits
# idle waiting on a slow one.
#
# Usage:
#   bash k8s/run_streaming.sh \
#       --gpus 3 --input article_ids_REV_pmc.csv \
#       --max-articles-per-slice 2000 \
#       --output-dir k8s/output/rev_pmc_streaming \
#       --seed-ontology data_gatherer/config/open_bio_data_repos.json \
#       --job-suffix -pmc-stream \
#       --model hf-vida-nyu/flan-t5-base-dataref-info-extract@dataset-page \
#       --semantic-retrieval true --top-k 3 --brute-force-regex true \
#       [--poll-interval 60] [--plot]
#
# Prerequisites: same as run_loop.sh (kubectl configured, ANTHROPIC_API_KEY set).

set -euo pipefail

if [[ -f ".env" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    export ANTHROPIC_API_KEY=$(grep -E "^ANTHROPIC_API_KEY=" .env | cut -d= -f2-)
fi
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is not set — set it or add it to .env}"

N_GPUS=1
INPUT_FILE="article_ids_test_500.csv"
MAX_ARTICLES_PER_SLICE=2000
OUTPUT_BASE="k8s/output/rev_pmc_streaming"
NAMESPACE="data-gatherer"
JOB_TEMPLATE="k8s/streaming-batch-job-template.yaml"
SEED_ONTOLOGY="data_gatherer/config/open_bio_data_repos_seed.json"
JOB_SUFFIX=""
SEMANTIC_RETRIEVAL="true"
BRUTE_FORCE_REGEX="true"
TOP_K="5"
SECTS_REQUIRED="5"
MODEL="hf-vida-nyu/flan-t5-base-dataref-info-extract"
PROMPT_NAME="T5_primer"
S3_BACKUP_KEY="cache/Local_fetched_data.parquet"
POLL_INTERVAL=60
PLOT_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)                    N_GPUS="$2";                  shift 2 ;;
        --input)                   INPUT_FILE="$2";              shift 2 ;;
        --max-articles-per-slice)  MAX_ARTICLES_PER_SLICE="$2";  shift 2 ;;
        --output-dir)               OUTPUT_BASE="$2";             shift 2 ;;
        --seed-ontology)            SEED_ONTOLOGY="$2";           shift 2 ;;
        --job-suffix)               JOB_SUFFIX="$2";              shift 2 ;;
        --semantic-retrieval)       SEMANTIC_RETRIEVAL="$2";      shift 2 ;;
        --brute-force-regex)       BRUTE_FORCE_REGEX="$2";       shift 2 ;;
        --top-k)                    TOP_K="$2";                   shift 2 ;;
        --sects-required)          SECTS_REQUIRED="$2";          shift 2 ;;
        --model)                    MODEL="$2";                   shift 2 ;;
        --prompt-name)              PROMPT_NAME="$2";              shift 2 ;;
        --s3-backup-key)            S3_BACKUP_KEY="$2";            shift 2 ;;
        --poll-interval)            POLL_INTERVAL="$2";            shift 2 ;;
        --plot)                     PLOT_FLAG="--plot";           shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export JOB_SUFFIX SEMANTIC_RETRIEVAL BRUTE_FORCE_REGEX TOP_K SECTS_REQUIRED MODEL PROMPT_NAME S3_BACKUP_KEY

mkdir -p "$OUTPUT_BASE"
BUCKET=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" \
             -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)

echo "[stream] Splitting $INPUT_FILE into $N_GPUS slices and uploading to S3..."
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
    aws s3 cp "k8s/input/slice_${i}.csv" "s3://${BUCKET}/input/slice_${i}.csv"
    echo "  uploaded slice_${i}.csv to S3"
done

echo "[stream] Submitting $N_GPUS self-looping job(s) (one-time, job-suffix '${JOB_SUFFIX}')..."
for i in $(seq 1 "$N_GPUS"); do
    kubectl delete job "data-gatherer-stream-slice-${i}${JOB_SUFFIX}" -n "$NAMESPACE" --ignore-not-found 2>/dev/null
    SLICE_ID=$i MAX_ARTICLES=$MAX_ARTICLES_PER_SLICE \
        envsubst '${SLICE_ID} ${JOB_SUFFIX} ${MAX_ARTICLES} ${MODEL} ${PROMPT_NAME} ${S3_BACKUP_KEY} ${SEMANTIC_RETRIEVAL} ${BRUTE_FORCE_REGEX} ${TOP_K} ${SECTS_REQUIRED}' \
        < "$JOB_TEMPLATE" | kubectl apply -f -
    echo "  submitted slice_${i} (streaming)"
done

echo "[stream] Jobs submitted — each will run until its whole partition is done."
echo "[stream] Handing off to the enrichment watcher (polls every ${POLL_INTERVAL}s)..."
bash k8s/watch_and_enrich.sh \
    --slices "$N_GPUS" --job-suffix "$JOB_SUFFIX" \
    --output-dir "$OUTPUT_BASE" \
    --seed-ontology "$SEED_ONTOLOGY" \
    --poll-interval "$POLL_INTERVAL" \
    $PLOT_FLAG

echo "[stream] All slices complete. Final ontology written under $OUTPUT_BASE."

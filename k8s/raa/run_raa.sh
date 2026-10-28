#!/usr/bin/env bash
# SciNoBo RAA on the OpenAIRE Marine Science pilot, one Job per GPU.
#
# Runs the image the RAA authors publish (intelligencenoborders/scinobo-raa:v0.1.0, built by their CI from commit
# 2b351ef, pinned here by digest); k8s/raa/raa_worker.py reaches the pods through a ConfigMap per run.
# --raa-commit runs the RAA pipeline files of another commit of --raa-repo instead (e.g. the batching PR,
# iNoBo/scinobo-raa#1), and --batch-size sets RAA_BATCH_SIZE, which only that code reads. --procs-per-gpu runs
# several workers per pod on the same GPU (RAA is CPU-bound and leaves the GPU mostly idle), so a GPU quota goes further.
#
#   1. Upload the TEI files (scripts/OpenAIRE/raa_adapter.py to-tei) to s3://<bucket>/raa<suffix>/tei/
#   2. Submit N Jobs (k8s/raa/raa-job-template.yaml); slice i takes every N-th TEI file
#   3. Wait for all of them
#   4. Download the RAA JSON (+ per-slice logs and GPU power) to --output-dir
#
# Pods upload each article as soon as it is done and skip what is already in S3, so rerunning this script resumes
# the run (Jobs are deleted and resubmitted). Use a new --job-suffix for an independent run.
#
# Usage:
#   bash k8s/raa/run_raa.sh --tei-dir scripts/OpenAIRE/output/raa/tei --output-dir scripts/OpenAIRE/output/raa/json \
#                           --gpus 4 [--image <scinobo-raa image>] [--job-suffix -mraa] [--timeout 21600] \
#                           [--raa-commit <sha> [--raa-repo <owner/repo>] --batch-size 16] [--procs-per-gpu 4]
#
# Prerequisites: kubectl pointing at the cluster, aws CLI with access to the data-gatherer S3 bucket.

set -euo pipefail

TEI_DIR=""
OUTPUT_DIR=""
N_GPUS=4
IMAGE="intelligencenoborders/scinobo-raa@sha256:342623650a003931cf538fc3c0d3558583036ebe721df2bbf0e4a2c2344bb5f6"
JOB_SUFFIX="-mraa"
TIMEOUT=21600
RAA_COMMIT=""
RAA_REPO="pietromarini00/scinobo-raa"
BATCH_SIZE=1
PROCS_PER_GPU=1
NAMESPACE="data-gatherer"
JOB_TEMPLATE="k8s/raa/raa-job-template.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tei-dir)     TEI_DIR="$2";     shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --gpus)        N_GPUS="$2";      shift 2 ;;
        --image)       IMAGE="$2";       shift 2 ;;
        --job-suffix)  JOB_SUFFIX="$2";  shift 2 ;;
        --timeout)     TIMEOUT="$2";     shift 2 ;;
        --raa-commit)  RAA_COMMIT="$2";  shift 2 ;;
        --raa-repo)    RAA_REPO="$2";    shift 2 ;;
        --batch-size)  BATCH_SIZE="$2";  shift 2 ;;
        --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
: "${TEI_DIR:?--tei-dir is required}"
: "${OUTPUT_DIR:?--output-dir is required}"

S3_PREFIX="raa${JOB_SUFFIX}"
BUCKET=$(kubectl get secret data-gatherer-s3-secret -n "$NAMESPACE" -o jsonpath='{.data.S3_OUTPUT_BUCKET}' | base64 -d)
RAA_REPO_RAW="https://raw.githubusercontent.com/${RAA_REPO}"
# about one core per worker plus headroom; 1 worker keeps the original 4 CPUs / 16Gi
CPU=$((PROCS_PER_GPU + 3))
MEMORY="$((4 * PROCS_PER_GPU + 12))Gi"
export N_SLICES="$N_GPUS" IMAGE JOB_SUFFIX S3_PREFIX RAA_COMMIT RAA_REPO_RAW BATCH_SIZE PROCS_PER_GPU CPU MEMORY
echo "[raa] run $S3_PREFIX | code: ${RAA_COMMIT:-published image} | batch size: $BATCH_SIZE | $N_GPUS GPUs x $PROCS_PER_GPU workers"

echo "[raa] Uploading $(ls "$TEI_DIR"/*.xml | wc -l | tr -d ' ') TEI files to s3://$BUCKET/$S3_PREFIX/tei/"
aws s3 sync "$TEI_DIR" "s3://$BUCKET/$S3_PREFIX/tei/" --exclude "*" --include "*.xml" --only-show-errors

kubectl create configmap "raa-worker${JOB_SUFFIX}" -n "$NAMESPACE" --from-file=k8s/raa/raa_worker.py \
    --dry-run=client -o yaml | kubectl apply -f -

echo "[raa] Submitting $N_GPUS jobs"
# every earlier job of this run, also from a run with more --gpus
kubectl delete job -n "$NAMESPACE" -l "raa-run=$S3_PREFIX" --ignore-not-found
for i in $(seq 1 "$N_GPUS"); do
    kubectl delete job "scinobo-raa-slice-${i}${JOB_SUFFIX}" -n "$NAMESPACE" --ignore-not-found
    # only the template's placeholders: $GPU_SAMPLER_PID and $JOB_EXIT_CODE are evaluated inside the pod
    SLICE_ID=$i envsubst '${SLICE_ID} ${N_SLICES} ${IMAGE} ${JOB_SUFFIX} ${S3_PREFIX} ${RAA_COMMIT} ${RAA_REPO_RAW} ${BATCH_SIZE} ${PROCS_PER_GPU} ${CPU} ${MEMORY}' < "$JOB_TEMPLATE" | kubectl apply -f -
done

echo "[raa] Waiting for the jobs (timeout ${TIMEOUT}s); follow one with:"
echo "      kubectl logs -f -n $NAMESPACE job/scinobo-raa-slice-1${JOB_SUFFIX}"
deadline=$((SECONDS + TIMEOUT))
while :; do
    n_ok=0; n_failed=0
    for i in $(seq 1 "$N_GPUS"); do
        job="scinobo-raa-slice-${i}${JOB_SUFFIX}"
        [[ "$(kubectl get job "$job" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null)" == "1" ]] && n_ok=$((n_ok + 1))
        kubectl get job "$job" -n "$NAMESPACE" \
            -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q True && n_failed=$((n_failed + 1))
    done
    echo "  $(date +%H:%M:%S) succeeded=$n_ok failed=$n_failed of $N_GPUS"
    (( n_ok + n_failed >= N_GPUS )) && break
    (( SECONDS >= deadline )) && { echo "[raa] Timed out; pods keep running, rerun this script to resume/collect"; break; }
    sleep 60
done

mkdir -p "$OUTPUT_DIR"
echo "[raa] Downloading results to $OUTPUT_DIR"
aws s3 sync "s3://$BUCKET/$S3_PREFIX/output/" "$OUTPUT_DIR" --exclude "*" --include "*.json" --only-show-errors
aws s3 sync "s3://$BUCKET/$S3_PREFIX/logs/" "$OUTPUT_DIR/logs" --only-show-errors
echo "[raa] $(ls "$OUTPUT_DIR"/*_reevaluated.json 2>/dev/null | wc -l | tr -d ' ') articles with output"
(( n_failed == 0 ))

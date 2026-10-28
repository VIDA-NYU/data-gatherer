"""SciNoBo RAA on one slice of the Marine pilot's TEI files (one pod of k8s/raa/raa-job-template.yaml).

Lists s3://$S3_OUTPUT_BUCKET/<prefix>/tei/*.xml, takes every n-th file for this slice and skips those whose
reevaluated output is already under <prefix>/output/, so a restarted pod resumes. Each article runs what the CLI runs
with `--xml_mode --perform_deduplication --reevaluate` (thresholds 0.8), and its two JSON files are uploaded as soon
as it is done. Per-article timings and errors go to <prefix>/logs/slice_<i>_log.csv; an article that fails is logged
and skipped, so the pod only fails on a crash.
"""
import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import boto3
import torch

# Same fix as data_gatherer/llm/hf_model_client.py: TF32 on Ampere+ GPUs makes Flan-T5 hallucinate. Set before the
# model loads, so RAA's code and published image stay unmodified.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from raa.pipeline import inference as raa  # noqa: E402  (loads Flan-T5 + LoRA and the gazetteers)
from raa.pipeline.model import model as raa_model  # noqa: E402

raa.compress_output = False  # module global that the CLI's main() sets and reevaluate() reads


def list_keys(s3, bucket: str, prefix: str) -> list:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, required=True, help="1-based")
    ap.add_argument("--n-slices", type=int, required=True)
    ap.add_argument("--prefix", required=True, help="S3 prefix holding tei/, output/ and logs/")
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/raa"))
    ap.add_argument("--gpu-log", type=Path, default=Path("/tmp/gpu_power.csv"))
    args = ap.parse_args()

    s3, bucket = boto3.client("s3"), os.environ["S3_OUTPUT_BUCKET"]
    tei_keys = sorted(k for k in list_keys(s3, bucket, f"{args.prefix}/tei/") if k.endswith(".xml"))
    mine = tei_keys[args.slice - 1::args.n_slices]
    done = {Path(k).name for k in list_keys(s3, bucket, f"{args.prefix}/output/") if k.endswith("_reevaluated.json")}
    todo = [k for k in mine if f"{Path(k).stem}_reevaluated.json" not in done]
    print(f"slice {args.slice}/{args.n_slices}: {len(mine)} articles, {len(mine) - len(todo)} already done, "
          f"device {raa_model.device}, RAA code {os.getenv('RAA_COMMIT') or 'published image'}, "
          f"batch size {getattr(raa, 'batch_size', 'n/a (unbatched code)')}", flush=True)

    (args.work_dir / "tei").mkdir(parents=True, exist_ok=True)
    (args.work_dir / "out").mkdir(parents=True, exist_ok=True)
    log_path = args.work_dir / f"slice_{args.slice}_log.csv"
    log_key = f"{args.prefix}/logs/slice_{args.slice}_log.csv"
    try:  # keep the rows of earlier attempts of this slice
        s3.download_file(bucket, log_key, str(log_path))
    except Exception:
        log_path.write_text("article,status,seconds,candidates,dataset_clusters,error\n")

    try:
        for n, key in enumerate(todo, 1):
            stem = Path(key).stem
            tei, out = args.work_dir / "tei" / f"{stem}.xml", args.work_dir / "out" / f"{stem}.json"
            s3.download_file(bucket, key, str(tei))
            t0, row = time.time(), {"article": stem, "status": "ok", "candidates": "", "dataset_clusters": "", "error": ""}
            try:
                res = raa.extract_research_artifacts_pdf(str(tei), xml_mode=True, filter_paragraphs=False,
                                                         perform_deduplication=True, insert_fast_mode_gazetteers=False,
                                                         dataset_gazetteers=None, verbose=False)
                out.write_text(json.dumps(res, indent=1))
                raa.reevaluate(str(out))  # writes <stem>_reevaluated.json (+ .xlsx)
                ra = res["research_artifacts"]
                row["candidates"] = len(ra["candidates_metadata"] or {})
                row["dataset_clusters"] = len((ra["grouped_clusters"] or {}).get("dataset", {}))
                for f in (out, out.with_name(f"{stem}_reevaluated.json")):
                    s3.upload_file(str(f), bucket, f"{args.prefix}/output/{f.name}")
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
                traceback.print_exc()
            row["seconds"] = round(time.time() - t0, 1)
            with log_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=["article", "status", "seconds", "candidates", "dataset_clusters",
                                              "error"]).writerow(row)
            print(f"s{args.slice} [{n}/{len(todo)}] {stem} {row['status']} {row['seconds']}s "
                  f"candidates={row['candidates']} dataset_clusters={row['dataset_clusters']}", flush=True)
            if n % 10 == 0:
                s3.upload_file(str(log_path), bucket, log_key)
    finally:
        s3.upload_file(str(log_path), bucket, log_key)
        if args.gpu_log.exists():
            s3.upload_file(str(args.gpu_log), bucket, f"{args.prefix}/logs/slice_{args.slice}_gpu_power.csv")


if __name__ == "__main__":
    main()

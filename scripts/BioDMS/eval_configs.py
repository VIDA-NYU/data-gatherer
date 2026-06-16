#!/usr/bin/env python3
"""
Evaluate retrieval configs against SciLite and Gold GT,
restricted to the REV test split (249 articles).

Usage:
    python scripts/BioDMS/eval_configs.py            # evaluate whatever is already local
    python scripts/BioDMS/eval_configs.py --download # pull missing T5 CSVs from S3 first
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.experiment_utils import evaluate_performance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIGS = {
    # Fine-tuned T5 (k8s, downloaded via --download or run_loop.sh)
    "T5 · c1 (no semantic, no regex)":  "k8s/output/rev_test_c1/iter1/dataset_citations.csv",
    "T5 · c2 (semantic k=3, no regex)": "k8s/output/rev_test_c2/iter1/dataset_citations.csv",
    "T5 · c3 (semantic k=3 + regex)":   "k8s/output/rev_test_c3/iter1/dataset_citations.csv",
    "T5 · c4 (chunked FDR)":            "k8s/output/rev_test_c4/iter1/dataset_citations.csv",
    # Claude Haiku (run locally)
    "Haiku · c1 (no semantic, no regex)":  "k8s/output/rev_test_haiku_c1/dataset_citations.csv",
    "Haiku · c2 (semantic k=3, no regex)": "k8s/output/rev_test_haiku_c2/dataset_citations.csv",
    "Haiku · c3 (semantic k=3 + regex)":   "k8s/output/rev_test_haiku_c3/dataset_citations.csv",
    # GPT-5-mini (run locally)
    "GPT-5-mini · c1 (no semantic, no regex)":  "k8s/output/rev_test_gpt5mini_c1/dataset_citations.csv",
    "GPT-5-mini · c2 (semantic k=3, no regex)": "k8s/output/rev_test_gpt5mini_c2/dataset_citations.csv",
    "GPT-5-mini · c3 (semantic k=3 + regex)":   "k8s/output/rev_test_gpt5mini_c3/dataset_citations.csv",
    # Gemini 3.5 Flash (run locally)
    "Gemini-3-flash · c1 (no semantic, no regex)":  "k8s/output/rev_test_gemini_c1/dataset_citations.csv",
    "Gemini-3-flash · c2 (semantic k=3, no regex)": "k8s/output/rev_test_gemini_c2/dataset_citations.csv",
    "Gemini-3-flash · c3 (semantic k=3 + regex)":   "k8s/output/rev_test_gemini_c3/dataset_citations.csv",
}

S3_KEYS = {
    "T5 · c1 (no semantic, no regex)":  "slice_1-tc1/dataset_citations.csv",
    "T5 · c2 (semantic k=3, no regex)": "slice_1-tc2/dataset_citations.csv",
    "T5 · c3 (semantic k=3 + regex)":   "slice_1-tc3/dataset_citations.csv",
}

TEST_PMCID_CSV = "k8s/input/article_ids_REV_test.csv"
SCILITE_GT_PATH = "scripts/output/scilite_ground_truth.parquet"
GOLD_GT_PATH = "scripts/output/gold/dataset_citation_records_Table.parquet"
OUTPUT_DIR = Path("scripts/BioDMS/config_eval")

NON_DATA_SUBTYPES = {
    "Gene Ontology (GO)", "RefSNP", "Pfam", "InterPro",
    "HGNC", "Brenda", "Rfam", "EFO", "Treefam",
}

# ---------------------------------------------------------------------------
# Minimal orchestrator shim (evaluate_performance needs orchestrator.logger)
# ---------------------------------------------------------------------------

class _Orchestrator:
    class _Logger:
        def __init__(self):
            self._log = logging.getLogger("eval")
        def info(self, m):    self._log.info(m)
        def debug(self, m):   self._log.debug(m)
        def warning(self, m): self._log.warning(m)
    def __init__(self):
        self.logger = self._Logger()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_from_s3(s3_key: str, local_path: str) -> bool:
    import os
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    bucket = os.environ.get("S3_OUTPUT_BUCKET", "data-gatherer-output")
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, s3_key, local_path)
        print(f"  ✓ {s3_key} → {local_path}")
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"  ✗ {s3_key}: {e}")
        return False


def load_test_pmcids() -> set:
    return set(pd.read_csv(TEST_PMCID_CSV)["pmcid"].str.upper().str.strip())


def test_gt_base(test_pmcids: set) -> list:
    """REV_test.txt URL format — evaluate_performance extracts PMCID from the path."""
    return [f"https://www.ncbi.nlm.nih.gov/pmc/articles/{p}" for p in test_pmcids]


def load_scilite_gt(test_pmcids: set) -> pd.DataFrame:
    gt = pd.read_parquet(SCILITE_GT_PATH)
    # handle both column name variants
    if "exact" in gt.columns and "identifier" not in gt.columns:
        gt = gt.rename(columns={"exact": "identifier"})
    gt = gt[~gt["subType"].isin(NON_DATA_SUBTYPES)].copy()
    gt["pmcid"] = gt["pmcid"].str.upper().str.strip()
    gt = gt[gt["pmcid"].isin(test_pmcids)].copy()
    print(f"SciLite GT (test subset): {len(gt):,} rows  |  {gt['pmcid'].nunique()} articles")
    return gt


def load_gold_gt(test_pmcids: set) -> pd.DataFrame:
    gt = pd.read_parquet(GOLD_GT_PATH)
    gt = gt[gt["pmcid"].notna()].copy()
    gt["pmcid"] = gt["pmcid"].str.upper().str.strip()
    gt["identifier"] = gt["identifier"].str.strip()
    gt = gt[gt["pmcid"].isin(test_pmcids)].copy()
    print(f"Gold GT    (test subset): {len(gt):,} rows  |  {gt['pmcid'].nunique()} articles")
    return gt


def run_eval(label, pred_df, gt, gt_base, tag, orchestrator):
    fp_file = str(OUTPUT_DIR / f"fp_{tag}_{label.split()[0]}.txt")
    fn_file = str(OUTPUT_DIR / f"fn_{tag}_{label.split()[0]}.txt")
    return evaluate_performance(
        pred_df, gt, orchestrator,
        fp_file,
        false_negatives_file=fn_file,
        repo_return=True,
        gt_base=gt_base,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download missing config CSVs from S3 before evaluating")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_pmcids = load_test_pmcids()
    print(f"Test set: {len(test_pmcids)} articles\n")

    if args.download:
        print("Downloading missing configs from S3...")
        for label, local_path in CONFIGS.items():
            if not Path(local_path).exists():
                download_from_s3(S3_KEYS[label], local_path)
        print()

    gt_scilite = load_scilite_gt(test_pmcids)
    gt_gold = load_gold_gt(test_pmcids)
    gt_base = test_gt_base(test_pmcids)
    orchestrator = _Orchestrator()

    print()
    rows = []
    for label, csv_path in CONFIGS.items():
        if not Path(csv_path).exists():
            print(f"[skip] {label}: file not found (pass --download to fetch from S3)")
            continue

        pred_df = pd.read_csv(csv_path)
        pred_df = pred_df[pred_df["dataset_identifier"].notna()].copy()
        n_articles = pred_df["source_url"].nunique()
        print(f"── {label}  ({len(pred_df)} rows, {n_articles} articles with predictions)")

        t0 = time.time()
        m_s = run_eval(label, pred_df, gt_scilite, gt_base, "scilite", orchestrator)
        m_g = run_eval(label, pred_df, gt_gold,    gt_base, "gold",    orchestrator)
        elapsed = time.time() - t0

        print(f"   SciLite  P={m_s['average_precision']:.3f}  R={m_s['average_recall']:.3f}  F1={m_s['f1_score']:.3f}")
        print(f"   Gold     P={m_g['average_precision']:.3f}  R={m_g['average_recall']:.3f}  F1={m_g['f1_score']:.3f}  ({elapsed:.1f}s)")
        print()

        rows.append({
            "config":         label,
            "articles":       n_articles,
            "scilite_P":      round(m_s["average_precision"], 3),
            "scilite_R":      round(m_s["average_recall"],    3),
            "scilite_F1":     round(m_s["f1_score"],          3),
            "gold_P":         round(m_g["average_precision"], 3),
            "gold_R":         round(m_g["average_recall"],    3),
            "gold_F1":        round(m_g["f1_score"],          3),
        })

    if not rows:
        print("No configs evaluated.")
        return

    results = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "results.csv"
    results.to_csv(out_csv, index=False)

    print("=" * 72)
    print(results.to_string(index=False))
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()

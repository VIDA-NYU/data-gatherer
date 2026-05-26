#!/usr/bin/env python3
"""
scripts/scrape_annotations.py

Fetch Europe PMC SciLite annotations for a list of PMC papers and write
ground-truth JSON output.  Designed to run against the full
article_ids_REV_pmc.csv (118 k IDs) with checkpointing so it can be
interrupted and resumed.

Supports two input CSV formats
  - pmcid column:                  PMC10238389
  - citing_publication_link col:   https://europepmc.org/article/PMC/PMC11895760

Output schema
  {
    "PMC10238389": [                  # key = bare PMCID (no colon prefix)
      {
        "type": "Gene_Proteins",
        "exact": "BRCA1",
        "section": "Abstract",
        "tags": [...],
        ...
      },
      ...
    ],
    ...
  }

Usage
  python scripts/scrape_annotations.py \\
      --input k8s/input/article_ids_eval.csv \\
      --output scripts/output/annotations_eval.json \\
      [--batch-size 25] [--pause 0.2] [--retries 3] [--resume]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

ANNOTATIONS_URL = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
_PMC_RE = re.compile(r"PMC\d+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=backoff,
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# ID handling
# ---------------------------------------------------------------------------

def extract_pmcid(raw: str) -> str | None:
    """Return bare PMCID (e.g. 'PMC10238389') from any string."""
    raw = (raw or "").strip()
    if not raw:
        return None
    m = _PMC_RE.search(raw)
    if not m:
        return None
    return m.group(0).upper()


def to_api_id(pmcid: str) -> str:
    """Convert 'PMC10238389' → 'PMC:PMC10238389' as required by the API."""
    return f"PMC:{pmcid}"


def load_ids_from_csv(path: str) -> list[str]:
    """Load PMC IDs from CSV; auto-detects column (pmcid or citing_publication_link)."""
    df = pd.read_csv(path)
    if "pmcid" in df.columns:
        col = df["pmcid"]
    elif "citing_publication_link" in df.columns:
        col = df["citing_publication_link"]
    else:
        col = df.iloc[:, 0]
        logger.warning("No recognised PMC ID column — using first column: %s", df.columns[0])

    ids = []
    for raw in col.dropna():
        pmcid = extract_pmcid(str(raw))
        if pmcid:
            ids.append(pmcid)
    return ids


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_path(output: Path) -> Path:
    return output.with_suffix(".checkpoint.json")


def load_checkpoint(output: Path) -> dict:
    cp = checkpoint_path(output)
    if cp.exists():
        with cp.open() as fh:
            return json.load(fh)
    return {}


def save_checkpoint(output: Path, done: dict) -> None:
    cp = checkpoint_path(output)
    with cp.open("w") as fh:
        json.dump(done, fh)


# ---------------------------------------------------------------------------
# Annotation fetching
# ---------------------------------------------------------------------------

def batch_list(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def parse_response(api_response: list) -> dict[str, list]:
    """Flatten API response list into {pmcid: [annotation, ...]} mapping."""
    result = {}
    for article_entry in api_response:
        # API returns extId=PubMedID, pmcid=PMCxxxxxxx — prefer the pmcid field
        raw_id = article_entry.get("pmcid") or article_entry.get("extId") or ""
        pmcid = extract_pmcid(raw_id) or raw_id
        result[pmcid] = [
            ann for ann in article_entry.get("annotations", [])
            if ann.get("type") == "Accession Numbers"
        ]
    return result


def extract_accession_numbers(annotations: dict[str, list]) -> dict[str, list]:
    """Filter a {pmcid: [annotation, ...]} mapping to accession-number entries only."""
    return {
        pmcid: [ann for ann in anns if ann.get("type") == "Accession Numbers"]
        for pmcid, anns in annotations.items()
    }


def to_dataframe(annotations: dict[str, list]) -> "pd.DataFrame":
    """Flatten a {pmcid: [annotation, ...]} mapping into a tidy DataFrame.

    Each row is one accession annotation with all original fields preserved.
    The tags list is exploded: each tag becomes its own row with tag_name / tag_uri columns.
    """
    import pandas as pd

    rows = []
    for pmcid, anns in annotations.items():
        for ann in anns:
            tags = ann.get("tags") or [{}]
            base = {k: v for k, v in ann.items() if k != "tags"}
            base["pmcid"] = pmcid
            for tag in tags:
                rows.append({**base, "tag_name": tag.get("name"), "tag_uri": tag.get("uri")})
    return pd.DataFrame(rows)


def fetch_annotations(
    pmcids: list[str],
    session: requests.Session,
    batch_size: int = 8,
    pause: float = 0.2,
    done: dict | None = None,
    output: Path | None = None,
    s3_bucket: str | None = None,
    s3_checkpoint_key: str | None = None,
) -> dict[str, list]:
    """Fetch annotations for all pmcids, skipping already-done ones.

    Saves a checkpoint every 10 batches so large runs can be resumed.
    """
    if done is None:
        done = {}

    remaining = [p for p in pmcids if p not in done]
    n_batches = (len(remaining) + batch_size - 1) // batch_size
    logger.info(
        "Total IDs: %d  |  already done: %d  |  to fetch: %d  |  batches: %d",
        len(pmcids), len(pmcids) - len(remaining), len(remaining), n_batches,
    )

    for i, chunk in enumerate(batch_list(remaining, batch_size)):
        api_ids = ",".join(to_api_id(p) for p in chunk)
        url = f"{ANNOTATIONS_URL}?articleIds={api_ids}"  # avoid requests encoding the colon in PMC:PMCxxxxx
        logger.info("[%d/%d] Fetching %d IDs", i + 1, n_batches, len(chunk))
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            batch_result = parse_response(resp.json())
            done.update(batch_result)
            for pmcid in chunk:
                done.setdefault(pmcid, [])
        except requests.RequestException as exc:
            logger.warning("Batch %d failed (%s) — skipping", i + 1, exc)

        if output and (i + 1) % 1000 == 0:
            save_checkpoint(output, done)
            if s3_checkpoint_key:
                _s3_upload(checkpoint_path(output), s3_bucket, s3_checkpoint_key)

        time.sleep(pause)

    return done


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client():
    import boto3
    return boto3.client("s3")


def _s3_upload(local: Path, bucket: str, key: str) -> None:
    try:
        _s3_client().upload_file(str(local), bucket, key)
        logger.info("S3 upload: s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning("S3 upload failed (%s) — continuing", exc)


def _s3_download(bucket: str, key: str, local: Path) -> bool:
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        _s3_client().download_file(bucket, key, str(local))
        logger.info("S3 download: s3://%s/%s → %s", bucket, key, local)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Europe PMC SciLite annotations (ground truth)")
    parser.add_argument("--input", required=True, help="CSV with PMC IDs (pmcid or citing_publication_link column)")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--batch-size", type=int, default=8, help="IDs per API request — API max is 8 (default 8)")
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between requests (default 0.2)")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry attempts (default 3)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket for input/output (uses AWS env creds)")
    parser.add_argument("--s3-input-key", default=None, help="S3 key to download input CSV from")
    parser.add_argument("--s3-output-key", default=None, help="S3 key to upload final JSON to")
    parser.add_argument("--s3-checkpoint-key", default=None, help="S3 key to sync checkpoint to/from")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Download input from S3 if requested
    if args.s3_bucket and args.s3_input_key:
        _s3_download(args.s3_bucket, args.s3_input_key, Path(args.input))

    pmcids = load_ids_from_csv(args.input)
    if not pmcids:
        logger.error("No valid PMC IDs found in %s", args.input)
        raise SystemExit(2)
    logger.info("Loaded %d unique PMC IDs from %s", len(set(pmcids)), args.input)

    # Resume from S3 checkpoint if available
    done: dict = {}
    if args.resume:
        if args.s3_bucket and args.s3_checkpoint_key:
            _s3_download(args.s3_bucket, args.s3_checkpoint_key, checkpoint_path(output))
        done = load_checkpoint(output)
        if done:
            logger.info("Resumed from checkpoint: %d IDs already fetched", len(done))

    session = make_session(retries=args.retries)
    done = fetch_annotations(
        pmcids, session,
        batch_size=args.batch_size, pause=args.pause,
        done=done, output=output,
        s3_bucket=args.s3_bucket, s3_checkpoint_key=args.s3_checkpoint_key,
    )

    with output.open("w") as fh:
        json.dump(done, fh, indent=2)
    logger.info("Wrote annotations for %d papers → %s", len(done), output)

    if args.s3_bucket and args.s3_output_key:
        _s3_upload(output, args.s3_bucket, args.s3_output_key)

    cp = checkpoint_path(output)
    if cp.exists():
        cp.unlink()
        logger.info("Removed checkpoint file")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

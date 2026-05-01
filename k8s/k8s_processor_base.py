"""
k8s batch processor for data-gatherer (single-article path).

Uses process_articles → process_url → generate (one section at a time, no padding),
which is the same path as the local notebook. Avoids batch_generate hallucinations.

Usage:
    python k8s/k8s_processor_base.py \
        --input /data/input/article_ids_eval.csv \
        --output-dir /data/output \
        --model hf-vida-nyu/flan-t5-base-dataref-info-extract
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

LOG_FMT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fmt = logging.Formatter(LOG_FMT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)


logger = logging.getLogger(__name__)

PMC_URL_TEMPLATE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"


def pmcid_to_url(pmcid: str) -> str:
    pmcid = str(pmcid).strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return PMC_URL_TEMPLATE.format(pmcid=pmcid)


def load_checkpoint(output_csv: str) -> set:
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        if "source_url" in df.columns:
            return set(df["source_url"].dropna().unique())
    except Exception as e:
        logger.warning(f"Could not read checkpoint from {output_csv}: {e}")
    return set()


def append_to_csv(df: pd.DataFrame, output_csv: str) -> None:
    if df is None or df.empty:
        return
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode="a", index=False, header=write_header, quoting=1)
    logger.info(f"Checkpoint saved: appended {len(df)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run data-gatherer single-article extraction on k8s")
    parser.add_argument("--input", required=True, help="Path to CSV with 'pmcid' column")
    parser.add_argument("--output-dir", required=True, help="Output directory (PVC mount)")
    parser.add_argument(
        "--model",
        default="hf-vida-nyu/flan-t5-base-dataref-info-extract",
        help="LLM model name (must start with 'hf-')",
    )
    parser.add_argument(
        "--section-filter",
        default=None,
        choices=["data_availability_statement", "supplementary_material"],
        help="Restrict extraction to one section type (default: both)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    output_csv = os.path.join(args.output_dir, "dataset_citations.csv")
    log_file = os.path.join(args.output_dir, "run.log")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    input_df = pd.read_csv(args.input)
    if "pmcid" not in input_df.columns:
        raise ValueError(f"Input CSV must have a 'pmcid' column. Found: {input_df.columns.tolist()}")
    pmcids = input_df["pmcid"].dropna().astype(str).str.strip().tolist()
    all_urls = [pmcid_to_url(p) for p in pmcids]
    logger.info(f"Loaded {len(all_urls)} URLs from {args.input}")

    done_urls = load_checkpoint(output_csv)
    pending_urls = [u for u in all_urls if u not in done_urls]
    logger.info(f"Checkpoint: {len(done_urls)} done, {len(pending_urls)} pending")

    if not pending_urls:
        logger.info("All URLs already processed. Nothing to do.")
        return

    from data_gatherer.data_gatherer import DataGatherer

    dg = DataGatherer(
        llm_name=args.model,
        save_to_cache=True,
        load_from_cache=True,
        log_file_override=log_file,
        log_level=logging.INFO,
    )

    total = len(pending_urls)
    start_time = time.time()

    for i, url in enumerate(pending_urls):
        logger.info(f"Article {i + 1}/{total}: {url}")
        try:
            results = dg.process_articles(
                url_list=[url],
                full_document_read=False,
                semantic_retrieval=True,
                brute_force_RegEx_ID_ptrs=True,
                prompt_name="T5_primer",
                section_filter=args.section_filter,
                use_portkey=False,
            )
            article_df = results.get(url)
            if isinstance(article_df, pd.DataFrame):
                append_to_csv(article_df, output_csv)
            else:
                logger.warning(f"Article {url} returned unexpected type: {type(article_df)}")
        except Exception as e:
            logger.error(f"Article {url} failed: {e}", exc_info=True)

        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        logger.info(f"Progress: {i + 1}/{total} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    logger.info(f"Done. Results at {output_csv}")


if __name__ == "__main__":
    main()

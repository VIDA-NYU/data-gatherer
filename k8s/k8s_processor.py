"""
k8s batch processor for data-gatherer.

Reads a CSV of PMCIDs, converts them to PMC article URLs, and runs
run_integrated_batch_processing in checkpointed batches using the
HuggingFace flan-t5 model. Results are saved to a CSV file on the PVC.

Usage:
    python scripts/k8s/k8s_processor.py \
        --input /data/input/pmcids.csv \
        --output-dir /data/output \
        --model hf-vida-nyu/flan-t5-base-dataref-info-extract \
        --batch-size 50
"""

import argparse
import os
import sys
import time
import logging

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
    """Return set of source_urls already in the output CSV."""
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        if "source_url" in df.columns:
            return set(df["source_url"].dropna().unique())
    except Exception as e:
        logger.warning(f"Could not read checkpoint from {output_csv}: {e}")
    return set()


def append_to_csv(batch_df: pd.DataFrame, output_csv: str) -> None:
    if batch_df.empty:
        return
    write_header = not os.path.exists(output_csv)
    batch_df.to_csv(output_csv, mode="a", index=False, header=write_header, quoting=1)  # QUOTE_ALL
    logger.info(f"Checkpoint saved: appended {len(batch_df)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run data-gatherer batch extraction on k8s")
    parser.add_argument("--input", required=True, help="Path to CSV with 'pmcid' column")
    parser.add_argument("--output-dir", required=True, help="Output directory (PVC mount)")
    parser.add_argument(
        "--model",
        default="hf-vida-nyu/flan-t5-base-dataref-info-extract",
        help="LLM model name (must start with 'hf-')",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="URLs per batch call")
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

    # Load input PMCIDs
    input_df = pd.read_csv(args.input)
    if "pmcid" not in input_df.columns:
        raise ValueError(f"Input CSV must have a 'pmcid' column. Found: {input_df.columns.tolist()}")
    pmcids = input_df["pmcid"].dropna().astype(str).str.strip().tolist()
    all_urls = [pmcid_to_url(p) for p in pmcids]
    logger.info(f"Loaded {len(all_urls)} URLs from {args.input}")

    # Checkpoint: skip already-processed URLs
    done_urls = load_checkpoint(output_csv)
    pending_urls = [u for u in all_urls if u not in done_urls]
    logger.info(f"Checkpoint: {len(done_urls)} done, {len(pending_urls)} pending")

    if not pending_urls:
        logger.info("All URLs already processed. Nothing to do.")
        return

    # Import here so the module is importable even without GPU during syntax checks
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
    processed = 0

    for batch_start in range(0, total, args.batch_size):
        batch = pending_urls[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size
        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} URLs")

        batch_file_path = os.path.join(args.output_dir, f"batch_requests_{batch_num}.jsonl")
        batch_output_path = os.path.join(args.output_dir, f"dataset_citations_batch_{batch_num}.csv")

        try:
            batch_df = dg.run_integrated_batch_processing(
                url_list=batch,
                batch_file_path=batch_file_path,
                output_file_path=batch_output_path,
                section_filter=args.section_filter,
                prompt_name='T5_primer',
                semantic_retrieval=True,
                brute_force_RegEx_ID_ptrs=True,
                use_portkey=False,
            )
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}", exc_info=True)
            continue

        if isinstance(batch_df, pd.DataFrame):
            append_to_csv(batch_df, output_csv)
            log_cols = ['source_url', 'pub_title', 'raw_data_format', 'n_all_sections', 'n_corpus_sections', 'retrieved_sections_title', 'top_k', 'n_das_sections']
            article_info = batch_df[[c for c in log_cols if c in batch_df.columns]].drop_duplicates('source_url') if not batch_df.empty else pd.DataFrame(columns=log_cols)
            log_df = pd.DataFrame({'source_url': batch}).merge(article_info, on='source_url', how='left')
            append_to_csv(log_df, os.path.join(args.output_dir, 'articles_log.csv'))
        else:
            logger.warning(f"Batch {batch_num} returned unexpected type: {type(batch_df)}")

        processed += len(batch)
        elapsed = time.time() - start_time
        avg = elapsed / processed
        eta = avg * (total - processed)
        logger.info(
            f"Progress: {processed}/{total} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
        )

    logger.info(f"Done. Results at {output_csv}")


if __name__ == "__main__":
    main()

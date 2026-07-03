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
import json
from data_gatherer.llm.response_schema import dataset_response_schema_gpt, dataset_response_schema_gpt_completions, dataset_response_schema_claude, Dataset_w_Page
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    BotoCoreError = ClientError = Exception

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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    # Some columns may contain unhashable types (lists/dicts). Normalize them to
    # deterministic JSON strings for deduplication purposes only.
    def _normalize_value(v):
        if isinstance(v, (list, dict)):
            try:
                return json.dumps(v, sort_keys=True)
            except Exception:
                return str(v)
        return v

    new_rows = batch_df.copy()
    # Normalize for dedupe
    normalized_new = new_rows.applymap(_normalize_value)
    # Keep only unique rows from the new batch (preserve original types in new_rows)
    keep_mask = ~normalized_new.duplicated()
    dedup_new_rows = new_rows.loc[keep_mask].reset_index(drop=True)

    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        # Normalize existing + new for global dedupe
        merged = pd.concat([existing_df, dedup_new_rows], ignore_index=True)
        normalized_merged = merged.applymap(_normalize_value)
        merged_unique = merged.loc[~normalized_merged.duplicated()].reset_index(drop=True)
        merged_unique.to_csv(output_csv, index=False, quoting=1)  # QUOTE_ALL
        added = len(merged_unique) - len(existing_df)
        logger.info(f"Checkpoint saved: added {added} deduplicated rows to {output_csv}")
    else:
        dedup_new_rows.to_csv(output_csv, index=False, quoting=1)  # QUOTE_ALL
        logger.info(f"Checkpoint saved: wrote {len(dedup_new_rows)} deduplicated rows to {output_csv}")


def s3_client():
    return boto3.client("s3")


def download_from_s3(s3_key: str, local_path: str) -> bool:
    bucket = os.environ.get("S3_OUTPUT_BUCKET")
    if not bucket or not s3_key:
        return False
    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        s3_client().download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{s3_key} → {local_path}")
        return True
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"S3 download failed for {s3_key}: {e}")
        return False


def upload_to_s3(local_path: str, s3_key: str) -> None:
    bucket = os.environ.get("S3_OUTPUT_BUCKET")
    if not bucket:
        return
    if not os.path.exists(local_path):
        logger.warning(f"S3 upload skipped: {local_path} not found")
        return
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} → s3://{bucket}/{s3_key}")
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 upload failed for {local_path}: {e}")


def _default_api_provider(model: str) -> str:
    m = model.lower()
    if "gemini" in m:
        return "portkey"
    if "claude" in m:
        return "anthropic"
    return "openai"


def _default_use_portkey(model: str) -> bool:
    return "gemini" in model.lower()


def _default_prompt(model: str, portkey: bool=True) -> str:
    m = model.lower()
    if m.startswith(("hf-", "local-")):
        return "T5_primer"
    if "claude" in m:
        return "CLAUDE_RTR_FewShot"
    if "gemini" in m and not portkey:
        return "GEMINI_RTR_FewShot"
    return "GPT_FewShot"


def _default_response_format(model: str):
    m = model.lower()
    if m.startswith("gpt") or m.startswith("openai"):
        return dataset_response_schema_gpt
    if "gemini" in m:
        # Portkey uses OpenAI chat/completions format, not native Gemini params
        return dataset_response_schema_gpt_completions
    if "claude" in m:
        return dataset_response_schema_claude
    return None


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
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Stop after processing this many new articles (for iterative enrichment)")
    parser.add_argument("--ontology-path", default=None,
                        help="Directory containing open_bio_data_repos.json to use instead of the bundled config")
    parser.add_argument("--s3-input-key", default=None,
                        help="S3 key to download the input CSV from (e.g. input/slice_1.csv)")
    parser.add_argument("--s3-ontology-key", default=None,
                        help="S3 key to download the ontology JSON from (e.g. ontology/open_bio_data_repos.json)")
    parser.add_argument("--s3-skip-urls-key", default=None,
                        help="S3 key to a JSON file containing a list of source_urls already processed in prior iterations")
    parser.add_argument(
        "--skip-already-processed",
        type=lambda v: v.lower() not in ("false", "0", "no", "off"),
        default=False, metavar="BOOL",
        help="Skip source_urls already listed in --s3-skip-urls-key (default: false). Set true to avoid reprocessing URLs done in prior iterations.",
    )
    parser.add_argument("--s3-backup-key", default=None,
                        help="S3 key to download the local-fetch backup parquet from (e.g. cache/Local_fetched_data.parquet)")
    parser.add_argument(
        "--section-filter",
        default=None,
        choices=["data_availability_statement", "supplementary_material"],
        help="Restrict extraction to one section type (default: both)",
    )
    parser.add_argument(
        "--semantic-retrieval",
        type=lambda v: v.lower() not in ("false", "0", "no", "off"),
        default=True, metavar="BOOL",
        help="Enable semantic section retrieval (default: true)",
    )
    parser.add_argument(
        "--brute-force-regex",
        type=lambda v: v.lower() not in ("false", "0", "no", "off"),
        default=True, metavar="BOOL",
        help="Enable brute-force regex ID pointer scan (default: true)",
    )
    parser.add_argument(
        "--top-k", type=lambda x: -1 if str(x).lower() == "all" else int(x), default=5,
        help="Top-k sections for semantic retrieval (default: 5). Pass 'all' or -1 to process every corpus chunk (chunked document read).",
    )
    parser.add_argument(
        "--sects-required", type=int, default=5,
        help="Minimum number of sections required for an article to be processed (default: 5).",
    )
    parser.add_argument(
        "--full-document-read",
        type=lambda v: v.lower() not in ("false", "0", "no", "off"),
        default=False, metavar="BOOL",
        help="Feed the entire normalized document to the LLM in one shot instead of retrieving sections "
             "(default: false). Only takes effect for models in entire_document_models.",
    )
    parser.add_argument(
        "--prompt-name", default=None,
        help="Prompt template name (auto-detected from model if not set)",
    )
    parser.add_argument(
        "--use-batch-api",
        type=lambda v: v.lower() not in ("false", "0", "no", "off"),
        default=True, metavar="BOOL",
        help="Use async Batch API (default: true). Set false for synchronous commercial API calls.",
    )
    # arg backup_file
    parser.add_argument(
        "--backup-file", default='scripts/exp_input/Local_fetched_data.parquet',
        help="Path to backup file for saving intermediate results (default: None). If set, the processor will periodically save the current state to this file.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    output_csv = os.path.join(args.output_dir, "dataset_citations.csv")

    # Download input CSV from S3 if provided and not already present
    if args.s3_input_key and not os.path.exists(args.input):
        download_from_s3(args.s3_input_key, args.input)

    # Download ontology from S3 into output_dir so it can be found by user_config_dir
    if args.s3_ontology_key:
        ontology_local = os.path.join(args.output_dir, "open_bio_data_repos.json")
        if download_from_s3(args.s3_ontology_key, ontology_local):
            args.ontology_path = args.output_dir

    # Download local-fetch backup parquet from S3 if provided and not already present
    if args.s3_backup_key:
        backup_local = os.path.join(args.output_dir, "Local_fetched_data.parquet")
        if os.path.exists(backup_local) or download_from_s3(args.s3_backup_key, backup_local):
            args.backup_file = backup_local
        else:
            logger.warning(f"Backup parquet not found at s3://.../{args.s3_backup_key} — proceeding without local-fetch cache")

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

    # Checkpoint: skip already-processed URLs (within this job run)
    done_urls = load_checkpoint(output_csv)

    # Also skip URLs processed in previous iterations (downloaded from S3)
    if args.skip_already_processed and args.s3_skip_urls_key:
        skip_local = os.path.join(args.output_dir, "skip_urls.json")
        if download_from_s3(args.s3_skip_urls_key, skip_local):
            import json as _json
            with open(skip_local) as _f:
                prior_done = set(_json.load(_f))
            logger.info(f"Loaded {len(prior_done)} skip URLs from S3 ({args.s3_skip_urls_key})")
            done_urls = done_urls | prior_done

    pending_urls = [u for u in all_urls if u not in done_urls]
    logger.info(f"Checkpoint: {len(done_urls)} done, {len(pending_urls)} pending")

    if not pending_urls:
        logger.info("All URLs already processed. Nothing to do.")
        return

    if args.max_articles is not None and len(pending_urls) > args.max_articles:
        logger.info(f"--max-articles {args.max_articles}: capping pending from {len(pending_urls)} to {args.max_articles}")
        pending_urls = pending_urls[:args.max_articles]

    # Import here so the module is importable even without GPU during syntax checks
    from data_gatherer.data_gatherer import DataGatherer

    dg = DataGatherer(
        llm_name=args.model,
        save_to_cache=True,
        load_from_cache=True,
        embeds_cache_read=True,
        embeds_cache_write=True,
        log_file_override=log_file,
        log_level=logging.INFO,
        user_config_dir=args.ontology_path,
        raw_data_df_parquet_filepath=args.backup_file,
        process_entire_document=args.full_document_read,
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
                prompt_name=args.prompt_name or _default_prompt(args.model, portkey=_default_use_portkey(args.model)),
                response_format=_default_response_format(args.model),
                semantic_retrieval=args.semantic_retrieval,
                top_k=args.top_k,
                sects_required=args.sects_required,
                brute_force_RegEx_ID_ptrs=args.brute_force_regex,
                use_portkey=_default_use_portkey(args.model),
                use_batch_api=args.use_batch_api,
                api_provider=_default_api_provider(args.model),
                local_fetch_file=args.backup_file,
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

    # Ensure output CSV exists even when no citations were found, so S3 always has a file
    if not os.path.exists(output_csv):
        pd.DataFrame().to_csv(output_csv, index=False)
        logger.info(f"No citations found — wrote empty CSV to {output_csv}")

    # Upload outputs to S3 (key prefix = output dir basename, e.g. slice_1)
    prefix = os.path.basename(args.output_dir.rstrip("/"))
    upload_to_s3(output_csv, f"{prefix}/dataset_citations.csv")
    upload_to_s3(os.path.join(args.output_dir, "articles_log.csv"), f"{prefix}/articles_log.csv")
    upload_to_s3(os.path.join(args.output_dir, "run.log"), f"{prefix}/run.log")


if __name__ == "__main__":
    main()

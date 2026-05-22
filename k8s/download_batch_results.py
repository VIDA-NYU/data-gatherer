"""
Download and convert OpenAI Batch API results to CSV files.

Each batch ID maps to its own output directory (one per config run).

Usage:
    python k8s/download_batch_results.py \
        --batch-ids batch_abc123 batch_def456 batch_ghi789 \
        --output-dirs k8s/output/rev_test_gpt5mini_c1 \
                      k8s/output/rev_test_gpt5mini_c2 \
                      k8s/output/rev_test_gpt5mini_c3 \
        --model gpt-5-mini
"""

import argparse
import logging
import os
import sys

import pandas as pd

LOG_FMT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download and process OpenAI batch results")
    parser.add_argument("--batch-ids", nargs="+", required=True, help="OpenAI batch job IDs (one per config)")
    parser.add_argument("--output-dirs", nargs="+", required=True, help="Output directories (one per batch ID)")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name used for the batch jobs")
    args = parser.parse_args()

    if len(args.batch_ids) != len(args.output_dirs):
        logger.error("--batch-ids and --output-dirs must have the same number of entries")
        sys.exit(1)

    from data_gatherer.data_gatherer import DataGatherer

    dg = DataGatherer(llm_name=args.model, save_to_cache=False, load_from_cache=False)
    dg.init_parser_by_input_type("XML", use_portkey=False)

    for bid, output_dir in zip(args.batch_ids, args.output_dirs):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Checking status for {bid} → {output_dir}")
        status_info = dg.parser.llm_client.check_batch_status(bid, api_provider="openai")
        status = status_info["status"]
        logger.info(f"  Status: {status}")

        if status != "completed":
            logger.warning(f"  Not completed yet, skipping.")
            continue

        jsonl_path = os.path.join(output_dir, "batch_results.jsonl")
        logger.info(f"  Downloading to {jsonl_path}")
        dg.parser.llm_client.download_batch_results(bid, jsonl_path, api_provider="openai")

        logger.info(f"  Converting to DataFrame")
        df = dg.from_batch_resp_file_to_df(jsonl_path, skip_validation=True)
        logger.info(f"  Got {len(df)} rows")

        output_csv = os.path.join(output_dir, "dataset_citations.csv")
        df.to_csv(output_csv, index=False, quoting=1)
        logger.info(f"  Saved to {output_csv}")


if __name__ == "__main__":
    main()

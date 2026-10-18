"""Find the marine benchmark publications in SciLaD (Foppiano et al., LREC 2026, https://aclanthology.org/2026.lrec-1.603/).

Scans the open English split (scilons/SciLaD-en-dedup-v1, ~11M rows, 700 parquet shards) reading only the id/DOI/title
columns, and keeps the rows whose DOI or normalized title matches a publication. Half of the SciLaD rows have no DOI,
hence the title match. Each hit carries the location of its GROBID TEI in the gated XML split
(scilons/SciLaD-all-xml-v1): `source_file_path` (partition) and `source_zip_content_path` (member of the partition's tei.zip),
which the notebook (§5a) reads.

Takes ~1.5 h (16 threads). Usage:
    python scilad_lookup.py gt/marine_publications.parquet gt/scilad_marine_hits.parquet
"""
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SHARDS = "datasets/scilons/SciLaD-en-dedup-v1/data/*.parquet"
COLS = ["id", "metadata.doi", "metadata.title", "metadata.publication_year",
        "metadata.source_file_path", "metadata.source_zip_content_path"]
MIN_TITLE_LEN = 25  # shorter normalized titles ("Editorial", "Introduction", …) match unrelated papers


def compact_title(t) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def main(pubs_path: str, out_path: str) -> None:
    pubs = pd.read_parquet(pubs_path)
    dois = set(pubs["doi"].dropna().str.lower().str.strip())
    titles = {t for t in pubs["title"].dropna().map(compact_title) if len(t) >= MIN_TITLE_LEN}
    fs = HfFileSystem()

    def scan(path):
        with fs.open(path, block_size=2**20) as f:
            df = pq.ParquetFile(f).read(columns=COLS).flatten().to_pandas()
        df.columns = [c.replace("metadata.", "") for c in df.columns]
        hit = df[df["doi"].fillna("").str.lower().str.strip().isin(dois) | df["title"].fillna("").map(compact_title).isin(titles)]
        return hit.assign(file=path.rsplit("/", 1)[1]), len(df)

    files = sorted(fs.glob(SHARDS))
    out, n_rows, t0 = [], 0, time.time()
    with ThreadPoolExecutor(16) as ex:
        for i, fut in enumerate(as_completed([ex.submit(scan, p) for p in files]), 1):
            hit, n = fut.result()
            out.append(hit)
            n_rows += n
            if i % 50 == 0:
                print(f"{i}/{len(files)} shards, {n_rows:,} rows, {sum(map(len, out))} hits, {time.time() - t0:.0f}s", flush=True)
    hits = pd.concat(out, ignore_index=True)
    hits.to_parquet(out_path)
    print(f"{len(hits)} hits in {n_rows:,} rows -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

#!/usr/bin/env python3
"""
Ontology enrichment script — runs between k8s batch jobs.

Pipeline:
  1. Load dataset_citations.csv from the completed batch run
  2. Filter FP candidates: data_repository=NaN AND identifier doesn't match
     any existing id_pattern in the current ontology
  3. 2D embed: char n-gram TF-IDF on dataset_identifier +
               MiniLM on repository_reference, concatenated
  4. HDBSCAN cluster (noise discarded)
  5. Claude reviews each cluster ranked by #articles: induces regex,
     self-verifies coverage and false-match rate, decides whether to promote
  6. Write new ontology JSON to --output

Usage:
    python scripts/enrich_ontology.py \\
        --citations k8s/output/dataset_citations.csv \\
        --current-ontology data_gatherer/config/open_bio_data_repos_seed.json \\
        --ground-truth scripts/exp_input/Full_REV_dataset_citation_records_Table.parquet \\
        --output /data/ontology/open_bio_data_repos.json
"""

import argparse
import ast
import json
import logging
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_gatherer.prompts.prompt_manager import PromptManager


class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return False

import anthropic
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    import umap
    import plotly.express as px
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    candidates_df: pd.DataFrame,
    agent_results: dict[int, dict],
    output_path: str,
) -> None:
    """Reduce embeddings to 2D via UMAP and write an interactive plotly scatter."""
    if not _PLOT_AVAILABLE:
        print("[plot] umap-learn or plotly not installed — skipping plot.")
        return

    print("\nReducing to 2D for plot (UMAP)...")
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
    coords = reducer.fit_transform(embeddings)

    df = candidates_df.copy().reset_index(drop=True)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = labels

    # Cluster label: promoted repo name > top repo_ref > cluster id > "noise"
    def cluster_label(cid):
        if cid == -1:
            return "noise"
        res = agent_results.get(cid, {})
        if res.get("promote") and res.get("repo_name"):
            return f"✓ {res['repo_name']} ({res.get('id_pattern','')})"
        return f"cluster {cid}"

    df["label"] = df["cluster"].apply(cluster_label)

    # Sample IDs for hover (truncate long lists)
    df["hover_id"] = df["dataset_identifier"].fillna("").astype(str)
    df["hover_ref"] = df["repository_reference"].fillna("").astype(str)

    fig = px.scatter(
        df,
        x="x", y="y",
        color="label",
        hover_data={"hover_id": True, "hover_ref": True, "x": False, "y": False},
        title="FP candidate clusters (UMAP projection)",
        labels={"hover_id": "identifier", "hover_ref": "repo_reference", "label": "cluster"},
        opacity=0.7,
        width=1100, height=750,
    )
    fig.update_traces(marker=dict(size=5))
    fig.write_html(output_path)
    print(f"  Plot written to {output_path}")


def plot_groups(
    summaries: list[dict],
    decisions: list[dict],
    n_hallucination_filtered: int,
    n_below_min: int,
    output_path: str,
) -> None:
    """Bar chart of group sizes colored by agent decision."""
    if not _PLOT_AVAILABLE:
        print("[plot] plotly not installed — skipping plot.")
        return

    decision_map = {d["repository_reference"]: d for d in decisions}

    COLOR = {
        "promote":        "#2ecc71",
        "update_pattern": "#f39c12",
        "skip":           "#95a5a6",
        "filtered_hallucination": "#e74c3c",
        "filtered_min":   "#bdc3c7",
        "no_decision":    "#d0d0d0",
    }

    rows = []
    for s in summaries:
        ref = s["repository_reference"]
        dec = decision_map.get(ref, {})
        action = dec.get("action", "no_decision")
        rows.append({
            "repository_reference": ref,
            "n_articles": s["n_articles"],
            "n_ids": s["n_ids"],
            "n_webpages": s.get("n_webpages", 0),
            "action": action,
            "repo_name": dec.get("repo_name") or ref,
            "id_pattern": dec.get("id_pattern") or "",
            "reason": dec.get("reason") or "",
            "color": COLOR.get(action, "#d0d0d0"),
        })

    rows.sort(key=lambda x: x["n_articles"], reverse=True)
    plot_df = pd.DataFrame(rows)

    import plotly.graph_objects as go
    fig = go.Figure()
    for action, color in COLOR.items():
        sub = plot_df[plot_df["action"] == action]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["repository_reference"],
            y=sub["n_articles"],
            name=action,
            marker_color=color,
            customdata=sub[["n_ids", "n_webpages", "repo_name", "id_pattern", "reason"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "articles=%{y}  ids=%{customdata[0]}  webpages=%{customdata[1]}<br>"
                "repo_name=%{customdata[2]}<br>"
                "pattern=%{customdata[3]}<br>"
                "reason=%{customdata[4]}<extra></extra>"
            ),
        ))

    title = (
        f"Group pipeline — {len(summaries)} groups reviewed  |  "
        f"{n_hallucination_filtered} hallucination-filtered  |  "
        f"{n_below_min} below min_cluster_size"
    )
    fig.update_layout(
        title=title,
        xaxis_title="repository_reference",
        yaxis_title="n_articles",
        barmode="stack",
        xaxis_tickangle=-45,
        width=max(900, len(rows) * 22),
        height=550,
        legend_title="agent decision",
    )
    fig.write_html(output_path)
    print(f"  Plot written to {output_path}")


# ---------------------------------------------------------------------------
# Ontology helpers
# ---------------------------------------------------------------------------

def load_ontology(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_id_patterns(ontology: dict) -> list[re.Pattern]:
    patterns = []
    for repo in ontology["repos"].values():
        pat = repo.get("id_pattern")
        if pat:
            try:
                patterns.append(re.compile(pat, re.IGNORECASE))
            except re.error:
                pass
    return patterns


def matches_any_pattern(identifier: str, patterns: list[re.Pattern]) -> bool:
    if not isinstance(identifier, str) or not identifier.strip():
        return False
    return any(p.search(identifier) for p in patterns)


def pattern_examples(ontology: dict) -> dict:
    """Returns {ontology_key: {repo_name, id_pattern}} for use in agent prompts."""
    return {
        key: {"repo_name": data.get("repo_name", key), "id_pattern": data["id_pattern"]}
        for key, data in ontology["repos"].items()
        if data.get("id_pattern")
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_candidates(df: pd.DataFrame, patterns: list[re.Pattern]) -> pd.DataFrame:
    """Keep rows that are genuine unknowns: no resolved repo, no pattern match,
    but the model did extract a repository name from the article text."""
    mask = (
        df["data_repository"].isna()
        & df["dataset_identifier"].notna()
        & df["repository_reference"].notna()
        & ~df["dataset_identifier"].apply(lambda x: matches_any_pattern(x, patterns))
    )
    return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_candidates(df: pd.DataFrame) -> np.ndarray:
    identifiers = df["dataset_identifier"].fillna("").tolist()
    repo_refs = df["repository_reference"].fillna("").tolist()

    # Char n-gram TF-IDF captures structural prefix patterns (GSE*, PXD*, syn*)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=512)
    id_vecs = normalize(tfidf.fit_transform(identifiers).toarray())

    # Sentence transformer for free-text repo names
    model = SentenceTransformer("all-MiniLM-L6-v2")
    repo_vecs = normalize(model.encode(repo_refs, batch_size=128, show_progress_bar=True))

    return np.hstack([id_vecs, repo_vecs])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def build_summaries(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    df = df.copy()
    df["_cluster"] = labels
    clustered = df[df["_cluster"] != -1]

    summaries = []
    for cid, grp in clustered.groupby("_cluster"):
        n_articles = grp["source_url"].nunique() if "source_url" in grp.columns else len(grp)
        sample_webpages = (
            grp["dataset_webpage"].dropna().unique().tolist()[:3]
            if "dataset_webpage" in grp.columns else []
        )
        summaries.append({
            "cluster_id": int(cid),
            "n_articles": n_articles,
            "n_ids": int(grp["dataset_identifier"].nunique()),
            "top_repo_refs": grp["repository_reference"].value_counts().head(5).to_dict(),
            "sample_ids": grp["dataset_identifier"].dropna().unique().tolist()[:30],
            "sample_webpages": sample_webpages,
        })

    summaries.sort(key=lambda x: x["n_articles"], reverse=True)
    return summaries


def build_group_summaries(df: pd.DataFrame) -> list[dict]:
    """Group candidates by repository_reference, sorted by article evidence desc."""
    summaries = []
    for repo_ref, grp in df.groupby("repository_reference", sort=False):
        n_articles = grp["source_url"].nunique() if "source_url" in grp.columns else len(grp)
        sample_webpages = (
            grp["dataset_webpage"].dropna().unique().tolist()[:3]
            if "dataset_webpage" in grp.columns else []
        )

        # das_ratio: average fraction of sections that are Data Availability Statements,
        # computed per article (deduped by source_url) to avoid inflating via multi-citation rows.
        # High das_ratio (≥0.4) is strong evidence this is a data repository, not a reagent.
        das_ratio = None
        if {"n_das_sections", "n_corpus_sections", "source_url"}.issubset(grp.columns):
            art = grp.drop_duplicates("source_url")
            corpus = art["n_corpus_sections"].fillna(0).astype(float)
            valid = corpus > 0
            if valid.any():
                das = art.loc[valid, "n_das_sections"].fillna(0).astype(float)
                das_ratio = round(float((das / corpus[valid]).mean()), 3)

        # top_sections: most frequent section types where these identifiers appeared.
        # "Data Availability Statement" dominance confirms a data repository.
        top_sections: list[str] = []
        if "retrieved_sections_title" in grp.columns:
            counts: dict[str, int] = {}
            for val in grp["retrieved_sections_title"].dropna():
                try:
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, list):
                        for title in parsed:
                            top = str(title).split(" > ")[0].strip()
                            counts[top] = counts.get(top, 0) + 1
                except Exception:
                    pass
            top_sections = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:5]]

        summaries.append({
            "repository_reference": str(repo_ref),
            "n_articles": n_articles,
            "n_ids": int(grp["dataset_identifier"].nunique()),
            "n_webpages": int(grp["dataset_webpage"].dropna().nunique()) if "dataset_webpage" in grp.columns else 0,
            "das_ratio": das_ratio,
            "top_sections": top_sections,
            "sample_ids": grp["dataset_identifier"].dropna().unique().tolist()[:20],
            "sample_webpages": sample_webpages,
        })
    summaries.sort(key=lambda x: x["n_articles"], reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# Agent review
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """\
You are auditing a cluster of dataset identifiers extracted from biomedical articles.
Decide the correct action for this cluster relative to the ontology.

## Existing ontology entries (key → repo_name + id_pattern):
{examples}

## Cluster:
- Distinct articles citing these IDs: {n_articles}
- Distinct IDs in cluster: {n_ids}
- Repository names mentioned in articles (extracted verbatim): {repo_refs}
- Sample identifiers: {sample_ids}
- Sample dataset webpage URLs: {sample_webpages}

## Instructions:
Choose exactly one action:

**"promote"** — cluster is a real, previously-unknown repository with a stable ID format
  not yet in the ontology. Write a new `id_pattern` regex for it.

**"update_pattern"** — cluster is a real repository ALREADY in the ontology, but the sample
  IDs include variants NOT matched by the existing pattern. Extend the existing pattern with
  the new alternatives (existing_pattern|new_alternatives). Set `existing_repo_key` to the
  ontology key (e.g. "geo", "synapse.org") that should be updated.

**"skip"** — cluster is noise, a funding body, too heterogeneous, or has no stable ID format.

CRITICAL consistency rules:
- If your reason says a repo "should be added" or "is not yet in the ontology" → action MUST be "promote".
- If your reason says IDs belong to an existing repo but aren't matched → action MUST be "update_pattern".
- A reason that recommends promotion/update while action is "skip" is a logical error.

Respond with JSON only — no markdown fences:
{{
  "action": "promote" | "update_pattern" | "skip",
  "reason": "...",
  "repo_name": "...",
  "existing_repo_key": "...",
  "id_pattern": "...",
  "pattern_coverage": 0.0,
  "over_broad": false,
  "confidence": "high" | "medium" | "low"
}}"""


def agent_review(client: anthropic.Anthropic, cluster_summary: dict, examples: dict) -> dict:
    prompt = REVIEW_PROMPT.format(
        examples=json.dumps(examples, indent=2),
        n_articles=cluster_summary["n_articles"],
        n_ids=cluster_summary["n_ids"],
        repo_refs=json.dumps(cluster_summary["top_repo_refs"], indent=2),
        sample_ids=json.dumps(cluster_summary["sample_ids"], indent=2),
        sample_webpages=json.dumps(cluster_summary.get("sample_webpages", []), indent=2),
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tolerate stray whitespace or a single markdown fence
        cleaned = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  [warn] Could not parse agent response: {e}\n  Raw: {text[:200]}")
            return {"promote": False, "reason": f"parse error: {e}"}


# ---------------------------------------------------------------------------
# Bulk agent review (group pipeline)
# ---------------------------------------------------------------------------

def bulk_agent_review(
    client: anthropic.Anthropic,
    group_summaries: list[dict],
    examples: dict,
    pm: PromptManager,
) -> list[dict]:
    tmpl = pm.load_prompt("bulk_repo_review", subdir="agent_for_FP_analysis")
    # Format directly (not via pm.render_prompt) because render_prompt escapes { in string
    # values, which corrupts the JSON ontology block.
    system_content = tmpl[0]["content"].format(existing_repos=json.dumps(examples, indent=2))
    user_content = tmpl[1]["content"].format(groups=json.dumps(group_summaries, indent=2))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_content,
        messages=[{"role": "user", "content": user_content}],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  [warn] Could not parse bulk response: {e}\n  Raw: {text[:400]}")
            return []


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

def verify(pattern_str: str, cluster_ids: list[str], all_ids: list[str]) -> tuple[float, float]:
    """Returns (coverage_on_cluster, false_match_rate_on_non_cluster)."""
    try:
        pat = re.compile(pattern_str, re.IGNORECASE)
    except re.error as e:
        print(f"  [warn] Invalid regex '{pattern_str}': {e}")
        return 0.0, 1.0

    cluster_set = set(cluster_ids)
    coverage = sum(1 for x in cluster_ids if pat.search(str(x))) / max(len(cluster_ids), 1)

    non_cluster = [x for x in all_ids if x not in cluster_set][:2000]
    false_rate = (
        sum(1 for x in non_cluster if pat.search(str(x))) / len(non_cluster)
        if non_cluster else 0.0
    )
    return coverage, false_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enrich the data-repos ontology from batch FP signal")
    parser.add_argument("--citations", required=True, nargs="+", metavar="PATH",
                        help="Path(s) to dataset_citations.csv. "
                             "Pass multiple paths to merge across iterations (cumulative mode).")
    parser.add_argument("--current-ontology", required=True,
                        help="Current ontology JSON (seed or previous iteration)")
    parser.add_argument("--output", required=True,
                        help="Path to write the enriched ontology JSON")
    parser.add_argument("--pipeline", choices=["cluster", "group"], default="cluster",
                        help="'cluster': embed→HDBSCAN→per-cluster Claude (default); "
                             "'group': group by repo_reference→Claude in batches")
    parser.add_argument("--group-batch-size", type=int, default=10,
                        help="Groups per Claude call in group pipeline (default: 10, 1 = one-by-one)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Min distinct articles to review a cluster/group (default: 5)")
    parser.add_argument("--min-ids", type=int, default=3,
                        help="Min distinct identifiers to review a group (default: 3)")
    parser.add_argument("--coverage-threshold", type=float, default=0.5,
                        help="Min regex coverage on cluster/group to accept (default: 0.5)")
    parser.add_argument("--false-rate-threshold", type=float, default=0.05,
                        help="Max false-match rate on non-cluster IDs to accept (default: 0.05)")
    parser.add_argument("--plot", default=None, metavar="PATH",
                        help="Write interactive cluster plot (cluster pipeline only; requires umap-learn + plotly)")
    parser.add_argument("--log", default=None, metavar="PATH",
                        help="Append all stdout output to this log file in addition to the console")
    args = parser.parse_args()

    import os
    if not os.environ.get("ANTHROPIC_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                break

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

    # --- Load ---
    print("Loading citations...")
    if len(args.citations) == 1:
        df = pd.read_csv(args.citations[0])
        print(f"  {len(df)} rows from {args.citations[0]}")
    else:
        print(f"  Merging {len(args.citations)} citation files (cumulative)...")
        dfs = []
        for p in args.citations:
            try:
                _df = pd.read_csv(p)
                dfs.append(_df)
                print(f"    {len(_df):>6} rows  {p}")
            except FileNotFoundError:
                print(f"    [warn] {p} not found — skipping")
        if not dfs:
            print("No citation files loaded — aborting.")
            sys.exit(1)
        df = pd.concat(dfs, ignore_index=True)
        before = len(df)
        dedup_cols = [c for c in ["dataset_identifier", "source_url"] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        print(f"  Merged total: {len(df)} rows ({before - len(df)} duplicates dropped)")

    ontology = load_ontology(args.current_ontology)
    patterns = get_id_patterns(ontology)
    examples = pattern_examples(ontology)
    print(f"  Ontology has {len(ontology['repos'])} repos, {len(patterns)} id_patterns")

    # --- Filter ---
    print("\nFiltering FP candidates...")
    candidates = filter_candidates(df, patterns)
    print(f"  {len(candidates)} candidates (from {len(df)} total rows, "
          f"{len(df) - len(candidates)} filtered out)")

    if candidates.empty:
        print("No candidates — ontology unchanged.")
        import shutil; shutil.copy(args.current_ontology, args.output)
        return

    all_ids = candidates["dataset_identifier"].dropna().tolist()

    new_entries: dict[str, dict] = {}
    pattern_updates: dict[str, str] = {}
    embeddings = labels = agent_results = None  # cluster pipeline only

    # ------------------------------------------------------------------ #
    # Pipeline A — embed → HDBSCAN → per-cluster Claude                   #
    # ------------------------------------------------------------------ #
    if args.pipeline == "cluster":
        print("\nEmbedding candidates...")
        embeddings = embed_candidates(candidates)

        print("\nClustering...")
        labels = cluster(embeddings, args.min_cluster_size)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  {n_clusters} clusters found, {n_noise} noise points discarded")

        if n_clusters == 0:
            print("No clusters formed — ontology unchanged.")
            import shutil; shutil.copy(args.current_ontology, args.output)
            return

        summaries = build_summaries(candidates, labels)
        before_halluc = len(summaries)
        summaries = [s for s in summaries if s["n_ids"] > 1]
        if before_halluc - len(summaries):
            print(f"  Hallucination filter: skipped {before_halluc - len(summaries)} clusters with single repeated identifier")
        client = anthropic.Anthropic()
        agent_results = {}

        print(f"\nAgent reviewing {len(summaries)} clusters (ranked by #articles)...\n")
        for i, summary in enumerate(summaries):
            top_repo = list(summary["top_repo_refs"].keys())[:1]

            print(f"[{i+1}/{len(summaries)}] cluster={summary['cluster_id']} "
                  f"articles={summary['n_articles']} ids={summary['n_ids']} "
                  f"top_ref={top_repo}")

            result = agent_review(client, summary, examples)     
            agent_results[summary["cluster_id"]] = result

            action = result.get("action") or ("promote" if result.get("promote") else "skip")
            reason = result.get("reason", "")
            pattern_str = (result.get("id_pattern") or "").strip()
            repo_name = (result.get("repo_name") or "").strip()
            confidence = result.get("confidence", "?")
            sample_ids = summary["sample_ids"]

            if action == "skip":
                print(f"  → skip: {reason}")
                continue

            if action == "update_pattern":
                existing_key = (result.get("existing_repo_key") or "").strip()
                if not existing_key or existing_key not in ontology["repos"]:
                    print(f"  → skip: update_pattern but key '{existing_key}' not found")
                    continue
                if not pattern_str:
                    print(f"  → skip: empty id_pattern")
                    continue
                coverage, false_rate = verify(pattern_str, sample_ids, all_ids)
                existing_name = ontology["repos"][existing_key].get("repo_name", existing_key)
                print(f"  → update_pattern: '{existing_name}' | {pattern_str} | "
                      f"coverage={coverage:.0%} false_rate={false_rate:.2%}")
                if coverage < args.coverage_threshold:
                    print(f"  ✗ rejected: coverage {coverage:.0%} < {args.coverage_threshold:.0%}"); continue
                if false_rate > args.false_rate_threshold:
                    print(f"  ✗ rejected: false_rate {false_rate:.2%} > {args.false_rate_threshold:.2%}"); continue
                pattern_updates[existing_key] = pattern_str
                print(f"  ✓ queued")
                continue

            # promote
            if not pattern_str or not repo_name:
                print(f"  → skip: empty pattern or name"); continue
            coverage, false_rate = verify(pattern_str, sample_ids, all_ids)
            print(f"  → promote: '{repo_name}' | {pattern_str} | "
                  f"coverage={coverage:.0%} false_rate={false_rate:.2%} confidence={confidence}")
            if coverage < args.coverage_threshold:
                print(f"  ✗ rejected: coverage {coverage:.0%} < {args.coverage_threshold:.0%}"); continue
            if result.get("over_broad"):
                print(f"  ✗ rejected: over-broad"); continue
            regex_eligible = bool(result.get("regex_match_eligible", False))
            if false_rate > args.false_rate_threshold:
                regex_eligible = False
                print(f"  [warn] false_rate {false_rate:.2%} > {args.false_rate_threshold:.2%} → regex_match_eligible forced False")
            entry_key = repo_name.lower().replace(" ", "_")
            mask = df["dataset_identifier"].isin(sample_ids)
            webpage_url = (
                df[mask]["dataset_webpage"].dropna().value_counts().idxmax()
                if "dataset_webpage" in df.columns and mask.any() and df[mask]["dataset_webpage"].dropna().any()
                else None
            )
            new_entries[entry_key] = {
                "repo_name": repo_name,
                "id_pattern": pattern_str,
                "regex_match_eligible": regex_eligible,
                "_sample_ids": sample_ids,
                **({"dataset_webpage_url_ptr": webpage_url} if webpage_url else {}),
            }
            print(f"  ✓ added  regex_match_eligible={regex_eligible}")

    # ------------------------------------------------------------------ #
    # Pipeline B — group by repo_reference → single bulk Claude call      #
    # ------------------------------------------------------------------ #
    else:
        print("\nGrouping by repository_reference...")
        summaries = build_group_summaries(candidates)
        before = len(summaries)
        summaries = [
            s for s in summaries
            if s["n_articles"] >= args.min_cluster_size and s["n_ids"] >= args.min_ids
        ]
        n_below_min = before - len(summaries)
        n_hallucination_filtered = 0  # folded into n_below_min
        print(f"  {len(summaries)} groups after filtering "
              f"({n_below_min} dropped: n_articles<{args.min_cluster_size} or n_ids<{args.min_ids})")

        # Drop groups whose repository_reference matches a known repo name AND whose
        # sample IDs are already well-covered by that repo's id_pattern.
        # Groups where the name matches but coverage is low are kept as update_pattern candidates.
        def _covered_by_known_repo(summary: dict) -> bool:
            ref = summary["repository_reference"].lower()
            for repo in ontology["repos"].values():
                known = repo.get("repo_name", "").lower()
                if not known:
                    continue
                if known not in ref and ref not in known:
                    continue
                pat_str = repo.get("id_pattern")
                if not pat_str:
                    return True  # name match, no pattern → nothing to update
                try:
                    pat = re.compile(pat_str, re.IGNORECASE)
                except re.error:
                    continue
                ids = summary.get("sample_ids", [])
                if not ids:
                    return True
                coverage = sum(1 for x in ids if pat.search(str(x))) / len(ids)
                if coverage >= args.coverage_threshold:
                    return True  # already well-covered → filter out
                return False  # low coverage → keep as update_pattern candidate
            return False

        before_known = len(summaries)
        summaries = [s for s in summaries if not _covered_by_known_repo(s)]
        n_known_filtered = before_known - len(summaries)
        if n_known_filtered:
            print(f"  {len(summaries)} groups after known-repo filter "
                  f"({n_known_filtered} already covered by existing ontology — skipped)")

        print(f"  Groups to review:")
        for s in summaries:
            print(f"    {s['repository_reference']!r:50s}  articles={s['n_articles']}  ids={s['n_ids']}  webpages={s['n_webpages']}")

        client = anthropic.Anthropic()
        pm = PromptManager(prompt_dir="data_gatherer/prompts/prompt_templates",
                           logger=logging.getLogger(__name__))
        bs = args.group_batch_size
        n_batches = (len(summaries) + bs - 1) // bs
        print(f"\nReviewing {len(summaries)} groups in {n_batches} batch(es) of up to {bs}...")

        decisions = []
        for b in range(n_batches):
            batch = summaries[b * bs:(b + 1) * bs]
            refs = [s["repository_reference"] for s in batch]
            print(f"\n  Batch {b+1}/{n_batches}: {refs}")
            batch_decisions = bulk_agent_review(client, batch, examples, pm)
            decisions.extend(batch_decisions)

        if not decisions:
            print("  [warn] No decisions returned — ontology unchanged.")
            import shutil; shutil.copy(args.current_ontology, args.output)
            return

        summary_by_ref = {s["repository_reference"]: s for s in summaries}
        print()
        for dec in decisions:
            ref = dec.get("repository_reference", "")
            action = dec.get("action", "skip")
            reason = dec.get("reason", "")
            pattern_str = (dec.get("id_pattern") or "").strip()
            repo_name = (dec.get("repo_name") or "").strip()
            confidence = dec.get("confidence", "?")
            summary = summary_by_ref.get(ref, {})
            sample_ids = summary.get("sample_ids", [])

            print(f"[{ref}]  action={action}  confidence={confidence}")

            if action == "skip":
                print(f"  → skip: {reason}")
                continue

            if action == "update_pattern":
                existing_key = (dec.get("existing_repo_key") or "").strip()
                if not existing_key or existing_key not in ontology["repos"]:
                    print(f"  → skip: update_pattern but key '{existing_key}' not found"); continue
                if not pattern_str:
                    print(f"  → skip: empty id_pattern"); continue
                coverage, false_rate = verify(pattern_str, sample_ids, all_ids)
                existing_name = ontology["repos"][existing_key].get("repo_name", existing_key)
                print(f"  → update_pattern '{existing_name}': {pattern_str} | "
                      f"coverage={coverage:.0%}  false_rate={false_rate:.2%}")
                if coverage < args.coverage_threshold:
                    print(f"  ✗ rejected: coverage {coverage:.0%} < {args.coverage_threshold:.0%}"); continue
                if false_rate > args.false_rate_threshold:
                    print(f"  ✗ rejected: false_rate {false_rate:.2%} > {args.false_rate_threshold:.2%}"); continue
                pattern_updates[existing_key] = pattern_str
                print(f"  ✓ queued")
                continue

            # promote
            if not pattern_str or not repo_name:
                print(f"  → skip: empty pattern or name"); continue
            coverage, false_rate = verify(pattern_str, sample_ids, all_ids)
            print(f"  → promote '{repo_name}': {pattern_str} | "
                  f"coverage={coverage:.0%}  false_rate={false_rate:.2%}")
            if coverage < args.coverage_threshold:
                print(f"  ✗ rejected: coverage {coverage:.0%} < {args.coverage_threshold:.0%}"); continue
            if dec.get("over_broad"):
                print(f"  ✗ rejected: over-broad"); continue
            regex_eligible = bool(dec.get("regex_match_eligible", False))
            if false_rate > args.false_rate_threshold:
                regex_eligible = False
                print(f"  [warn] false_rate {false_rate:.2%} > {args.false_rate_threshold:.2%} → regex_match_eligible forced False")
            entry_key = repo_name.lower().replace(" ", "_")
            webpage_url = (summary.get("sample_webpages") or [None])[0]
            new_entries[entry_key] = {
                "repo_name": repo_name,
                "id_pattern": pattern_str,
                "regex_match_eligible": regex_eligible,
                "_sample_ids": sample_ids,
                **({"dataset_webpage_url_ptr": webpage_url} if webpage_url else {}),
            }
            print(f"  ✓ added  regex_match_eligible={regex_eligible}")

    # --- Apply pattern updates to existing repos ---
    for key, new_pat in pattern_updates.items():
        ontology["repos"][key]["id_pattern"] = new_pat
        print(f"  [updated] {ontology['repos'][key].get('repo_name', key)}: {new_pat}")

    # --- Dedup: exact-match + intra-batch superset check ---
    seen_patterns: dict[str, str] = {
        repo.get("id_pattern"): repo.get("repo_name", key)
        for key, repo in ontology["repos"].items()
        if repo.get("id_pattern")
    }
    # First pass: exact-match dedup against ontology
    deduped: dict[str, dict] = {}
    for key, entry in new_entries.items():
        pat = entry["id_pattern"]
        if pat in seen_patterns:
            print(f"  [dedup] '{entry['repo_name']}' has same pattern as '{seen_patterns[pat]}' — skipped")
        else:
            seen_patterns[pat] = entry["repo_name"]
            deduped[key] = entry

    # Second pass: intra-batch superset check (catches UniProt-style P ⊂ Q pairs)
    new_keys = list(deduped.keys())
    subsumed: set[str] = set()
    for i, kA in enumerate(new_keys):
        if kA in subsumed:
            continue
        for kB in new_keys[i + 1:]:
            if kB in subsumed:
                continue
            try:
                patB = re.compile(deduped[kB]["id_pattern"], re.IGNORECASE)
            except re.error:
                continue
            ids_A = deduped[kA].get("_sample_ids", [])
            if not ids_A:
                continue
            overlap = sum(1 for x in ids_A if patB.search(str(x))) / len(ids_A)
            if overlap >= 0.8:
                print(f"  [dedup] '{deduped[kA]['repo_name']}' subsumed by "
                      f"'{deduped[kB]['repo_name']}' ({overlap:.0%} overlap) — dropped")
                subsumed.add(kA)
                break
    new_entries = {k: v for k, v in deduped.items() if k not in subsumed}

    # Remove internal field before writing
    for entry in new_entries.values():
        entry.pop("_sample_ids", None)


    # --- Plot ---
    if args.plot and args.pipeline == "cluster" and embeddings is not None:
        plot_clusters(embeddings, labels, candidates, agent_results, args.plot)
    elif args.plot and args.pipeline == "group":
        plot_groups(summaries, decisions, n_hallucination_filtered, n_below_min, args.plot)

    # --- Write ---
    ontology["repos"].update(new_entries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ontology, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {len(new_entries)} new repos added, {len(pattern_updates)} patterns updated.")
    for entry in new_entries.values():
        print(f"  + {entry['repo_name']}: {entry['id_pattern']}")
    for key in pattern_updates:
        print(f"  ~ {ontology['repos'][key].get('repo_name', key)}: {pattern_updates[key]}")
    print(f"Ontology written to {args.output}")


if __name__ == "__main__":
    main()

"""Construction-aware evaluation for the OpenAIRE Marine Science benchmark.

The curated MES graph (Irrera et al., JDIQ 2023, 10.1145/3597310) is not an exhaustive, text-derived gold standard:
- only `References` / `Cites` edges come from the publication PDF (GROBID); `IsSupplementedBy`, `IsReferencedBy`,
  `Documents` and `IsCitedBy` were scraped from dataset landing pages (§3). Only 14.27% of pairs are text-evidenced (§4);
- curation only checked pairs already linked in the OpenAIRE graph, after dropping nodes with incomplete metadata (§3–4),
  so datasets a paper cites but OpenAIRE never linked are missing;
- many text links were matched on the title only: the reference is the paper describing the dataset (§4, Table 8);
- `Removed` edges are curated negatives; supplementary copies on figshare are titled "Additional file N of <paper>".

This module assigns every ground-truth article–dataset pair to a reachability tier, scores recall per tier (with a
file-label rule for supplementary copies) and triages the predictions that match nothing. It sits next to the official
`evaluate_performance` scores and does not change them.

Tiers (first rule that applies):
    no_fulltext   article not in the full-text cache
    not_dataset   PLOS self identifiers (the paper's DOI, its .g/.t/.s parts) and other PLOS article DOIs
    text_id       Cites/References with a DOI/URL mention, or the identifier occurs in the article text
    text_no_id    Cites/References whose identifier is not in the text (mostly title-only references)
    suppl_label   the dataset title names a supplementary file of the article ("Additional file 4", "MOESM2", ...)
    page_only     everything else: links taken from the dataset page, identifier not in the text
"""
import html
import re
from collections import defaultdict
from urllib.parse import unquote

import pandas as pd

PMC_RE = re.compile(r"\b(PMC\d+)\b", re.I)
TEXT_SEMANTICS = {"Cites", "References"}
TIERS = ["text_id", "text_no_id", "suppl_label", "page_only", "not_dataset", "no_fulltext"]
SCORED_TIERS = ["text_id", "text_no_id", "suppl_label", "page_only"]
FP_CLASSES = ["removed", "in_graph_other", "suppl_label", "out_of_graph_in_text", "name_in_text", "not_in_text"]
MIN_ALIAS_LEN = 6  # shorter strings ("s1", "moesm") match too many identifiers by substring
MIN_TEXT_LEN = 4
PLOS_ARTICLE_RE = re.compile(r"^10\.1371/journal\.[a-z]+\.\d{7}$")
DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)
# the article's own files: PLOS .s001, Springer _MOESM1_ESM, Pensoft treatments, or a data-file extension
OWN_FILE_RE = re.compile(r"(\.s\d{3}\b|moesm\d|_esm\b|-treatment\d|\.(xlsx?|docx?|csv|pdf|zip|rar|gz|txt|tiff?|mov|mp4|"
                         r"fas|fasta|nex|nxs|bed|ppt|pptx)$)")
# lists and accession ranges (ky565269-ky565305; T5 writes the en dash as a literal "u2013")
SPLIT_RE = re.compile(r"[\s,;'\"\[\]]+|(?<=\d)(?:-|u201[0-5])(?=[a-z]*\d)")

# Supplementary file labels, normalized to "<kind> <n>" so the GT title, the run's supplementary list and the
# predictions can be compared. MOESM numbers differ from "Additional file" numbers, so each is its own kind.
_KIND = r"(table|tab|fig|figure|data|dataset|text|video|movie|file|appendix|material|materials|information|note)"
LABEL_PATTERNS = [
    (re.compile(r"additional\s+file\s*(\d+)"), "additional file {}"),
    (re.compile(r"\bmoesm(\d+)"), "moesm {}"),
    (re.compile(r"\bappendix\s+([a-z])\b"), "appendix {}"),
    (re.compile(rf"\b{_KIND}\s+s(\d+)\b"), "{} s{}"),          # Table S3, Data S1
    (re.compile(rf"\bs(\d+)\s+{_KIND}\b"), "{1} s{0}"),        # S1 Table (PLOS)
    (re.compile(rf"\bsupplementa(?:ry|l)\s+{_KIND}\s+(\d+)"), "{} s{}"),  # Supplementary figure 1
]
_KIND_NORM = {"tab": "table", "fig": "figure", "dataset": "data", "movie": "video", "materials": "material"}


def norm_doi(s: str) -> str:
    return unquote(s).strip().lower().rstrip("/.").removesuffix(".pdf")


def norm_pred(s) -> str:
    return norm_doi(DOI_PREFIX_RE.sub("", str(s).strip()))


def url_to_key(url: str) -> str | None:
    if m := PMC_RE.search(url):
        return m.group(1).upper()
    return url.split("doi.org/", 1)[1].lower() if "doi.org/" in url else None


def plain_text(markup: str) -> str:
    """Text without markup, followed by the link targets (ext-link / supplementary-material hrefs)."""
    hrefs = " ".join(HREF_RE.findall(markup))
    return WS_RE.sub(" ", html.unescape(TAG_RE.sub(" ", markup) + " " + unquote(hrefs))).lower()


def load_fulltext(cache_path) -> dict:
    """{article_key: lowercased plain text} from the fetch cache (index = URL, `fetched_data` = XML/HTML)."""
    cache = pd.read_parquet(cache_path, columns=["fetched_data"])
    return {k: plain_text(str(t)) for u, t in cache["fetched_data"].items() if (k := url_to_key(str(u)))}


def file_labels(text) -> set:
    """Normalized supplementary-file labels mentioned in `text`, e.g. {"additional file 4", "table s3"}."""
    if not isinstance(text, str) or not text:
        return set()
    t = WS_RE.sub(" ", text.lower().replace("_", " "))
    out = set()
    for rx, fmt in LABEL_PATTERNS:
        for m in rx.finditer(t):
            g = [_KIND_NORM.get(x, x) for x in m.groups()]
            out.add(fmt.format(*g))
    return out


def title_labels(title) -> set:
    """Labels in the file-name part of a supplementary dataset title ("Additional file 4: Table S2. of <paper>")."""
    if not isinstance(title, str):
        return set()
    head = re.split(r"\s+(?:of|from)\s+", title.strip(), maxsplit=1)[0]
    return file_labels(head[:120])


def suppl_labels(suppl_csv) -> dict:
    """{article_key: labels} from a run's `*_suppl.csv` (labels live in `description`, `link` and `id`)."""
    df = pd.read_csv(suppl_csv, usecols=["source_url", "link", "id", "description"])
    out = defaultdict(set)
    for r in df.itertuples():
        key = url_to_key(str(r.source_url))
        for col in (r.link, r.id, r.description):
            out[key] |= file_labels(col)
    return dict(out)


def _alias(a: str, b) -> bool:
    """Same identifier: equal, or one contains the other including the longer one's final number
    (`pangaea.863126` ~ `10.1594/pangaea.863126`, but not `10.1594/pangaea` ~ any PANGAEA DOI)."""
    if not isinstance(b, str) or not b or not a:
        return False
    b = b.lower()
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < MIN_ALIAS_LEN or short not in long_:
        return False
    nums = DIGITS_RE.findall(long_)
    return not nums or nums[-1] in short


def _id_tail(identifier: str) -> str:
    """Repository-specific part of a DOI (`pangaea.863126` for `10.1594/pangaea.863126`)."""
    return identifier.split("/", 1)[1] if identifier.startswith("10.") and "/" in identifier else identifier


def in_text(identifier, text: str | None) -> bool:
    if text is None or not isinstance(identifier, str):
        return False
    i = identifier.lower()
    tail = _id_tail(i)
    return i in text or (len(tail) >= MIN_ALIAS_LEN and tail in text)


def pred_in_text(pred: str, text: str | None) -> bool:
    """The prediction, its DOI tail, or every numbered part of a list/range occurs in the article text."""
    if text is None or len(pred) < MIN_TEXT_LEN:
        return False
    if pred in text or _id_tail(pred) in text:
        return True
    parts = [s for s in SPLIT_RE.split(pred) if DIGITS_RE.search(s) and len(s) >= MIN_TEXT_LEN]
    return len(parts) > 1 and all(s in text for s in parts)


def build_pairs(gt_edges: pd.DataFrame, fulltext: dict) -> pd.DataFrame:
    """One row per (article_key, identifier) with its tier. `gt_edges` = dataset edges of the evaluated articles."""
    def _join(s):
        return "|".join(sorted({x for v in s.dropna() for x in str(v).split("|")}))

    pairs = (gt_edges.groupby(["article_key", "identifier"], as_index=False)
             .agg(semantics=("semantics", _join), mentioned_element=("mentioned_element", _join),
                  status=("status", _join), pub_doi=("pub_doi", "first"), doi_prefix=("doi_prefix", "first"),
                  artifact_title=("artifact_title", "first"), parent_identifier=("parent_identifier", "first")))
    pairs["in_text"] = [in_text(i, fulltext.get(k)) for k, i in zip(pairs["article_key"], pairs["identifier"])]
    pairs["labels"] = pairs["artifact_title"].map(title_labels)

    def tier(r) -> str:
        ident, pub_doi = str(r.identifier).lower(), str(r.pub_doi or "").lower()
        if r.article_key not in fulltext:
            return "no_fulltext"
        if ident.startswith("10.1371/") and ((pub_doi and (ident == pub_doi or ident.startswith(pub_doi + ".")))
                                            or PLOS_ARTICLE_RE.match(ident)):
            return "not_dataset"
        text_sem = bool(TEXT_SEMANTICS & set(r.semantics.split("|")))
        mentioned_id = bool({"doi", "persistent_url"} & set(r.mentioned_element.split("|")))
        if r.in_text or (text_sem and mentioned_id):
            return "text_id"
        if text_sem:
            return "text_no_id"
        return "suppl_label" if r.labels else "page_only"

    pairs["tier"] = pairs.apply(tier, axis=1)
    return pairs


def load_predictions(pred_csv, eval_ids) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    df = df[df["dataset_identifier"].notna()].copy()
    df["article_key"] = df["source_url"].astype(str).map(url_to_key)
    df["pred"] = df["dataset_identifier"].map(norm_pred)
    df = df[df["article_key"].isin(eval_ids) & ~df["pred"].isin(["", "n/a", "na", "none"])]
    return df.drop_duplicates(["article_key", "pred"])


def score_config(pairs: pd.DataFrame, pred: pd.DataFrame, neg: pd.DataFrame, graph_ids: set, fulltext: dict,
                 suppl: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Found flags per GT pair and a class per prediction.

    Pair columns: found_id (identifier alias match), found_parent (… or its IsPartOf parent),
    found_label_pred (supplementary label in a prediction), found_label_suppl (… in the run's supplementary list).
    """
    preds_by = pred.groupby("article_key")["pred"].apply(list).to_dict()
    pred_labels = {k: set().union(*map(file_labels, v)) for k, v in preds_by.items()}
    suppl = suppl or {}

    p = pairs.copy()

    def per_pair(col: str, hit) -> pd.Series:
        return pd.Series([hit(k, v) for k, v in zip(p["article_key"], p[col])], index=p.index, dtype=bool)

    p["found_id"] = per_pair("identifier", lambda k, i: any(_alias(x, i) for x in preds_by.get(k, ())))
    p["found_parent"] = p["found_id"] | per_pair("parent_identifier", lambda k, par: any(_alias(x, par) for x in preds_by.get(k, ())))
    p["found_label_pred"] = p["found_id"] | per_pair("labels", lambda k, lbl: bool(lbl & pred_labels.get(k, set())))
    p["found_label_suppl"] = p["found_label_pred"] | per_pair("labels", lambda k, lbl: bool(lbl & suppl.get(k, set())))

    gt_by = p[p["tier"] != "not_dataset"].groupby("article_key")
    gt_ids, gt_lbls = gt_by["identifier"].apply(list).to_dict(), gt_by["labels"].apply(lambda s: set().union(*s)).to_dict()
    self_ids = p[p["tier"] == "not_dataset"].groupby("article_key")["identifier"].apply(list).to_dict()
    neg_ids = neg.groupby("article_key")["identifier"].apply(list).to_dict()
    graph_tails = {_id_tail(g) for g in graph_ids}

    def classify(k: str, x: str) -> str:
        if any(_alias(x, i) for i in gt_ids.get(k, ())) or (file_labels(x) & gt_lbls.get(k, set())):
            return "tp"
        if any(_alias(x, i) for i in self_ids.get(k, ())):
            return "not_dataset"
        if any(_alias(x, i) for i in neg_ids.get(k, ())):
            return "removed"
        if x in graph_ids or x in graph_tails:
            return "in_graph_other"
        if file_labels(x) or OWN_FILE_RE.search(x):
            return "suppl_label"
        if pred_in_text(x, fulltext.get(k)):
            # identifiers carry a number; plain names ("medits", "census of marine life") are not citations of a record
            return "out_of_graph_in_text" if DIGITS_RE.search(x) else "name_in_text"
        return "not_in_text"

    pr = pred[["article_key", "pred", "data_repository"]].copy()
    pr["cls"] = [classify(k, x) for k, x in zip(pr["article_key"], pr["pred"])]
    return p, pr


def summarize(label: str, p: pd.DataFrame, pr: pd.DataFrame, has_suppl: bool) -> dict:
    row = {"config": label, "predictions": len(pr)}
    for t in SCORED_TIERS:
        s = p[p["tier"] == t]
        row[f"n_{t}"] = len(s)
        row[f"R_{t}"] = s["found_id"].mean() if len(s) else float("nan")
    t1 = p[p["tier"] == "text_id"]
    row["R_text_id_macro"] = t1.groupby("article_key")["found_id"].mean().mean() if len(t1) else float("nan")
    row["R_text_id_parent"] = t1["found_parent"].mean() if len(t1) else float("nan")
    s3 = p[p["tier"] == "suppl_label"]
    row["R_suppl_label_pred"] = s3["found_label_pred"].mean() if len(s3) else float("nan")
    row["R_suppl_label_list"] = s3["found_label_suppl"].mean() if len(s3) and has_suppl else float("nan")
    scored = p[p["tier"].isin(SCORED_TIERS)]
    row["R_all_id"] = scored["found_id"].mean()
    counts = pr["cls"].value_counts()
    n = max(len(pr) - counts.get("not_dataset", 0), 1)
    row["P_strict"] = counts.get("tp", 0) / n
    row["P_text_supported"] = (counts.get("tp", 0) + counts.get("out_of_graph_in_text", 0)) / n
    row["hallucination_rate"] = counts.get("not_in_text", 0) / n
    for c in FP_CLASSES:
        row[f"fp_{c}"] = int(counts.get(c, 0))
    return row


def run_graph_aware_eval(gt_edges: pd.DataFrame, neg_edges: pd.DataFrame, artifacts: pd.DataFrame, fulltext: dict,
                         configs: dict, eval_ids: set, out_dir=None, name: str = "pilot"):
    """Score every config. `configs` = {label: (dataset_citations.csv, suppl csv or None)}.

    Returns (summary, pairs with found flags per config, {label: classified predictions}); writes them to `out_dir`
    as graph_aware_summary_<name>.csv, graph_aware_pairs_<name>.csv and fp_triage_<name>_<cfg>.csv when given.
    """
    gt = gt_edges[(gt_edges["artifact_type"] == "dataset") & gt_edges["article_key"].isin(eval_ids)]
    neg = neg_edges[(neg_edges["artifact_type"] == "dataset") & neg_edges["article_key"].isin(eval_ids)]
    graph_ids = set(artifacts["identifier"].dropna().map(norm_pred))
    pairs = build_pairs(gt, fulltext)
    out, rows, triage = pairs.copy(), [], {}
    for label, (pred_csv, suppl_csv) in configs.items():
        if not pred_csv.exists():
            print(f"[skip] {label}: no {pred_csv}")
            continue
        has_suppl = suppl_csv is not None and suppl_csv.exists()
        pred = load_predictions(pred_csv, eval_ids)
        p, pr = score_config(pairs, pred, neg, graph_ids, fulltext, suppl_labels(suppl_csv) if has_suppl else None)
        out[f"found_id · {label}"] = p["found_id"]
        out[f"found_label · {label}"] = p["found_label_suppl"]
        triage[label] = pr
        rows.append(summarize(label, p, pr, has_suppl))
    summary = pd.DataFrame(rows)
    if out_dir is not None and len(summary):
        summary.to_csv(out_dir / f"graph_aware_summary_{name}.csv", index=False)
        out.drop(columns=["labels"]).assign(labels=pairs["labels"].map(lambda s: "|".join(sorted(s)))) \
           .to_csv(out_dir / f"graph_aware_pairs_{name}.csv", index=False)
        for label, pr in triage.items():
            pr.to_csv(out_dir / f"fp_triage_{name}_{_slug(label)}.csv", index=False)
    return summary, out, triage


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def tier_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairs per tier and the share of each tier in the scored ground truth."""
    n = pairs["tier"].value_counts().reindex(TIERS, fill_value=0)
    return pd.DataFrame({"pairs": n, "share_of_scored": (n / n[SCORED_TIERS].sum()).where(n.index.isin(SCORED_TIERS))})

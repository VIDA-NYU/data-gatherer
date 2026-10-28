"""Run SciNoBo RAA (https://github.com/iNoBo/scinobo-raa) on the Marine Science pilot and score it like a Data Gatherer run.

RAA reads GROBID TEI (`--xml_mode`): sections of paragraphs of `<s>` sentences, with `<ref type="bibr">` pointers
into a `<listBibl>`. The fetch cache holds PMC JATS, so `to-tei` writes one GROBID-like TEI per article:
- header: title, the article's own biblStruct (RAA's reevaluate reads its authors) and abstract;
- one `<div>` per JATS `sec` (flattened, like GROBID), body then back matter (data availability, acknowledgments, notes),
  then the front-matter statements (`notes`, PLOS data-availability `custom-meta`), which GROBID also puts at the back;
  figure and table captions are left out, as GROBID keeps them outside the divs RAA reads;
- a link whose text hides its target gets the target in parentheses, as Data Gatherer reads hrefs from the XML;
- bibliography with the DOI of each reference (JATS `pub-id`, else a DOI in the citation text).

`to-citations` reads RAA's `*_reevaluated.json` (mentions with artifact score >= the reevaluate threshold) and writes
`dataset_citations.csv` (`source_url`, `dataset_identifier`, `data_repository`) for two configs:
- `raa`       RAA as is: every DOI/URL RAA attached to a dataset cluster, else the cluster name;
- `raa_refs`  same, plus the DOIs of the references cited in the sentences of the cluster's mentions.

    python -I raa_adapter.py to-tei --cache data/marine_fetched.parquet --pilot input/article_ids_marine.csv --out output/raa/tei
    python -I raa_adapter.py to-citations --tei output/raa/tei --raa-out output/raa/json --out output
"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
from lxml import etree
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{}/"
DOI_URL = "https://doi.org/{}"
PMC_RE = re.compile(r"\b(PMC\d+)\b", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
UNNAMED_RE = re.compile(r"^unnamed(_\d+)?$", re.I)
SKIP_IN_SEC = {"sec", "fig", "table-wrap", "ref-list", "fig-group", "table-wrap-group", "disp-formula"}
# untrained Punkt splits at every abbreviation ("Ihnken et al. [43]", "Pseudo-nitzschia cf. turgiduloides")
_ABBREV = ("al cf e.g i.e fig figs tab ref refs eq eqs approx ca vs sp spp ssp var subsp no nos vol pp st dr "
           "jan feb mar apr jun jul aug sep sept oct nov dec n.d resp suppl")
SENT = PunktSentenceTokenizer(PunktParameters())
SENT._params.abbrev_types.update(_ABBREV.split())


def url_to_key(url: str) -> str | None:
    if m := PMC_RE.search(url):
        return m.group(1).upper()
    return url.split("doi.org/", 1)[1].lower() if "doi.org/" in url else None


def key_to_url(key: str) -> str:
    return PMC_URL.format(key) if key.upper().startswith("PMC") else DOI_URL.format(key)


def file_stem(key: str) -> str:
    """Article key as a file name (DOIs carry slashes); `to-citations` maps it back through the TEI idno."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)


def norm_doi(s: str) -> str:
    return unquote(s).strip().lower().rstrip("/.,;)]").removesuffix(".pdf")


def _txt(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


# ---- JATS -> TEI ---------------------------------------------------------------------------------------------------

def _tei(tag: str, parent=None, text: str | None = None, **attrs):
    el = etree.Element(f"{{{TEI_NS}}}{tag}", nsmap={None: TEI_NS}) if parent is None \
        else etree.SubElement(parent, f"{{{TEI_NS}}}{tag}")
    for k, v in attrs.items():
        el.set(XML_ID if k == "xml_id" else k, v)
    if text:
        el.text = text
    return el


def _segments(p, rid_to_b: dict) -> list:
    """Paragraph as [(text, ref target or None)]: bibliography xrefs become refs, everything else plain text."""
    out = []

    def walk(el):
        if el.text:
            out.append((el.text, None))
        for ch in el:
            if ch.tag == "xref" and ch.get("ref-type") == "bibr":
                targets = [rid_to_b[r] for r in (ch.get("rid") or "").split() if r in rid_to_b]
                out.append((_txt(ch) or " ", f"#{targets[0]}" if targets else None))
            elif ch.tag not in SKIP_IN_SEC:
                walk(ch)
                # a link whose text hides its target ("German PANGAEA database"): the target follows in parentheses
                href = (ch.get(XLINK_HREF) or "").strip() if ch.tag in {"ext-link", "uri"} else ""
                if href.startswith(("http", "doi:", "10.")) and unquote(href).lower() not in _txt(ch).lower():
                    out.append((f" ({href})", None))
            if ch.tail:
                out.append((ch.tail, None))

    walk(p)
    return out


def _add_paragraph(div, p, rid_to_b: dict) -> None:
    """`<p>` of `<s>` sentences; a bibliography ref belongs to the sentence it starts in, which is extended to cover it."""
    segs = [(re.sub(r"\s+", " ", t), r) for t, r in _segments(p, rid_to_b)]
    text = "".join(t for t, _ in segs)
    if not text.strip():
        return
    bounds = [e for _, e in SENT.span_tokenize(text)]
    pe, s_el, pos, b = _tei("p", div), None, 0, 0
    for t, ref in segs:
        if s_el is None:
            s_el = _tei("s", pe)
        if ref:
            r = _tei("ref", s_el, t, type="bibr", target=ref)
            r.tail = ""
        else:  # plain text may span sentence ends
            while t:
                cut = bounds[b] - pos if b < len(bounds) else len(t)
                head, t = t[:max(cut, 0)], t[max(cut, 0):]
                last = s_el[-1] if len(s_el) else None
                if last is not None:
                    last.tail = (last.tail or "") + head
                else:
                    s_el.text = (s_el.text or "") + head
                pos += len(head)
                if t or (b < len(bounds) and pos >= bounds[b]):
                    b += 1
                    if t:
                        s_el = _tei("s", pe)
            continue
        pos += len(t)
        while b < len(bounds) and bounds[b] <= pos:  # the ref ran past a sentence end: drop that boundary
            b += 1
    for s in list(pe):
        if not "".join(s.itertext()).strip():
            pe.remove(s)
        elif s.text:
            s.text = s.text.lstrip()


def _add_sections(parent_tei, jats_root, rid_to_b: dict) -> None:
    """One TEI div per JATS sec/ack/notes, holding only its own paragraphs (nested secs get their own div)."""
    secs = [s for s in jats_root.iter("sec", "ack", "notes", "fn-group", "app")]
    loose = [p for p in jats_root.findall("p")]
    if loose:  # paragraphs directly under body
        div = _tei("div", parent_tei)
        for p in loose:
            _add_paragraph(div, p, rid_to_b)
    for sec in secs:
        if sec.getparent() is not None and sec.getparent().tag in {"fig", "table-wrap"}:
            continue
        div = _tei("div", parent_tei)
        title = sec.find("title")
        if title is None and sec.tag == "ack":
            _tei("head", div, "Acknowledgements")
        elif title is not None:
            _tei("head", div, _txt(title))
        own = [p for p in sec.iter("p") if _nearest_sec(p) is sec and not _inside(p, {"fig", "table-wrap", "ref-list"})]
        for p in own:
            _add_paragraph(div, p, rid_to_b)
        if not len(div.findall(f"{{{TEI_NS}}}p")):
            parent_tei.remove(div)


def _nearest_sec(el):
    a = el.getparent()
    while a is not None and a.tag not in {"sec", "ack", "notes", "fn-group", "app"}:
        a = a.getparent()
    return a


def _inside(el, tags: set) -> bool:
    a = el.getparent()
    while a is not None:
        if a.tag in tags:
            return True
        a = a.getparent()
    return False


def _ref_doi(ref) -> str | None:
    for pid in ref.iter("pub-id", "ext-link"):
        v = pid.get("pub-id-type") or pid.get("ext-link-type") or ""
        if v.lower() == "doi" and _txt(pid):
            return norm_doi(_txt(pid))
        href = pid.get(XLINK_HREF) or ""
        if "doi.org/" in href:
            return norm_doi(href.split("doi.org/", 1)[1])
    m = DOI_RE.search(_txt(ref))
    return norm_doi(m.group(1)) if m else None


def _add_front_statements(parent_tei, root, rid_to_b: dict) -> None:
    """Statements kept in the front matter: `notes` (e.g. Data Availability) and PLOS's data-availability `custom-meta`,
    which usually repeats a note. GROBID puts these in the back matter, so they go there."""
    seen = set()
    for notes in root.findall("front//notes"):
        div = _tei("div", parent_tei)
        if (title := notes.find("title")) is not None:
            _tei("head", div, _txt(title))
        for p in notes.iter("p"):
            seen.add(_txt(p).lower())
            _add_paragraph(div, p, rid_to_b)
        if not len(div.findall(f"{{{TEI_NS}}}p")):
            parent_tei.remove(div)
    for cm in root.findall("front/article-meta/custom-meta-group/custom-meta"):
        name, value = cm.find("meta-name"), cm.find("meta-value")
        if value is None or "data availability" not in _txt(name).lower() or _txt(value).lower() in seen:
            continue
        div = _tei("div", parent_tei)
        _tei("head", div, _txt(name))
        _add_paragraph(div, value, rid_to_b)


def jats_to_tei(jats: str, key: str) -> bytes:
    root = etree.fromstring(jats.encode(), etree.XMLParser(recover=True, huge_tree=True))
    meta = root.find("front/article-meta")
    refs = root.findall(".//back//ref")
    rid_to_b = {r.get("id"): f"b{i + 1}" for i, r in enumerate(refs) if r.get("id")}

    tei = _tei("TEI")
    fdesc = _tei("fileDesc", _tei("teiHeader", tei))
    _tei("title", _tei("titleStmt", fdesc), _txt(meta.find("title-group/article-title")) if meta is not None else "")
    src = _tei("sourceDesc", fdesc)
    own = _tei("biblStruct", src)
    analytic = _tei("analytic", own)
    for c in (meta.findall("contrib-group/contrib[@contrib-type='author']") if meta is not None else []):
        pers = _tei("persName", _tei("author", analytic))
        _tei("forename", pers, _txt(c.find("name/given-names")))
        _tei("surname", pers, _txt(c.find("name/surname")) or _txt(c.find("collab")))
    _tei("idno", own, key, type="PMCID" if key.startswith("PMC") else "DOI")
    abstract = _tei("abstract", _tei("profileDesc", tei[0]))
    abs_div = _tei("div", abstract)
    for p in (meta.findall("abstract//p") if meta is not None else []):
        _add_paragraph(abs_div, p, rid_to_b)
    if not len(abs_div):  # RAA's parser needs an abstract paragraph
        _tei("s", _tei("p", abs_div), "")

    text = _tei("text", tei)
    body_tei = _tei("body", text)
    if (body := root.find("body")) is not None:
        _add_sections(body_tei, body, rid_to_b)
    back_tei = _tei("back", text)
    if (back := root.find("back")) is not None:
        _add_sections(back_tei, back, rid_to_b)
    _add_front_statements(back_tei, root, rid_to_b)
    lb = _tei("listBibl", _tei("div", back_tei, type="references"))
    for r in refs:
        bs = _tei("biblStruct", lb, xml_id=rid_to_b.get(r.get("id"), ""))
        title = r.find(".//article-title")
        _tei("title", _tei("analytic", bs), _txt(title if title is not None else r.find(".//source")))
        if doi := _ref_doi(r):
            _tei("idno", bs, doi, type="DOI")
    return etree.tostring(tei, encoding="utf-8", xml_declaration=True)


def to_tei(cache_path: Path, pilot_csv: Path, out_dir: Path) -> None:
    pilot = pd.read_csv(pilot_csv, dtype=str)
    keys = set(pilot["pmcid"].fillna(pilot["doi"].str.lower()) if "pmcid" in pilot else pilot["doi"].str.lower())
    cache = pd.read_parquet(cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, []
    for url, row in cache.iterrows():
        key = url_to_key(str(url))
        if key not in keys:
            continue
        if not str(row["fetched_data"]).lstrip().startswith(("<article", "<?xml", "<!DOCTYPE article")):
            skipped.append((key, row["raw_data_format"]))  # HTML or TEI: not handled here
            continue
        (out_dir / f"{file_stem(key)}.xml").write_bytes(jats_to_tei(str(row["fetched_data"]), key))
        n += 1
    print(f"{n} TEI files in {out_dir} | skipped {len(skipped)} non-JATS: {skipped[:5]}")


# ---- RAA output -> dataset_citations.csv ---------------------------------------------------------------------------

def _strip_pct(s: str) -> list:
    """RAA cluster fields are "value (NN.N%)" lines."""
    return [re.sub(r"\s*\(\d+(?:\.\d+)?%\)$", "", x).strip() for x in str(s or "").split("\n") if x.strip()]


def _identifiers(urls: list) -> list:
    """DOIs from doi.org/... or bare 10.x strings, else the URL."""
    out = []
    for u in urls:
        m = DOI_RE.search(unquote(u))
        out.append(norm_doi(m.group(1)) if m else u.strip().rstrip(".,;)"))
    return out


def raa_rows(res: dict, key: str) -> tuple[list, list]:
    """(rows of `raa`, extra rows of `raa_refs`) for one reevaluated RAA output."""
    src = key_to_url(key)
    ref_doi = {f"#{c['Id']}": c["Doi"] for c in res.get("citations", []) if c.get("Doi")}
    cites_at = {}
    for c in res.get("citances", []):
        if c.get("Target") in ref_doi:
            cites_at.setdefault(tuple(c["Indices"]), set()).add(ref_doi[c["Target"]])
    mentions_by = {}
    for m in res.get("mentions", []):
        mentions_by.setdefault(m["RA Cluster"], []).append(m)

    base, extra = [], []
    for ra in res.get("research_artifacts", []):
        if ra["Type"] != "dataset":
            continue
        common = {"source_url": src, "raa_name": ra["Research Artifact"], "raa_cluster": ra["RA Cluster"],
                  "raa_score": ra["Research Artifact Score"], "raa_owned": ra["Owned"], "raa_reused": ra["Reused"],
                  "raa_mentions": ra["Mentions Count"]}
        ids = _identifiers(_strip_pct(ra["URLs"]))
        # clusters RAA could not name are labelled "Unnamed_<n>": a label, not a name from the text
        names = [] if UNNAMED_RE.match(str(ra["Research Artifact"])) else [ra["Research Artifact"]]
        for i in dict.fromkeys(ids) or names:
            base.append({**common, "dataset_identifier": i, "data_repository": "", "evidence": "url" if ids else "name"})
        cited = {d for m in mentions_by.get(ra["RA Cluster"], []) for d in cites_at.get(tuple(m["Indices"]), ())}
        for d in sorted(cited - set(ids)):
            extra.append({**common, "dataset_identifier": d, "data_repository": "", "evidence": "cited_ref"})
    return base, extra


def to_citations(tei_dir: Path, raa_dir: Path, out_dir: Path, name: str = "raa") -> None:
    stem_to_key = {}
    for p in tei_dir.glob("*.xml"):
        idno = etree.parse(str(p)).find(f".//{{{TEI_NS}}}sourceDesc//{{{TEI_NS}}}idno")
        stem_to_key[p.stem] = idno.text
    base, extra, missing = [], [], 0
    for stem, key in stem_to_key.items():
        f = raa_dir / f"{stem}_reevaluated.json"
        if not f.exists():
            missing += 1
            continue
        b, e = raa_rows(json.loads(f.read_text()), key)
        base += b
        extra += e
    cols = ["source_url", "dataset_identifier", "data_repository", "evidence", "raa_name", "raa_cluster", "raa_score",
            "raa_owned", "raa_reused", "raa_mentions"]
    for cfg, rows in ((name, base), (f"{name}_refs", base + extra)):
        (out_dir / cfg).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / cfg / "dataset_citations.csv", index=False)
    print(f"{len(stem_to_key) - missing}/{len(stem_to_key)} articles with RAA output | {name}: {len(base)} rows | "
          f"{name}_refs: {len(base) + len(extra)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-tei")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--pilot", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("to-citations")
    b.add_argument("--tei", type=Path, required=True)
    b.add_argument("--raa-out", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", default="raa", help="configs <name>/ and <name>_refs/ under --out")
    args = ap.parse_args()
    if args.cmd == "to-tei":
        to_tei(args.cache, args.pilot, args.out)
    else:
        to_citations(args.tei, args.raa_out, args.out, args.name)

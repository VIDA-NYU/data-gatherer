"""
Apply reviewed corrections to the ground truth training dataset files.

Reads:  scripts/training_data_corrections.jsonl
Target: scripts/Local_model_finetuning/ground_truth/gt_dataset_info_no_dspage_extraction_from_snippet.csv
        scripts/Local_model_finetuning/ground_truth/gt_dataset_info_no_dspage_extraction_from_snippet.xlsx

Rules:
  drop  — remove the matching entry from the datasets list
  edit  — replace the matching entry with corrected_entries
  keep  — unchanged (not in corrections file)

Deduplication is applied per-row after corrections: entries with the same
dataset_identifier are collapsed, preferring the one with a non-empty
repository_reference.
"""

import json
import os
from collections import defaultdict

CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), 'training_data_corrections.jsonl')
GT_DIR           = os.path.join(os.path.dirname(__file__), 'Local_model_finetuning', 'ground_truth')
GT_CSV           = os.path.join(GT_DIR, 'gt_dataset_info_no_dspage_extraction_from_snippet.csv')
GT_XLSX          = os.path.join(GT_DIR, 'gt_dataset_info_no_dspage_extraction_from_snippet.xlsx')


def load_corrections():
    """Returns {idx -> list[correction]} for all drop/edit rows."""
    index = defaultdict(list)
    with open(CORRECTIONS_FILE, encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            index[c['idx']].append(c)
    return index


def fix_malformed_entry(entry):
    """
    Repair entries where the user typed REPO:ID (single colon) instead of REPO::ID,
    so the parser left the full token in dataset_identifier with an empty repo.
    E.g. {"dataset_identifier": "EMDB:EMD-15269", "repository_reference": ""}
    → {"dataset_identifier": "EMD-15269", "repository_reference": "EMDB"}
    Only applied when repository_reference is empty and ':' appears in the identifier.
    """
    did = entry.get('dataset_identifier', '')
    repo = entry.get('repository_reference', '')
    if not repo and ':' in did and '://' not in did:
        repo_part, _, id_part = did.partition(':')
        return {'dataset_identifier': id_part.strip(), 'repository_reference': repo_part.strip()}
    return entry


def dedup_datasets(entries):
    """
    Remove duplicate dataset_identifiers, keeping the entry with the richer
    repository_reference (non-empty preferred). Preserves insertion order.
    """
    seen = {}
    result = []
    for entry in entries:
        did = entry.get('dataset_identifier', '').strip()
        if not did or did == 'n/a':
            result.append(entry)
            continue
        if did not in seen:
            seen[did] = len(result)
            result.append(entry)
        else:
            # Upgrade if current has a repo and the stored one doesn't
            existing_idx = seen[did]
            if entry.get('repository_reference') and not result[existing_idx].get('repository_reference'):
                result[existing_idx] = entry
    return result


def apply_corrections_to_row(output_text_raw, corrections_for_row):
    """
    Apply a list of corrections to one row's output_text JSON string.
    Returns the updated JSON string.
    """
    try:
        obj = json.loads(output_text_raw)
    except json.JSONDecodeError:
        return output_text_raw

    datasets = obj.get('datasets', [])

    for correction in corrections_for_row:
        original_id = correction['identifier']
        decision    = correction['decision']

        if decision == 'd':
            datasets = [e for e in datasets if e.get('dataset_identifier') != original_id]

        elif decision == 'e':
            # Build replacement entries (fix single-colon entries too)
            replacements = [fix_malformed_entry(e) for e in correction['corrected_entries']]
            new_datasets = []
            replaced = False
            for entry in datasets:
                if entry.get('dataset_identifier') == original_id:
                    if not replaced:
                        new_datasets.extend(replacements)
                        replaced = True
                    # skip the original entry (drop it, replaced by corrected_entries)
                else:
                    new_datasets.append(entry)
            if not replaced:
                # original not found — just append corrections
                new_datasets.extend(replacements)
            datasets = new_datasets

    obj['datasets'] = dedup_datasets(datasets)
    return json.dumps(obj, ensure_ascii=False)


def main():
    import pandas as pd

    corrections = load_corrections()
    affected_idxs = set(corrections.keys())

    print(f"Loading ground truth CSV…")
    df = pd.read_csv(GT_CSV)

    total   = len(df)
    changed = 0

    print(f"Applying corrections to {len(affected_idxs)} rows…")
    for idx in sorted(affected_idxs):
        mask = df['Unnamed: 0'] == idx
        if not mask.any():
            print(f"  WARNING: idx {idx} not found in CSV, skipping")
            continue
        row_pos = df.index[mask][0]
        original = df.at[row_pos, 'output_text']
        updated  = apply_corrections_to_row(original, corrections[idx])
        if updated != original:
            df.at[row_pos, 'output_text'] = updated
            changed += 1

    print(f"Writing {total} rows to CSV…")
    df.to_csv(GT_CSV, index=False)

    print(f"Writing {total} rows to XLSX…")
    df.to_excel(GT_XLSX, index=False)

    print(f"\nDone. {changed} rows updated in:\n  {GT_CSV}\n  {GT_XLSX}")


if __name__ == '__main__':
    main()

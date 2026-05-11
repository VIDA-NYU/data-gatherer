"""
Interactive review of flagged training data entries.
For each flagged identifier not found in its snippet, decide:
  k  — keep   (valid inference, GPT-4o-mini was right even without explicit mention)
  d  — drop   (hallucination, remove this identifier from output_text)
  e  — edit   (wrong value, replace with correct identifier)
  s  — skip   (decide later)
  q  — quit   (save progress and exit)

Results saved to scripts/training_data_reviewed.csv
A corrected dataset is written to scripts/training_data_corrected.jsonl
"""

import csv
import json
import os
import sys
import textwrap

FLAGGED_CSV     = os.path.join(os.path.dirname(__file__), 'training_data_flagged.csv')
REVIEWED_CSV    = os.path.join(os.path.dirname(__file__), 'training_data_reviewed.csv')
CORRECTED_JSONL = os.path.join(os.path.dirname(__file__), 'training_data_corrections.jsonl')

DECISION_COL = 'decision'     # keep / drop / edit
CORRECTED_ID = 'corrected_identifier'

SNIPPET_WIDTH = 100


def load_dataset_index():
    """Load full input_text from HF dataset, keyed by row index."""
    print("Loading full snippets from HuggingFace dataset…")
    from datasets import load_dataset
    ds = load_dataset("vida-nyu/pmc-articles-dataset-mentions-snippets", split="full")
    return {i: row['input_text'] for i, row in enumerate(ds)}


def load_flagged():
    with open(FLAGGED_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_reviewed():
    """Returns dict of (idx, identifier) -> row already reviewed."""
    if not os.path.exists(REVIEWED_CSV):
        return {}
    with open(REVIEWED_CSV, newline='', encoding='utf-8') as f:
        done = {}
        for row in csv.DictReader(f):
            done[(row['idx'], row['identifier'])] = row
    return done


def save_reviewed(rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(REVIEWED_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def color(text, code):
    return f"\033[{code}m{text}\033[0m"

def bold(t):    return color(t, '1')
def red(t):     return color(t, '31')
def green(t):   return color(t, '32')
def yellow(t):  return color(t, '33')
def cyan(t):    return color(t, '36')
def dim(t):     return color(t, '2')


def show_entry(i, total, row, done_count, full_snippets):
    os.system('clear')
    print(bold(f"═══  Entry {i}/{total}  (reviewed so far: {done_count})  ═══\n"))
    print(f"  {bold('idx')}:        {row['idx']}")
    print(f"  {bold('article')}:    {row['url'].split('/')[-1]}  —  {row['url']}")
    print(f"  {bold('section')}:    {row['section_title']}")
    print()
    print(bold("  IDENTIFIER (not found in snippet):"))
    print(f"    {red(bold(row['identifier']))}   [{dim(row['repository'])}]")
    print()
    print(bold("  FULL SNIPPET:"))
    snippet = (full_snippets.get(int(row['idx'])) or row['input_text']).replace('\\n', '\n')
    id_low = row['identifier'].lower()
    for line in textwrap.wrap(snippet, width=SNIPPET_WIDTH):
        if id_low[:4] in line.lower():
            print(f"    {yellow(line)}")
        else:
            print(f"    {line}")
    print()


def prompt_decision(row):
    while True:
        print(f"  Decision for {red(bold(row['identifier']))}:")
        print(f"    {green('k')} keep   — valid inference, identifier is correct")
        print(f"    {red('d')} drop   — hallucination, remove from output_text")
        print(f"    {yellow('e')} edit   — wrong value, enter the correct identifier")
        print(f"    {dim('s')} skip   — undecided, revisit later")
        print(f"    {dim('q')} quit   — save progress and exit")
        choice = input("\n  > ").strip().lower()
        if choice in ('k', 'd', 's', 'q'):
            return choice, row['identifier']
        if choice == 'e':
            print(dim("  Syntax: plain IDs comma-separated, or REPO::ID for repo+identifier pairs."))
            print(dim("  Examples:  GSE12345, GSE67890"))
            print(dim("             PDB::8A99, PDB::8A94, EMDB::EMD-15273"))
            corrected = input("  > ").strip()
            if not corrected:
                return 'd', row['identifier']
            return 'e', corrected
        print(red("  Invalid choice, try again.\n"))


def main():
    flagged  = load_flagged()
    reviewed = load_reviewed()

    # Build working list: add decision columns if missing
    all_rows = []
    for row in flagged:
        key = (row['idx'], row['identifier'])
        if key in reviewed:
            all_rows.append(reviewed[key])
        else:
            r = dict(row)
            r[DECISION_COL]  = ''
            r[CORRECTED_ID]  = ''
            all_rows.append(r)

    pending = [r for r in all_rows if not r[DECISION_COL] or r[DECISION_COL] == 's']
    done_count = sum(1 for r in all_rows if r[DECISION_COL] and r[DECISION_COL] != 's')

    print(f"\n  {bold('Training data review')}")
    print(f"  Total flagged: {len(all_rows)}")
    print(f"  Already reviewed: {done_count}")
    print(f"  Pending (incl. skipped): {len(pending)}")
    input("\n  Press Enter to start…")

    full_snippets = load_dataset_index()

    for i, row in enumerate(pending, 1):
        show_entry(i, len(pending), row, done_count, full_snippets)
        decision, value = prompt_decision(row)

        if decision == 'q':
            save_reviewed(all_rows)
            print(green("\n  Progress saved. Bye."))
            sys.exit(0)

        row[DECISION_COL] = decision
        row[CORRECTED_ID] = value if decision == 'e' else (row['identifier'] if decision == 'k' else '')
        done_count += 1 if decision != 's' else 0
        save_reviewed(all_rows)

    # Final summary
    os.system('clear')
    decisions = [r[DECISION_COL] for r in all_rows if r[DECISION_COL]]
    print(bold("\n  ✓ Review complete!\n"))
    print(f"  keep:  {decisions.count('k'):>3}")
    print(f"  drop:  {decisions.count('d'):>3}")
    print(f"  edit:  {decisions.count('e'):>3}")
    print(f"  skip:  {decisions.count('s'):>3}")
    print(f"\n  Results saved to:\n    {REVIEWED_CSV}")

    # Write corrections as JSONL for easy ingestion
    corrections = [r for r in all_rows if r[DECISION_COL] in ('d', 'e')]
    with open(CORRECTED_JSONL, 'w', encoding='utf-8') as f:
        for r in corrections:
            corrected_raw = r[CORRECTED_ID]
            corrected_entries = []
            for token in (corrected_raw or '').split(','):
                token = token.strip()
                if not token:
                    continue
                if '::' in token:
                    repo, _, acc = token.partition('::')
                    corrected_entries.append({
                        'dataset_identifier':   acc.strip(),
                        'repository_reference': repo.strip(),
                    })
                else:
                    corrected_entries.append({
                        'dataset_identifier':   token,
                        'repository_reference': '',
                    })
            f.write(json.dumps({
                'idx':               int(r['idx']),
                'url':               r['url'],
                'identifier':        r['identifier'],
                'decision':          r[DECISION_COL],
                'corrected_entries': corrected_entries,
            }) + '\n')
    print(f"  Corrections (drop/edit) saved to:\n    {CORRECTED_JSONL}")


if __name__ == '__main__':
    main()

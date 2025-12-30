---
language: en
license: apache-2.0
task_categories:
- text2text-generation
tags:
- information-extraction
- dataset-extraction
- scientific-literature
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: "gt_dataset_info_no_dspage_extraction_from_snippet.csv"
---

# PMC Articles Dataset Mentions Snippets

Text snippets from PubMed Central articles paired with structured dataset citations. Designed for training models to extract dataset references from scientific literature.

## Description

- **Task**: Extract structured dataset info (identifier, repository, webpage) from article text
- **Source**: PMC open-access articles
- **Format**: Text snippet → JSON output
- **Examples**: Positive (with datasets) and negative (no datasets)

## Fields

- `input_text`: Article snippet potentially containing dataset mentions
- `output_text`: JSON with `dataset_identifier`, `data_repository`
- `section_title`, `sec_type`: Source section metadata
- `L2_distance`: Semantic retrieval relevance score
- `url`, `article_id`: Source article identifiers

## Construction

**Snippet Extraction:**
1. Retrieve-Then-Read (RTR) with semantic retrieval (top-k=3)
2. Rule-based filtering for data availability sections
3. Regex pattern matching for known repository identifiers

**Annotation:**
- Model: GPT-4o-mini via OpenAI Batch API
- Schema: Structured JSON output
- Deduplication: Word frequency analysis + manual review
- Tool: [Data Gatherer](https://github.com/VIDA-NYU/data-gatherer)

**Versions:**
- `gt_dataset_info_extraction_from_snippet`: With webpage URLs
- `gt_dataset_info_no_dspage_extraction_from_snippet`: Without webpages (prevents from halluciations on unseen datasets)

## Limitations

- **Domain**: Biomedical/life sciences articles from PMC only
- **Language**: English only
- **Coverage**: May not represent all dataset citation styles
- **Negatives**: Distribution may not reflect real-world proportions
- **Temporal**: Snapshot from mid 2025

## Usage

Trained model: [vida-nyu/flan-t5-base-dataref-info-extract](https://huggingface.co/vida-nyu/flan-t5-base-dataref-info-extract)

## Citation

```bibtex
@dataset{vida_nyu_pmc_dataset_mentions_2025,
  title={PMC Articles Dataset Mentions Snippets},
  author={VIDA Lab, New York University},
  year={2025},
  url={https://huggingface.co/datasets/vida-nyu/pmc-articles-dataset-mentions-snippets}
}
```

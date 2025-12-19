---
language: en
license: apache-2.0
tags:
- text-generation
- information-extraction
- dataset-extraction
- scientific-literature
- flan-t5
- seq2seq
- seq2struct
- semantic-parsing
datasets:
- vida-nyu/pmc-articles-dataset-mentions-snippets
metrics:
- rouge
- exact_match
library_name: transformers
pipeline_tag: text-generation
base_model: google/flan-t5-base
---

# Dataset Information Extraction from Scientific Text

Fine-tuned [flan-t5-base](https://huggingface.co/google/flan-t5-base) to extract dataset identifiers and repository references from scientific publications.

**What it does:** Finds dataset IDs (GSE123456, DOIs, etc.) and repository names (GEO, Zenodo, etc.) in text snippets from papers.

**Trained on:** [vida-nyu/pmc-articles-dataset-mentions-snippets](https://huggingface.co/datasets/vida-nyu/pmc-articles-dataset-mentions-snippets) - PMC article snippets extracted using [Data Gatherer](https://github.com/VIDA-NYU/data-gatherer)

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("vida-nyu/flan-t5-base-dataref-info-extract")
tokenizer = AutoTokenizer.from_pretrained("vida-nyu/flan-t5-base-dataref-info-extract")

text = "Extract dataset information: The data are in GEO under GSE123456."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=256, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

- **Base:** flan-t5-base (250M params)
- **Data:** PMC snippets with/without dataset mentions. Used 80/20 train-test split
- **Config:** 5 epochs, lr=3e-4, batch=16 (effective)
- **Input:** Max 512 tokens with prefix "Extract dataset information: "
- **Output:** Structured dataset info or null for no-dataset cases

## Limitations

- Max 512 tokens input (truncates long passages)
- Trained on biomedical literature (may underperform on other domains)
- English only
- May miss novel/obscure repositories

## Citation

```bibtex
@software{flan_t5_dataset_extraction,
  title={Flan-T5 for Dataset Information Extraction},
  author={VIDA Lab, NYU},
  year={2025},
  url={https://huggingface.co/vida-nyu/flan-t5-base-dataref-info-extract}
}
```

**Contact:** [GitHub Issues](https://github.com/VIDA-NYU/data-gatherer/issues) | VIDA Lab, NYU

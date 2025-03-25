# Data Gatherer

The Data Gatherer Tool is designed for automating the extraction of datasets from scientific articles webpages. It integrates LLMs, dynamic prompt management, and rule-based parsing, to facilitate data harmonization in biomedical research, and hopefully other domains.

## Installation

```bash
pip install git+https://github.com/yourusername/data-gatherer.git
```

## Usage

### Command-line Interface

```bash
# Process URLs from a file
data-gatherer --input-file input/urls.txt --output-file output/results.csv

# Process URLs directly
data-gatherer --urls https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC789012
```

### Python API

```python
from data_gatherer import DataGatherer

# Initialize with default configuration
gatherer = DataGatherer()

# Process a list of URLs
urls = [
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC789012"
]
results = gatherer.process_urls(urls, output_file="output/results.csv")

# Process URLs from a file
results = gatherer.process_file("input/urls.txt", output_file="output/results.csv")

# Print the results
print(results)
```

## Requirements

- Python 3.6+
- Selenium WebDriver (Firefox, Chrome, or Edge)
- Required Python packages (installed automatically)
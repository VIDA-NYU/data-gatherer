.. image:: https://readthedocs.org/projects/data-gatherer/badge/?version=latest
   :target: https://data-gatherer.readthedocs.io/en/latest/
   :alt: Documentation Status

********************
Data Gatherer
********************

*Data Gatherer* is a tool for automating the extraction of datasets from scientific article webpages.
It integrates large language models (LLMs), dynamic prompt management, and rule-based parsing
to support data harmonization in biomedical research—and potentially beyond.

When a dataset is found, Data Gatherer classifies its access type into one of four categories:

1. **Easy download**
   Three or fewer files, publicly accessible.

2. **Complex download**
   Four or more files, publicly accessible.

3. **Application to access**
   Requires application through a centralized process with clear procedures.

4. **Contact to access**
   Requires reaching out to the data provider; procedures are unclear or informal.
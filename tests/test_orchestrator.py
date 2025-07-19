from data_gatherer.data_fetcher import *
from conftest import get_test_data_path
import logging
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from data_gatherer.data_gatherer import DataGatherer
from data_gatherer.data_fetcher import EntrezFetcher
from data_gatherer.parser.xml_parser import XMLParser

def load_mock_xml(get_test_data_path, filename="test_2.xml"):
    with open(get_test_data_path(filename), "r", encoding="utf-8") as f:
        return ET.fromstring(f.read())


def mock_datasets_info(self, *args, **kwargs):
    return [
            {'dataset_identifier': 'PXD043612', 'data_repository': 'www.ebi.ac.uk', 'dataset_webpage': 'https://www.ebi.ac.uk/pride/archive/projects/PXD043612'},
            {'dataset_identifier': '10.17632/3wfxrz66w2.1', 'data_repository': 'doi.org', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': '10.17632/bvdn865y9c.1', 'data_repository': 'doi.org', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000234', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000127', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000204', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000221', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000198', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000110', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000270', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000125', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
            {'dataset_identifier': 'PDC000153', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'},
        ]


def test_process_url_with_mocks(monkeypatch, get_test_data_path):
    # Setup
    orchestrator = DataGatherer(log_level="INFO")

    if orchestrator.data_fetcher is None:
        orchestrator.data_fetcher = EntrezFetcher(requests, logger=logging.getLogger("data_gatherer"))

    # Monkeypatch fetch_data
    orchestrator.data_fetcher.fetch_data = lambda *args, **kwargs: open(get_test_data_path("test_2.xml"), "r", encoding="utf-8").read()
    # Monkeypatch extract_datasets_info_from_content
    monkeypatch.setattr(XMLParser, "extract_datasets_info_from_content", mock_datasets_info)
    monkeypatch.setenv("OPENAI_API_KEY", "test-gpt-key")

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11129317/'
    result = orchestrator.process_url(url)

    # Assertions
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 22

    expected_columns = [
        'dataset_identifier', 'data_repository', 'dataset_webpage',
        'source_section', 'retrieval_pattern', 'access_mode', 'link',
        'source_url', 'download_link', 'title', 'content_type', 'id',
        'caption', 'description', 'context_description', 'file_extension', 'pub_title'
    ]
    assert list(result.columns) == expected_columns, f"Columns do not match. Got: {list(result.columns)}"
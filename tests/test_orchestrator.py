from data_gatherer.data_fetcher import *
from conftest import get_test_data_path
import logging
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from data_gatherer.data_gatherer import DataGatherer
from data_gatherer.data_fetcher import EntrezFetcher
from data_gatherer.parser.xml_parser import XMLParser
import lxml.etree as LET  # Add this import

def load_mock_xml(get_test_data_path, filename="test_2.xml"):
    with open(get_test_data_path(filename), "rb") as f:  # Open in binary mode
        return LET.fromstring(f.read())  # Pass bytes to fromstring


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
    # Monkeypatch the fetcher BEFORE creating the orchestrator!
    monkeypatch.setattr(
        "data_gatherer.data_fetcher.EntrezFetcher.fetch_data",
        lambda self, *args, **kwargs: load_mock_xml(get_test_data_path)
    )
    # Setup
    orchestrator = DataGatherer(log_level="INFO", llm_name="gemini-2.0-flash")

    # Monkeypatch extract_datasets_info_from_content
    monkeypatch.setattr(XMLParser, "extract_datasets_info_from_content", mock_datasets_info)
    monkeypatch.setenv("OPENAI_API_KEY", "test-gpt-key")
    monkeypatch.setenv("PORTKEY_API_KEY", "test-port-key")
    monkeypatch.setenv("PORTKEY_ROUTE", "gemini-vertexai-test-key")
    monkeypatch.setenv("PORTKEY_CONFIG", "test-portkey-config")
    monkeypatch.setenv("PORTKEY_GATEWAY_URL", "https://test-portkey-gateway-url.com")
    monkeypatch.setenv("NCBI_API_KEY", "test-ncbi-key")

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
        'caption', 'description', 'context_description', 'file_extension', 'pub_title', 'raw_data_format'
    ]
    assert list(result.columns) == expected_columns, f"Columns do not match. Got: {list(result.columns)}"
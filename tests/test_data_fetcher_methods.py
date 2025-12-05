from data_gatherer.data_fetcher import *
from data_gatherer.data_gatherer import DataGatherer
from lxml import etree
from conftest import get_test_data_path
from unittest.mock import patch, Mock
import logging

class DummyFetcher(DataFetcher):
    def fetch_data(self, *args, **kwargs):
        pass  # Minimal implementation for testing

@patch("requests.get")
def test_PMCID_to_doi(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'ok', 'response-date': '2025-07-10 11:14:58', 'request': {'warnings': ['query param `email` is missing.', 'query param `tool` is missing.'], 'format': 'json', 'ids': ['PMC3531190'], 'echo': 'ids=PMC3531190&format=json', 'versions': 'no', 'showaiid': 'no', 'idtype': 'pmcid'}, 'records': [{'doi': '10.1093/nar/gks1195', 'pmcid': 'PMC3531190', 'pmid': 23193287, 'requested-id': 'PMC3531190'}]}
    mock_get.return_value = mock_response
    fetcher = DummyFetcher(logger=logging.getLogger("test_logger"))
    pmcid = "PMC3531190"
    doi = fetcher.PMCID_to_doi(pmcid)
    assert doi is not None
    assert isinstance(doi, str)
    assert doi == "10.1093/nar/gks1195"

def test_is_full_text_complete(get_test_data_path):
    dg = DataGatherer(log_level='DEBUG')
    checker = DataCompletenessChecker(logger=dg.logger)

    xml_cont = etree.parse(get_test_data_path("pmc_element_set_1.xml"))

    assert checker.is_fulltext_complete(xml_cont,'test_url') is True
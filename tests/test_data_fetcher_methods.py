from data_gatherer.data_fetcher import *
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


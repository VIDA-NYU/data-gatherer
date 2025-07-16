import tempfile
from data_gatherer.data_gatherer import DataGatherer
from data_gatherer.data_fetcher import *
from conftest import get_test_data_path
import requests
import pandas as pd

def test_process_url_with_mocked_fetch_data_and_parser(monkeypatch):
    orchestrator = DataGatherer()

    if orchestrator.data_fetcher is None:
        orchestrator.data_fetcher = EntrezFetcher(requests, logger=logging.getLogger("data_gatherer"))
    assert orchestrator.data_fetcher is not None, "Data fetcher could not be set up."

    # Mock fetch_data to return the contents of a test XML file
    def mock_fetch_data(get_test_data_path, *args, **kwargs):
        with open(get_test_data_path("test_2.xml"), "r", encoding="utf-8") as f:
            return f.read()
    orchestrator.data_fetcher.fetch_data = mock_fetch_data

    # Mock XMLParser.extract_datasets_info_from_content to return a controlled value
    def mock_extract_datasets_info_from_content(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            if len(args[0]) > 4000:
                # print(f"Mocking extract_datasets_info_from_content with a long string {args[0]}\n\n\n")
                return [{'dataset_identifier': 'PXD043612', 'data_repository': 'www.ebi.ac.uk', 'dataset_webpage': 'https://www.ebi.ac.uk/pride/archive/projects/PXD043612'}, {'dataset_identifier': '10.17632/3wfxrz66w2.1', 'data_repository': 'doi.org', 'dataset_webpage': 'n/a'}, {'dataset_identifier': '10.17632/bvdn865y9c.1', 'data_repository': 'doi.org', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000234', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000127', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000204', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000221', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000198', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000110', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000270', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000125', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}, {'dataset_identifier': 'PDC000153', 'data_repository': 'pdc.cancer.gov', 'dataset_webpage': 'n/a'}]
            else:
                # print(f"Mocking extract_datasets_info_from_content with a short string {args[0]}\n\n\n")
                return []
    from data_gatherer.parser.xml_parser import XMLParser
    monkeypatch.setattr(XMLParser, "extract_datasets_info_from_content", mock_extract_datasets_info_from_content)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11129317/'
    orchestrator.current_url = url
    orchestrator.publisher = orchestrator.data_fetcher.url_to_publisher_domain(url)

    # Run the orchestrator's process_url method
    result = orchestrator.process_url(url)

    # Basic assertion: result should not be None
    assert result is not None
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    expected_columns = [
        'dataset_identifier', 'data_repository', 'dataset_webpage',
        'source_section', 'retrieval_pattern', 'access_mode', 'link',
        'source_url', 'download_link', 'title', 'content_type', 'id',
        'caption', 'description', 'context_description', 'file_extension', 'pub_title'
    ]
    assert list(result.columns) == expected_columns, f"Columns do not match. Got: {list(result.columns)}"
    # print(f"Result DataFrame:\n{result.columns}")
    assert len(result) == 22, "Expected 22 datasets in the result"

    # Clean up the temporary log file

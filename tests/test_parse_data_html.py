from data_gatherer.parser import LLMParser
from data_gatherer.resources_loader import load_config
from data_gatherer.logger_setup import setup_logging
from bs4 import BeautifulSoup, NavigableString, CData, Comment

import os
from dotenv import load_dotenv

load_dotenv()

def test_get_data_availability_elements_from_HTML():
    logger = setup_logging("test_logger", log_file="logs/scraper.log")
    parser = LLMParser("open_bio_data_repos.json", logger, llm_name='gemini-2.0-flash')
    html_file_path = os.path.join('test_data', 'test_extract_1.html')
    with open(html_file_path, 'rb') as f:
        raw_html = f.read()
    preprocessed_data = parser.normalize_full_DOM(raw_html)
    DAS_elements = parser.get_data_availability_elements_from_webpage(preprocessed_data)
    assert isinstance(DAS_elements, list)
    assert len(DAS_elements) == 5
    assert all(isinstance(sm, dict) for sm in DAS_elements)
    print('\n')
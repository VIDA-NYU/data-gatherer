from data_gatherer.parser.base_parser import LLMParser
from data_gatherer.parser.xml_parser import XMLParser
from data_gatherer.parser.html_parser import HTMLParser
from data_gatherer.logger_setup import setup_logging
from lxml import etree

import os
from dotenv import load_dotenv

load_dotenv()

# To see logs in the test output, configure the logger to also log to the console (StreamHandler), or set log_file=None in setup_logging.

def test_get_data_availability_elements_from_HTML():
    logger = setup_logging("test_logger", log_file="logs/scraper.log")
    parser = HTMLParser("open_bio_data_repos.json", logger, llm_name='gemini-2.0-flash')
    html_file_path = os.path.join('test_data', 'test_extract_1.html')
    with open(html_file_path, 'rb') as f:
        raw_html = f.read()
    preprocessed_data = parser.normalize_HTML(raw_html)
    DAS_elements = parser.retriever.get_data_availability_elements_from_webpage(preprocessed_data)
    assert isinstance(DAS_elements, list)
    assert len(DAS_elements) == 5
    assert all(isinstance(sm, dict) for sm in DAS_elements)
    print('\n')


def test_extract_paragraphs_from_xml():
    logger = setup_logging("test_logger", log_file="../logs/scraper.log")
    parser = XMLParser("open_bio_data_repos.json", logger, log_file_override=None,
                       llm_name='gemini-2.0-flash')
    xml_file_path = os.path.join('test_data', 'test_1.xml')
    with open(xml_file_path, 'rb') as f:  # ✅ open in binary mode
        xml_root = etree.fromstring(f.read())
    paragraphs = parser.extract_paragraphs_from_xml(xml_root)
    assert isinstance(paragraphs, list)
    assert len(paragraphs) > 0
    assert all(isinstance(p, dict) for p in paragraphs)
    print('\n')

def test_extract_sections_from_xml():
    logger = setup_logging("test_logger", log_file="../logs/scraper.log")
    parser = XMLParser("open_bio_data_repos.json", logger, log_file_override=None,
                       llm_name='gemini-2.0-flash')
    xml_file_path = os.path.join('test_data', 'test_1.xml')
    with open(xml_file_path, 'rb') as f:  # ✅ open in binary mode
        xml_root = etree.fromstring(f.read())
    section = parser.extract_sections_from_xml(xml_root)
    assert isinstance(section, list)
    assert len(section) > 0
    assert all(isinstance(s, dict) for s in section)
    print('\n')


def test_resolve_data_repository():
    logger = setup_logging("test_logger", log_file="../logs/scraper.log", level="INFO",
                                    clear_previous_logs=True)
    parser = XMLParser("open_bio_data_repos.json", logger, log_file_override=None,
                       llm_name='gemini-2.0-flash')
    test_cases = {
        "NCBI GEO": "GEO",
        "NCBI Gene Expression Omnibus (GEO)": "GEO"
        # add more test cases as needed
    }

    for url,tgt in test_cases.items():
        print(f"Testing URL: {url}")
        data_repo = parser.resolve_data_repository(url)
        assert isinstance(data_repo, str)
        assert data_repo.lower() == tgt.lower()
        print('\n')

def test_extract_title_from_xml():
    logger = setup_logging("test_logger", log_file="../logs/scraper.log")
    parser = XMLParser("open_bio_data_repos.json", logger, log_file_override=None,
                       llm_name='gemini-2.0-flash')
    xml_file_path = os.path.join('test_data', 'test_1.xml')
    with open(xml_file_path, 'rb') as f:  # ✅ open in binary mode
        xml_root = etree.fromstring(f.read())
    title = parser.extract_publication_title(xml_root)
    assert isinstance(title, str)
    assert len(title) > 0
    assert title == "Dual molecule targeting HDAC6 leads to intratumoral CD4+ cytotoxic lymphocytes recruitment through MHC-II upregulation on lung cancer cells"
    print('\n')

def test_extract_title_from_html():
    logger = setup_logging("test_logger", log_file="../logs/scraper.log")
    parser = HTMLParser("open_bio_data_repos.json", logger, llm_name='gemini-2.0-flash')
    html_file_path = os.path.join('test_data', 'test_extract_1.html')
    with open(html_file_path, 'rb') as f:
        raw_html = f.read()
    title = parser.extract_publication_title(raw_html)
    assert isinstance(title, str)
    assert len(title) > 0
    assert "Proteogenomic insights suggest druggable pathways in endometrial carcinoma" in title
    print('\n')
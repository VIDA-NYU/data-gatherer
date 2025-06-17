from data_gatherer.parser import RuleBasedParser, LLMParser
from data_gatherer.logger_setup import setup_logging
from lxml import etree

import os
from dotenv import load_dotenv

load_dotenv()

def test_extract_paragraphs_from_xml():
    logger = setup_logging("test_logger", log_file="logs/scraper.log")
    parser = LLMParser("open_bio_data_repos.json", logger, log_file_override=None)
    xml_file_path = os.path.join('test_data', 'test_1.xml')
    with open(xml_file_path, 'rb') as f:  # ✅ open in binary mode
        xml_root = etree.fromstring(f.read())
    paragraphs = parser.extract_paragraphs_from_xml(xml_root)
    assert isinstance(paragraphs, list)
    assert len(paragraphs) > 0
    assert all(isinstance(p, dict) for p in paragraphs)
    print('\n')

def test_extract_sections_from_xml():
    logger = setup_logging("test_logger", log_file="logs/scraper.log")
    parser = LLMParser("open_bio_data_repos.json", logger, log_file_override=None)
    xml_file_path = os.path.join('test_data', 'test_1.xml')
    with open(xml_file_path, 'rb') as f:  # ✅ open in binary mode
        xml_root = etree.fromstring(f.read())
    section = parser.extract_sections_from_xml(xml_root)
    assert isinstance(section, list)
    assert len(section) > 0
    assert all(isinstance(s, dict) for s in section)
    print('\n')


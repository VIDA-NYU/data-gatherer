# retrievers/base.py
from abc import ABC, abstractmethod
from data_gatherer.resources_loader import load_config


class BaseRetriever(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(self, publisher='general'):
        """
        Initialize the BaseRetriever with retrieval patterns.

        :param retrieval_patterns_file: Path to the file containing retrieval patterns.
        """
        self.retrieval_patterns = load_config('retrieval_patterns.json')
        self.css_selectors = self.retrieval_patterns[publisher]['css_selectors']
        self.xpaths = self.retrieval_patterns[publisher]['xpaths']
        self.xml_tags = self.retrieval_patterns[publisher]['xml_tags']
        self.bad_patterns = self.retrieval_patterns[publisher].get('bad_patterns', [])

    def update_class_patterns(self, publisher):
        patterns = self.retrieval_patterns[publisher]
        self.css_selectors.update(patterns['css_selectors'])
        self.xpaths.update(patterns['xpaths'])
        if 'bad_patterns' in patterns.keys():
            self.bad_patterns.extend(patterns['bad_patterns'])
        if 'xml_tags' in patterns.keys():
            self.xml_tags.update(patterns['xml_tags'])

    def has_target_section(self, raw_data, section_name: str) -> bool:
        """
        Check if the target section (data availability or supplementary data) exists in the raw data.

        :param raw_data: Raw XML data.

        :param section_name: Name of the section to check.

        :return: True if the section is found with relevant links, False otherwise.
        """

        if raw_data is None:
            self.logger.info("No raw data to check for sections.")
            return False

        self.logger.debug(f"type of raw_data: {type(raw_data)}, raw_data: {raw_data}")

        self.logger.info(f"----Checking for {section_name} section in raw data.")
        section_patterns = self.load_target_sections_ptrs(section_name)
        self.logger.debug(f"Section patterns: {section_patterns}")
        namespaces = self.extract_namespaces(raw_data)
        self.logger.debug(f"Namespaces: {namespaces}")

        for pattern in section_patterns:
            self.logger.debug(f"Checking pattern: {pattern}")
            sections = raw_data.findall(pattern, namespaces=namespaces)
            if sections:
                for section in sections:
                    self.logger.info(f"----Found section: {ET.tostring(section, encoding='unicode')}")
                    if self.has_links_in_section(section, namespaces):
                        return True

        return False

    def load_target_sections_ptrs(self, section_name):
        """
        Load the XML tag patterns for the target section from the configuration.

        :param section_name: str — name of the section to load.

        :return: str — XML tag patterns for the target section.
        """

        if self.publisher in self.retrieval_patterns:
            if 'xml_tags' not in self.retrieval_patterns[self.publisher]:
                self.logger.error(f"XML tags not set for publisher '{self.publisher}' in retrieval patterns.")
                return None
            else:
                section_patterns = self.retrieval_patterns[self.publisher]
                if section_name in section_patterns.keys():
                    return section_patterns[section_name]

                else:
                    self.logger.error(f"Section name '{section_name}' not found in section patterns.")
                    return None

        else:
            self.logger.warning(f"Publisher '{self.publisher}' not found in retrieval patterns. Using default patterns.")
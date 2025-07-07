# retrievers/base.py
from abc import ABC, abstractmethod
from data_gatherer.resources_loader import load_config


class BaseRetriever(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(self, publisher='general', retrieval_patterns_file='retrieval_patterns.json'):
        """
        Initialize the BaseRetriever with retrieval patterns.

        :param retrieval_patterns_file: Path to the file containing retrieval patterns.
        """
        self.publisher = publisher
        self.retrieval_patterns = load_config(retrieval_patterns_file)
        self.bad_patterns = self.retrieval_patterns[publisher].get('bad_patterns', [])

    def update_class_patterns(self, publisher):
        patterns = self.retrieval_patterns[publisher]
        self.css_selectors.update(patterns['css_selectors'])
        self.xpaths.update(patterns['xpaths'])
        if 'bad_patterns' in patterns.keys():
            self.bad_patterns.extend(patterns['bad_patterns'])
        if 'xml_tags' in patterns.keys():
            self.xml_tags.update(patterns['xml_tags'])

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
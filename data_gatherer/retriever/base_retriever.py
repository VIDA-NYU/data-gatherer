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

    @abstractmethod
    def search(self, *args, **kwargs):
        """Retrieve relevant data from the corpus."""
        pass

    def update_class_patterns(self, publisher):
        patterns = self.retrieval_patterns[publisher]
        self.css_selectors.update(patterns['css_selectors'])
        self.xpaths.update(patterns['xpaths'])
        if 'bad_patterns' in patterns.keys():
            self.bad_patterns.extend(patterns['bad_patterns'])
        if 'xml_tags' in patterns.keys():
            self.xml_tags.update(patterns['xml_tags'])
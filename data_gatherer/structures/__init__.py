"""
Data structures for the data gatherer system.

This module provides standardized data structures for corpus management,
document representation, and retrieval workflows.
"""

from .corpus import CorpusDocument, Corpus
from .retrieval import RetrievalPipeline, RetrievalResult

__all__ = [
    'CorpusDocument', 
    'Corpus', 
    'RetrievalPipeline', 
    'RetrievalResult'
]
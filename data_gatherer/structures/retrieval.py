"""
Retrieval pipeline and result structures.

This module provides classes for orchestrating document processing and retrieval
workflows, including result representation and pipeline management.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
import logging
import time
from .corpus import Corpus, CorpusDocument


@dataclass
class RetrievalResult:
    """
    Standardized representation of a retrieval result.
    
    Contains the retrieved document along with relevance information
    and metadata about the retrieval process.
    """
    document: CorpusDocument
    score: float
    rank: int
    distance: Optional[float] = None  # For L2/cosine distance
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def preview_text(self) -> str:
        """Get a preview of the document text for display."""
        return self.document.display_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backwards compatibility."""
        result_dict = self.document.to_dict()
        result_dict.update({
            'score': self.score,
            'rank': self.rank,
            'retrieval_metadata': self.metadata
        })
        if self.distance is not None:
            result_dict['L2_distance'] = self.distance
        return result_dict
    
    @classmethod
    def from_embeddings_result(cls, result_dict: Dict[str, Any], rank: int) -> 'RetrievalResult':
        """
        Create RetrievalResult from embeddings retriever result.
        
        Args:
            result_dict: Dictionary from embeddings retriever
            rank: Rank in the result list
            
        Returns:
            RetrievalResult instance
        """
        # Extract document info
        doc = CorpusDocument.from_dict(result_dict)
        
        # Extract retrieval info
        score = result_dict.get('score', 0.0)
        distance = result_dict.get('L2_distance')
        
        # Metadata includes everything else
        excluded_keys = {'text', 'sec_txt', 'sec_txt_clean', 'section_title', 'sec_type', 
                        'source_file', 'chunk_id', 'score', 'L2_distance'}
        metadata = {k: v for k, v in result_dict.items() if k not in excluded_keys}
        
        return cls(
            document=doc,
            score=score,
            rank=rank,
            distance=distance,
            metadata=metadata
        )


class EmbeddingsRetrieverProtocol(Protocol):
    """Protocol for embeddings retriever dependency injection."""
    
    def embed_corpus(self, corpus: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """Embed a corpus of documents."""
        ...
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the embedded corpus."""
        ...
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text (if available)."""
        ...
    
    @property
    def corpus(self) -> List[Dict[str, Any]]:
        """Get the current corpus."""
        ...
    
    @corpus.setter 
    def corpus(self, value: List[Dict[str, Any]]) -> None:
        """Set the corpus."""
        ...


class DocumentParserProtocol(Protocol):
    """Protocol for document parser dependency injection."""
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from document content."""
        ...
    
    def normalize_content(self, content: str) -> str:
        """Normalize/preprocess document content."""
        ...


class RetrievalPipeline:
    """
    Orchestrates the complete document processing and retrieval workflow.
    
    This class provides a high-level interface for:
    1. Document parsing → sections
    2. Sections → standardized corpus
    3. Corpus processing (deduplication, chunking)
    4. Embedding and search
    """
    
    def __init__(self, 
                 parser: DocumentParserProtocol,
                 retriever: EmbeddingsRetrieverProtocol,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline with parser and retriever.
        
        Args:
            parser: Document parser implementation
            retriever: Embeddings retriever implementation  
            logger: Optional logger for pipeline operations
        """
        self.parser = parser
        self.retriever = retriever
        self.logger = logger or logging.getLogger(__name__)
        
        # Pipeline configuration
        self.config = {
            'deduplicate': True,
            'deduplication_strategy': 'merge_titles',
            'chunk_tokens': None,
            'min_section_length': 8,
            'embedding_batch_size': 32
        }
        
        # Pipeline state
        self.last_corpus: Optional[Corpus] = None
        self.last_processing_time: Optional[float] = None
        
    def configure(self, **kwargs) -> None:
        """
        Update pipeline configuration.
        
        Available options:
        - deduplicate: bool
        - deduplication_strategy: str ('merge_titles', 'keep_first', 'keep_longest') 
        - chunk_tokens: Optional[int]
        - min_section_length: int
        - embedding_batch_size: int
        """
        self.config.update(kwargs)
        self.logger.info(f"Pipeline configured: {self.config}")
    
    def process_document(self, content: str) -> Corpus:
        """
        Process a document into a retrieval-ready corpus.
        
        Args:
            content: Raw document content (HTML, XML, etc.)
            
        Returns:
            Processed Corpus ready for embedding/search
        """
        start_time = time.time()
        self.logger.info("Starting document processing pipeline")
        
        # Step 1: Parse document to sections
        self.logger.debug("Step 1: Parsing document content")
        if hasattr(self.parser, 'normalize_content'):
            content = self.parser.normalize_content(content)
        
        raw_sections = self.parser.extract_sections(content)
        self.logger.info(f"Extracted {len(raw_sections)} raw sections")
        
        # Step 2: Convert to standardized corpus
        self.logger.debug("Step 2: Creating standardized corpus")
        corpus = Corpus.from_dict_list(raw_sections, self.logger)
        
        # Step 3: Apply processing steps
        corpus = self._process_corpus(corpus)
        
        # Store for debugging/analysis
        self.last_corpus = corpus
        self.last_processing_time = time.time() - start_time
        
        self.logger.info(f"Document processing complete in {self.last_processing_time:.2f}s")
        self.logger.info(f"Final corpus stats: {corpus.stats}")
        
        return corpus
    
    def _process_corpus(self, corpus: Corpus) -> Corpus:
        """Apply configured processing steps to corpus."""
        
        # Filter by minimum length
        if self.config['min_section_length'] > 0:
            original_count = len(corpus)
            corpus = corpus.filter_by_min_length(self.config['min_section_length'])
            filtered_count = original_count - len(corpus)
            if filtered_count > 0:
                self.logger.info(f"Filtered out {filtered_count} short sections")
        
        # Deduplication
        if self.config['deduplicate']:
            corpus = corpus.deduplicate(self.config['deduplication_strategy'])
        
        # Chunking
        if self.config['chunk_tokens']:
            # Try to use retriever's tokenizer if available
            tokenizer = None
            if hasattr(self.retriever, 'get_token_count'):
                class RetrieverTokenizer:
                    def __init__(self, retriever):
                        self.retriever = retriever
                    
                    def count_tokens(self, text: str) -> int:
                        return self.retriever.get_token_count(text)
                    
                    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
                        # Fallback to simple chunking - could be enhanced
                        target_chars = max_tokens * 4
                        return [text[i:i+target_chars] for i in range(0, len(text), target_chars)]
                
                tokenizer = RetrieverTokenizer(self.retriever)
            
            corpus = corpus.chunk_by_tokens(self.config['chunk_tokens'], tokenizer)
        
        return corpus
    
    def search_corpus(self, corpus: Corpus, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Perform semantic search on a processed corpus.
        
        Args:
            corpus: Processed corpus to search
            query: Search query
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        self.logger.info(f"Performing semantic search with query: '{query[:100]}...'")
        
        # Convert corpus to embeddings format
        embeddings_corpus = corpus.to_embeddings_format()
        
        # Set corpus in retriever and embed
        self.retriever.corpus = embeddings_corpus
        self.retriever.embed_corpus(batch_size=self.config['embedding_batch_size'])
        
        # Perform search
        raw_results = self.retriever.search(query, k)
        
        # Convert to RetrievalResult objects
        results = []
        for i, result_dict in enumerate(raw_results):
            result = RetrievalResult.from_embeddings_result(result_dict, rank=i + 1)
            results.append(result)
        
        self.logger.info(f"Search complete: found {len(results)} results")
        
        return results
    
    def process_and_search(self, content: str, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Complete pipeline: process document and perform search in one call.
        
        Args:
            content: Raw document content
            query: Search query  
            k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        corpus = self.process_document(content)
        return self.search_corpus(corpus, query, k)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the last pipeline run."""
        stats = {
            'processing_time_seconds': self.last_processing_time,
            'configuration': self.config.copy()
        }
        
        if self.last_corpus:
            stats['corpus_stats'] = self.last_corpus.stats
        
        return stats


def create_simple_pipeline(parser, retriever, logger: Optional[logging.Logger] = None) -> RetrievalPipeline:
    """
    Convenience function to create a pipeline with sensible defaults.
    
    Args:
        parser: Document parser (HTMLParser, XMLParser, etc.)
        retriever: Embeddings retriever
        logger: Optional logger
        
    Returns:
        Configured RetrievalPipeline
    """
    pipeline = RetrievalPipeline(parser, retriever, logger)
    
    # Configure with reasonable defaults
    pipeline.configure(
        deduplicate=True,
        deduplication_strategy='merge_titles',
        chunk_tokens=512,  # Common token limit
        min_section_length=8,
        embedding_batch_size=32
    )
    
    return pipeline
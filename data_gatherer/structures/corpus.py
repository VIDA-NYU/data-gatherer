"""
Corpus and document data structures for embeddings retrieval.

This module provides standardized representations for documents and collections
of documents, with built-in processing capabilities like deduplication and chunking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Protocol
import logging
import re


@dataclass
class CorpusDocument:
    """
    Standardized document representation for embeddings retrieval.
    
    This class provides a consistent interface for documents across all parsers,
    ensuring compatibility with embeddings retriever and other components.
    """
    text: str
    section_title: str
    section_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    chunk_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Document text cannot be empty")
        if not self.section_title:
            self.section_title = "Untitled"
        if not self.section_type:
            self.section_type = "unknown"
    
    @property
    def display_text(self) -> str:
        """Get formatted text for display/logging purposes."""
        preview = self.text[:100].replace('\n', ' ').strip()
        chunk_info = f" (chunk {self.chunk_id})" if self.chunk_id else ""
        return f"[{self.section_title}]{chunk_info} {preview}..."
    
    @property
    def is_chunked(self) -> bool:
        """Check if this document is part of a chunked section."""
        return self.chunk_id is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for backwards compatibility.
        
        This ensures existing code that expects dict format continues to work.
        """
        base_dict = {
            'text': self.text,
            'section_title': self.section_title,
            'sec_type': self.section_type,
            'sec_txt': self.text,  # Legacy field
            'sec_txt_clean': self.text,  # Legacy field
        }
        
        # Add chunk info if present
        if self.chunk_id is not None:
            base_dict['chunk_id'] = self.chunk_id
            
        # Add source info if present
        if self.source_file:
            base_dict['source_file'] = self.source_file
            
        # Merge metadata
        base_dict.update(self.metadata)
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorpusDocument':
        """
        Create CorpusDocument from dictionary (for backwards compatibility).
        
        Args:
            data: Dictionary with document data (from existing parsers)
            
        Returns:
            CorpusDocument instance
        """
        # Extract core fields with fallbacks for different naming conventions
        text = data.get('text') or data.get('sec_txt_clean') or data.get('sec_txt', '')
        section_title = data.get('section_title', 'Untitled')
        section_type = data.get('sec_type') or data.get('section_type', 'unknown')
        
        # Extract optional fields
        source_file = data.get('source_file')
        chunk_id = data.get('chunk_id')
        
        # Everything else goes to metadata
        excluded_keys = {'text', 'sec_txt', 'sec_txt_clean', 'section_title', 'sec_type', 'source_file', 'chunk_id'}
        metadata = {k: v for k, v in data.items() if k not in excluded_keys}
        
        return cls(
            text=text,
            section_title=section_title,
            section_type=section_type,
            metadata=metadata,
            source_file=source_file,
            chunk_id=chunk_id
        )


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer dependency injection."""
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...
    
    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks respecting token limits."""
        ...


class Corpus:
    """
    Container for a collection of documents with processing capabilities.
    
    This class manages collections of CorpusDocument objects and provides
    methods for deduplication, chunking, and format conversion.
    """
    
    def __init__(self, documents: Optional[List[CorpusDocument]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize corpus with optional documents and logger.
        
        Args:
            documents: Initial list of documents
            logger: Logger for processing information
        """
        self.documents = documents or []
        self.logger = logger or logging.getLogger(__name__)
        self._stats = {
            'original_count': 0,
            'duplicates_removed': 0,
            'title_merges': 0,
            'chunks_created': 0
        }
    
    def __len__(self) -> int:
        """Return number of documents in corpus."""
        return len(self.documents)
    
    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)
    
    def __getitem__(self, index):
        """Get document by index."""
        return self.documents[index]
    
    def add_document(self, doc: CorpusDocument) -> None:
        """Add a document to the corpus."""
        self.documents.append(doc)
        self._stats['original_count'] += 1
    
    def add_from_dict(self, doc_dict: Dict[str, Any]) -> None:
        """Add a document from dictionary format."""
        doc = CorpusDocument.from_dict(doc_dict)
        self.add_document(doc)
    
    def extend(self, other_corpus: 'Corpus') -> None:
        """Add all documents from another corpus."""
        self.documents.extend(other_corpus.documents)
        self._stats['original_count'] += len(other_corpus.documents)
    
    def deduplicate(self, strategy: str = 'merge_titles') -> 'Corpus':
        """
        Remove duplicates using specified strategy.
        
        Args:
            strategy: Deduplication strategy ('merge_titles', 'keep_first', 'keep_longest')
            
        Returns:
            New Corpus with deduplicated documents
        """
        self.logger.info(f"Deduplicating {len(self.documents)} documents using strategy: {strategy}")
        
        if strategy == 'merge_titles':
            result_corpus = self._dedupe_merge_titles()
        elif strategy == 'keep_first':
            result_corpus = self._dedupe_keep_first()
        elif strategy == 'keep_longest':
            result_corpus = self._dedupe_keep_longest()
        else:
            raise ValueError(f"Unknown deduplication strategy: {strategy}")
        
        # Update stats
        duplicates_removed = len(self.documents) - len(result_corpus.documents)
        result_corpus._stats.update(self._stats)
        result_corpus._stats['duplicates_removed'] += duplicates_removed
        
        self.logger.info(f"Deduplication complete: {len(self.documents)} → {len(result_corpus.documents)} documents "
                        f"(removed {duplicates_removed} duplicates, {result_corpus._stats['title_merges']} title merges)")
        
        return result_corpus
    
    def _dedupe_merge_titles(self) -> 'Corpus':
        """Deduplicate by merging section titles for same content."""
        seen_texts = {}
        unique_docs = []
        title_merges = 0
        
        for doc in self.documents:
            text_key = doc.text.strip().lower()
            
            if text_key not in seen_texts:
                seen_texts[text_key] = doc
                unique_docs.append(doc)
                self.logger.debug(f"Added new document: '{doc.section_title}'")
            else:
                existing_doc = seen_texts[text_key]
                if doc.section_title and doc.section_title != existing_doc.section_title:
                    # Check if title is already included
                    if doc.section_title not in existing_doc.section_title:
                        existing_doc.section_title += f" | {doc.section_title}"
                        title_merges += 1
                        self.logger.debug(f"Merged titles: '{existing_doc.section_title}'")
                    else:
                        self.logger.debug(f"Title '{doc.section_title}' already included")
                else:
                    self.logger.debug(f"Skipping duplicate with same title: '{doc.section_title}'")
        
        result_corpus = Corpus(unique_docs, self.logger)
        result_corpus._stats['title_merges'] = title_merges
        return result_corpus
    
    def _dedupe_keep_first(self) -> 'Corpus':
        """Deduplicate by keeping first occurrence of each text."""
        seen_texts = set()
        unique_docs = []
        
        for doc in self.documents:
            text_key = doc.text.strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_docs.append(doc)
        
        return Corpus(unique_docs, self.logger)
    
    def _dedupe_keep_longest(self) -> 'Corpus':
        """Deduplicate by keeping the longest version of each text."""
        text_groups = {}
        
        # Group by normalized text
        for doc in self.documents:
            text_key = doc.text.strip().lower()
            if text_key not in text_groups:
                text_groups[text_key] = []
            text_groups[text_key].append(doc)
        
        # Keep longest from each group
        unique_docs = []
        for group in text_groups.values():
            longest_doc = max(group, key=lambda d: len(d.text))
            unique_docs.append(longest_doc)
        
        return Corpus(unique_docs, self.logger)
    
    def chunk_by_tokens(self, max_tokens: int, tokenizer: Optional[TokenizerProtocol] = None) -> 'Corpus':
        """
        Split documents that exceed token limits.
        
        Args:
            max_tokens: Maximum tokens per document
            tokenizer: Optional tokenizer for accurate token counting
            
        Returns:
            New Corpus with chunked documents
        """
        self.logger.info(f"Chunking documents with max_tokens={max_tokens}")
        
        if tokenizer is None:
            # Fallback to character-based estimation
            tokenizer = self._create_fallback_tokenizer()
        
        chunked_docs = []
        chunks_created = 0
        
        for doc in self.documents:
            token_count = tokenizer.count_tokens(doc.text)
            
            if token_count <= max_tokens:
                # No chunking needed
                chunked_docs.append(doc)
                self.logger.debug(f"Document '{doc.section_title}' fits in {token_count} tokens")
            else:
                # Need to chunk
                chunks = tokenizer.chunk_text(doc.text, max_tokens)
                self.logger.debug(f"Chunking '{doc.section_title}': {token_count} tokens → {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks):
                    chunked_doc = CorpusDocument(
                        text=chunk,
                        section_title=doc.section_title,
                        section_type=doc.section_type,
                        metadata=doc.metadata.copy(),
                        source_file=doc.source_file,
                        chunk_id=i + 1
                    )
                    chunked_docs.append(chunked_doc)
                    chunks_created += 1
        
        result_corpus = Corpus(chunked_docs, self.logger)
        result_corpus._stats.update(self._stats)
        result_corpus._stats['chunks_created'] = chunks_created
        
        self.logger.info(f"Chunking complete: {len(self.documents)} → {len(result_corpus.documents)} documents "
                        f"({chunks_created} chunks created)")
        
        return result_corpus
    
    def _create_fallback_tokenizer(self) -> TokenizerProtocol:
        """Create a simple character-based tokenizer fallback."""
        class FallbackTokenizer:
            def count_tokens(self, text: str) -> int:
                # Rough estimation: 1 token ≈ 4 characters
                return len(text) // 4
            
            def chunk_text(self, text: str, max_tokens: int) -> List[str]:
                target_chars = max_tokens * 4
                # Simple sentence-aware chunking
                sentences = re.split(r'(?<=[.!?]) +', text)
                
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= target_chars:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        return FallbackTokenizer()
    
    def to_embeddings_format(self) -> List[Dict[str, Any]]:
        """
        Convert to format expected by embeddings retriever.
        
        Returns:
            List of dictionaries compatible with existing retrieval code
        """
        return [doc.to_dict() for doc in self.documents]
    
    def filter_by_type(self, section_types: List[str]) -> 'Corpus':
        """Filter documents by section type."""
        filtered_docs = [doc for doc in self.documents if doc.section_type in section_types]
        return Corpus(filtered_docs, self.logger)
    
    def filter_by_min_length(self, min_chars: int) -> 'Corpus':
        """Filter out documents shorter than specified length."""
        filtered_docs = [doc for doc in self.documents if len(doc.text) >= min_chars]
        return Corpus(filtered_docs, self.logger)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        if not self.documents:
            return {
                'document_count': 0,
                'avg_text_length': 0,
                'section_types': [],
                'processing_stats': self._stats
            }
        
        return {
            'document_count': len(self.documents),
            'avg_text_length': sum(len(doc.text) for doc in self.documents) / len(self.documents),
            'total_text_length': sum(len(doc.text) for doc in self.documents),
            'section_types': list(set(doc.section_type for doc in self.documents)),
            'section_titles': [doc.section_title for doc in self.documents],
            'chunked_documents': len([doc for doc in self.documents if doc.is_chunked]),
            'processing_stats': self._stats
        }
    
    @classmethod
    def from_dict_list(cls, dict_list: List[Dict[str, Any]], logger: Optional[logging.Logger] = None) -> 'Corpus':
        """
        Create Corpus from list of dictionaries (backwards compatibility).
        
        Args:
            dict_list: List of document dictionaries from existing parsers
            logger: Optional logger
            
        Returns:
            Corpus instance
        """
        documents = [CorpusDocument.from_dict(d) for d in dict_list]
        corpus = cls(documents, logger)
        corpus._stats['original_count'] = len(documents)
        return corpus


def create_corpus_from_sections(sections: List[Dict[str, Any]], 
                              deduplicate: bool = True,
                              chunk_tokens: Optional[int] = None,
                              logger: Optional[logging.Logger] = None) -> Corpus:
    """
    Convenience function to create and process corpus from raw sections.
    
    This function provides a simple interface for the most common workflow:
    sections → corpus → deduplicate → chunk
    
    Args:
        sections: List of section dictionaries from parsers
        deduplicate: Whether to remove duplicates with title merging
        chunk_tokens: Optional token limit for chunking
        logger: Optional logger
        
    Returns:
        Processed Corpus ready for embeddings
    """
    if logger:
        logger.info(f"Creating corpus from {len(sections)} sections")
    
    # Create initial corpus
    corpus = Corpus.from_dict_list(sections, logger)
    
    # Apply processing
    if deduplicate:
        corpus = corpus.deduplicate('merge_titles')
    
    if chunk_tokens:
        corpus = corpus.chunk_by_tokens(chunk_tokens)
    
    if logger:
        logger.info(f"Final corpus: {corpus.stats}")
    
    return corpus
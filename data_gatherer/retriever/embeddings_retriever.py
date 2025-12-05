import numpy as np
import time
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer, AutoConfig
from data_gatherer.retriever.base_retriever import BaseRetriever
from data_gatherer.env import CACHE_BASE_DIR
import torch
import os
import json

class EmbeddingsRetriever(BaseRetriever):
    """
    Embeddings-based retriever for text passages, inspired by DSPy's approach.
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', corpus=None, device=None, logger=None, embed_corpus=True, read_cache=False, write_cache=False):
        """
        Initialize the EmbeddingsRetriever.

        Args:
            model_name (str): Name of the sentence transformer model to use.
            corpus (List[str]): List of text documents to embed.
            device (str): Device to run the model on ('cpu' or 'cuda').
            logger: Logger instance.
            embed_corpus (bool): Whether to embed the corpus during initialization. Default True for backward compatibility.
            read_cache (bool): Whether to read embeddings from cache.
            write_cache (bool): Whether to write embeddings to cache.
        """
        super().__init__(publisher='general', retrieval_patterns_file='retrieval_patterns.json')
        self.logger = logger
        if model_name is None:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.logger.warning(f"No embeddings model name provided under parameter 'model_name'. Defaulting to {model_name}")
        self.model_name = model_name
        self.corpus = corpus
        self.embeddings_cache = {}

        if os.path.exists(os.path.join(CACHE_BASE_DIR, "corpus_embeddings_cache.npz")) and (read_cache or write_cache):
            corpus_emb_cache = np.load(os.path.join(CACHE_BASE_DIR, "corpus_embeddings_cache.npz"), 'r')
            self.embeddings_cache = {key: corpus_emb_cache[key] for key in corpus_emb_cache.files}
        
        # Auto-detect best available device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info("CUDA available - using GPU acceleration")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Metal Performance Shaders for Apple Silicon
                self.logger.info("Metal Performance Shaders available - using Apple Silicon acceleration")
            else:
                device = "cpu"
                self.logger.info("Using CPU - no GPU acceleration available")
        
        self.device = device
        
        self.config = AutoConfig.from_pretrained(model_name)

        if "BiomedBERT" in model_name or "biomedbert" in model_name.lower():
            self.model = self._initialize_biomedbert_model(model_name, device)
        else:
            self.model = SentenceTransformer(model_name, device=device)
        self.logger.info(f"Initialized model: {self.model}")

        try:
            self.max_seq_length = self.model.get_max_seq_length()
            self.logger.info(f"Using model's actual max sequence length: {self.max_seq_length}")
        except:
            # Fallback for models without this method
            self.max_seq_length = 512  # Safe default for most sentence transformers
            self.logger.warning(f"Could not get model max seq length, using default: {self.max_seq_length}")
            
        # Initialize tokenizer for chunk size calculation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embeddings = None
        if corpus and embed_corpus:
            self.embed_corpus(read_cache=read_cache, write_cache=write_cache)
        self.query_embedding = None
    
    def cnt_tokens(self, text):
        """
        Count the number of tokens in the given text using the model's tokenizer.
        
        Args:
            text (str): Input text to tokenize.
        Returns:
            int: Number of tokens in the text.
        """
        tokens = self.tokenizer(text)
        self.logger.debug(f"Text tokenized into {len(tokens['input_ids'])} tokens.")
        return len(tokens['input_ids'])

    def embed_corpus(self, corpus=None, enable_chunking=True, chunk_size=None, chunk_overlap=20, batch_size=32, src=None, read_cache=False, write_cache=False,
                    fmt=None):
        """
        Embed the corpus using the initialized model with intelligent chunking to prevent truncation.
        
        Args:
            corpus: Optional corpus to embed. If None, uses self.corpus.
            enable_chunking (bool): Whether to enable intelligent chunking. Default True.
            chunk_size (int): Maximum tokens per chunk. If None, uses 80% of max_seq_length.
            chunk_overlap (int): Number of tokens to overlap between chunks.
            batch_size (int): Batch size for encoding. Default 32. Larger batches may be faster but use more memory.
            src (str): Source identifier for caching.
            read_cache (bool): Whether to read embeddings from cache.
            write_cache (bool): Whether to write embeddings to cache.
            fmt (str): format of the source string embedded (xml, html).
        """
        if corpus is not None:
            self.corpus = corpus
        corpus_texts = []

        if fmt is not None:
            src = src + '-' + fmt

        process_id = None

        if src is not None and (read_cache or write_cache):
            process_id = f"{src}_{self.model_name.replace('/', '_')}"
        
        if self.corpus is None:
            raise ValueError("No corpus provided for embedding")
        elif isinstance(self.corpus, str):
            raise ValueError("Corpus should be a list of documents, not a single string")
        elif isinstance(self.corpus, list):
            self.logger.info(f"Corpus contains {len(self.corpus)} documents of type ({type(self.corpus[0])})")
            corpus_texts = self.corpus
        elif isinstance(self.corpus, dict):
            self.logger.info(f"Corpus is a dict with {len(self.corpus)} entries.")
            corpus_texts = [doc['sec_txt'] if 'sec_txt' in doc else doc['text'] for doc in self.corpus]
        else:
            raise ValueError("Corpus should be a list or dict of documents")

        self.logger.info(f"Embedding {type(corpus_texts)} corpus of {len(corpus_texts)} documents using {self.model_name}")

        embed_start = time.time()
        if process_id is not None and read_cache and process_id in self.embeddings_cache:
            self.logger.info(f"Loading results from cache for process ID: {process_id}")
            self.embeddings =self.embeddings_cache[process_id]
        else:
            self.embeddings = self.model.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
        embed_time = time.time() - embed_start

        if src is not None and write_cache:
            self.embeddings_cache[process_id] = self.embeddings
            self.logger.info(f"Writing embeddings to cache for process ID: {process_id}")
            try:
                cache_path = os.path.join(CACHE_BASE_DIR, "corpus_embeddings_cache.npz")
                np.savez(cache_path, **self.embeddings_cache)
                self.logger.info(f"Successfully wrote embeddings cache with {len(self.embeddings_cache)} entries")
            except Exception as e:
                self.logger.error(f"Failed to write embeddings cache: {e}")

        self.logger.info(f"Embedding time: {embed_time:.2f}s ({embed_time/len(corpus_texts):.3f}s per chunk)")
        self.logger.info(f"Corpus embedding completed. Shape: {self.embeddings.shape}")
        
        if enable_chunking:
            # Log chunking statistics
            original_docs = len(corpus_texts)
            final_chunks = self.embeddings.shape[0]
            self.logger.debug(f"Chunking results: {original_docs} original documents → {final_chunks} embedded chunks")

    def _initialize_biomedbert_model(self, model_name, device):
        """
        Initialize a SentenceTransformer model using BiomedBERT for embedding generation.
        Args:
            model_name (str): HuggingFace model name for BiomedBERT.
            device (str): Device for embedding model.
        Returns:
            SentenceTransformer: Initialized SentenceTransformer model.
        """

        return SentenceTransformer(
            modules=[
                models.Transformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", max_seq_length=self.config.max_position_embeddings),
                models.Pooling("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", pooling_mode='mean')
            ], device=device
        )

    @classmethod
    def create_model_only(cls, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu', logger=None):
        """
        Create an EmbeddingsRetriever instance with only the model initialized (no corpus embedding).
        Useful for performance optimization when you want to reuse the same model for multiple corpora.
        
        Args:
            model_name (str): Name of the sentence transformer model to use.
            device (str): Device to run the model on ('cpu' or 'cuda').
            logger: Logger instance.
            
        Returns:
            EmbeddingsRetriever: Instance with model loaded but no corpus embedded.
        """
        return cls(model_name=model_name, corpus=None, device=device, logger=logger, embed_corpus=False)

    def _l2_search(self, query_emb, k):
        """
        Perform L2 distance search using numpy.
        Args:
            query_emb (np.ndarray): Query embedding of shape (1, dim).
            k (int): Number of results to return.
        Returns:
            indices (np.ndarray): Indices of top-k nearest neighbors.
            distances (np.ndarray): L2 distances of top-k nearest neighbors.
        """
        # Compute squared L2 distances
        self.logger.info("Computing L2 distances using numpy.")
        dists = np.sum((self.embeddings - query_emb) ** 2, axis=1)
        idxs = np.argpartition(dists, k)[:k]
        # Sort the top-k indices by distance
        sorted_idxs = idxs[np.argsort(dists[idxs])]
        return sorted_idxs, dists[sorted_idxs]

    def search(self, query=None, k=5):
        """
        Retrieve top-k most similar passages to the query.

        Args:
            query (str): Query string. Pass None to use pre-embedded query.
            k (int): Number of results to return.

        Returns:
            List[dict]: List of result dictionaries with text, metadata, and scores.
        """
        self.logger.info(f"Searching for top-{k} passages similar to the query by embeddings.")
        if k > len(self.corpus):
            raise ValueError(f"top-k k-parameter ({k}) is greater than the corpus size {len(self.corpus)}. Please set k "
                             f"to a smaller value.")
        query_emb = self.model.encode([query], convert_to_numpy=True)[0] if query is not None else self.query_embedding
        idxs, dists = self._l2_search(query_emb, k)
        results = []
        for idx, score in zip(idxs, dists):
            doc = self.corpus[idx]

            if isinstance(doc, str):
                text_content = doc
            else:
                text_content = doc['sec_txt'] if 'sec_txt' in doc else doc['text']
                
            result = {
                'text': text_content,
                'section_title': doc.get('section_title', None),
                'sec_type': doc.get('sec_type', None),
                'L2_distance': float(score)
            }
            
            # Add chunk metadata if available
            if 'chunk_id' in doc:
                result.update({
                    'chunk_id': doc['chunk_id'],
                    'is_chunked': True
                })
            else:
                result['is_chunked'] = False
            
            passage = result.get('text', 'sec_txt')
            chunk_info = f" (chunk {doc.get('chunk_id', 'N/A')+1}/{doc.get('total_chunks', 'N/A')})" if 'chunk_id' in doc else ""
            passage_preview = str(passage)[:100] if passage is not None else ""
            self.logger.debug(f"Retrieved passage{chunk_info}: {passage_preview}... with L2 distance: {score}")
            results.append(result)
        return results

    def embed_query(self, query):
        """
        Store query embedding as attribute for the retriever
        """
        if len(query) > self.max_seq_length*4:
            self.logger.warning(f"Query maybe longer than max tokens limit for model {self.model}.")
        self.query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
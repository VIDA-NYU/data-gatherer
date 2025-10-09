import numpy as np
import time
from sentence_transformers import SentenceTransformer, models
from data_gatherer.retriever.base_retriever import BaseRetriever

class EmbeddingsRetriever(BaseRetriever):
    """
    Embeddings-based retriever for text passages, inspired by DSPy's approach.
    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', corpus=None, device='cpu', logger=None, embed_corpus=True):
        """
        Initialize the EmbeddingsRetriever.

        Args:
            model_name (str): Name of the sentence transformer model to use.
            corpus (List[str]): List of text documents to embed.
            device (str): Device to run the model on ('cpu' or 'cuda').
            logger: Logger instance.
            embed_corpus (bool): Whether to embed the corpus during initialization. Default True for backward compatibility.
        """
        super().__init__(publisher='general', retrieval_patterns_file='retrieval_patterns.json')
        self.logger = logger
        self.model_name = model_name
        self.device = device
        self.corpus = corpus
        if "BiomedBERT" in model_name or "biomedbert" in model_name.lower():
            self.model = self._initialize_biomedbert_model(model_name, device)
        else:
            self.model = SentenceTransformer(model_name, device=device)
        self.logger.info(f"Initialized model: {self.model}")
        self.embeddings = None
        if corpus and embed_corpus:
            self.embed_corpus()
        self.query_embedding = None

    def embed_corpus(self, corpus=None):
        """
        Embed the corpus using the initialized model.
        
        Args:
            corpus: Optional corpus to embed. If None, uses self.corpus.
        """
        if corpus is not None:
            self.corpus = corpus
        
        if self.corpus is None:
            raise ValueError("No corpus provided for embedding")
            
        self.logger.info(f"Embedding corpus of {len(self.corpus)} documents using {self.model_name}")
        corpus_texts = [doc['sec_txt'] if 'sec_txt' in doc else doc['text'] for doc in self.corpus]
        embed_start = time.time()
        self.embeddings = self.model.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)
        embed_time = time.time() - embed_start
        self.logger.info(f"Embedding time: {embed_time:.2f}s ({embed_time/len(self.corpus):.3f}s per doc)")
        self.logger.info(f"Corpus embedding completed. Shape: {self.embeddings.shape}")

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
                models.Transformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", max_seq_length=512),
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
            query (str): Query string.
            k (int): Number of results to return.

        Returns:
            List[Tuple[str, float]]: List of (passage, score) tuples.
        """
        self.logger.info(f"Searching for top-{k} passages similar to the query by embeddings.")
        if k > len(self.corpus):
            raise ValueError(f"top-k k-parameter ({k}) is greated than the corpus size {len(self.corpus)}. Please set k "
                             f"to a smaller value.")
        query_emb = self.model.encode([query], convert_to_numpy=True)[0] if query is not None else self.query_embedding
        idxs, dists = self._l2_search(query_emb, k)
        results = []
        for idx, score in zip(idxs, dists):
            results.append({
                'text': self.corpus[idx]['sec_txt'] if 'sec_txt' in self.corpus[idx] else self.corpus[idx]['text'],
                'section_title': self.corpus[idx]['section_title'] if 'section_title' in self.corpus[idx] else None,
                'sec_type': self.corpus[idx]['sec_type'] if 'sec_type' in self.corpus[idx] else None,
                'L2_distance': float(score)
            })
            passage = results[-1]['text']
            self.logger.debug(f"Retrieved passage: {passage[:100]}... with L2 distance: {score}")
        return results

    def embed_query(self, query, max_tokens=512):
        """
        Store query embedding as attribute for the retriever
        """
        if len(query) > max_tokens/4:
            self.logger.warning(f"Query is longer than max tokens limit for model {self.model}.")
        self.query_embedding = self.model.encode([query], convert_to_numpy=True)[0]



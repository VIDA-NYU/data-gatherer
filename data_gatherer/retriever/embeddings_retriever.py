import numpy as np
from sentence_transformers import SentenceTransformer
from data_gatherer.retriever.base_retriever import BaseRetriever

class EmbeddingsRetriever(BaseRetriever):
    """
    Embeddings-based retriever for text passages, inspired by DSPy's approach.
    """

    def __init__(self, corpus, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu', logger=None):
        """
        Args:
            corpus (List[str]): List of text passages to index.
            model_name (str): HuggingFace model name for sentence embeddings.
            device (str): Device for embedding model.
        """
        self.logger = logger
        self.corpus = corpus
        print(f"Initializing EmbeddingsRetriever with: {model_name} on corpus: {len(corpus), type(corpus)}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)

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
        print("Computing L2 distances using numpy.")
        dists = np.sum((self.embeddings - query_emb) ** 2, axis=1)
        idxs = np.argpartition(dists, k)[:k]
        # Sort the top-k indices by distance
        sorted_idxs = idxs[np.argsort(dists[idxs])]
        return sorted_idxs, dists[sorted_idxs]

    def search(self, query, k=5):
        """
        Retrieve top-k most similar passages to the query.

        Args:
            query (str): Query string.
            k (int): Number of results to return.

        Returns:
            List[Tuple[str, float]]: List of (passage, score) tuples.
        """
        print(f"Searching for top-{k} passages similar to the query by embeddings.")
        if k > len(self.corpus):
            raise ValueError(f"top-k k-parameter ({k}) is greated than the corpus size {len(self.corpus)}. Please set k "
                             f"to a smaller value.")
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
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

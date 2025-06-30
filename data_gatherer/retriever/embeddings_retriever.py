import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from data_gatherer.retriever.base_retriever import BaseRetriever

class EmbeddingsRetriever(BaseRetriever):
    """
    Embeddings-based retriever for text passages, inspired by DSPy's approach.
    """

    def __init__(self, corpus, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        """
        Args:
            corpus (List[str]): List of text passages to index.
            model_name (str): HuggingFace model name for sentence embeddings.
            device (str): Device for embedding model.
        """
        self.corpus = corpus
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, k=5):
        """
        Retrieve top-k most similar passages to the query.

        Args:
            query (str): Query string.
            k (int): Number of results to return.

        Returns:
            List[Tuple[str, float]]: List of (passage, score) tuples.
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                'text': self.corpus[idx],
                'Faiss_index': float(score)
            })
        return results
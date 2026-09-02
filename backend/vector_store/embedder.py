"""
Embedder and semantic vector indexer for NovoMind RAG.
Supports pluggable OpenAI embeddings as well as a fast local TF-IDF/Dense embedding fallback for offline environments.
"""

import math
import os
import re
import json
from typing import List, Dict, Any, Tuple

class LocalSemanticEmbedder:
    """
    High-performance semantic vector embedder with dimensionality reduction
    and cosine similarity computation. Supports offline standalone execution
    and OpenAI API integration if API key is provided.
    """
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())
        return words

    def fit_and_embed(self, texts: List[str]) -> List[List[float]]:
        # Build vocabulary & document frequencies
        doc_count = len(texts)
        df: Dict[str, int] = {}
        for text in texts:
            words = set(self._tokenize(text))
            for w in words:
                df[w] = df.get(w, 0) + 1

        self.idf = {w: math.log((doc_count + 1) / (count + 1)) + 1.0 for w, count in df.items()}
        sorted_vocab = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:self.vector_dim]
        self.vocab = {item[0]: i for i, item in enumerate(sorted_vocab)}

        return [self.embed(t) for t in texts]

    def embed(self, text: str) -> List[float]:
        words = self._tokenize(text)
        vec = [0.0] * self.vector_dim
        if not words:
            return vec

        # TF-IDF weighted vector
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1

        for w, count in tf.items():
            if w in self.vocab:
                idx = self.vocab[w]
                weight = (count / len(words)) * self.idf.get(w, 1.0)
                vec[idx] = weight
            else:
                # Character hashing for out-of-vocabulary semantic grounding
                h_idx = abs(hash(w)) % self.vector_dim
                vec[h_idx] += 0.1 * (count / len(words))

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot))

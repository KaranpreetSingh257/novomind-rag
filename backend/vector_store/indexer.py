"""
Document ingestion pipeline and Vector Index Manager for NovoMind RAG.
"""

import os
import json
import uuid
from typing import List, Dict, Any, Optional
from core.state import DocumentChunk
from vector_store.embedder import LocalSemanticEmbedder, cosine_similarity
from core.config import VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

class VectorIndexer:
    def __init__(self, index_file: str = VECTOR_DB_PATH):
        self.index_file = index_file
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = LocalSemanticEmbedder()
        self.load_index()

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Break nicely on paragraph or newline if possible
            if end < len(text):
                last_newline = chunk.rfind('\n')
                if last_newline > chunk_size * 0.5:
                    end = start + last_newline + 1
                    chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
            if start < 0 or start >= len(text):
                break
        return chunks

    def ingest_document(self, content: str, source_name: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not metadata:
            metadata = {}

        raw_chunks = self.chunk_text(content)
        if not raw_chunks:
            return 0

        # Remove existing chunks for this source
        self.chunks = [c for c in self.chunks if c.get("source") != source_name]

        # Embed and register new chunks
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{uuid.uuid4().hex[:8]}_{i}"
            chunk_metadata = {**metadata, "chunk_index": i, "total_chunks": len(raw_chunks)}
            
            self.chunks.append({
                "doc_id": chunk_id,
                "source": source_name,
                "content": chunk_text,
                "metadata": chunk_metadata
            })

        self.rebuild_embeddings()
        self.save_index()
        return len(raw_chunks)

    def rebuild_embeddings(self):
        if not self.chunks:
            return
        all_texts = [c["content"] for c in self.chunks]
        vectors = self.embedder.fit_and_embed(all_texts)
        for chunk, vec in zip(self.chunks, vectors):
            chunk["vector"] = vec

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.15) -> List[DocumentChunk]:
        if not self.chunks:
            return []

        query_vec = self.embedder.embed(query)
        results = []

        for chunk in self.chunks:
            vec = chunk.get("vector")
            if not vec:
                continue
            sim = cosine_similarity(query_vec, vec)
            if sim >= score_threshold:
                results.append(DocumentChunk(
                    doc_id=chunk["doc_id"],
                    source=chunk["source"],
                    content=chunk["content"],
                    score=sim,
                    metadata=chunk.get("metadata", {})
                ))

        # Sort descending by similarity score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def save_index(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                if self.chunks:
                    self.rebuild_embeddings()
            except Exception as e:
                print(f"Error loading index: {e}")
                self.chunks = []

    def get_document_stats(self) -> List[Dict[str, Any]]:
        sources: Dict[str, int] = {}
        for c in self.chunks:
            s = c["source"]
            sources[s] = sources.get(s, 0) + 1
        return [{"source": k, "chunks_count": v} for k, v in sources.items()]

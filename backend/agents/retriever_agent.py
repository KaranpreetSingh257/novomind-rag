"""
Agent 2: Semantic Vector Retriever Agent.
Executes multi-hop retrieval across vector embeddings and aggregates contextual chunks with deduplication.
"""

from typing import List, Dict, Any
from core.state import NovoMindState, DocumentChunk, AgentStepTrace
from vector_store.indexer import VectorIndexer

class RetrieverAgent:
    def __init__(self, indexer: VectorIndexer):
        self.name = "VectorRetriever"
        self.indexer = indexer

    def run(self, state: NovoMindState) -> NovoMindState:
        all_chunks: Dict[str, DocumentChunk] = {}

        sub_queries = state.sub_queries or [state.query]
        for sub_q in sub_queries:
            results = self.indexer.search(sub_q, top_k=4)
            for chunk in results:
                if chunk.doc_id not in all_chunks:
                    all_chunks[chunk.doc_id] = chunk
                else:
                    # Keep highest score
                    if chunk.score > all_chunks[chunk.doc_id].score:
                        all_chunks[chunk.doc_id] = chunk

        # If cyclical retry was triggered with feedback, search on original query too
        if state.iteration_count > 0 and state.critique_feedback:
            fallback_results = self.indexer.search(state.query, top_k=5, score_threshold=0.05)
            for chunk in fallback_results:
                if chunk.doc_id not in all_chunks:
                    all_chunks[chunk.doc_id] = chunk

        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)[:6]
        state.retrieved_docs = sorted_chunks

        sources_found = list({c.source for c in sorted_chunks})
        avg_score = sum(c.score for c in sorted_chunks) / len(sorted_chunks) if sorted_chunks else 0.0

        thought = (
            f"Retrieved {len(sorted_chunks)} candidate chunks from {len(sources_found)} document(s) "
            f"(Sources: {', '.join(sources_found) if sources_found else 'None'}). Mean similarity: {avg_score:.2f}."
        )

        state.traces.append(AgentStepTrace(
            agent_name=self.name,
            action="VECTOR_RETRIEVAL",
            thought=thought,
            details={
                "retrieved_count": len(sorted_chunks),
                "sources": sources_found,
                "avg_score": round(avg_score, 4)
            }
        ))
        return state

"""
Automated unit & workflow tests for NovoMind Multi-Agent RAG System.
"""

import os
import unittest
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from vector_store.indexer import VectorIndexer
from graph.workflow import MultiAgentRAGWorkflow
from core.state import NovoMindState

class TestNovoMindMultiAgentRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_db_path = os.path.join(BASE_DIR, "storage", "test_vector_index.json")
        cls.indexer = VectorIndexer(index_file=test_db_path)
        
        # Ingest test sample documents
        doc1 = """
        NovoMind uses LangGraph for cyclical multi-agent orchestration.
        The Query Decomposer agent parses multi-hop questions into sub-tasks.
        The Vector Retriever agent queries ChromaDB and FAISS for embeddings.
        """
        doc2 = """
        All API requests require JWT Bearer authentication.
        Rate limits are capped at 60 requests/minute for standard tier and 1200 for enterprise.
        """
        cls.indexer.ingest_document(doc1, "novomind_architecture.md")
        cls.indexer.ingest_document(doc2, "security_policy.md")
        cls.workflow = MultiAgentRAGWorkflow(cls.indexer)

    def test_chunking_and_indexing(self):
        self.assertGreater(len(self.indexer.chunks), 0)
        stats = self.indexer.get_document_stats()
        sources = [s["source"] for s in stats]
        self.assertIn("novomind_architecture.md", sources)
        self.assertIn("security_policy.md", sources)

    def test_semantic_vector_search(self):
        results = self.indexer.search("How does Query Decomposer work in LangGraph?", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].source, "novomind_architecture.md")

    def test_multi_agent_workflow_execution(self):
        state = self.workflow.run("What are the API rate limits and JWT authentication rules?")
        
        # Verify state properties
        self.assertTrue(len(state.sub_queries) >= 1)
        self.assertTrue(len(state.retrieved_docs) >= 1)
        self.assertTrue(state.is_grounded)
        self.assertGreater(state.critique_score, 0.20)
        self.assertTrue(len(state.citations) >= 1)
        self.assertIn("security_policy.md", [c["source"] for c in state.citations])

        # Verify agent trace log
        agent_names = [t.agent_name for t in state.traces]
        self.assertIn("QueryDecomposer", agent_names)
        self.assertIn("VectorRetriever", agent_names)
        self.assertIn("CritiqueGrader", agent_names)
        self.assertIn("AnswerSynthesizer", agent_names)

if __name__ == "__main__":
    unittest.main()

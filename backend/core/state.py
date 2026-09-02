"""
Graph State definition for the NovoMind Multi-Agent RAG cyclical workflow.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentStepTrace:
    agent_name: str
    action: str
    thought: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NovoMindState:
    query: str
    sub_queries: List[str] = field(default_factory=list)
    retrieved_docs: List[DocumentChunk] = field(default_factory=list)
    is_grounded: bool = False
    critique_score: float = 0.0
    critique_feedback: str = ""
    iteration_count: int = 0
    max_iterations: int = 2
    final_answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    traces: List[AgentStepTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sub_queries": self.sub_queries,
            "retrieved_docs": [
                {
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "content": d.content,
                    "score": round(d.score, 4),
                    "metadata": d.metadata
                }
                for d in self.retrieved_docs
            ],
            "is_grounded": self.is_grounded,
            "critique_score": round(self.critique_score, 2),
            "critique_feedback": self.critique_feedback,
            "iteration_count": self.iteration_count,
            "final_answer": self.final_answer,
            "citations": self.citations,
            "traces": [
                {
                    "agent_name": t.agent_name,
                    "action": t.action,
                    "thought": t.thought,
                    "details": t.details
                }
                for t in self.traces
            ]
        }

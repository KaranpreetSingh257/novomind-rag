"""
Agent 4: Answer Synthesis & Citation Agent.
Synthesizes contextually grounded answers with source citations.
"""

from typing import List, Dict, Any
from core.state import NovoMindState, AgentStepTrace

class SynthesizerAgent:
    def __init__(self):
        self.name = "AnswerSynthesizer"

    def run(self, state: NovoMindState) -> NovoMindState:
        docs = state.retrieved_docs
        query = state.query

        if not docs:
            state.final_answer = (
                f"I searched the project directory for **'{query}'**, but could not find relevant documentation or source code chunks. "
                "Please verify that the corresponding files have been uploaded/indexed in the repository manager."
            )
            state.citations = []
            state.traces.append(AgentStepTrace(
                agent_name=self.name,
                action="SYNTHESIS_EMPTY",
                thought="No documents available to synthesize grounded response.",
                details={}
            ))
            return state

        # Build citations
        citations = []
        for i, d in enumerate(docs):
            citations.append({
                "citation_id": f"[{i+1}]",
                "source": d.source,
                "doc_id": d.doc_id,
                "score": round(d.score, 3),
                "snippet": d.content[:150] + ("..." if len(d.content) > 150 else "")
            })
        state.citations = citations

        # Synthesize clear, grounded answer using the retrieved context
        context_summary_lines = []
        for i, d in enumerate(docs):
            context_summary_lines.append(f"- **From {d.source}** (Relevance: {d.score:.2f}):\n> {d.content}")

        context_block = "\n\n".join(context_summary_lines)

        answer_markdown = (
            f"### Contextual Answer for: *{query}*\n\n"
            f"Based on the multi-agent retrieval and contextual verification of your indexed files, here are the key findings:\n\n"
            f"{context_block}\n\n"
            f"#### 📚 Source Citations:\n"
        )

        for c in citations:
            answer_markdown += f"- **{c['citation_id']}** `{c['source']}` (Confidence: {c['score']})\n"

        state.final_answer = answer_markdown

        state.traces.append(AgentStepTrace(
            agent_name=self.name,
            action="SYNTHESIZE_ANSWER",
            thought=f"Synthesized comprehensive answer with {len(citations)} verified citations.",
            details={"citation_count": len(citations)}
        ))
        return state

"""
Cyclical Multi-Agent Graph Coordinator for NovoMind RAG.
Orchestrates:
QueryDecomposer -> VectorRetriever -> CritiqueGrader -> (if not grounded and iterations < max -> Rewrite & Re-retrieve) -> Synthesizer
"""

from core.state import NovoMindState, AgentStepTrace
from agents.query_decomposer import QueryDecomposerAgent
from agents.retriever_agent import RetrieverAgent
from agents.critique_grader import CritiqueGraderAgent
from agents.synthesizer_agent import SynthesizerAgent
from vector_store.indexer import VectorIndexer

class MultiAgentRAGWorkflow:
    def __init__(self, indexer: VectorIndexer):
        self.indexer = indexer
        self.decomposer = QueryDecomposerAgent()
        self.retriever = RetrieverAgent(indexer)
        self.grader = CritiqueGraderAgent()
        self.synthesizer = SynthesizerAgent()

    def run(self, query: str, max_iterations: int = 2) -> NovoMindState:
        state = NovoMindState(query=query, max_iterations=max_iterations)

        # Step 1: Query Decomposition & Planning
        state = self.decomposer.run(state)

        # Cyclical Execution Loop
        while state.iteration_count < state.max_iterations:
            # Step 2: Vector Retrieval
            state = self.retriever.run(state)

            # Step 3: Critique & Grounding Grading
            state = self.grader.run(state)

            # Check termination condition
            if state.is_grounded or state.iteration_count >= state.max_iterations - 1:
                break

            # If not grounded and iterations remaining, trigger cyclical re-planning
            state.iteration_count += 1
            state.traces.append(AgentStepTrace(
                agent_name="GraphCoordinator",
                action="CYCLICAL_RETRY",
                thought=f"Triggering cyclical iteration {state.iteration_count} to expand search terms based on critique feedback.",
                details={"iteration": state.iteration_count}
            ))

        # Step 4: Final Answer Synthesis
        state = self.synthesizer.run(state)
        return state

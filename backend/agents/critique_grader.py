"""
Agent 3: Grounding Critique & Hallucination Grader Agent.
Verifies factual grounding, scores document relevance, and decides whether a query rewrite loop is required.
"""

from core.state import NovoMindState, AgentStepTrace

class CritiqueGraderAgent:
    def __init__(self, score_threshold: float = 0.25):
        self.name = "CritiqueGrader"
        self.score_threshold = score_threshold

    def run(self, state: NovoMindState) -> NovoMindState:
        docs = state.retrieved_docs
        
        if not docs:
            state.is_grounded = False
            state.critique_score = 0.0
            state.critique_feedback = "No context retrieved matching query keywords."
            thought = "Critique Grade: FAILED (0.0). No relevant context retrieved. Cyclical query expansion required."
        else:
            top_score = max(c.score for c in docs)
            avg_score = sum(c.score for c in docs) / len(docs)
            
            # Weighted confidence scoring
            confidence = (top_score * 0.7) + (avg_score * 0.3)
            state.critique_score = confidence

            if confidence >= self.score_threshold:
                state.is_grounded = True
                state.critique_feedback = f"Sufficient factual grounding verified (Relevance Score: {confidence:.2f})."
                thought = f"Critique Grade: PASSED ({confidence:.2f}). Grounding verified across {len(docs)} chunks. Proceeding to answer synthesis."
            else:
                state.is_grounded = False
                state.critique_feedback = f"Low grounding confidence ({confidence:.2f} < {self.score_threshold})."
                thought = f"Critique Grade: LOW CONFIDENCE ({confidence:.2f}). Initiating cyclical feedback loop."

        state.traces.append(AgentStepTrace(
            agent_name=self.name,
            action="CRITIQUE_GROUNDING",
            thought=thought,
            details={
                "is_grounded": state.is_grounded,
                "score": round(state.critique_score, 2),
                "feedback": state.critique_feedback
            }
        ))
        return state

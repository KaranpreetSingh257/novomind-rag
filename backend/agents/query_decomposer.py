"""
Agent 1: Query Decomposer & Planner Agent.
Analyzes user queries, identifies complexity, and decomposes into targeted sub-questions.
"""

import re
from typing import List
from core.state import NovoMindState, AgentStepTrace

class QueryDecomposerAgent:
    def __init__(self):
        self.name = "QueryDecomposer"

    def run(self, state: NovoMindState) -> NovoMindState:
        query = state.query.strip()
        sub_queries: List[str] = []

        # Complex multi-hop pattern detection (e.g., comparing X and Y, multiple questions, and/or)
        conjunctions = re.split(r'\band\b|\balso\b|\bcompare\b|\bvs\b|\bhow does .* differ from\b|\?|,', query, flags=re.IGNORECASE)
        valid_parts = [p.strip() for p in conjunctions if len(p.strip().split()) >= 2]

        if len(valid_parts) > 1:
            sub_queries = valid_parts[:3]
            thought = f"Identified multi-aspect query. Decomposed '{query}' into {len(sub_queries)} distinct sub-tasks."
        else:
            sub_queries = [query]
            thought = f"Query is direct and cohesive. Proceeding with targeted single-hop retrieval plan for: '{query}'."

        state.sub_queries = sub_queries
        state.traces.append(AgentStepTrace(
            agent_name=self.name,
            action="DECOMPOSE_QUERY",
            thought=thought,
            details={"sub_queries": sub_queries}
        ))
        return state

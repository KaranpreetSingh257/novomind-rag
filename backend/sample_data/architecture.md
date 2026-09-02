# NovoMind Microservice & Multi-Agent Architecture

## 1. System Overview
NovoMind is an enterprise-grade Autonomous Multi-Agent Retrieval-Augmented Generation (RAG) platform. It orchestrates intelligent LLM agents to decompose user queries, extract high-dimensional semantic vector embeddings, evaluate factual grounding, and stream synthesized responses in real-time.

## 2. Core Agents
- **Query Decomposer Agent**: Analyzes user intent, separates compound questions into discrete search operations, and generates multi-hop execution plans.
- **Semantic Vector Retriever Agent**: Interfaces with vector indexes (ChromaDB / FAISS) using cosine similarity ranking over high-dimensional dense embeddings.
- **Grounding Critique & Grader Agent**: Assesses whether candidate retrieved passages contain factual answers to avoid hallucinations. If relevance is low, it triggers a cyclical re-query loop.
- **Answer Synthesizer Agent**: Generates structured markdown output with precise inline file and chunk citations.

## 3. Asynchronous Streaming Architecture
FastAPI provides asynchronous Server-Sent Events (SSE) token streaming via `/api/chat/stream`. Frontends receive live agent thought states (`DECOMPOSE_QUERY`, `VECTOR_RETRIEVAL`, `CRITIQUE_GROUNDING`, `SYNTHESIZE_ANSWER`) followed by the streamed text tokens.

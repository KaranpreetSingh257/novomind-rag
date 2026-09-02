# NovoMind: Autonomous Multi-Agent RAG System

NovoMind is an enterprise-grade Autonomous Retrieval-Augmented Generation (RAG) platform leveraging LangGraph cyclical multi-agent workflows, semantic vector retrieval (ChromaDB / FAISS), and real-time asynchronous token streaming with FastAPI and React.

---

## 🏗️ Multi-Agent Architecture

```
                    +--------------------+
                    |     User Query     |
                    +--------------------+
                               |
                               v
               +-------------------------------+
               | 1. Query Decomposer & Planner |
               +-------------------------------+
                               |
                               v
               +-------------------------------+
               |   2. Semantic Vector Search   |
               |      (ChromaDB / FAISS)       |
               +-------------------------------+
                               |
                               v
               +-------------------------------+
               | 3. Grounding Critique & Grader|<-------+
               +-------------------------------+        |
                               |                        |
                   Is Grounded? / Score >= Threshold    | Retry Loop
                      /                 \               | (Rewritten Query)
                    YES                  NO ------------+
                     |
                     v
               +-------------------------------+
               | 4. Answer Synthesis & Citation|
               +-------------------------------+
                               |
                               v
               +-------------------------------+
               | Real-Time SSE Token Streaming |
               +-------------------------------+
```

---

## 🤖 Core Agents
1. **Query Decomposer Agent**: Deconstructs multi-hop and ambiguous user prompts into discrete sub-questions and keyword plans.
2. **Vector Retriever Agent**: Ingests and performs high-dimensional semantic search across multi-format documents (PDF, Markdown, Code) with cosine similarity ranking.
3. **Critique & Grounding Grader Agent**: Assesses whether candidate retrieved passages factually answer the inquiry. If score is below threshold, it triggers a cyclical re-query loop.
4. **Answer Synthesizer Agent**: Synthesizes verified responses with precise source citations (`[Source: file.md#chunk]`).

---

## 🚀 Quickstart & Execution

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
cd backend
python -m unittest tests/test_novomind.py
```

### 3. Launch the Backend Server
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Interactive Frontend UI
Open `frontend/index.html` in your web browser or run:
```bash
python -m http.server 3000 --directory frontend
```

---

## 📂 Project Structure
```
novomind-rag/
├── backend/
│   ├── agents/
│   │   ├── query_decomposer.py     # Multi-hop task decomposition
│   │   ├── retriever_agent.py      # Semantic vector search
│   │   ├── critique_grader.py      # Grounding and hallucination verification
│   │   └── synthesizer_agent.py    # Answer synthesis & inline citations
│   ├── core/
│   │   ├── config.py               # Environment & system configuration
│   │   └── state.py                # Graph state definitions
│   ├── graph/
│   │   └── workflow.py             # Cyclical agent graph coordinator
│   ├── vector_store/
│   │   ├── embedder.py             # Semantic embedding generator
│   │   └── indexer.py              # Text chunking & document indexer
│   ├── sample_data/                # Sample repository documents
│   ├── tests/
│   │   └── test_novomind.py        # Automated test suite
│   ├── main.py                     # FastAPI SSE streaming server
│   └── requirements.txt
├── frontend/
│   └── index.html                  # Interactive React/Tailwind Web UI
└── README.md
```

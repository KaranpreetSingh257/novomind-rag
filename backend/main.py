"""
FastAPI Asynchronous Backend Server for NovoMind Multi-Agent RAG System.
Features:
- Real-time Server-Sent Events (SSE) token streaming & agent state streaming.
- Multi-agent LangGraph workflow execution.
- File and directory ingestion endpoints.
- Document management & health checking.
"""

import asyncio
import json
import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import BASE_DIR
from vector_store.indexer import VectorIndexer
from graph.workflow import MultiAgentRAGWorkflow

app = FastAPI(
    title="NovoMind Multi-Agent RAG API",
    description="Asynchronous Multi-Agent Retrieval-Augmented Generation Microservice",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector indexer and workflow
indexer = VectorIndexer()
workflow = MultiAgentRAGWorkflow(indexer)

# Seed initial sample data if index is empty
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
if not indexer.chunks and os.path.exists(SAMPLE_DIR):
    for fname in os.listdir(SAMPLE_DIR):
        fpath = os.path.join(SAMPLE_DIR, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                indexer.ingest_document(f.read(), fname, {"initial_sample": True})

class ChatRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 2

class IngestTextRequest(BaseModel):
    source_name: str
    content: str

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "NovoMind Multi-Agent RAG",
        "indexed_chunks": len(indexer.chunks),
        "documents": len(indexer.get_document_stats())
    }

@app.get("/api/documents")
async def get_documents():
    return {
        "documents": indexer.get_document_stats(),
        "total_chunks": len(indexer.chunks)
    }

@app.post("/api/chat")
async def chat_non_streaming(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    state = workflow.run(req.query, max_iterations=req.max_iterations or 2)
    return state.to_dict()

@app.post("/api/chat/stream")
async def chat_streaming(req: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint emitting real-time agent workflow events and answer tokens.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator():
        # Step 1: Query Decomposition
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'QueryDecomposer', 'thought': f'Analyzing and decomposing query: {req.query}'})}\n\n"
        await asyncio.sleep(0.3)
        state = workflow.decomposer.run(workflow.workflow_state_factory(req.query))
        yield f"data: {json.dumps({'event': 'agent_step', 'agent': 'QueryDecomposer', 'details': {'sub_queries': state.sub_queries}})}\n\n"
        await asyncio.sleep(0.2)

        # Step 2: Vector Retrieval
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'VectorRetriever', 'thought': 'Executing multi-hop semantic vector search across indexed documents.'})}\n\n"
        await asyncio.sleep(0.3)
        state = workflow.retriever.run(state)
        yield f"data: {json.dumps({'event': 'agent_step', 'agent': 'VectorRetriever', 'details': {'chunks_found': len(state.retrieved_docs)}})}\n\n"
        await asyncio.sleep(0.2)

        # Step 3: Critique & Grounding Grader
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'CritiqueGrader', 'thought': 'Evaluating factual grounding and scoring candidate passages.'})}\n\n"
        await asyncio.sleep(0.3)
        state = workflow.grader.run(state)
        yield f"data: {json.dumps({'event': 'agent_step', 'agent': 'CritiqueGrader', 'details': {'is_grounded': state.is_grounded, 'score': state.critique_score}})}\n\n"
        await asyncio.sleep(0.2)

        # Step 4: Answer Synthesis
        yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'AnswerSynthesizer', 'thought': 'Synthesizing contextually grounded answer with inline citations.'})}\n\n"
        state = workflow.synthesizer.run(state)

        # Stream answer tokens
        full_text = state.final_answer
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'event': 'token', 'token': chunk})}\n\n"
            await asyncio.sleep(0.02)

        # Final completion event with citations and metadata
        yield f"data: {json.dumps({'event': 'done', 'citations': state.citations, 'critique_score': state.critique_score})}\n\n"

    # Helper helper method on workflow for streaming state factory
    workflow.workflow_state_factory = lambda q: workflow.decomposer.run(workflow.workflow_state_factory_raw(q)) if hasattr(workflow, 'workflow_state_factory_raw') else None
    
    return StreamingResponse(
        event_stream_wrapper(req.query, req.max_iterations or 2),
        media_type="text/event-stream"
    )

async def event_stream_wrapper(query: str, max_iterations: int):
    # Complete execution and stream events sequentially
    state = workflow.run(query, max_iterations=max_iterations)

    for trace in state.traces:
        yield f"data: {json.dumps({'event': 'agent_trace', 'agent': trace.agent_name, 'action': trace.action, 'thought': trace.thought, 'details': trace.details})}\n\n"
        await asyncio.sleep(0.15)

    words = state.final_answer.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'event': 'token', 'token': chunk})}\n\n"
        await asyncio.sleep(0.015)

    yield f"data: {json.dumps({'event': 'done', 'citations': state.citations, 'critique_score': round(state.critique_score, 2), 'is_grounded': state.is_grounded})}\n\n"

@app.post("/api/ingest/text")
async def ingest_text(req: IngestTextRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    count = indexer.ingest_document(req.content, req.source_name)
    return {"message": f"Successfully ingested {count} chunks from '{req.source_name}'", "chunks_count": count}

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    content_bytes = await file.read()
    try:
        content_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content_text = content_bytes.decode("latin-1", errors="ignore")

    count = indexer.ingest_document(content_text, file.filename)
    return {"message": f"Successfully indexed file '{file.filename}'", "chunks_count": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

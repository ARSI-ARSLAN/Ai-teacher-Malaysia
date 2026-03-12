"""
api.py - FastAPI application for AI Teacher Malaysia
Endpoints:
  POST /api/chat    - Student chat with streaming SSE response
  POST /api/ingest  - Trigger PDF ingestion (admin protected)
  GET  /api/health  - Health check
  GET  /            - Serve frontend
"""

import logging
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.config import config
from backend.rag_chain import query as rag_query, get_embed_model
from backend.vector_store import init_collection, get_collection_stats

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Teacher Malaysia",
    description="RAG-based AI Math Tutor for Malaysian KSSM Syllabus",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ────────────────────────────────────────────────────────────
_collection = None


def get_collection():
    """Get or initialize the Zilliz collection."""
    global _collection
    if _collection is None:
        logger.info("🔌 Initializing Zilliz collection...")
        _collection = init_collection()
    return _collection


# ─── Request / Response Models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    language: str = "auto"  # "en", "bm", or "auto"
    form_filter: Optional[str] = None  # "T1" .. "T5"


class IngestRequest(BaseModel):
    admin_key: str
    dry_run: bool = False
    pdf_dir: Optional[str] = None


# ─── Startup Event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 AI Teacher Malaysia starting up...")
    try:
        # Pre-warm embedding model in background
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, get_embed_model)
        # Initialize Zilliz
        get_collection()
        logger.info("✅ Startup complete.")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        logger.warning("⚠️  Make sure .env is configured with ZILLIZ and GROK credentials.")


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    try:
        collection = get_collection()
        stats = get_collection_stats(collection)
        return {
            "status": "ok",
            "collection": config.COLLECTION_NAME,
            "total_chunks": stats["num_entities"],
            "grok_model": config.GROK_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


# ─── Chat Endpoint (Streaming SSE) ───────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if len(request.question) > 2000:
        raise HTTPException(
            status_code=400, detail="Question too long (max 2000 chars)."
        )

    async def event_stream():
        try:
            collection = get_collection()
            loop = asyncio.get_event_loop()

            # Run synchronous generator in thread pool
            def run_rag():
                return list(
                    rag_query(
                        question=request.question,
                        collection=collection,
                        language=request.language,
                        form_filter=request.form_filter,
                    )
                )

            tokens = await loop.run_in_executor(None, run_rag)

            for token in tokens:
                # SSE format: "data: <content>\n\n"
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Ingestion Endpoint (Admin Protected) ────────────────────────────────────
@app.post("/api/ingest")
async def trigger_ingestion(request: IngestRequest):
    # Simple admin key check (set ADMIN_KEY in .env for production)
    admin_key = os.getenv("ADMIN_KEY", "kssm-admin-2024")
    if request.admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    async def ingestion_stream():
        try:
            from backend.ingest import run_ingestion
            from backend.vector_store import init_collection

            collection = init_collection() if not request.dry_run else None
            stats = run_ingestion(
                pdf_dir=request.pdf_dir or config.PDF_DIR,
                collection=collection,
                dry_run=request.dry_run,
            )
            yield f"data: {json.dumps({'status': 'complete', 'stats': stats})}\n\n"
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            yield f"data: {json.dumps({'status': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(ingestion_stream(), media_type="text/event-stream")


# ─── Serve Frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from indexer import Indexer, IndexerMetrics
from fastapi.encoders import jsonable_encoder
from query_engine import QueryEngine
import traceback

app = FastAPI()

@app.get("/api/v1/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

class IndexRequest(BaseModel):
    directory: str
    file_extensions: Optional[List[str]] = None  # Example: [".txt", ".md"]


@app.post("/api/v1/index")
def index_directory(request: IndexRequest):
    directory = request.directory
    file_extensions = request.file_extensions
    try:
        indexer = Indexer()
        metrics: IndexerMetrics = indexer.index_directory(directory, file_extensions=file_extensions)
        return JSONResponse(content=jsonable_encoder(metrics))
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/query")
def query_endpoint(request: QueryRequest):
    try:
        engine = QueryEngine()
        formatted = engine.query(request.query, n_results=5)
        return JSONResponse(content={"results": formatted})
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

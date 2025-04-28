from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/api/v1/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

from typing import List, Optional

class IndexRequest(BaseModel):
    directory: str
    file_extensions: Optional[List[str]] = None  # Example: [".txt", ".md"]

from indexer import Indexer, IndexerMetrics
from fastapi.encoders import jsonable_encoder

@app.post("/api/v1/index")
def index_directory(request: IndexRequest):
    directory = request.directory
    file_extensions = request.file_extensions
    try:
        indexer = Indexer()
        metrics: IndexerMetrics = indexer.index_directory(directory, file_extensions=file_extensions)
        metrics_dict = jsonable_encoder(metrics)
        return JSONResponse(content=metrics_dict)
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

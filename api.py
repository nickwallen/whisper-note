from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from indexer import Indexer, IndexerMetrics
from openrouter import OpenRouterLangModel
from query import ContextChunk, QueryEngine
from vector_store import VectorStore
import traceback
import logging
import dotenv

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
for mod in [
    "api",
    "chunker",
    "embeddings",
    "indexer",
    "ollama",
    "openrouter",
    "query",
    "time_range",
    "vector_store",
]:
    logging.getLogger(mod).setLevel(logging.DEBUG)

app = FastAPI()


class IndexMetricsResponse(BaseModel):
    file_count: int
    chunk_count: int
    failed_files: List[dict] = []


@app.get("/api/v1/health")
def health_check():
    return JSONResponse(content={"status": "ok"})


class IndexRequest(BaseModel):
    directory: str
    file_extensions: Optional[List[str]] = None  # Example: [".txt", ".md"]


def get_collection_name():
    return "notes"  # Default collection for production


@app.post("/api/v1/index", response_model=IndexMetricsResponse)
def index_directory(
    request: IndexRequest,
    collection_name: str = Depends(get_collection_name),
):
    directory = request.directory
    file_extensions = request.file_extensions
    try:
        indexer = Indexer(vector_store=VectorStore(collection_name=collection_name))
        metrics = indexer.index_dir(directory, file_exts=file_extensions)
        response = IndexMetricsResponse(
            file_count=metrics.file_count,
            chunk_count=metrics.chunk_count,
            failed_files=metrics.failed_files,
        )
        return response
    except Exception as e:
        logging.getLogger(__name__).error(
            f"500 Internal Server Error: {e}\n{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


class QueryRequest(BaseModel):
    query: str


@app.get("/api/v1/index", response_model=IndexMetricsResponse)
def get_index(collection_name: str = Depends(get_collection_name)):
    """Return current index metrics (files, chunks, failed files)."""
    try:
        indexer = Indexer(vector_store=VectorStore(collection_name=collection_name))
        metadata = indexer.vector_store.get_all_metadata()
        metrics = build_indexer_metrics_from_metadata(metadata)
        return metrics
    except Exception as e:
        logging.getLogger(__name__).error(
            f"500 Internal Server Error: {e}\n{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


class QueryResponse(BaseModel):
    answer: str
    context: List[ContextChunk]


@app.post("/api/v1/query", response_model=QueryResponse)
def query(request: QueryRequest, collection_name: str = Depends(get_collection_name)):
    try:
        engine = QueryEngine(
            vector_store=VectorStore(collection_name=collection_name),
            lang_model=OpenRouterLangModel(),
        )
        result = engine.query(request.query)
        resp = QueryResponse(answer=result.answer, context=result.context)
        return resp
    except Exception as e:
        logging.getLogger(__name__).error(
            f"500 Internal Server Error: {e}\n{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


def build_indexer_metrics_from_metadata(metadata_list):
    file_set = set()
    chunk_count = 0
    for metadata in metadata_list:
        if metadata.file:
            file_set.add(metadata.file)
            chunk_count += 1
    return IndexerMetrics(
        file_count=len(file_set), chunk_count=chunk_count, failed_files=[]
    )

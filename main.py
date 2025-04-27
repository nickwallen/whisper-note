from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/api/v1/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

class IndexRequest(BaseModel):
    directory: str

@app.post("/api/v1/index")
def index_directory(request: IndexRequest):
    directory = request.directory
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return {"message": f"Found {count} file(s) to index."}

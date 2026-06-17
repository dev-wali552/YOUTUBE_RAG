from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from ingest import ingest_channel
from graph import graph
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://strong-meerkat-a392da.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class IngestRequest(BaseModel):
    channel_url: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Youtube RAG is running"}

@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_channel, request.channel_url)
    return {"message": "Ingestion started — this takes a few minutes for large channels"}

@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=request.message)]
    }, config=config)
    return {"response": result["messages"][-1].content}
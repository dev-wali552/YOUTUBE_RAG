from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from ingest import ingest_channel
from graph import graph

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
class Ingest_Request(BaseModel):
    channel_url: str

@app.get("/")
def root():
    return {"message": "Youtube RAG is running"}

@app.post("/ingest")
async def ingest(request: Ingest_Request):
    message = ingest_channel(request.channel_url)
    return {"messages": message}



@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=request.message)]
    }, config=config)
    response = result["messages"][-1].content
    return {"response": response}
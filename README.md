# ARIA — AI Research Intelligence Assistant

> **Chat with any YouTube channel. Ask questions. Get answers. No scrubbing through hours of content.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-ARIA-blueviolet?style=for-the-badge&logo=netlify)](https://strong-meerkat-a392da.netlify.app)
[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://youtube-rag-j76u.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11.8-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20RAG-FF6B6B?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

---

## What is ARIA?

ARIA lets you drop a YouTube channel URL and instantly turn **hours of video content** into a conversational knowledge base. No timestamps. No scrubbing. Just ask.

Point it at a tech channel and ask *"What's their take on AI agents?"* — ARIA retrieves relevant transcript chunks across every video, synthesizes an answer, and cites the source videos. It's RAG, but for YouTube.

---

## Demo

![ARIA Demo](https://strong-meerkat-a392da.netlify.app)

**Try it live → [strong-meerkat-a392da.netlify.app](https://strong-meerkat-a392da.netlify.app)**

Paste any YouTube channel URL → wait for ingestion → start asking questions.

---

## Architecture

```
User Input (Channel URL)
        │
        ▼
┌──────────────────┐
│   FastAPI        │  ◄── /ingest  /chat
│   (Render)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│           LangGraph Agent                │
│                                          │
│  ┌─────────────┐    ┌─────────────────┐  │
│  │  Ingestion  │    │  RAG Retrieval  │  │
│  │   Pipeline  │    │     Pipeline    │  │
│  └──────┬──────┘    └────────┬────────┘  │
│         │                    │           │
└─────────┼────────────────────┼───────────┘
          │                    │
          ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  YouTube Data    │  │    ChromaDB      │
│  API v3          │  │  (Vector Store)  │
│  + Transcript    │  │                  │
│  API + Webshare  │  │  FastEmbed       │
│  Proxies         │  │  BAAI/bge-small  │
└──────────────────┘  └──────────────────┘
          │
          ▼
   Video IDs → Transcripts → Chunks → Embeddings → Chroma
```

**Request flow at query time:**

```
User question
    → Embed query (FastEmbed)
    → Similarity search (ChromaDB)
    → Top-k chunks retrieved
    → Groq LLM (llama-3.3-70b-versatile) synthesizes answer
    → Response streamed back
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Framework** | LangGraph |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Embeddings** | Cohere — `embed-english-v3.0` |
| **Vector Store** | ChromaDB (in-memory on Render) |
| **Transcript Fetching** | `youtube-transcript-api` |
| **Proxy Layer** | Webshare residential proxies (bypasses Render IP bans) |
| **YouTube Metadata** | YouTube Data API v3 |
| **Backend** | FastAPI |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Deployment** | Render (backend) + Netlify (frontend) |

---

## Key Engineering Decisions

### 1. Webshare Proxies — Bypassing YouTube's Cloud IP Ban
YouTube blocks transcript API calls from cloud provider IPs (Render, AWS, GCP — all flagged). Cookie-based auth wasn't enough. Solution: route all transcript requests through Webshare's **residential proxy pool**, making requests look like they're coming from real home IPs.

```python
ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=os.getenv("WEBSHARE_USERNAME"),
        proxy_password=os.getenv("WEBSHARE_PASSWORD"),
    )
)
```

### 2. LangGraph for Stateful Ingestion + Retrieval
Both pipelines (ingest and chat) are modeled as LangGraph graphs with typed state. This gives clean node separation, easy debugging, and the ability to extend later (re-ranking node, source citation node, multi-channel support).

---

## Project Structure

```
aria/
├── api.py              # FastAPI app — /ingest and /chat endpoints
├── graph.py            # LangGraph graph definitions (ingest + RAG)
├── state.py            # TypedDict state schemas
├── ingest.py           # YouTube fetch → transcript → chunk → embed → store
├── rag.py              # Query → retrieve → LLM → response
├── requirements.txt
├── runtime.txt         # Python 3.11.8
└── index.html          # Frontend (deployed separately on Netlify)
```

---

## Running Locally

### Prerequisites
- Python 3.11.8
- A [Groq API key](https://console.groq.com)
- A [YouTube Data API v3 key](https://console.cloud.google.com)
- A [Webshare](https://webshare.io) account (free tier works)

### Setup

```bash
git clone https://github.com/dev-wali/aria-youtube-rag
cd aria-youtube-rag
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
YOUTUBE_API_KEY=your_yt_key
COHERE_API_KEY=your_cohere_key
WEBSHARE_USERNAME=your_webshare_username
WEBSHARE_PASSWORD=your_webshare_password
```

Run the backend:

```bash
uvicorn api:app --reload
```

Open `index.html` in your browser or serve it locally.

---

## API Reference

### `POST /ingest`
Ingests a YouTube channel — fetches up to 15 video transcripts, chunks them, embeds and stores in ChromaDB.

```json
{
    "channel_url": "https://www.youtube.com/@channelname" 
}
```

**Response:**
```json
{
  "message": "Successfully ingested your YouTube channel"
}
```

### `POST /chat`
Ask a question against the ingested channel's content.

```json
{
  "message": "What does this channel say about LLM fine-tuning?"
}
```

**Response:**
```json
{
  "response": "Based on the channel's content, ..."
}
```

---

## Deployment

| Service | Purpose | Config |
|---------|---------|--------|
| **Render** | FastAPI backend | Free tier, `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| **Netlify** | Static frontend | Drop `index.html`, done |

**Required Render environment variables:**
```
GROQ_API_KEY
YOUTUBE_API_KEY
WEBSHARE_USERNAME
WEBSHARE_PASSWORD
```

---

## Limitations

- **15 videos per channel** — YouTube Data API quota is the bottleneck. Expandable with a paid quota increase.
- **ChromaDB resets on cold start** — Render's free tier has no persistent disk. Re-ingest after backend sleeps. Fixable with Render paid tier or external vector DB (Pinecone, Qdrant Cloud).
- **English transcripts only** — `youtube-transcript-api` fetches auto-generated captions; accuracy varies by creator.

---

## What's Next

- [ ] Persistent vector storage (Qdrant Cloud)
- [ ] Multi-channel ingestion
- [ ] Source citation with video timestamps
- [ ] Streaming responses
- [ ] Channel comparison mode ("What does X say vs Y say about topic Z?")

---

## Built By

**Wali Khan** — First-year CSAM @ IIIT Delhi. Building agentic AI systems.

[![GitHub](https://img.shields.io/badge/GitHub-dev--wali-181717?style=flat-square&logo=github)](https://github.com/dev-wali)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-wali--khan-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/wali-khan-42b656409)

---

*Part of a portfolio of production-grade agentic AI systems. Other projects: Flight Agent, Sports Agent, NEXUS Voice Agent, NEXUS Debate.*

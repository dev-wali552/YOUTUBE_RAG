from dotenv import load_dotenv
load_dotenv()

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
import shutil
import time

youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=os.getenv("WEBSHARE_USERNAME"),
        proxy_password=os.getenv("WEBSHARE_PASSWORD"),
    )
)

embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0"
)

def channel_id(yt_url: str) -> str:
    if "/channel/" in yt_url:
        return yt_url.split("/channel/")[1].split("/")[0]
    handle = yt_url.split("@")[1].split("/")[0] if "@" in yt_url else yt_url.split("/c/")[1].split("/")[0]
    response = youtube.search().list(q=handle, type="channel", part="id", maxResults=1).execute()
    return response["items"][0]["id"]["channelId"]


def get_vid_ids(channel_id: str, max_videos: int = 15) -> list:
    vid_ids, next_page_token = [], None
    while len(vid_ids) < max_videos:
        response = youtube.search().list(
            channelId=channel_id, type="video", part="id",
            maxResults=min(50, max_videos - len(vid_ids)),
            pageToken=next_page_token
        ).execute()
        vid_ids += [item["id"]["videoId"] for item in response["items"]]
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return vid_ids[:max_videos]


def get_transcripts(vid_ids: list) -> list[Document]:
    docs = []
    for vid in vid_ids:
        try:
            transcript = ytt_api.fetch(vid)
            full_text = " ".join(entry.text for entry in transcript)
            if full_text.strip():
                docs.append(Document(page_content=full_text, metadata={"video_id": vid}))
                print(f"✓ {vid}")
        except Exception as e:
            print(f"✗ {vid}: {e}")
        time.sleep(1)
    return docs


def ingest_channel(yt_url: str) -> str:

    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

    channel  = channel_id(yt_url)
    vid_ids  = get_vid_ids(channel)
    docs     = get_transcripts(vid_ids)

    print(f"Channel: {channel} | Videos: {len(vid_ids)} | Transcripts: {len(docs)}")

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    splits = [c for c in splits if c.page_content.strip()]

    print(f"Chunks: {len(splits)}")

    if not splits:
        return "No transcripts found for this channel"

    Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    return "Successfully ingested your YouTube channel"
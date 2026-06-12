from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from googleapiclient.discovery import build 
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
import os
# 3 types of yt_url
# https://www.youtube.com/@channelname
# https://www.youtube.com/channel/UCxxxxxxxxxxxxxx
# https://www.youtube.com/c/channelname

youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

def channel_id(yt_url: str) -> str:
    if "/channel/" in yt_url:
        return yt_url.split("/channel/")[1]
    else:
        handle = yt_url.split("@")[1] if "@" in yt_url else yt_url.split("/c/")[1]
        response = youtube.search().list(
            q=handle,
            type="channel",
            part="id",
            maxResults=1
        ).execute()
        return response["items"][0]["id"]["channelId"]


    
def get_vid_ids(channel_id: str) -> list:
    vid_ids = []
    next_page_token = None

    while True:
        response = youtube.search().list(
            channelId=channel_id,
            type="video",
            part="id",
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        for item in response["items"]:
            vid_ids.append(item["id"]["videoId"])
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return vid_ids

  
def get_transcripts(vid_ids: list) -> list[Document]:
    transcriptions = []
    ytt_api = YouTubeTranscriptApi()
    
    for vids in vid_ids:
        try:
            transcript = ytt_api.fetch(vids)
            full_text = " ".join(entry.text for entry in transcript)
            if full_text.strip():
                doc = Document(
                    page_content=full_text,
                    metadata={"video_id": vids}
                )
            
                transcriptions.append(doc)
        except Exception:
            continue  # skip videos with no transcript
        
    
    return transcriptions

def ingest_channel(yt_url: str) -> str:

    channelID = channel_id(yt_url)
    vidID = get_vid_ids(channelID)
    transcripts = get_transcripts(vidID)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    splits = text_splitter.split_documents(transcripts)
    splits = [chunk for chunk in splits if chunk.page_content.strip()]

    embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0")
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")

    return "Succesfully transcribed ur youtube channel"

    




        
        




        














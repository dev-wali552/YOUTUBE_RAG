from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = CohereEmbeddings(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model="embed-english-v3.0"
)

template = """You are ARIA, a helpful AI assistant for YouTube channels.

If the user is making small talk or being conversational (greetings, thanks, etc.), respond naturally and friendly.

For content-related questions, answer ONLY using the context below. If the answer isn't in the context, say "I couldn't find that in this channel's content."

Never make up facts or numbers.

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)


def ask(question: str) -> str:
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


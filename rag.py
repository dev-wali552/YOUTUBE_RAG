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


template = """You are a helpful AI Agent.Answer the user's question using only the YouTube channel content provided in the context. If u cant find the desired answer then just say " I couldnt find the desired answer" .Never make up facts or number
  context: {context},
  question: {question}
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


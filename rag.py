from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_API_KEY")
)

vectorstore = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

template = """You are a helpful AI Agent.Answer the user from only the youtube link the user has attached. If u cant find the desired answer then just say " I couldnt find the desired answer" .Never make up facts or number
  context: {context},
  question: {question}
 """
prompt = ChatPromptTemplate.from_template(template)
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
def ask(query:str) ->str:
    return rag_chain.invoke(query)


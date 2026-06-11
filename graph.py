from langgraph.graph import StateGraph
from state import State
from langgraph.graph import START,END
from langgraph.checkpoint.memory import MemorySaver
from rag import ask
from langchain_core.messages import AIMessage
def rag_node(state: State):
    last_message = state["messages"][-1].content
    response = ask(last_message)
    return {"messages": [AIMessage(content=response)]}
builder = StateGraph(State)
builder.add_node("RAG",rag_node)
builder.add_edge(START,"RAG")
builder.add_edge("RAG",END)
memory = MemorySaver()
graph = builder.compile(name="Youtube_RAG",checkpointer=memory)
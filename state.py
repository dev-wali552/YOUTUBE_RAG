from typing import Sequence
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict,Annotated
class State(TypedDict):
    messages: Annotated[Sequence[AnyMessage],add_messages]
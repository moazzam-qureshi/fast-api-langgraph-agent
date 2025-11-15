import os
from typing import TypedDict, Annotated
from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model


class State(TypedDict):
  user_query: str
  messages: Annotated[list[BaseMessage], add_messages]


def llm(state:State):
    model = init_chat_model("gpt-4o-mini")
    # LangGraph automatically streams this even with .invoke()
    response = model.invoke([
        {"role": "user", "content": state["user_query"]}
    ])
    return {"messages": [HumanMessage(content=state["user_query"]), AIMessage(content=response.content)]}


s_graph = (
  StateGraph(State)
  .add_node(llm)
  .add_edge(START, "llm")
  .add_edge("llm", END)
)
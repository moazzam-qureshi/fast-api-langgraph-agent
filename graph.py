import os
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langchain.messages import AIMessage, SystemMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from uuid import UUID
import logging

logger = logging.getLogger(__name__)


class State(TypedDict):
    user_query: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: Optional[str]
    use_rag: bool
    rag_k: int
    rag_threshold: float
    retrieved_context: Optional[str]


def retrieve_context(state: State):
    """Retrieve relevant context from user's documents."""
    if not state.get("use_rag", True) or not state.get("user_id"):
        return {"retrieved_context": None}
    
    try:
        # Import here to avoid circular imports
        from database import SessionLocal
        from services.vectorstore import vector_store_service
        
        db = SessionLocal()
        try:
            user_id = UUID(state["user_id"])
            
            # Search for relevant chunks
            results = vector_store_service.search_similar_chunks(
                query=state["user_query"],
                user_id=user_id,
                k=state.get("rag_k", 5),
                score_threshold=state.get("rag_threshold", 0.7),
                db=db
            )
            if results:
                # Format context
                context = vector_store_service.format_context_for_prompt(results)
                logger.info(f"Retrieved {len(results)} chunks for context")
                return {"retrieved_context": context}
            else:
                logger.info("No relevant chunks found")
                return {"retrieved_context": None}
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return {"retrieved_context": None}


def llm(state: State):
    """Generate response using LLM with optional RAG context."""
    model = init_chat_model("gpt-4o-mini")
    
    # Build messages
    messages = []
    
    # Add system message with context if available
    if state.get("retrieved_context"):
        system_prompt = f"""You are a helpful AI assistant with access to the user's knowledge base.
Use the following context from their documents to answer their question. If the context doesn't contain 
relevant information, you can still answer based on your general knowledge, but mention that the specific 
information wasn't found in their documents.

Context from user's documents:
{state['retrieved_context']}
"""
        messages.append(SystemMessage(content=system_prompt))
    
    # Add user query
    messages.append(HumanMessage(content=state["user_query"]))
    
    # Generate response
    response = model.invoke(messages)
    
    # Return with message history
    return {
        "messages": [
            HumanMessage(content=state["user_query"]), 
            AIMessage(content=response.content)
        ]
    }


# Build the graph with conditional RAG
s_graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve_context)
    .add_node("llm", llm)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "llm")
    .add_edge("llm", END)
)
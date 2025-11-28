# -*- coding: utf-8 -*-
"""RAG Chain with Tools - Production ready for Streamlit deployment"""

import os
import json
import torch
from datetime import datetime
from typing import Optional, Dict, List, Any

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "youtube-qa-index"
TOP_K = 5

# ============================================================================
# GLOBAL STATE
# ============================================================================

_initialized = False
retriever = None
pc = None
index = None
llm = None
llm_with_tools = None
rag_agent_chain = None
memory = []  # Global conversation memory

# ============================================================================
# ENVIRONMENT SETUP (works with Streamlit secrets)
# ============================================================================

def _setup_env():
    """Load environment variables - works in Colab and Streamlit"""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent")

# ============================================================================
# RETRIEVER - CACHED FOR PERFORMANCE
# ============================================================================

@st.cache_resource
def get_retriever():
    """Load embedding model - cached for performance"""
    print("📥 Loading SentenceTransformer model (768 dims)...")
    device = "cpu"  # Force CPU for Streamlit Cloud
    
    retriever = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",  # 768 dims - MATCHES YOUR PINECONE INDEX
        device=device
    )
    print("✅ SentenceTransformer loaded (768 dims)")
    return retriever

# ============================================================================
# TOOLS
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2 * 5' or '10 / 3'"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_count(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"

@tool
def convert_case(text: str, case_type: str) -> str:
    """Convert text to uppercase, lowercase, or title case. case_type options: 'upper', 'lower', 'title'"""
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    else:
        return f"Error: Unknown case type '{case_type}'. Use 'upper', 'lower', or 'title'."

tools = [calculator, get_current_time, word_count, convert_case]

# ============================================================================
# RAG RETRIEVAL
# ============================================================================

def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict:
    """Query Pinecone with embedded version of user query"""
    if retriever is None or index is None:
        return {"matches": []}
    
    try:
        xq = retriever.encode(query).tolist()
        res = index.query(vector=xq, top_k=top_k, include_metadata=True, timeout=10)
        return res
    except Exception as e:
        print(f"⚠️ Pinecone retrieval error: {e}")
        return {"matches": []}

def context_string_from_matches(matches: List) -> str:
    """Build context string from Pinecone matches"""
    parts = []
    for m in matches:
        meta = m.get("metadata", {})
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)

# ============================================================================
# CHAIN RUNNABLES
# ============================================================================

def _build_messages(inputs: dict) -> dict:
    """STEP 1: Build messages with RAG context"""
    user_message = inputs["user_message"]
    
    # RAG: retrieve context from Pinecone
    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))
    
    # Start from global memory
    messages = list(memory)
    messages.append(HumanMessage(content=user_message))
    
    if context:
        messages.append(HumanMessage(content=f"📚 Relevant context from knowledge base:\n{context}"))
    
    return {
        "messages": messages,
        "rag_context": context,
    }

def _first_llm_call(state: dict) -> dict:
    """STEP 2: Call LLM with tools enabled (decision step)"""
    messages = state["messages"]
    first_response = llm_with_tools.invoke(messages)
    
    return {
        **state,
        "first_response": first_response,
    }

def _run_tools_if_needed(state: dict) -> dict:
    """STEP 3: Execute tools if LLM requested them"""
    first_response = state["first_response"]
    messages = state["messages"]
    
    # Extract tool calls
    tool_calls = getattr(first_response, "tool_calls", None)
    if not tool_calls and hasattr(first_response, "additional_kwargs"):
        tool_calls = first_response.additional_kwargs.get("tool_calls")
    
    # If no tool calls, propagate state forward
    if not tool_calls:
        return {
            **state,
            "messages_with_tools": messages,
        }
    
    # Execute tools
    tool_results_messages = []
    for call in tool_calls:
        tool_name = call.get("name") or call.get("function", {}).get("name")
        raw_args = (
            call.get("args")
            or call.get("arguments")
            or call.get("function", {}).get("arguments", {})
        )
        
        # Parse args if JSON string
        if isinstance(raw_args, str):
            try:
                tool_args = json.loads(raw_args)
            except Exception:
                tool_args = {}
        else:
            tool_args = raw_args or {}
        
        tool_id = call.get("id", "tool_call")
        
        # Find matching tool
        matching = [t for t in tools if t.name == tool_name]
        if not matching:
            result_text = f"Tool '{tool_name}' not found."
        else:
            try:
                result_text = matching[0].invoke(tool_args)
            except Exception as e:
                result_text = f"Error in tool '{tool_name}': {e}"
        
        tool_results_messages.append(
            ToolMessage(content=str(result_text), tool_call_id=tool_id)
        )
    
    # Build new message list with tools results
    messages_with_tools = messages + [first_response] + tool_results_messages
    
    return {
        **state,
        "messages_with_tools": messages_with_tools,
    }

def _final_llm_call(state: dict) -> dict:
    """STEP 4: Call plain LLM for final answer"""
    messages_with_tools = state["messages_with_tools"]
    final_response = llm.invoke(messages_with_tools)
    
    return {
        "final_response": final_response,
        "rag_context": state.get("rag_context", ""),
    }

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_chain():
    """Initialize all components - call once at startup"""
    global _initialized, retriever, pc, index, llm, llm_with_tools, rag_agent_chain
    
    if _initialized:
        return
    
    _setup_env()
    
    print("🔧 Initializing RAG chain...")
    
    # Get cached retriever (768 dims - matches your Pinecone index)
    retriever = get_retriever()
    
    # Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not set")
    
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME} (768 dims)")
    
    # LangChain LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"✅ Loaded {len(tools)} tools and LLM")
    
    # Build chain
    build_messages = RunnableLambda(_build_messages)
    first_llm_call = RunnableLambda(_first_llm_call)
    run_tools_if_needed = RunnableLambda(_run_tools_if_needed)
    final_llm_call = RunnableLambda(_final_llm_call)
    
    rag_agent_chain = (
        RunnableLambda(lambda user_message: {"user_message": user_message})
        | build_messages
        | first_llm_call
        | run_tools_if_needed
        | final_llm_call
    )
    print("✅ RAG chain initialized and ready")
    
    _initialized = True

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_rag_and_tools(user_message: str) -> str:
    """Main chat function - call this with user input"""
    global memory
    
    if not _initialized:
        raise RuntimeError("Chain not initialized. Call initialize_chain() first.")
    
    try:
        result = rag_agent_chain.invoke(user_message)
        final_response = result["final_response"]
        response_text = final_response.content
        
        # Update memory
        memory.append(HumanMessage(content=user_message))
        memory.append(AIMessage(content=response_text))
        
        # Keep memory from getting too long (last 20 messages)
        if len(memory) > 20:
            memory = memory[-20:]
        
        return response_text
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        raise

# -*- coding: utf-8 -*-
"""RAG Chain with Tools - Production ready for Streamlit deployment (LCEL Refactored)"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "youtube-qa-index"
TOP_K = 5
MEMORY_WINDOW_SIZE = 20
SESSION_ID_KEY = "langchain_session" 

# ============================================================================
# GLOBAL STATE
# ============================================================================

_initialized = False
retriever = None
pc = None
index = None
llm = None
llm_with_tools = None
rag_agent_chain_with_history = None
tools = [] 

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def _setup_env():
    """Load environment variables - works in Colab and Streamlit"""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent-v3") 

# ============================================================================
# RETRIEVER - CACHED FOR PERFORMANCE
# ============================================================================

@st.cache_resource
def get_retriever():
    """Load embedding model - cached for performance"""
    # FIX RETAINED: Moving torch import inside this function for stability
    import torch 
    
    print("📥 Loading SentenceTransformer model (768 dims)...")
    device = "cpu"
    
    retriever = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
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

@tool
def estimate_targets(weight_kg: float, sex: str, activity: str, goal: str) -> str:
    """
    Estimate daily calories and protein for a user.
    Args:
        weight_kg: Body weight in kilograms.
        sex: 'male' or 'female'.
        activity: 'sedentary', 'light', 'moderate', 'active'.
        goal: 'maintain', 'lose', 'gain'.
    """
    factor = {
        "sedentary": 28,
        "light": 31,
        "moderate": 34,
        "active": 37
    }.get(activity, 31)

    maintenance_cals = weight_kg * factor

    if goal == "lose":
        target_cals = maintenance_cals - 400
        goal_text = "weight loss"
    elif goal == "gain":
        target_cals = maintenance_cals + 400
        goal_text = "muscle gain"
    else:
        target_cals = maintenance_cals
        goal_text = "weight maintenance"

    protein_low = weight_kg * 1.6
    protein_high = weight_kg * 2.2

    return (
        f"Estimated daily targets for {goal_text}:\n"
        f"- Calories: {int(target_cals)} kcal per day\n"
        f"- Protein: {protein_low:.1f}–{protein_high:.1f} g per day\n"
        "These are simplified estimates and should be adjusted for age, body composition, and training volume."
    )

tools = [calculator, get_current_time, word_count, convert_case, estimate_targets]


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
# LCEL RUNNABLES 
# ============================================================================

def _tool_executor(call: dict) -> ToolMessage:
    """Executes a single tool call requested by the LLM and returns the result."""
    
    tool_name = call.get("name") or call.get("function", {}).get("name")
    raw_args = call.get("args") or call.get("function", {}).get("arguments", {})
    tool_id = call.get("id", "tool_call")

    if isinstance(raw_args, str):
        try:
            tool_args = json.loads(raw_args)
        except Exception:
            tool_args = {}
    else:
        tool_args = raw_args or {}

    matching = [t for t in tools if t.name == tool_name]
    if not matching:
        result_text = f"Tool '{tool_name}' not found."
    else:
        try:
            result_text = matching[0].invoke(tool_args)
        except Exception as e:
            result_text = f"Error in tool '{tool_name}': {e}"
            
    return ToolMessage(content=str(result_text), tool_call_id=tool_id)


def _build_full_prompt_messages(inputs: dict) -> dict:
    """
    STEP 1: Combines history, RAG context, and the current user message 
    into a complete list of messages for the LLM.
    """
    user_message = inputs["user_message"]
    history = inputs.get("chat_history", []) 
    
    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))
    
    messages = []

    # 1. System Prompt
    messages.append(
        SystemMessage(
            content=(
                "You are a friendly, evidence-based personal trainer and RAG assistant. "
                "Your goals are to: (1) give safe, practical fitness advice; "
                "(2) tailor suggestions to the user's level and goals; "
                "(3) clearly explain reasoning in simple language.\n\n"
                "Always use the retrieved knowledge base context when it is relevant.\n\n"
                "Tool usage rules:\n"
                "- If the user asks for general arithmetic or numeric computations (e.g. 75 * 22, percentages), "
                "you MUST call the `calculator` tool.\n"
                "- If the user asks for word counts, you MUST call the `word_count` tool.\n"
                "- If the user asks for case conversion, you MUST call the `convert_case` tool.\n"
                "- If the user asks for the current time or date, you MUST call the `get_current_time` tool.\n"
                "- If the user asks for calorie or protein targets, daily macro targets, or bodyweight-based "
                "nutrition targets, you MUST call ONLY the `estimate_targets` tool and NOT the `calculator` tool.\n\n"
                "When you use any tool, explicitly mention in your explanation that you used that tool, and base "
                "your answer directly on the tool's output instead of estimating."
            )
        )
    )

    # 2. Conversation History
    messages.extend(history)
    
    # 3. Current User Message
    messages.append(HumanMessage(content=user_message))
    
    # 4. Optional RAG Context Message 
    if context:
        messages.append(
            HumanMessage(
                content=f"📚 Relevant context from knowledge base:\n{context}"
            )
        )
    
    return {
        "messages": messages,
        "rag_context": context,
        "original_user_message": user_message 
    }

# ============================================================================
# MEMORY INTEGRATION
# ============================================================================

def _get_session_history(session_id: str) -> ConversationBufferWindowMemory:
    """Retrieve or create a memory object isolated by Streamlit session."""
    session_key = f"{SESSION_ID_KEY}_{session_id}" 
    if session_key not in st.session_state:
        st.session_state[session_key] = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            return_messages=True,
            input_key="user_message",    
            output_key="final_response", 
            memory_key="chat_history"    
        )
    return st.session_state[session_key]


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_chain():
    """Initialize all components and build the final chain with LCEL."""
    global _initialized, retriever, pc, index, llm, llm_with_tools, rag_agent_chain_with_history
    
    if _initialized:
        return
    
    _setup_env()
    print("🔧 Initializing RAG chain...")
    
    retriever = get_retriever()
    
    # Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("⚠️ PINECONE_API_KEY not set. RAG will likely fail.") 

    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME} (768 dims)")
    
    # LangChain LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"✅ Loaded {len(tools)} tools and LLM")
    
    # --- BUILD THE LCEL CHAIN ---
    
    prompt_builder = RunnableLambda(_build_full_prompt_messages)

    messages_extractor = RunnablePassthrough.assign(
        messages=lambda x: x["messages"]
    )
    
    def _is_tool_call(response: AIMessage) -> bool:
        """Check if the LLM response is a tool call."""
        return response.tool_calls is not None and len(response.tool_calls) > 0

    tool_execution_loop = RunnableBranch(
        # CONDITION: If the LLM output is a tool call
        (_is_tool_call,
            # SEQUENCE A: Tool Execution -> Second LLM Call
            RunnableMap({
                # Execute tool, returning a ToolMessage object
                "tool_results": RunnableLambda(lambda x: [
                    _tool_executor(call) for call in x.tool_calls
                ]),
                "original_context": RunnablePassthrough()
            })
            | RunnableMap({
                # Re-build the message list for the second LLM call
                "messages": lambda x: (
                    x["original_context"]["messages"] + 
                    [x["original_context"]["llm_response"]] + 
                    x["tool_results"]
                )
            })
            | RunnableMap({
                "llm_response": llm, # Second LLM call (final answer)
                "final_response": lambda x: x["llm_response"]
            })
        ),
        # DEFAULT/ELSE: No tool call, the first LLM response is the final answer
        RunnableMap({
            "llm_response": RunnablePassthrough(), 
            "final_response": lambda x: x["llm_response"]
        })
    )
    
    # 3. Main LCEL Chain (The core logic)
    core_rag_chain = (
        prompt_builder  
        | RunnableMap({
            "llm_response": messages_extractor | llm_with_tools, 
            "messages": lambda x: x["messages"] 
        })
        | tool_execution_loop 
    )

    # 4. Wrap the core chain with history management
    rag_agent_chain_with_history = RunnableWithMessageHistory(
        core_rag_chain,
        _get_session_history,
        input_messages_key="user_message",
        output_messages_key="final_response",
        history_messages_key="chat_history",
        input_keys=["user_message"], 
        output_keys=["final_response"]
    )
    
    print("✅ RAG chain initialized and ready (LCEL + Memory)")
    
    _initialized = True

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_rag_and_tools(user_message: str) -> str:
    """
    Main chat function - invokes the chain wrapped with history.
    """
    if not _initialized:
        raise RuntimeError("Chain not initialized. Call initialize_chain() first.")
    
    try:
        result = rag_agent_chain_with_history.invoke(
            {"user_message": user_message},
            config={"configurable": {"session_id": "streamlit_user"}} 
        )
        
        response_text = result["final_response"].content
        
        return response_text
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        raise

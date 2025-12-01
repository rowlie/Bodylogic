# -*- coding: utf-8 -*-
"""RAG Chain with Tools - Streamlit-ready (LCEL + Tools + Memory)"""

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
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableMap, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# ChatMessageHistory from langchain-community for Streamlit memory
from langchain_community.chat_message_histories import ChatMessageHistory

# ============================================================================
# CONFIG
# ============================================================================
INDEX_NAME = "youtube-qa-index"
TOP_K = 5
MEMORY_WINDOW_SIZE = 20
SESSION_ID_KEY = "langchain_session"

# ============================================================================
# GLOBALS
# ============================================================================
_initialized = False
retriever = None
pc = None
index = None
llm = None
llm_with_tools = None
rag_agent_chain_with_history = None

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def _setup_env():
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent-v3")

# ============================================================================
# RETRIEVER
# ============================================================================
@st.cache_resource
def get_retriever():
    import torch
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    return model

# ============================================================================
# TOOLS (explicit keyword args)
# ============================================================================
@tool
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_count(text: str) -> str:
    return f"Word count: {len(text.split())}"

@tool
def convert_case(text: str, case_type: str) -> str:
    case_type = case_type.lower()
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    return f"Error: Unknown case_type '{case_type}'. Use upper/lower/title."

@tool
def estimate_targets(weight_kg: float, sex: str, activity: str, goal: str) -> str:
    factors = {"sedentary": 28, "light": 31, "moderate": 34, "active": 37}
    maintenance = weight_kg * factors.get(activity.lower(), 31)

    if goal.lower() == "lose":
        target = maintenance - 400
        goal_text = "weight loss"
    elif goal.lower() == "gain":
        target = maintenance + 400
        goal_text = "muscle gain"
    else:
        target = maintenance
        goal_text = "maintenance"

    protein_low = weight_kg * 1.6
    protein_high = weight_kg * 2.2

    return (
        f"Estimated daily targets for {goal_text}:\n"
        f"- Calories: {int(target)} kcal/day\n"
        f"- Protein: {protein_low:.1f}-{protein_high:.1f} g/day"
    )

tools = [calculator, get_current_time, word_count, convert_case, estimate_targets]

# ============================================================================
# PINECONE RAG RETRIEVAL
# ============================================================================
def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    global retriever, index
    if retriever is None or index is None:
        return {"matches": []}
    try:
        vector = retriever.encode(query).tolist()
        return index.query(vector=vector, top_k=top_k, include_metadata=True, timeout=10)
    except Exception as e:
        print(f"⚠️ Pinecone retrieval error: {e}")
        return {"matches": []}

def context_string_from_matches(matches: List[dict]) -> str:
    parts: List[str] = []
    for m in matches:
        meta = m.get("metadata", {}) or {}
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)

# ============================================================================
# TOOL EXECUTOR
# ============================================================================
def _tool_executor(call: dict) -> ToolMessage:
    tool_name = call.get("name") or (call.get("function") or {}).get("name")
    raw_args = call.get("args") or (call.get("function") or {}).get("arguments") or {}

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    match = [t for t in tools if getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name]

    if not match:
        result = f"Tool '{tool_name}' not found."
    else:
        try:
            result = match[0].invoke(**args)
        except Exception as e:
            result = f"Tool error: {e}"

    return ToolMessage(content=str(result), tool_call_id=call.get("id", "tool_call"))

# ============================================================================
# PROMPT BUILDER
# ============================================================================
def _build_full_prompt_messages(inputs: dict) -> List[Any]:
    user_message = inputs["user_message"]
    history = inputs.get("chat_history", [])

    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))

    messages: List[Any] = [
        SystemMessage(
            content=(
                "You are a friendly, evidence-based personal trainer and RAG assistant. "
                "Give safe, practical recommendations and explain reasoning.\n\n"
                "Tool rules:\n"
                "- Use `calculator` for arithmetic\n"
                "- Use `word_count` for word counting\n"
                "- Use `convert_case` for case conversions\n"
                "- Use `get_current_time` for current date/time\n"
                "- Use ONLY `estimate_targets` for calorie/protein targets\n"
                "Always explicitly state when you call a tool."
            )
        )
    ]
    messages.extend(history)
    messages.append(HumanMessage(content=user_message))
    if context:
        messages.append(HumanMessage(content=f"📚 Relevant context:\n{context}"))
    return messages

# ============================================================================
# MEMORY
# ============================================================================
def _get_session_history(session_id: str):
    key = f"{SESSION_ID_KEY}_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = ChatMessageHistory()
    return st.session_state[key]

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_chain():
    global _initialized, retriever, pc, index, llm, llm_with_tools, rag_agent_chain_with_history
    if _initialized:
        return

    _setup_env()
    retriever = get_retriever()
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    prompt_builder = RunnableLambda(_build_full_prompt_messages)
    messages_extractor = RunnablePassthrough()

    def _is_tool_call(response: AIMessage) -> bool:
        return getattr(response, "tool_calls", None) and len(response.tool_calls) > 0

    tool_execution_loop = RunnableBranch(
        (
            _is_tool_call,
            RunnableMap({
                "tool_results": RunnableLambda(lambda x: [_tool_executor(c) for c in x.tool_calls]),
                "original_context": RunnablePassthrough(),
            })
            | RunnableMap({
                "messages": lambda x: x["original_context"] + [x["original_context"]["llm_response"]] + x["tool_results"]
            })
            | RunnableMap({
                "llm_response": llm,
                "final_response": lambda x: x["llm_response"]
            })
        ),
        RunnableMap({
            "llm_response": RunnablePassthrough(),
            "final_response": lambda x: x["llm_response"]
        })
    )

    core_rag_chain = (
        prompt_builder
        | RunnableMap({
            "llm_response": messages_extractor | llm_with_tools,
            "messages": lambda x: x["llm_response"]
        })
        | tool_execution_loop
    )

    rag_agent_chain_with_history = RunnableWithMessageHistory(
        core_rag_chain,
        _get_session_history,
        input_messages_key="user_message",
        output_messages_key="final_response",
        history_messages_key="chat_history",
        input_keys=["user_message"],
        output_keys=["final_response"],
    )

    _initialized = True
    print("✅ RAG chain initialized")

# ============================================================================
# CHAT FUNCTION
# ============================================================================
def chat_with_rag_and_tools(user_message: str) -> str:
    if not _initialized:
        raise RuntimeError("Chain not initialized.")
    try:
        result = rag_agent_chain_with_history.invoke(
            {"user_message": user_message},
            config={"configurable": {"session_id": "streamlit_user"}}
        )
        return getattr(result["final_response"], "content", str(result["final_response"]))
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return f"Error: {str(e)}"

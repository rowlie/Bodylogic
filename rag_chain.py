# -*- coding: utf-8 -*-
"""RAG Chain with Tools - Streamlit-ready, LangChain >=0.3.1"""

import os
import json
from datetime import datetime
from typing import Dict, List

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnableMap, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Correct memory import for LangChain >=0.3.1
from langchain.memory.buffer import ConversationBufferMemory

# =============================
# Configuration
# =============================
INDEX_NAME = "youtube-qa-index"
TOP_K = 5
SESSION_ID_KEY = "langchain_session"

# =============================
# Global state
# =============================
_initialized = False
retriever = None
pc = None
index = None
llm = None
llm_with_tools = None
rag_agent_chain_with_history = None
tools = []

# =============================
# Environment setup
# =============================
def _setup_env():
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent-v3")

# =============================
# Retriever
# =============================
@st.cache_resource
def get_retriever():
    import torch
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    return model

# =============================
# Tools
# =============================
@tool
def calculator(expression: str) -> str:
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_count(text: str) -> str:
    return f"Word count: {len(text.split())}"

@tool
def convert_case(text: str, case_type: str) -> str:
    case_map = {"upper": text.upper(), "lower": text.lower(), "title": text.title()}
    return case_map.get(case_type, "Error: Unknown case type. Use upper/lower/title.")

@tool
def estimate_targets(weight_kg: float, sex: str, activity: str, goal: str) -> str:
    factors = {"sedentary": 28, "light": 31, "moderate": 34, "active": 37}
    maintenance = weight_kg * factors.get(activity, 31)
    if goal == "lose":
        target = maintenance - 400
        g = "weight loss"
    elif goal == "gain":
        target = maintenance + 400
        g = "muscle gain"
    else:
        target = maintenance
        g = "maintenance"
    protein_low = weight_kg * 1.6
    protein_high = weight_kg * 2.2
    return f"Daily targets for {g}:\n- Calories: {int(target)} kcal\n- Protein: {protein_low:.1f}–{protein_high:.1f} g"

tools = [calculator, get_current_time, word_count, convert_case, estimate_targets]

# =============================
# Pinecone retrieval
# =============================
def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict:
    if retriever is None or index is None:
        return {"matches": []}
    try:
        xq = retriever.encode(query).tolist()
        return index.query(vector=xq, top_k=top_k, include_metadata=True)
    except Exception as e:
        print(f"⚠️ Pinecone error: {e}")
        return {"matches": []}

def context_string_from_matches(matches: List) -> str:
    parts = []
    for m in matches:
        meta = m.get("metadata", {})
        text = meta.get("text") or meta.get("passage_text") or ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)

# =============================
# Tool execution
# =============================
def _tool_executor(call: dict) -> ToolMessage:
    name = call.get("name") or call.get("function", {}).get("name")
    raw_args = call.get("args") or call.get("function", {}).get("arguments", {})
    tool_id = call.get("id", "tool_call")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except:
            args = {}
    else:
        args = raw_args or {}
    match = [t for t in tools if t.name == name]
    if not match:
        result = f"Tool '{name}' not found."
    else:
        try:
            result = match[0].invoke(args)
        except Exception as e:
            result = f"Tool error: {e}"
    return ToolMessage(content=str(result), tool_call_id=tool_id)

# =============================
# Prompt builder
# =============================
def _build_full_prompt_messages(inputs: dict) -> dict:
    user_msg = inputs["user_message"]
    history = inputs.get("chat_history", [])
    pinecone_res = retrieve_pinecone_context(user_msg)
    context = context_string_from_matches(pinecone_res.get("matches", []))

    msgs = [
        SystemMessage(
            content=(
                "You are a friendly, evidence-based personal trainer and RAG assistant.\n"
                "Always give safe, practical advice.\n\n"
                "Tool rules:\n"
                "- Use `calculator` for arithmetic.\n"
                "- Use `word_count` for word counting.\n"
                "- Use `convert_case` for case changes.\n"
                "- Use `get_current_time` for current date/time.\n"
                "- Use ONLY `estimate_targets` for calorie/protein targets.\n"
            )
        )
    ]
    msgs.extend(history)
    msgs.append(HumanMessage(content=user_msg))
    if context:
        msgs.append(HumanMessage(content=f"📚 Context:\n{context}"))

    return {"messages": msgs, "rag_context": context, "original_user_message": user_msg}

# =============================
# Memory
# =============================
def _get_session_history(session_id: str):
    key = f"{SESSION_ID_KEY}_{session_id}"
    if key not in st.session_state:
        st.session_state[key] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="user_message",
            output_key="final_response",
            return_messages=True
        )
    return st.session_state[key]

# =============================
# Initialization
# =============================
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
    messages_extractor = RunnablePassthrough.assign(messages=lambda x: x["messages"])

    def _is_tool_call(msg: AIMessage) -> bool:
        return msg.tool_calls is not None and len(msg.tool_calls) > 0

    tool_execution_loop = RunnableBranch(
        (
            _is_tool_call,
            RunnableMap({
                "tool_results": RunnableLambda(lambda x: [_tool_executor(c) for c in x.tool_calls]),
                "original_context": RunnablePassthrough()
            })
            | RunnableMap({"messages": lambda x: x["original_context"]["messages"] + [x["original_context"]["llm_response"]] + x["tool_results"]})
            | RunnableMap({"llm_response": llm, "final_response": lambda x: x["llm_response"]})
        ),
        RunnableMap({"llm_response": RunnablePassthrough(), "final_response": lambda x: x["llm_response"]})
    )

    core_chain = prompt_builder | RunnableMap({"llm_response": messages_extractor | llm_with_tools, "messages": lambda x: x["messages"]}) | tool_execution_loop

    rag_agent_chain_with_history = RunnableWithMessageHistory(
        core_chain,
        _get_session_history,
        input_messages_key="user_message",
        output_messages_key="final_response",
        history_messages_key="chat_history",
        input_keys=["user_message"],
        output_keys=["final_response"]
    )

    _initialized = True

# =============================
# Main chat
# =============================
def chat_with_rag_and_tools(user_message: str) -> str:
    if not _initialized:
        raise RuntimeError("Chain not initialized.")
    try:
        result = rag_agent_chain_with_history.invoke(
            {"user_message": user_message},
            config={"configurable": {"session_id": "streamlit_user"}}
        )
        return result["final_response"].content
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return f"Error: {str(e)}"

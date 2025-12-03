# -*- coding: utf-8 -*-
"""RAG Chain with Tools and AgentExecutor for Streamlit deployment"""

import os
import json
from datetime import datetime
from typing import Dict, List
import streamlit as st

# Dependency imports
# NOTE: Ensure all these packages are installed:
# pip install streamlit langchain-openai pinecone-client sentence-transformers openai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "youtube-qa-index"
TOP_K = 5
# System prompt that defines the agent's role and rules
SYSTEM_PROMPT = (
    "You are a friendly, evidence-based personal trainer and RAG assistant. "
    "Your goals are to: (1) give safe, practical fitness advice; "
    "(2) tailor suggestions to the user's level and goals; "
    "(3) clearly explain reasoning in simple language.\n\n"
    "Always use the retrieved knowledge base context when it is relevant.\n\n"
    "The user's input is prefixed with 'USER_QUERY:' and any relevant, retrieved "
    "context from the knowledge base is prefixed with 'RAG_CONTEXT:'.\n\n"
    "Tool usage rules:\n"
    "- If the user asks for general arithmetic or numeric computations (e.g. 75 * 22, percentages), "
    "you MUST call the `calculator` tool.\n"
    "- If the user asks for word counts, you MUST call the `word_count` tool.\n"
    "- If the user asks for case conversion, you MUST call the `convert_case` tool.\n"
    "- If the user asks for the current time or date, you MUST call the `get_current_time` tool.\n"
    "- If the user asks for calorie or protein targets, daily macro targets, or bodyweight-based "
    "nutrition targets, you MUST call ONLY the `estimate_targets` tool and NOT the "
    "`calculator` tool.\n\n"
    "When you use any tool, explicitly mention in your explanation that you used that tool, and base "
    "your answer directly on the tool's output instead of estimating."
)

# ============================================================================
# GLOBAL STATE (Minimized)
# ============================================================================

_initialized = False
retriever = None
pc = None
index = None
rag_agent_chain = None
# Note: The 'memory' object is now managed in st.session_state

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def _setup_env():
    """Load environment variables - works in Colab and Streamlit"""
    # NOTE: Set your environment variables (e.g., OPENAI_API_KEY, PINECONE_API_KEY)
    # The LangChain tracing variables below are optional but highly recommended.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", "agent-executor-rag-memory")

# ============================================================================
# RETRIEVER (CACHED)
# ============================================================================

@st.cache_resource
def get_retriever():
    """Load embedding model - cached for performance"""
    print("📥 Loading SentenceTransformer model (768 dims)...")
    device = "cpu"  # Force CPU for Streamlit Cloud
    
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
        # Warning: eval is generally unsafe for production, but common in agent examples.
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
    """Convert text to uppercase, lowercase, or title case. case_type options: 'upper', 'lower', or 'title'"""
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
# RAG RETRIEVAL AND PROMPT FORMATTING
# ============================================================================

def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict:
    """Query Pinecone with embedded version of user query"""
    if retriever is None or index is None:
        return {"matches": []}
    
    try:
        xq = get_retriever().encode(query).tolist() # Use get_retriever() for thread safety/caching
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

def _retrieve_and_format_context(user_message: str) -> dict:
    """
    Retrieves Pinecone context and packages it into the single 'input' key 
    required by the AgentExecutor.
    """
    # RAG: retrieve context from Pinecone
    pinecone_result = retrieve_pinecone_context(user_message)
    context = context_string_from_matches(pinecone_result.get("matches", []))
    
    # Format the full prompt to be passed to the agent
    rag_context_prefix = f"RAG_CONTEXT:\n{context}\n\n" if context else ""
    input_for_agent = f"{rag_context_prefix}USER_QUERY: {user_message}"

    # Return required inputs for the AgentExecutor
    return {
        "input": input_for_agent,  # The main input string the agent sees
        "rag_context_text": context, # Optional: For logging/debugging
    }

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_chain():
    """Initialize all components - call once at startup"""
    global _initialized, retriever, pc, index, rag_agent_chain
    
    if _initialized:
        return
    
    _setup_env()
    
    print("🔧 Initializing RAG Agent with Executor...")
    
    # 1. Get cached retriever (768 dims - matches your Pinecone index)
    retriever = get_retriever()
    
    # 2. Pinecone connection
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not set. Cannot connect to Pinecone.")
    
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME} (768 dims)")
    
    # 3. LangChain LLM and Memory
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Use st.session_state to persist memory across Streamlit reruns
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferWindowMemory(
            memory_key="chat_history", # Must match the key in the prompt template
            return_messages=True,
            k=20 # Keeps the last 20 messages
        )
    memory_object = st.session_state.agent_memory
    print("✅ Initialized ConversationBufferWindowMemory (k=20) in session_state")

    # 4. Define Agent Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"), # History inserted by memory_object
            ("human", "{input}"), # Combined RAG context and user query
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Internal agent thoughts/tool calls
        ]
    )

    # 5. Create the Agent and Agent Executor
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory_object, # The key integration point for memory
        handle_parsing_errors=True
    )
    print(f"✅ Created AgentExecutor with {len(tools)} tools and memory")

    # 6. Build the Final RAG Chain (LCEL)
    rag_agent_chain = (
        # Step 1: Takes user_message string, returns {'input': combined_prompt, ...}
        RunnableLambda(_retrieve_and_format_context)
        |
        # Step 2: Passes the dictionary output of Step 1 to the Agent Executor.
        # The Agent Executor reads the 'input' key, runs, and the result is mapped to 'final_response'.
        RunnablePassthrough.assign(
            final_response=agent_executor
        )
    )
    
    print("✅ RAG Agent chain initialized and ready")
    _initialized = True

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_rag_and_tools(user_message: str) -> str:
    """
    Main chat function - call this with user input.
    Memory update is handled automatically by the AgentExecutor.
    """
    
    if not _initialized:
        raise RuntimeError("Chain not initialized. Call initialize_chain() first.")
    
    try:
        # Invoke the chain with the raw user message
        result = rag_agent_chain.invoke(user_message)
        
        # The final answer is under the 'final_response' dictionary, which is the 
        # output of the AgentExecutor. The 'output' key contains the text.
        response_text = result["final_response"].get("output")
        
        return response_text
    except Exception as e:
        # In a real Streamlit app, this would be an st.error() call
        print(f"❌ Error in chat: {e}") 
        return "Sorry, an unexpected error occurred while processing your request."

# ============================================================================
# STREAMLIT APP EXAMPLE (Optional, for testing)
# ============================================================================

if __name__ == "__main__":
    # Example of how you would run this code in a Streamlit app
    st.set_page_config(page_title="RAG Agent with AgentExecutor")

    # Initialize components (Run once)
    try:
        initialize_chain()
    except ValueError as e:
        st.error(f"Initialization Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"A general error occurred during initialization: {e}")
        st.stop()

    st.title("🏋️ Fitness RAG Agent (AgentExecutor + Memory)")

    # Initialize chat history display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a fitness question or try a tool (e.g., 'What is 100*5 + 2?')"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Get response from the agent
            response = chat_with_rag_and_tools(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Update chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

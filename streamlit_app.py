import streamlit as st
import os
from datetime import datetime

# Import your fully working RAG chain
from rag_chain import initialize_chain, chat_with_rag_and_tools, SESSION_ID_KEY

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="LCEL RAG Agent with Tools",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Session State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# ----------------------------
# Sidebar: Configuration
# ----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")

    if all([api_key, pinecone_key, langsmith_key]):
        st.success("✅ All API keys configured")
        st.info("LangSmith tracing enabled")
    else:
        st.error("⚠️ Missing environment variables. Check secrets.")
        st.stop()

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        mem_key = f"{SESSION_ID_KEY}_streamlit_user"
        if mem_key in st.session_state:
            del st.session_state[mem_key]
        st.rerun()

    st.divider()
    st.caption(
        """
**Demo Prompts**

**RAG Retrieval**
- What is the most dangerous type of fat?

**Tool Use**
- I am an 80kg male, light activity, and I want to lose weight. What should my calories and protein be?

**Conversational Memory**
1. Ask the tool question above.
2. Ask: "What if I switch to active training but keep my weight loss goal?" (Agent remembers weight and sex from step 1)

**LCEL Structure**
- Uses RunnableBranch for tool logic
- RunnableWithMessageHistory for memory
        """
    )

# ----------------------------
# Main Title
# ----------------------------
st.title("🤖 Body Logic - LCEL Agent with Tools & Memory")
st.markdown(
    "Ask questions about your Fitness Goals — the agent uses RAG, tools, and conversational memory."
)

# ----------------------------
# Initialize RAG Chain
# ----------------------------
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG chain..."):
            initialize_chain()
            st.session_state.chain_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize chain: {str(e)}")
        st.stop()

# ----------------------------
# Display conversation history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat input
# ----------------------------
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Thinking..."):
                response = chat_with_rag_and_tools(prompt)
                placeholder.markdown(response)

                # Add assistant response to session
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

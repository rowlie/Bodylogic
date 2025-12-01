import streamlit as st
import os
from datetime import datetime

# Import your chain logic
from rag_chain import initialize_chain, chat_with_rag_and_tools, SESSION_ID_KEY

# Page config
st.set_page_config(
    page_title="LCEL RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for UI history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# Sidebar for configuration & settings
with st.sidebar:
    st.title("⚙️ Configuration")

    # Check if required API keys are set (uses os.getenv for keys)
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")

    if all([api_key, pinecone_key, langsmith_key]):
        st.success("✅ All API keys configured")
        st.info("✅ **LangSmith tracing is enabled!**")
    else:
        st.error("⚠️ Environment variables missing. Check secrets.")
        st.stop()
        

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        # Clear the LangChain memory object from session_state 
        if f"{SESSION_ID_KEY}_streamlit_user" in st.session_state:
            del st.session_state[f"{SESSION_ID_KEY}_streamlit_user"]
        st.rerun()

    st.divider()
    st.caption(
    """
**Demo Prompts**

**RAG Retrieval**
- What is the most dangerous type of fat?

**Tool Use**
- I am a 80kg male, light activity, and I want to lose weight. What should my calories and protein be? (Uses `estimate_targets` tool)

**Conversational Memory**
1. Ask the tool question above.
2. Ask: "What if I switch to active training but keep my weight loss goal?" (Agent remembers weight and sex from step 1)

**LCEL Structure**
- The app uses `RunnableBranch` for tool logic and `RunnableWithMessageHistory` for memory.
    """
)


# Main title
st.title("🤖 Body Logic - LCEL Agent with Tools & Memory")
st.markdown(
    "Ask questions about your Fitness Goals - The agent will use RAG, tools, and conversational memory."
)

# Initialize chain once
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG chain..."):
            initialize_chain()
            st.session_state.chain_initialized = True
    except Exception as e:
        st.error(f"❌ Failed to initialize chain: {str(e)}")
        st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to UI history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Thinking... (A new trace is being sent to LangSmith!)"):
                # Call your RAG chain (it now handles memory internally)
                response = chat_with_rag_and_tools(prompt)

                # Display response
                message_placeholder.markdown(response)

                # Add to UI history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )

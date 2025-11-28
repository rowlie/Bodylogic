import streamlit as st
import os
from datetime import datetime

# Import your chain logic
from rag_chain import initialize_chain, chat_with_rag_and_tools

# Page config
st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# Sidebar for configuration & settings
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Check if required API keys are set
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    
    if not all([api_key, pinecone_key, langsmith_key]):
        st.warning("⚠️ Missing environment variables!")
        st.info("""
        Please set these in your Streamlit Cloud secrets:
        - OPENAI_API_KEY
        - PINECONE_API_KEY
        - LANGCHAIN_API_KEY (LangSmith)
        """)
    else:
        st.success("✅ All API keys configured")
    
    st.divider()
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chain_initialized = False
        st.rerun()
    
    st.divider()
    st.caption("RAG Agent with Tool Use • Powered by LangChain + Streamlit")

# Main title
st.title("🤖 RAG Agent with Tools")
st.markdown("Ask questions about your knowledge base. I can use tools and retrieve relevant context.")

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
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                # Call your RAG chain
                response = chat_with_rag_and_tools(prompt)
                
                # Display response
                message_placeholder.markdown(response)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Footer
st.divider()
st.caption("💡 Tip: This agent can use tools like calculator, time, word count, and case conversion.")

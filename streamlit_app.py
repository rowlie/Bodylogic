import streamlit as st
import os
from datetime import datetime

# Import your chain logic
from rag_chain import initialize_chain, chat_with_rag_and_tools

# Page config
st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
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
        st.info(
            """
            Please set these in your Streamlit Cloud secrets:
            - OPENAI_API_KEY
            - PINECONE_API_KEY
            - LANGCHAIN_API_KEY (LangSmith)
            """
        )
    else:
        st.success("✅ All API keys configured")

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chain_initialized = False
        st.rerun()

    st.divider()
    st.caption(
    """
**Demo Prompts**

**Pinecone Database**
- What is the most dangerous type of fat?
- What are the best exercises for your body type?
- What does alcohol do to your brain?

**Tools**
- I am a 50 year old male 80kg what should be my calories targets?

**Conversational Memory**
1. I am a 75 kg male, office job, training 3 times per week. I want to lose a bit of fat but keep my strength. How would you structure my training and nutrition?
   - Calls the estimator tool.
2. Based on what I told you earlier about my weight, job, and training schedule, adjust your plan if I can only train twice per week now. Please remind me what targets you gave me before and how they change.
   - In the second reply, the agent correctly recalls that you are a 75 kg office worker and that the original plan assumed 3 sessions per week, then explicitly adjusts to “training twice per week” without you restating those details

    
    """
)


# Main title
st.title("🤖 Welcome to Body Logic ")
st.markdown(
    "Ask questions about your Fitness Goals - I can use tools and conversational memory and retrieve relevant content from hand curated youtube videos."
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
    # Add user message to history
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
            with st.spinner("Thinking..."):
                # Call your RAG chain
                response = chat_with_rag_and_tools(prompt)

                # Display response
                message_placeholder.markdown(response)

                # Add to history
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

# Footer
st.divider()
st.caption(
    "💡 Tip: This agent can use a calorie estimator, a current time tool, and has conversational memory."
)

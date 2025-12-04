# V1 BodyLogic — RAG Agent with Tools and Memory

BodyLogic is a Streamlit app that delivers evidence-based fitness guidance using a Retrieval-Augmented Generation (RAG) pipeline, tool-calling, and lightweight conversational memory. It indexes curated YouTube transcripts into Pinecone with all-mpnet-base-v2 embeddings, then answers user questions with OpenAI's GPT via LangChain Runnables. 

The streamlit app uses - rag_chain.py, requirements.txt and streamlit_app.py

## Features

- RAG on curated YouTube content using Pinecone (768-dim embeddings via all-mpnet-base-v2). 
- Tool-aware agent: calorie/protein estimator with strict usage rules. 
- Conversational memory: sliding window of the last 20 messages to preserve context. 
- Streamlit UI with demo prompts and LangSmith tracing enabled for observability. 

## Architecture

- **Ingestion**: YouTubeTranscriptApi → chunking (500 characters, 50 overlap) → HuggingFace embeddings → Pinecone upsert.
- This is done with the videotranscripts.py file 
- **Serving chain** (LangChain RunnableSequence):
  - Retriever step: encodes user query and fetches top-k (5) matches from Pinecone. 
  - Prompt builder: system message + memory + optional knowledge base context message. 
  - First LLM call with bound tools decides whether to call tools. 
  - Tool execution and final LLM response. 
- **Memory**: global Python list capped at 20 messages. 

## Repo Structure

- `rag_chain.py` — Core chain: initialization, retriever, tools, runnable steps, and chat function. 
- `app.py` — Streamlit UI, sidebar checks, demo prompts, and chat loop. 
- `video transcripts.py` — Script/notebook to fetch transcript, chunk, embed, and index into Pinecone. 

## Requirements

- Python 3.10+
- Recommended packages: `streamlit`, `langchain`, `langchain-openai`, `langchain-core`, `langchain-text-splitters`, `sentence-transformers`, `pinecone-client`, `youtube-transcript-api`, `huggingface-hub`, `langchain-huggingface`, `tqdm`, `python-dotenv`.

## Environment Variables

Set via Streamlit secrets


# V2 Documentation for the V2 BodyLogic.py file

This script is a Python CLI chatbot that combines LangChain agents, Pinecone-based RAG, and simple fitness tools to act as a personal-trainer-style assistant.​

## Project overview
This project provides a command-line chatbot that uses a LangChain agent with tools, short-term conversation memory, and a Pinecone vector index to answer fitness questions and general queries.​
The assistant behaves like a friendly personal trainer, can search a YouTube Q&A knowledge base, and offers utility tools such as word counting, current time, and simple training plan generation.​

## Features
LangChain ZERO_SHOT_REACT_DESCRIPTION agent with ConversationBufferWindowMemory for short conversation context.​

RAG over a Pinecone index (youtube-qa-index) using SentenceTransformers for semantic search.​

Fitness-focused tools: weekly training plan generator plus helper tools for time and word counting.​

## Prerequisites
Python 3.9+ installed on your system.​

Accounts and API keys for:

OpenAI (chat model gpt-3.5-turbo)

Pinecone (vector index already created as youtube-qa-index)

LangSmith / LangChain tracing (optional but used here)​

## Installation
Clone your project repository and change into the directory.

Create and activate a virtual environment, then install requirements such as langchain, langchain-openai, pinecone-client, sentence-transformers, python-dotenv, and openai via pip.​

Ensure your Pinecone index name matches INDEX_NAME = "youtube-qa-index" or update the script accordingly.​

## Environment variables
Create a .env file in the project root with at least:

OPENAI_API_KEY=...

PINECONE_API_KEY=...

LANGCHAIN_API_KEY=...

LANGCHAIN_PROJECT=memory-and-tools-rag-agent (or your preferred project name)​

The script also enables LangSmith tracing via:

LANGCHAIN_TRACING_V2=true

LANGCHAIN_ENDPOINT=https://api.smith.langchain.com​

## How it works
Embeddings: Uses the SentenceTransformer model flax-sentence-embeddings/all_datasets_v3_mpnet-base to encode queries and retrieve relevant passages from Pinecone.​

RAG: rag_search_func queries the Pinecone index, pulls text or passage_text from metadata, and returns concatenated context for the agent to use in responses.​

Agent: initialize_agent wires the tools, ChatOpenAI model, and ConversationBufferWindowMemory into a ZERO_SHOT_REACT_DESCRIPTION agent that selects tools based on their descriptions.​

## Tools exposed to the agent
rag_search: Looks up information in the YouTube QA knowledge base using Pinecone.​

get_current_time: Returns the current date and time string (input ignored).

word_count: Returns the word count for arbitrary text.

training_plan: Generates a simple weekly training split from natural language preferences (days per week, level, equipment).​

The system prompt also instructs the assistant when each tool should be preferred for specific query types.​

## Running the chatbot
Ensure your .env is correctly configured and the virtual environment is active.

Run the script:

bash
python agentexecutor_rag_memory.py
Use the CLI loop:

Type fitness questions like “Can you design a 3-day beginner routine for the gym?” to get a plan using both training_plan and RAG when relevant.​

Ask knowledge-base questions such as “What did the YouTube coach say about deload weeks?” to trigger rag_search over the Pinecone index.​

Ask “What’s the current time?” or “How many words are in: …” to exercise the utility tools.

Type exit or quit to end the session.




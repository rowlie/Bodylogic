# agentexecutor_rag_memory.py

import os
from datetime import datetime

from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.schema import SystemMessage  # for initial system prompt

# ==============================================================
# Load env vars from .env
# ==============================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "memory-and-tools-rag-agent")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env")
if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY not set in .env")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# ==============================================================
# Core config
# ==============================================================
INDEX_NAME = "youtube-qa-index"
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
retriever = SentenceTransformer(
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    device=device,
)

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ==============================================================
# LangChain memory + initial system prompt
# ==============================================================
memory = ConversationBufferWindowMemory(
    k=5,
    return_messages=True,
    memory_key="chat_history",
)

system_prompt = (
    "You are a friendly, evidence-based personal trainer and RAG assistant. "
    "Your goals are to: (1) give safe, practical fitness advice; "
    "(2) tailor suggestions to the user's level and goals; "
    "(3) clearly explain reasoning in simple language.\n\n"
    "Always use the retrieved knowledge base context when it is relevant.\n\n"
    "Tool usage rules:\n"
    "- If the user asks for word counts, you MUST call the `word_count` tool.\n"
    "- If the user asks for the current time or date, you MUST call the `get_current_time` tool.\n"
    "- If the user asks for a weekly workout split or training schedule, you SHOULD call the "
    "`training_plan` tool.\n\n"
    "When you use any tool, explicitly mention in your explanation that you used that tool, and base "
    "your answer directly on the tool's output instead of estimating."
)
memory.chat_memory.add_message(SystemMessage(content=system_prompt))

# ==============================================================
# RAG helper
# ==============================================================
def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> str:
    xq = retriever.encode(query).tolist()
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    parts = []
    for m in res.get("matches", []):
        meta = m["metadata"]
        passage = meta.get("text") or meta.get("passage_text") or ""
        if passage:
            parts.append(passage)
    return "\n\n".join(parts)

def rag_search_func(query: str) -> str:
    """
    Search the YouTube QA knowledge base and return relevant context.
    Use this when the user asks about content that might be in the index.
    """
    context = retrieve_pinecone_context(query)
    if not context:
        return "No relevant context found in the knowledge base."
    return context

# ==============================================================
# Simple tools
# ==============================================================
def get_current_time_func() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def word_count_func(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"

# Optional helper, not exposed as a tool
def convert_case_func(text: str, case_type: str) -> str:
    """
    Convert text to uppercase, lowercase, or title case.
    case_type options: 'upper', 'lower', 'title'
    """
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    else:
        return f"Error: Unknown case type '{case_type}'. Use 'upper', 'lower', or 'title'."

# ==============================================================
# Training plan tool (single string input)
# ==============================================================
def training_plan_func(prefs: str) -> str:
    """
    Generate a simple weekly training plan based on user preferences.

    Input format (natural language), e.g.:
      "3 days per week, beginner, full gym"
      "4 days per week, intermediate, home dumbbells only"
      "2 days per week, complete beginner, no equipment"
    """
    text = prefs.lower()

    # Defaults
    days = 3
    level = "beginner"
    equipment = "gym"

    if "2 days" in text:
        days = 2
    elif "3 days" in text:
        days = 3
    elif "4 days" in text:
        days = 4
    elif "5 days" in text:
        days = 5

    if "intermediate" in text:
        level = "intermediate"
    elif "advanced" in text:
        level = "advanced"
    elif "beginner" in text:
        level = "beginner"

    if "no equipment" in text or "bodyweight" in text:
        equipment = "no equipment"
    elif "home" in text or "dumbbell" in text:
        equipment = "home / dumbbells"
    else:
        equipment = "gym"

    plan_lines = [
        f"Weekly training plan ({days} days, {level}, {equipment}):",
        "",
    ]

    if days == 2:
        plan_lines += [
            "Day 1: Full body strength (squat, push, hinge, pull).",
            "Day 2: Full body strength + 15–20 min brisk walking or light cardio.",
        ]
    elif days == 3:
        plan_lines += [
            "Day 1: Upper body strength (push + pull).",
            "Day 2: Lower body strength (squats, hinges, lunges).",
            "Day 3: Full body circuit + 15–20 min cardio.",
        ]
    elif days == 4:
        plan_lines += [
            "Day 1: Upper body (push emphasis).",
            "Day 2: Lower body (squat emphasis).",
            "Day 3: Upper body (pull emphasis).",
            "Day 4: Lower body (hinge/glute emphasis) + short cardio.",
        ]
    else:
        plan_lines += [
            "Mix 3–4 strength-focused days with 1–2 cardio-focused days.",
            "Ensure at least 1 full rest day per week.",
        ]

    plan_lines.append(
        "Adjust sets/reps so the last 2–3 reps of each set feel challenging but controlled, "
        "keeping 1–2 reps in reserve for most sets."
    )

    return "\n".join(plan_lines)

# ==============================================================
# Wrap functions as LangChain Tools (0.0.354 style)
# ==============================================================
tools = [
    Tool(
        name="rag_search",
        func=rag_search_func,
        description=(
            "Use this to look up information in the YouTube QA knowledge base. "
            "Input should be a natural language question."
        ),
    ),
    Tool(
        name="get_current_time",
        func=lambda _: get_current_time_func(),
        description="Get the current date and time. Input is ignored.",
    ),
    Tool(
        name="word_count",
        func=word_count_func,
        description="Count the number of words in the given text.",
    ),
    Tool(
        name="training_plan",
        func=training_plan_func,
        description=(
            "Generate a simple weekly training plan based on user preferences. "
            "Input should be a short natural-language description like "
            '"3 days per week, beginner, full gym" or '
            '"4 days per week, intermediate, home dumbbells only".'
        ),
    ),
]

print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")

# ==============================================================
# Initialize agent (ZERO_SHOT_REACT_DESCRIPTION) with memory
# and robust parsing
# ==============================================================
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# ==============================================================
# Chat wrapper
# ==============================================================
def chat_with_agent(user_message: str) -> str:
    result = agent.run(user_message)
    return result

# ==============================================================
# CLI loop
# ==============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 LangChain Agent (ZERO_SHOT_REACT_DESCRIPTION) with RAG + Memory + Fitness Tools")
    print("=" * 70)
    print("Tools:", ", ".join([t.name for t in tools]))
    print(f"Pinecone index: '{INDEX_NAME}'")
    print("Type 'exit' or 'quit' to stop\n")

    while True:
        try:
            user_message = input("You: ").strip()
            if not user_message:
                continue
            if user_message.lower() in ["exit", "quit"]:
                print("\n✅ Session ended.")
                break

            print("\n🔄 Processing...\n")
            reply = chat_with_agent(user_message)
            print(f"Assistant: {reply}\n")

        except KeyboardInterrupt:
            print("\n\n✅ Session ended.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

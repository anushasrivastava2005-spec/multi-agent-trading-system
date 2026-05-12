"""Quick test: verify Groq tool calling + vision works."""
import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

key = os.getenv('GROQ_API_KEY')

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Test 1: Tool calling
print("Test 1: Tool calling with llama-3.3-70b-versatile...")
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1, api_key=key)
chain = llm.bind_tools([add_numbers])
r = chain.invoke('What is 2 + 3?')
print(f"  Tool calls: {r.tool_calls}")
print("  OK!")

# Test 2: Vision model basic
print("\nTest 2: Llama 4 Scout text...")
llm2 = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0.1, api_key=key)
r2 = llm2.invoke('Say hello in one word')
print(f"  Response: {r2.content}")
print("  OK!")

print("\nAll tests passed!")

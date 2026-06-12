from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()
import asyncio
import sys
from pathlib import Path

# Resolve the Math server next to this file so the path does not depend on the
# current working directory.
MATH_SERVER = str(Path(__file__).resolve().parent / "mathservr.py")

async def main():
    client = MultiServerMCPClient(
        {
            "Weather": {
                "transport": "streamable-http",
                "url": "http://localhost:8000/mcp",
            },
            "Math": {
                "transport": "stdio",
                # Use the current interpreter (which has `mcp` installed) instead
                # of bare "python", which may be missing or point elsewhere.
                "command": sys.executable,
                "args": [MATH_SERVER],
            },
        }
    )

    tools = await client.get_tools()
    #print("Available tools:", tools)

    model = ChatGroq(model="qwen/qwen3-32b")

    agent = create_agent(model, tools)

    math_response = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "What is (5 + 7)*8?"},
            ]
        }
    )
    print("Agent response:", math_response["messages"][-1].content)

    print("\n---\n")
    
    weather_response = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "What is the weather like in New York?"},
            ]
        }    )
    print("Agent response:", weather_response["messages"][-1].content)

asyncio.run(main())

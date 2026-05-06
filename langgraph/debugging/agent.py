from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START,END
from langgraph.graph.state import  StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")

llm = init_chat_model("groq:qwen/qwen3-32b")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
def make_agent():
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tools=[add]

    tool_node = ToolNode(tools)

    llm_with_tools = llm.bind_tools(tools)

    def call_llm_model(state:State):
        return {"messages":[llm_with_tools.invoke(state['messages'])]}
    ## Grpah
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))

    ## Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition
    )
    builder.add_edge("tools","tool_calling_llm")

    ## compile the graph
    graph=builder.compile()

    return graph

tool_agent=make_agent()
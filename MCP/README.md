# Model Context Protocol (MCP)

This folder contains a practical Python example of using the **Model Context Protocol** with LangChain. The main idea is simple: define external tools in MCP servers, let a client discover those tools, then pass them to an agent so the model can use them while answering user questions.

## Introduction

AI applications often need more than a language model. They may need context and tools: reading files, calling APIs, running calculations, querying a database, or fetching information from another service. MCP provides a standard way to connect models and agents to those external capabilities.

You can think of MCP as a shared layer between:

- **MCP Client**: the application or agent that wants to use external capabilities.
- **MCP Server**: a program that exposes tools, resources, or prompts.
- **Transport**: the communication mechanism used to move MCP messages between the client and server, such as `stdio` or `streamable-http`.

In this project:

- `weather.py` is an MCP server that exposes a `get_weather` tool.
- `mathservr.py` is an MCP server that exposes `add`, `subtract`, and `multiply` tools.
- `client.py` connects to both servers, collects their tools, and gives them to a LangChain agent backed by Groq.

## MCP Theory

MCP is an open protocol for standardizing how language model applications connect to external systems. Without MCP, every application might invent its own custom way to expose tools. With MCP, servers expose capabilities through a common protocol, and clients can discover and call those capabilities consistently.

Important MCP concepts:

- **Base Protocol**: the core message format, built on JSON-RPC 2.0.
- **Lifecycle**: the connection starts with initialization, where the client and server negotiate protocol version and capabilities.
- **Server Features**: servers can expose tools, resources, and prompt templates.
- **Client Features**: clients can declare features such as roots or sampling, depending on the application.

This example focuses on the most beginner-friendly feature: **tools**. The server exposes Python functions as tools. The client lists those tools, passes them to an agent, and the agent chooses which tool to call based on the user request.

## What Is The Protocol?

A protocol is an agreement about message structure and behavior. MCP uses JSON-RPC 2.0, so messages usually fall into three categories.

### Request

A request asks the other side to perform an operation. For example, a client can ask a server to list its tools:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

### Response

A response answers a previous request and uses the same `id`:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": []
  }
}
```

### Notification

A notification is a one-way message that does not expect a response:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

When a client calls a tool, the MCP request looks conceptually like this:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "New York"
    }
  }
}
```

In normal code, you usually do not write these JSON-RPC messages manually. Libraries such as `mcp` and `langchain-mcp-adapters` handle them for you.

## Communication Transports

A transport is the channel used to carry MCP messages between a client and a server. The messages remain MCP/JSON-RPC messages, but the communication channel changes.

### 1. stdio

With `stdio`, the client starts the MCP server as a subprocess and communicates with it through standard input and output:

- `stdin`: the client sends JSON-RPC messages to the server.
- `stdout`: the server sends JSON-RPC responses back to the client.
- `stderr`: the server can write logs here.

This is useful for local tools and command-line integrations.

In this project, `mathservr.py` uses `stdio`:

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

In `client.py`, the math server is configured like this:

```python
"Math": {
    "transport": "stdio",
    "command": "python",
    "args": ["mathservr.py"],
}
```

The client starts `mathservr.py` automatically as a subprocess.

### 2. Streamable HTTP

With `streamable-http`, the MCP server runs as an independent HTTP service. The server exposes a single MCP endpoint. In FastMCP, the default endpoint path is:

```text
http://localhost:8000/mcp
```

This endpoint is not a normal web page and it is not a REST API. If you open `/mcp` in a browser, you may see `406 Not Acceptable`. If you open `/` or `/weather`, you may see `404 Not Found`. That is expected because the endpoint expects MCP protocol messages from an MCP client, not browser HTML requests.

In this project, `weather.py` uses `streamable-http`:

```python
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

In `client.py`, the weather server is configured like this:

```python
"Weather": {
    "transport": "streamable-http",
    "url": "http://localhost:8000/mcp",
}
```

### 3. Legacy SSE

Some libraries still support `sse` for backward compatibility with older MCP versions. For new HTTP-based examples, prefer `streamable-http`.

## Server Code Walkthrough

### weather.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a high of 25°C."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Key parts:

- `FastMCP("Weather")` creates an MCP server named `Weather`.
- `@mcp.tool()` registers the Python function as an MCP tool.
- `location: str` helps MCP build the tool input schema.
- `mcp.run(transport="streamable-http")` starts the server over HTTP.

The current weather function is a placeholder. In a real application, you could replace the hardcoded response with a real weather API call.

### mathservr.py

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first."""
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

This server does not open a port. The client starts it as a subprocess and communicates with it through standard input/output.

## Client Code Walkthrough

`client.py` is the coordinator. It connects to multiple MCP servers:

```python
client = MultiServerMCPClient(
    {
        "Weather": {
            "transport": "streamable-http",
            "url": "http://localhost:8000/mcp",
        },
        "Math": {
            "transport": "stdio",
            "command": "python",
            "args": ["mathservr.py"],
        },
    }
)
```

Then it gets the tools from both servers:

```python
tools = await client.get_tools()
```

This sends tool-listing requests to the MCP servers and converts the MCP tools into LangChain-compatible tools.

Next, the code creates the model and agent:

```python
model = ChatGroq(model="qwen/qwen3-32b")
agent = create_agent(model, tools)
```

Then it asks a math question:

```python
math_response = await agent.ainvoke(
    {
        "messages": [
            {"role": "user", "content": "What is (5 + 7)*8?"},
        ]
    }
)
print("Agent response:", math_response["messages"][-1].content)
```

The agent can decide to use `add` and `multiply` from the math server.

Then it asks a weather question:

```python
weather_response = await agent.ainvoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather like in New York?"},
        ]
    }
)
print("Agent response:", weather_response["messages"][-1].content)
```

The agent can decide to use `get_weather` from the weather server.

Important detail: LangChain returns message objects such as `AIMessage`, so you read the final content with `.content`, not `["content"]`.

## Communication Flow

The full flow looks like this:

```text
User question
    |
    v
LangChain Agent in client.py
    |
    v
MultiServerMCPClient
    |
    +--> Weather server
    |       transport: streamable-http
    |       endpoint: http://localhost:8000/mcp
    |       tool: get_weather(location)
    |
    +--> Math server
            transport: stdio
            command: python mathservr.py
            tools: add(a, b), subtract(a, b), multiply(a, b)
```

Step by step:

1. Start `weather.py` in one terminal because it runs as a persistent HTTP server.
2. Start `client.py` in another terminal.
3. The client connects to the weather server at `/mcp`.
4. The client starts `mathservr.py` automatically as a subprocess.
5. The client requests the available tools from both servers.
6. The LangChain agent uses the relevant tools based on the user question.
7. The final answer is printed by `client.py`.

## How To Run

From inside the `MCP` folder, start the weather server:

```bash
uv run weather.py
```

Leave this process running. You should see output similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

Then open another terminal in the same `MCP` folder and run:

```bash
uv run client.py
```

If you see a warning like this:

```text
VIRTUAL_ENV=.venv does not match the project environment path
```

that is a `uv` virtual environment warning, not an MCP error. If you want `uv` to use the currently active virtual environment, run:

```bash
uv run --active client.py
```

## Environment Requirements

The main dependencies are listed in `pyproject.toml`:

- `mcp`
- `langchain-mcp-adapters`
- `langchain-groq`
- `langgraph`

Because `client.py` uses Groq, your `.env` file should include:

```env
GROQ_API_KEY=your_key_here
```

## Common Errors

### Unknown transport: streamable-httpe

This is a typo. Use:

```python
mcp.run(transport="streamable-http")
```

### GET / or GET /weather returns 404

This is expected. An MCP server is not automatically a website or REST API. The `get_weather` tool does not create a `/weather` URL.

### GET /mcp returns 406

This is also expected when you open the MCP endpoint in a browser. `/mcp` expects MCP-compatible requests and headers from an MCP client.

### AIMessage object is not subscriptable

This happens when you treat a LangChain message object like a dictionary:

```python
response["messages"][-1]["content"]
```

Use attribute access instead:

```python
response["messages"][-1].content
```

## Security Notes

- For local `streamable-http` servers, bind to `127.0.0.1` or `localhost`.
- Do not expose an MCP server to the public internet without proper authentication and authorization.
- Sensitive tools should require user confirmation or enforce clear permissions.
- A `stdio` server should not print normal logs to `stdout`, because that can corrupt the MCP message stream. Use `stderr` for logs.

## References

- [MCP Specification 2025-06-18: Base Protocol Overview](https://modelcontextprotocol.io/specification/2025-06-18/basic)
- [MCP Specification 2025-06-18: Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [MCP Specification 2025-06-18: Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Math")

@mcp.tool()
def add(a:float, b:float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def subtract(a:float, b:float) -> float:
    """Subtract the second number from the first."""
    return a - b

@mcp.tool()
def multiply(a:float, b:float) -> float:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
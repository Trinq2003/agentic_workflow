from mcp.server.fastmcp import FastMCP


mcp = FastMCP("HelloMCP")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers. This is a simple addition function. It takes two integers as input and returns their sum."""
    return a + b

@mcp.tool()
def mul(a: int, b: int) -> int:
    """Multiply two numbers. This is a simple multiplication function. It takes two integers as input and returns their product."""
    return a * b


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.resource("bye://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized goobye"""
    return f"Goodbye, {name}!"


@mcp.prompt()
def translation_ja(txt: str) -> str:
    """Translating to Japanese"""
    return f"Please translate this sentence into Japanese:\n\n{txt}"
@mcp.prompt()
def translation_vi(txt: str) -> str:
    """Translating to Vietnamese"""
    return f"Please translate this sentence into Vietnamese:\n\n{txt}"
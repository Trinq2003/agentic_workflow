from fastmcp import Client
import asyncio

# Standard MCP configuration with multiple servers
config = {
    "mcpServers": {
        "system": {
            "command": "node",
            "args": ["C:/Users/ADMIN/Code/VTNET/dev_tool/Benchmarking-dataset/mcp/servers/system/build/index.js"]
        }
    }
}

# Create a client that connects to all servers
client = Client(config)
lst_tools = []
async def main():
    async with client:
        tools = await client.list_tools()
        for tool in tools:
            lst_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            )

asyncio.run(main())
for tool in lst_tools:
    print(f"tool: {tool}")
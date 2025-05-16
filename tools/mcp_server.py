import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging

from mcp.server.fastmcp import FastMCP
from typing import List, Dict

from tools.python_code_runner import PythonCodeRunnerTool
# from tools.demonstration_sampling import DemonstrationSamplingTool
# from tools.tool_chooser import ToolChooserTool
from configuration.tool_configuration import ToolConfiguration

mcp = FastMCP("SystemMCPServer")

# Config initialization
python_code_runner_config = ToolConfiguration()
python_code_runner_config.load("configuration/yaml/tools/python_code_runner.yaml")

# Tool initialization
python_code_runner = PythonCodeRunnerTool(python_code_runner_config)

# Tool registration
@mcp.tool()
async def run_python_code_blocks(
    input_code_list: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    This function is used to run a list of Python code blocks and return the value of local variables.
    Parameters:
    - input_code_list (List[Dict[str, str]]): A list of dictionaries containing the code blocks to be executed
    Returns:
    - List[Dict[str, str]]: A list of dictionaries containing the results of the executed code blocks
    """
    global python_code_runner
    results = await python_code_runner.execute(input_code_list = input_code_list)
    return results

import logging
import sys
import asyncio

# Configurations
from configuration.llm_inference_configuration import APILLMConfiguration
from configuration.embedding_inference_configuration import APIEmbeddingModelConfiguration, LocalEmbeddingModelConfiguration
from configuration.nlp_configuration import NLPConfiguration
from configuration.tool_configuration import ToolConfiguration, DemonstrationSamplingToolConfiguration, ToolChooserToolConfiguration
from configuration.operator_configuration import CoTOperatorConfiguration, DebateOperatorConfiguration, ReActOperatorConfiguration
# System components
from llm.request_llm import RequestLLM
from embedding.request_embedding import RequestEmbeddingModel
from nlp.spacy_nlp import SpacyNLP
from tools.python_code_runner import PythonCodeRunnerTool
from tools.demonstration_sampling import DemonstrationSamplingTool
from tools.tool_chooser import ToolChooserTool
# Prompt
from prompt.zero_shot import ZeroShotPrompt
from prompt.few_shot import FewShotPrompt
from prompt.user_message import UserMessagePrompt
from prompt.assistant_message import AssistantMessagePrompt
from prompt.tool_message import ToolMessagePrompt
# Memory
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.memory_topic import AbstractMemoryTopic
from base_classes.memory.datatypes.data_item import PromptDataItem
from base_classes.memory.memory_worker import MemoryWorker
# MCP
# from tools.mcp_server
# Operator
from base_classes.operator import AbstractOperator
from operators.cot import CoTOperator
from operators.debate import DebateOperator
from operators.react import ReactOperator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t\t%(levelname)s\t%(message)s',
    handlers=[logging.StreamHandler()]
)

deepseek_configuration = APILLMConfiguration()
deepseek_configuration.load("configuration/yaml/llm/deepseek.yaml")
deepseek = RequestLLM(deepseek_configuration)

embed_english_v3_config = APIEmbeddingModelConfiguration()
embed_english_v3_config.load("configuration/yaml/embedding/embed-english-v3.yaml")
embed_english_v3 = RequestEmbeddingModel(embed_english_v3_config)

tool_chooser_tool_config = ToolChooserToolConfiguration()
tool_chooser_tool_config.load("configuration/yaml/tools/tool_chooser.yaml")
tool_chooser_tool = ToolChooserTool(tool_chooser_tool_config)

react_demonstration_sampling_tool_config = DemonstrationSamplingToolConfiguration()
react_demonstration_sampling_tool_config.load("configuration/yaml/tools/react_demonstration_sampling.yaml")
react_demonstration_sampling_tool = DemonstrationSamplingTool(tool_config=react_demonstration_sampling_tool_config)

react_operator_config = ReActOperatorConfiguration()
react_operator_config.load("configuration/yaml/operators/react.yaml")
react_operator = ReactOperator(react_operator_config)

all_logger_names = list(logging.root.manager.loggerDict.keys())
if '' not in all_logger_names:
    all_logger_names.append('')

log_file_handler = logging.FileHandler('test_utils/logs/react_log.txt', mode='a+', encoding='utf-8')
log_file_handler.setFormatter(logging.Formatter('%(asctime)s\t\t%(levelname)s\t%(message)s'))

for logger_name in all_logger_names:
    logger = logging.getLogger(logger_name)
    if logger_name.startswith("Orbit"):
        logger.setLevel(logging.DEBUG)
        logger.addHandler(log_file_handler)
    elif logger_name == '':
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.CRITICAL)


react_test_message = UserMessagePrompt(
    [
        {
            "role": "user",
            "content": "Tell me the total number of running processes on my system, and their details. Call tools to get the information."
        }
    ]
)

react_test_answer, react_test_dependency_graph = asyncio.run(react_operator.run(input_message = react_test_message))
print("*"*100)
print(f"React test answer: {react_test_answer.prompt}")
print("*"*100)
print(f"React test dependency graph: {react_test_dependency_graph}")
print("*"*100)
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
from base_classes.memory.memory_stack import AbstractMemoryStack
from base_classes.memory.datatypes.data_item import PromptDataItem
from base_classes.memory.memory_worker import MemoryWorker

## Chat turn 1:
prompt_data_1 = [{
    'role': 'user',
    'content': 'Calculate the sum of 11 and 20.'
}]
prompt_data_2 = [{
    'role': 'assistant',
    'content': 'In order to calculate the sum of two numbers, we need to do the following:\n\t1. Write a Python code with add_sum() function, receiving 2 variables.\n\t2. Execute the code with the given variables.\n\t3. Return the result of the sum.'
}]
prompt_data_3 = [{
    'role': 'assistant',
    'content': 'Here is the Python code to calculate the sum of two numbers:\n\n```python\na = 11\nb = 20\nresult = a + b\n\nprint(f"The sum of {a} and {b} is {result}.")\n```'
}]
prompt_data_4 = [{
    'role': 'tool',
    'content': {
        'name': 'run_python_code',
        'arguments': {
            'code': 'a = 11\nb = 20\nresult = a + b\n\nprint(f"The sum of {a} and {b} is {result}.")\nreturn result'
        }
    },
    'result': '31'
}]
prompt_data_5 = [{
    'role': 'assistant',
    'content': 'After executing the code, we got the sum of 11 and 20 is 31.'
}]
## Chat turn 2:
prompt_data_6 = [{
    'role': 'user',
    'content': 'Can you show me how to use run_python_code() to add 100 and 250?'
}]
prompt_data_7 = [{
    'role': 'assistant',
    'content': 'Sure! To use run_python_code() for 100 and 250, you would call:\n\nrun_python_code(code="a = 100\nb = 250\nresult = a + b\n\nprint(f"The sum of {a} and {b} is {result}.")\nreturn result")'
}]
prompt_data_8 = [{
    'role': 'tool',
    'content': {
        'name': 'run_python_code',
        'arguments': {
            'code': 'a = 100\nb = 250\nresult = a + b\n\nprint(f"The sum of {a} and {b} is {result}.")\nreturn result'
        }
    },
    'result': '350'
}]
prompt_data_9 = [{
    'role': 'assistant',
    'content': 'The result of run_python_code(code="a = 100\nb = 250\nresult = a + b\n\nprint(f"The sum of {a} and {b} is {result}.")\nreturn result") is 350.'
}]
## Chat turn 3:
prompt_data_10 = [{
    'role': 'user',
    'content': 'What if I want to add a list of numbers, like [5, 10, 15, 20]?'
}]
prompt_data_11 = [{
    'role': 'assistant',
    'content': (
        'For a list of numbers, you can use Python\'s built-in sum() function:\n'
        'numbers = [5, 10, 15, 20]\n'
        'total = sum(numbers)\n'
        'print(total)  # Output: 50\n'
    )
}]
## Chat turn 4:
prompt_data_12 = [{
    'role': 'user',
    'content': 'Can you write a function that multiplies all numbers in a list?'
}]
prompt_data_13 = [{
    'role': 'assistant',
    'content': (
        'Certainly! Here is a Python function to multiply all numbers in a list:\n'
        'def multiply_list(numbers):\n'
        '    result = 1\n'
        '    for n in numbers:\n'
        '        result *= n\n'
        '    return result\n'
        'Example: multiply_list([2, 3, 4]) returns 24.'
    )
}]
## Chat turn 5:
prompt_data_14 = [{
    'role': 'user',
    'content': 'Let\'s try your multiply_list function with [1, 2, 3, 4, 5]. What is the result?'
}]
prompt_data_15 = [{
    'role': 'assistant',
    'content': (
        'Sure! I will use multiply_list([1, 2, 3, 4, 5]).\n'
        'Let me calculate it for you.'
    )
}]
prompt_data_16 = [{
    'role': 'tool',
    'content': {
        'name': 'run_python_code',
        'arguments': {
            'code': 'numbers = [1, 2, 3, 4, 5]\nresult = multiply_list(numbers)\n\nprint(f"The result of multiplying all numbers in {numbers} is {result}.")\nreturn result'
        }
    },
    'result': 120
}]
prompt_data_17 = [{
    'role': 'assistant',
    'content': 'The result of multiplying all numbers in [1, 2, 3, 4, 5] is 120.'
}]
## Chat turn 6:
prompt_data_18 = [{
    'role': 'user',
    'content': 'If I have 10 apples, eat 3, add 7, and then eat 2, how many apples do I have?'
}]
prompt_data_19 = [{
    'role': 'assistant',
    'content': (
        'Let\'s break it down step by step:\n'
        '1. Start with 10 apples.\n'
        '2. Eat 3: 10 - 3 = 7 apples left.\n'
        '3. Add 7: 7 + 7 = 14 apples.\n\n'
        '4. Eat 2: 14 - 2 = 12 apples.\n'
        'You have 12 apples.'
    )
}]

# Memory Atom
mem_atom_1 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_1)))
mem_atom_2 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_2)))
mem_atom_3 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_3)))
mem_atom_4 = AbstractMemoryAtom(data=PromptDataItem(ToolMessagePrompt(prompt_data_4)))
mem_atom_5 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_5)))
mem_atom_6 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_6)))
mem_atom_7 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_7)))
mem_atom_8 = AbstractMemoryAtom(data=PromptDataItem(ToolMessagePrompt(prompt_data_8)))
mem_atom_9 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_9)))
mem_atom_10 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_10)))
mem_atom_11 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_11)))
mem_atom_12 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_12)))
mem_atom_13 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_13)))
mem_atom_14 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_14)))
mem_atom_15 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_15)))
mem_atom_16 = AbstractMemoryAtom(data=PromptDataItem(ToolMessagePrompt(prompt_data_16)))
mem_atom_17 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_17)))
mem_atom_18 = AbstractMemoryAtom(data=PromptDataItem(UserMessagePrompt(prompt_data_18)))
mem_atom_19 = AbstractMemoryAtom(data=PromptDataItem(AssistantMessagePrompt(prompt_data_19)))

# Memory Block
## Memory Block for Chat Turn 1
mem_block_1 = AbstractMemoryBlock()
mem_block_1.add_memory_atom(mem_atom_1)
mem_block_1.add_memory_atom(mem_atom_2)
mem_block_1.add_memory_atom(mem_atom_3)
mem_block_1.add_memory_atom(mem_atom_4)
mem_block_1.add_memory_atom(mem_atom_5)

mem_block_1.mem_atom_graph = {
    mem_atom_1.mem_atom_id: [mem_atom_2.mem_atom_id],
    mem_atom_2.mem_atom_id: [mem_atom_3.mem_atom_id, mem_atom_4.mem_atom_id],
    mem_atom_3.mem_atom_id: [mem_atom_5.mem_atom_id],
    mem_atom_4.mem_atom_id: [mem_atom_5.mem_atom_id],
    mem_atom_5.mem_atom_id: []
}

mem_block_1.output_response = mem_atom_5.data.content.prompt[0]['content']

## Memory Block for Chat Turn 2
mem_block_2 = AbstractMemoryBlock()
mem_block_2.add_memory_atom(mem_atom_6)
mem_block_2.add_memory_atom(mem_atom_7)
mem_block_2.add_memory_atom(mem_atom_8)
mem_block_2.add_memory_atom(mem_atom_9)

mem_block_2.mem_atom_graph = {
    mem_atom_6.mem_atom_id: [mem_atom_7.mem_atom_id],
    mem_atom_7.mem_atom_id: [mem_atom_8.mem_atom_id],
    mem_atom_8.mem_atom_id: [mem_atom_9.mem_atom_id],
    mem_atom_9.mem_atom_id: []
}

mem_block_2.output_response = mem_atom_9.data.content.prompt[0]['content']

## Memory Block for Chat Turn 3
mem_block_3 = AbstractMemoryBlock()
mem_block_3.add_memory_atom(mem_atom_10)
mem_block_3.add_memory_atom(mem_atom_11)

mem_block_3.mem_atom_graph = {
    mem_atom_10.mem_atom_id: [mem_atom_11.mem_atom_id],
    mem_atom_11.mem_atom_id: []
}

mem_block_3.output_response = mem_atom_11.data.content.prompt[0]['content']

## Memory Block for Chat Turn 4
mem_block_4 = AbstractMemoryBlock()
mem_block_4.add_memory_atom(mem_atom_12)
mem_block_4.add_memory_atom(mem_atom_13)

mem_block_4.mem_atom_graph = {
    mem_atom_12.mem_atom_id: [mem_atom_13.mem_atom_id],
    mem_atom_13.mem_atom_id: []
}

mem_block_4.output_response = mem_atom_13.data.content.prompt[0]['content']

## Memory Block for Chat Turn 5
mem_block_5 = AbstractMemoryBlock()
mem_block_5.add_memory_atom(mem_atom_14)
mem_block_5.add_memory_atom(mem_atom_15)
mem_block_5.add_memory_atom(mem_atom_16)
mem_block_5.add_memory_atom(mem_atom_17)

mem_block_5.mem_atom_graph = {
    mem_atom_14.mem_atom_id: [mem_atom_15.mem_atom_id],
    mem_atom_15.mem_atom_id: [mem_atom_16.mem_atom_id],
    mem_atom_16.mem_atom_id: [mem_atom_17.mem_atom_id],
    mem_atom_17.mem_atom_id: []
}

mem_block_5.output_response = mem_atom_17.data.content.prompt[0]['content']

## Memory Block for Chat Turn 6
mem_block_6 = AbstractMemoryBlock()
mem_block_6.add_memory_atom(mem_atom_18)
mem_block_6.add_memory_atom(mem_atom_19)

mem_block_6.mem_atom_graph = {
    mem_atom_18.mem_atom_id: [mem_atom_19.mem_atom_id],
    mem_atom_19.mem_atom_id: []
}

mem_block_6.output_response = mem_atom_19.data.content.prompt[0]['content']

# Memory Stack
mem_stack = AbstractMemoryStack()

# Memory Topic
mem_topic_1 = AbstractMemoryTopic()
mem_topic_2 = AbstractMemoryTopic()
mem_topic_3 = AbstractMemoryTopic()
mem_stack.add_mem_topic(mem_topic_1)
mem_stack.add_mem_topic(mem_topic_2)
mem_stack.add_mem_topic(mem_topic_3)

mem_topic_1.add_mem_block(mem_block_1)
mem_topic_1.add_mem_block(mem_block_2)
mem_topic_1.add_mem_block(mem_block_3)

mem_topic_2.add_mem_block(mem_block_4)
mem_topic_2.add_mem_block(mem_block_5)

mem_topic_3.add_mem_block(mem_block_6)
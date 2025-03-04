from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Union, List

from base_classes.operator import AbstractOperator
from configuration.operator_configuration import DebateOperatorConfiguration
from base_classes.tool import AbstractTool
from base_classes.llm import AbstractLanguageModel
from prompt.user_message import UserMessagePrompt
from prompt.few_shot import FewShotPrompt
from prompt.assistant_message import AssistantMessagePrompt
from tools.demonstration_sampling import DemonstrationSamplingTool
from base_classes.memory.memory_atom import AbstractMemoryAtom
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.memory.datatypes.data_item import PromptDataItem

class DebateOperator(AbstractOperator):
    """
    Inspired by Improving Factuality and Reasoning in Language Models through Multiagent Debate paper by MIT and Google Brain.
    Link: https://arxiv.org/pdf/2305.14325
    """
    _config: DebateOperatorConfiguration = None
    _num_of_rounds: int = None
    _num_of_debaters: int = None
    def __init__(self, config: DebateOperatorConfiguration) -> None:
        super().__init__(config = config)
        self._num_of_rounds = self._config.debate_num_of_round
        self._num_of_debaters = self._config.debate_num_of_debaters
    
    def run(self, input_message: Union[UserMessagePrompt, AssistantMessagePrompt]) -> AssistantMessagePrompt:
        """
        This method is used to run the CoT operator.
        """
        debate_contexts = [[{"role": "user", "content": input_message.text}] for debater in range(self._num_of_debaters)]
        
        for round in range(self._num_of_rounds):
            for i, debate_context in enumerate(debate_contexts):

                if round != 0:
                    debate_contexts_other = debate_contexts[:i] + debate_contexts[i+1:]
                    message = self.construct_message(debate_contexts_other, input_message.text, 2 * round - 1)
                    debate_context.append(message)

                completion = self.generate_answer(debate_context)

                assistant_message = self.construct_assistant_message(completion)
                debate_context.append(assistant_message)
        
    @staticmethod
    def construct_message(agents, question, idx):
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

        prefix_string = "These are the solutions to the problem from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent solution: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
        return {"role": "user", "content": prefix_string}

    @staticmethod
    def construct_assistant_message(completion):
        content = completion["choices"][0]["message"]["content"]
        return {"role": "assistant", "content": content}
    @staticmethod
    def generate_answer(answer_context):
        try:
            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages=answer_context,
                    n=1)
        except:
            print("retrying due to an error......")
            time.sleep(20)
            return generate_answer(answer_context)

        return completion
    @staticmethod
    def parse_question_answer(df, ix):
        question = df.iloc[ix, 0]
        a = df.iloc[ix, 1]
        b = df.iloc[ix, 2]
        c = df.iloc[ix, 3]
        d = df.iloc[ix, 4]

        question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

        answer = df.iloc[ix, 5]

        return question, answer
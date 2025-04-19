from abc import abstractmethod
from typing import List, Dict
from torch import Tensor

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.prompt import ICIOPrompt
from base_classes.memory.memory_topic import AbstractMemoryTopic
from base_classes.memory.management_term import MemoryBlockState

MULTITURN_INPUT_REFINEMENT_PROMPT = """
You are an advanced conversational AI designed for multiturn refinement interactions. Your goal is to provide coherent, logical, concise, and clear responses that progressively refine and improve based on user feedback and context. Follow these guidelines for every interaction:

1. **Coherence**: Maintain a consistent tone, style, and context throughout the conversation. Ensure responses align with prior messages and the user's intent.
2. **Logic**: Provide well-reasoned answers grounded in accurate information or sound reasoning. Avoid contradictions or logical fallacies.
3. **Conciseness**: Deliver precise, to-the-point responses without unnecessary verbosity, while ensuring all relevant details are included.
4. **Clarity**: Use simple, unambiguous language to ensure the user easily understands the response. Avoid jargon unless requested or contextually appropriate.
5. **Context Awareness**: Track the conversation history to reference prior user inputs, preferences, or clarifications. Adapt responses to reflect evolving user needs.
6. **Refinement**: Actively incorporate user feedback to improve the accuracy, relevance, or depth of responses. Ask clarifying questions when needed to better understand user intent.
7. **Engagement**: Maintain a polite, professional, and user-focused tone. Anticipate user needs and proactively offer helpful insights or suggestions when appropriate.
8. **Flexibility**: Handle diverse topics and adjust the level of detail or formality based on user cues or explicit requests.
9. **Error Handling**: If unsure or unable to provide an accurate response, acknowledge the limitation transparently and offer alternative ways to assist (e.g., rephrasing, narrowing the scope, or suggesting resources).
10. **Proactivity**: When appropriate, ask targeted follow-up questions to deepen the conversation or confirm understanding, ensuring the interaction remains productive.

For each user input:
- Analyze the query for intent, context, and any specific instructions.
- Reference conversation history to ensure continuity and relevance.
- Craft a response that balances brevity with completeness, refining based on prior exchanges.
- If clarification is needed, pose a concise, relevant question to guide the conversation.

Example Interaction Flow:
User: "Tell me about AI."
AI: "Artificial Intelligence (AI) refers to systems that mimic human intelligence, such as learning, reasoning, and problem-solving. It includes applications like machine learning, natural language processing, and computer vision. Would you like me to focus on a specific AI topic, like its history, current uses, or future trends?"
User: "Focus on current uses."
AI: "Current AI uses include virtual assistants (e.g., Siri), recommendation systems (e.g., Netflix), autonomous vehicles, and medical diagnostics. In business, AI optimizes supply chains and personalizes marketing. Would you like examples in a specific industry or details on how one of these works?"

Your responses should always aim to advance the conversation, delivering value with each turn while remaining open to further refinement based on user input.
"""

MULTITURN_OUTPUT_REFINEMENT_PROMPT = """
You are an advanced AI system designed for multiturn conversational applications, tasked with refining the last output to ensure coherence with the conversational history. Your goal is to rewrite the most recent response to align seamlessly with prior exchanges, user intent, and context while improving accuracy, clarity, and relevance. Adhere to these guidelines for each refinement:

 1. **Conversational Coherence**: Ensure the rewritten output maintains a consistent tone, style, and context with the entire conversation history. Reference prior user inputs, clarifications, or preferences naturally to preserve continuity.
 2. **Accuracy**: Verify that the rewritten output is factually correct and logically consistent, correcting any errors or inconsistencies from the last output while aligning with the conversation’s established facts or reasoning.
 3. **Clarity**: Rewrite the response using clear, concise language with a logical structure (e.g., paragraphs, lists, or headings) to enhance readability and ensure the user easily understands the refined content.
 4. **Relevance**: Focus the rewritten output on the user’s intent as inferred from the conversational history. Remove irrelevant details and emphasize information that directly addresses the user’s needs or queries.
 5. **Iterative Improvement**: Enhance the last output by incorporating user feedback, new context, or clarifications from the conversation history. Improve precision, depth, or specificity without altering the core intent unless explicitly required.
 6. **Contextual Integration**: Seamlessly weave in relevant details from prior turns, such as user preferences, specific questions, or recurring themes, to make the response feel like a natural continuation of the dialogue.
 7. **Conciseness**: Streamline the rewritten output to eliminate redundancy or verbosity while retaining all necessary details to fully address the user’s query.
 8. **User-Centric Adaptation**: Adjust the tone, detail level, or format (e.g., explanatory, instructional, or technical) based on the user’s evolving needs as reflected in the conversation history.
 9. **Proactive Alignment**: If the last output was misaligned with the user’s intent or context, rewrite it to correct the course, and include a subtle acknowledgment of the refinement (e.g., “To better address your question…”). If clarification is needed, pose a concise follow-up question.
10. **Ethical Integrity**: Ensure the rewritten output is neutral, unbiased, and respects user privacy. If the last output contained errors or limitations, address them transparently while offering a constructive alternative.

For each task:

- Receive the full conversational history and the last output.
- Analyze the history to understand the user’s intent, preferences, and any feedback or clarifications provided across turns.
- Identify misalignments, inaccuracies, or areas for improvement in the last output based on the conversation’s context.
- Rewrite the last output to align with the history, enhancing clarity, coherence, and relevance while preserving or refining the original intent.
- Structure the response for maximum readability, using formatting as needed (e.g., bullet points for lists, headings for sections).
- If appropriate, include a brief prompt for further input (e.g., “Does this address your question, or would you like further details?”) to sustain the conversation.

Example Multiturn Flow: Conversation History: User (Turn 1): “Explain how AI is used in healthcare.” AI (Turn 1): “AI in healthcare improves diagnostics, treatment planning, and patient monitoring. Machine learning analyzes medical images, like X-rays, to detect diseases early. Predictive models forecast patient outcomes, and chatbots assist with triage. Would you like to focus on a specific application?” User (Turn 2): “Focus on diagnostics, especially for cancer.” AI (Last Output, Turn 2): “AI diagnostics use machine learning to analyze data. In cancer, AI processes images to detect tumors. It also predicts disease progression.”

Refined Output: AI (Rewritten): “In cancer diagnostics, AI leverages machine learning to enhance accuracy and early detection. For example, deep learning models analyze medical images, such as mammograms or CT scans, to identify tumors with high precision, often outperforming traditional methods. AI also integrates patient data to assess cancer risk or stage, supporting oncologists in tailoring treatments. This aligns with your interest in diagnostics from our earlier discussion. Would you like details on a specific cancer type or AI technique used in this process?”

Your rewritten output should ensure the conversation feels fluid and responsive, refining the last response to be a coherent, accurate, and valuable continuation of the multiturn dialogue.
"""

class MemoryWorker:
    _llm: AbstractLanguageModel
    _emb_model: AbstractEmbeddingModel
    
    def __init__(self, llm: AbstractLanguageModel, emb_model: AbstractEmbeddingModel) -> None:
        """
        Initialize the Memory Language Model instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self._llm = llm
        self._emb_model = emb_model
    
    # Memory block context refinement methods
    def refine_input_query(self, mem_block: AbstractMemoryBlock) -> None:
        _refine_prompt = ICIOPrompt(
            instruction="",
            context="",
            input_indicator="",
            output_indicator="Go directly to the answer without any introduction.",
            role="user",
        )
        
        list_of_relevant_mem_blocks = self.memory_block_retrieval(mem_block)
        context_str = ""
        for turn_index, mem_block in enumerate(list_of_relevant_mem_blocks):
            context_str += f"""
                Turn {turn_index + 1}:
                User: {mem_block.input_query}
                Assistant: {mem_block.output_response}
                """
        _refine_prompt.context(context_str)
        
        raw_refined_input_response = self._llm.query(str(_refine_prompt.instruction(MULTITURN_INPUT_REFINEMENT_PROMPT)))
        mem_block.refined_input_query = self._llm.get_response_texts(raw_refined_input_response)[0]
        raw_refined_output_response = self._llm.query(str(_refine_prompt.instruction(MULTITURN_OUTPUT_REFINEMENT_PROMPT)))
        mem_block.refined_output_response = self._llm.get_response_texts(raw_refined_output_response)[0]
        
        mem_block.mem_block_state = MemoryBlockState.REFINED_INPUT_AND_OUTPUT
    
    # Memory block feature engineering methods
    def _extract_keywords_for_memory_block(self, mem_block: AbstractMemoryBlock) -> List[str]:
        """
        Extract important keywords from the input string using the LLM.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[str]: A list of extracted keywords.
        """
        _keyword_extraction_prompt = ICIOPrompt(
            instruction="Extract the most important keywords from the following text and list them separated by commas.",
            context="",
            input_indicator="",
            output_indicator="Go directly to the answer without any introduction. Output on one line of keywords, seperated by `,`.",
            role="user",
        )
        
        keyword_dict = {}
        # Extract keywords for refined context
        _keyword_extraction_prompt.context(mem_block.refined_input_query + "\n" + mem_block.refined_output_response)
        raw_response = self._llm.query(str(_keyword_extraction_prompt))
        keywords = [keyword.strip().lower() for keyword in self._llm.get_response_texts(raw_response)[0].split(",")]
        keyword_dict["refined"] = keywords
        # Extract keywords for raw context
        _keyword_extraction_prompt.context = mem_block.input_query + "\n" + mem_block.output_response
        raw_response = self._llm.query(str(_keyword_extraction_prompt))
        keywords = [keyword.strip().lower() for keyword in self._llm.get_response_texts(raw_response)[0].split(",")]
        keyword_dict["raw"] = keywords
        
        return keywords
    
    def _generate_embedding_for_memory_block(self, mem_block: AbstractMemoryBlock) -> List[Tensor] | Tensor:
        """
        Generate an embedding vector for the input string using the embedding model.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[Tensor] | Tensor: The generated embedding vector.
        """
        # Use the embedding model's encode method (assumes it returns List[float])
        return {
            'refined': {
                'refined_input_embedding': self._emb_model.encode(mem_block.refined_input_query),
                'refined_output_embedding': self._emb_model.encode(mem_block.refined_output_response)
            },
            'raw': {
                'input_embedding': self._emb_model.encode(mem_block.input_query),
                'output_embedding': self._emb_model.encode(mem_block.output_response),
                'context_embedding': self._emb_model.encode(str(mem_block))
            }
        }
    
    def feature_engineer_for_memory_block(self, mem_block: AbstractMemoryBlock) -> None:
        """
        Perform feature engineering on the memory block.

        Args:
            mem_block (AbstractMemoryBlock): The memory block to be processed.

        Returns:
            None
        """
        assert mem_block.mem_block_state == MemoryBlockState.REFINED_INPUT_AND_OUTPUT, "❌ Memory Block State should be REFINE."
        
        container_topic_id = mem_block.topic_container_id
        # Set the topic container ID for the memory block
        mem_block.identifying_features["address_in_topic"] = AbstractMemoryTopic.get_memtopic_instance_by_id(container_topic_id).get_address_of_block_by_id(mem_block.mem_block_id)
        
        keywords: Dict = self._extract_keywords_for_memory_block(mem_block)
        embedding_vector: Dict = self._generate_embedding_for_memory_block(mem_block)
        
        # Store the keyword features in the memory block
        mem_block.identifying_features["feature_for_raw_context"]["keywords"] = keywords["raw"]
        mem_block.identifying_features["feature_for_refined_context"]["keywords"] = keywords["refined"]
        
        # Store the embedding features in the memory block
        mem_block.identifying_features["feature_for_raw_context"]["context_embedding"] = embedding_vector["raw"]["context_embedding"]
        mem_block.identifying_features["feature_for_raw_context"]["input_embedding"] = embedding_vector["raw"]["input_embedding"]
        mem_block.identifying_features["feature_for_raw_context"]["output_embedding"] = embedding_vector["raw"]["output_embedding"]
        mem_block.identifying_features["feature_for_refined_context"]["refined_input_embedding"] = embedding_vector["refined"]["refined_input_embedding"]
        mem_block.identifying_features["feature_for_refined_context"]["refined_output_embedding"] = embedding_vector["refined"]["refined_output_embedding"]
        
        mem_block.mem_block_state = MemoryBlockState.FEATURE_ENGINEERED
    
    # Memory block retrieval method
    def memory_block_retrieval(self, mem_block: AbstractMemoryBlock) -> List[AbstractMemoryBlock]:
        # TODO: Implement this method to retrieve relevant memory blocks based on the input query and context.
        pass
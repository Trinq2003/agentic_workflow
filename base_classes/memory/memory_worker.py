from abc import abstractmethod
from typing import List, Dict
import torch
from torch import Tensor
import numpy as np
import spacy
from datetime import datetime

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.nlp import AbstractNLPModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.prompt import ICIOPrompt
from base_classes.memory.memory_topic import AbstractMemoryTopic
from base_classes.memory.management_term import MemoryBlockState
from base_classes.memory.memory_stack import AbstractMemoryStack
from base_classes.logger import HasLoggerClass

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

class MemoryWorker(HasLoggerClass):
    _llm: AbstractLanguageModel
    _emb_model: AbstractEmbeddingModel
    _nlp: AbstractNLPModel
    _tunable_hyperparameters: Dict[str, float] = {
        "temporal_score_weight": 1/3,
        "access_count_weight": 1/3,
        "semantic_score_weight": 1/3,
        "semantic_weights": {
            "refined_input_embedding": 1/4,
            "refined_output_embedding": 1/4,
            "input_embedding": 1/4,
            "output_embedding": 1/4,
        }   
    }
    
    def __init__(self, llm: AbstractLanguageModel, emb_model: AbstractEmbeddingModel, nlp: AbstractNLPModel) -> None:
        """
        Initialize the Memory Language Model instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        self._llm = llm
        self._emb_model = emb_model
        self._nlp = nlp
        super().__init__()
    
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
    
    def _generate_embedding_for_memory_block(self, mem_block: AbstractMemoryBlock) -> Dict:
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
    
    # Memory block retrieval method (inside topic)
    def _retrieval_temporal_score(self, turn_number, current_turn) -> float:
        diff = turn_number - current_turn
        x = (2/3) * (7 - diff)
        return torch.sigmoid(torch.tensor(x, dtype=torch.float32)).item()
    def _retrieval_access_score(self, access_count: int) -> float:
        return np.log(access_count + 1)/np.log(30)
    def _retrieval_semantic_similarity_score(self, raw_input_emb_similarity: float, raw_output_emb_similarity: float, refined_input_emb_similarity: float, refined_output_emb_similarity: float) -> float:
        return self._tunable_hyperparameters["semantic_weights"]["input_embedding"] * raw_input_emb_similarity + self._tunable_hyperparameters["semantic_weights"]["output_embedding"] * raw_output_emb_similarity + self._tunable_hyperparameters["semantic_weights"]["refined_input_embedding"] * refined_input_emb_similarity + self._tunable_hyperparameters["semantic_weights"]["refined_output_embedding"] * refined_output_emb_similarity
    def _retrieval_score(self, temporal_score: float, access_score: float, semantic_similarity_score: float) -> float:
        return self._tunable_hyperparameters["temporal_score_weight"] * temporal_score + self._tunable_hyperparameters["access_count_weight"]* access_score + self._tunable_hyperparameters["semantic_score_weight"] * semantic_similarity_score
    
    def memory_block_retrieval(self, mem_block: AbstractMemoryBlock, top_k: int = 5) -> List[AbstractMemoryBlock]:
        container_topic_id = mem_block.topic_container_id
        list_of_mem_blocks = AbstractMemoryTopic.get_memtopic_instance_by_id(container_topic_id).chain_of_memblocks
        current_turn = len(list_of_mem_blocks)
        
        input_query = mem_block.input_query
        input_query_emb = self._emb_model.encode(input_query)
        
        score_list = []
        
        for (turn_number, mem_block) in enumerate(list_of_mem_blocks):
            # Calculate the temporal score
            temporal_score = self._retrieval_temporal_score(turn_number, current_turn)
            
            # Calculate the access score
            access_count = mem_block.access_count
            access_score = self._retrieval_access_score(access_count)
            
            # Calculate the similarity score
            raw_input_emb = mem_block.identifying_features["feature_for_raw_context"]["input_embedding"]
            raw_input_emb_similarity = self._emb_model.similarity(input_query, raw_input_emb)
            raw_output_emb = mem_block.identifying_features["feature_for_raw_context"]["output_embedding"]
            raw_output_emb_similarity = self._emb_model.similarity(input_query, raw_output_emb)
            refined_input_emb = mem_block.identifying_features["feature_for_refined_context"]["refined_input_embedding"]
            refined_input_emb_similarity = self._retrieval_semantic_similarity_score(input_query_emb, refined_input_emb)
            refined_output_emb = mem_block.identifying_features["feature_for_refined_context"]["refined_output_embedding"]
            refined_output_emb_similarity = self._retrieval_semantic_similarity_score(input_query_emb, refined_output_emb)
            
            semantic_similarity_score = self._retrieval_semantic_similarity_score(raw_input_emb_similarity, raw_output_emb_similarity, refined_input_emb_similarity, refined_output_emb_similarity)
            # Calculate the overall retrieval score
            retrieval_score = self._retrieval_score(temporal_score, access_score, semantic_similarity_score)

            score_list.append((mem_block, retrieval_score))
        
        score_list.sort(key=lambda x: x[1], reverse=True)
        
        return [mem_block for mem_block, score in score_list[:top_k]]
    
    # Memory block arrangement method (choose memory topic to store memory block)
    def _is_directly_connected(self, new_query: str, prev_response: str) -> bool:
        """
        Check if the new query is directly connected to the previous response.
        Uses coreference resolution, subject detection, and entity overlap.
        """
        # Combine previous response and new query for coreference resolution
        combined_text = prev_response + " " + new_query
        doc = self._nlp(combined_text)
        
        # Check for coreferences linking the new query to the previous response
        for token in doc:
            if token.dep_ in ["conj", "appos"]:
                return True
        
        # Check if the new query lacks a subject (indicating reliance on previous context)
        query_doc = self._nlp(new_query)
        has_subject = any(token.dep_ == "nsubj" for token in query_doc)
        if not has_subject:
            return True
        
        # Check if the new query introduces a new subject or entity
        prev_entities = set(ent.text.lower() for ent in self._nlp(prev_response).ents)
        query_entities = set(ent.text.lower() for ent in query_doc.ents)
        new_entities = query_entities - prev_entities
        if not new_entities and query_entities:  # Shared entities, no new ones
            return True
        
        return False
    
    def select_relevant_topics(self, new_query: str, prev_response: str, topics: List[AbstractMemoryTopic], top_n: int = 1) -> List[AbstractMemoryTopic]:
        """
        Select the most relevant memory topic(s) for the new query.
        If directly connected to the previous response, return the current topic.
        Otherwise, use semantic similarity, keyword matching, NER, and temporal weighting.
        """
        if not topics:
            return []
        
        # If directly connected, return the most recent topic (assumed current)
        if self._is_directly_connected(new_query, prev_response):
            return [topics[-1]]
        
        # Encode the new query
        query_embedding = self._emb_model.encode(new_query)
        query_doc = self._nlp(new_query)
        query_entities = set(ent.text.lower() for ent in query_doc.ents)
        query_tokens = set(new_query.lower().split())
        
        # Compute similarity scores for all topics
        similarity_scores = []
        current_time = datetime.now()
        
        for topic in topics:
            # Semantic similarity
            sem_sim = torch.cosine_similarity(query_embedding, topic.identifying_features["refined_context_embedding"]).item()
            
            # Keyword matching (simple overlap)
            context_tokens = set(topic.context.lower().split())
            keyword_overlap = len(query_tokens.intersection(context_tokens)) / max(len(query_tokens), 1)
            
            # NER-based matching
            topic_doc = self._nlp(topic.context)
            topic_entities = set(ent.text.lower() for ent in topic_doc.ents)
            entity_overlap = len(query_entities.intersection(topic_entities)) / max(len(query_entities), 1) if query_entities else 0
            
            # Temporal weighting (more recent topics get a slight boost)
            time_diff = (current_time - topic.timestamp).total_seconds() / (24 * 3600)  # Days difference
            temporal_weight = max(0.5, 1 - time_diff / 30)  # Decay over 30 days, min 0.5
            
            # Combine scores with weights
            combined_score = (0.5 * sem_sim) + (0.2 * keyword_overlap) + (0.2 * entity_overlap) + (0.1 * temporal_weight)
            similarity_scores.append((topic, combined_score))
        
        # Sort by score and return top N topics
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in similarity_scores[:top_n]]
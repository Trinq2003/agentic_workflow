from abc import abstractmethod
from typing import List, Dict
import torch
from torch import Tensor
import spacy
from datetime import datetime

from base_classes.llm import AbstractLanguageModel
from base_classes.embedding import AbstractEmbeddingModel
from base_classes.nlp import AbstractNLPModel
from base_classes.memory.memory_block import AbstractMemoryBlock
from base_classes.prompt import ICIOPrompt, AbstractPrompt
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
 2. **Accuracy**: Verify that the rewritten output is factually correct and logically consistent, correcting any errors or inconsistencies from the last output while aligning with the conversation's established facts or reasoning.
 3. **Clarity**: Rewrite the response using clear, concise language with a logical structure (e.g., paragraphs, lists, or headings) to enhance readability and ensure the user easily understands the refined content.
 4. **Relevance**: Focus the rewritten output on the user's intent as inferred from the conversational history. Remove irrelevant details and emphasize information that directly addresses the user's needs or queries.
 5. **Iterative Improvement**: Enhance the last output by incorporating user feedback, new context, or clarifications from the conversation history. Improve precision, depth, or specificity without altering the core intent unless explicitly required.
 6. **Contextual Integration**: Seamlessly weave in relevant details from prior turns, such as user preferences, specific questions, or recurring themes, to make the response feel like a natural continuation of the dialogue.
 7. **Conciseness**: Streamline the rewritten output to eliminate redundancy or verbosity while retaining all necessary details to fully address the user's query.
 8. **User-Centric Adaptation**: Adjust the tone, detail level, or format (e.g., explanatory, instructional, or technical) based on the user's evolving needs as reflected in the conversation history.
 9. **Proactive Alignment**: If the last output was misaligned with the user's intent or context, rewrite it to correct the course, and include a subtle acknowledgment of the refinement (e.g., "To better address your question…"). If clarification is needed, pose a concise follow-up question.
10. **Ethical Integrity**: Ensure the rewritten output is neutral, unbiased, and respects user privacy. If the last output contained errors or limitations, address them transparently while offering a constructive alternative.

For each task:

- Receive the full conversational history and the last output.
- Analyze the history to understand the user's intent, preferences, and any feedback or clarifications provided across turns.
- Identify misalignments, inaccuracies, or areas for improvement in the last output based on the conversation's context.
- Rewrite the last output to align with the history, enhancing clarity, coherence, and relevance while preserving or refining the original intent.
- Structure the response for maximum readability, using formatting as needed (e.g., bullet points for lists, headings for sections).
- If appropriate, include a brief prompt for further input (e.g., "Does this address your question, or would you like further details?") to sustain the conversation.

Example Multiturn Flow: Conversation History: User (Turn 1): "Explain how AI is used in healthcare." AI (Turn 1): "AI in healthcare improves diagnostics, treatment planning, and patient monitoring. Machine learning analyzes medical images, like X-rays, to detect diseases early. Predictive models forecast patient outcomes, and chatbots assist with triage. Would you like to focus on a specific application?" User (Turn 2): "Focus on diagnostics, especially for cancer." AI (Last Output, Turn 2): "AI diagnostics use machine learning to analyze data. In cancer, AI processes images to detect tumors. It also predicts disease progression."

Refined Output: AI (Rewritten): "In cancer diagnostics, AI leverages machine learning to enhance accuracy and early detection. For example, deep learning models analyze medical images, such as mammograms or CT scans, to identify tumors with high precision, often outperforming traditional methods. AI also integrates patient data to assess cancer risk or stage, supporting oncologists in tailoring treatments. This aligns with your interest in diagnostics from our earlier discussion. Would you like details on a specific cancer type or AI technique used in this process?"

Your rewritten output should ensure the conversation feels fluid and responsive, refining the last response to be a coherent, accurate, and valuable continuation of the multiturn dialogue.
"""

class MemoryWorker(HasLoggerClass):
    _llm: AbstractLanguageModel
    _emb_model: AbstractEmbeddingModel
    _nlp: AbstractNLPModel
    
    def __init__(self, llm: AbstractLanguageModel, emb_model: AbstractEmbeddingModel, nlp_model: AbstractNLPModel) -> None:
        """
        Initialize the Memory Language Model instance with configuration, model details, and caching options.

        :param llm_config: The LLM configuration object.
        :type llm_config: LLMConfiguration
        """
        super().__init__()
        self._llm = llm
        self._emb_model = emb_model
        self._nlp = nlp_model
        
        self.logger.debug(f"Memory Worker initialized with LLM: {self._llm.llm_id}, Embedding Model: {self._emb_model.emb_id}, NLP Model: {self._nlp.nlp_model_id}.")
    
    # Memory Block Methods
    ## Memory block context refinement methods
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
        _refine_prompt.context = context_str
        raw_input_refinement_prompt = AbstractPrompt([_refine_prompt.to_dict()])
        raw_refined_input_response = self._llm.query(raw_input_refinement_prompt)
        
        # Check if the response is None (query failed)
        if raw_refined_input_response is None:
            self.logger.warning(f"LLM query failed for input refinement of memory block {mem_block.mem_block_id}, using original input")
            mem_block.refined_input_query = mem_block.input_query
        else:
            mem_block.refined_input_query = self._llm.get_response_texts(raw_refined_input_response)[0]
        
        mem_block.mem_block_state = MemoryBlockState.REFINED_INPUT_AND_OUTPUT
    
    ## Memory block feature engineering methods
    def _extract_keywords_for_memory_block(self, mem_block: AbstractMemoryBlock) -> Dict[str, List[str]]:
        """
        Extract important keywords (nouns) from the input string using NLP.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            Dict[str, List[str]]: A dictionary with "raw" and "refined" keyword lists.
        """
        # Combine input and output text for keyword extraction
        combined_text = str(mem_block)

        # Use NLP to extract nouns (excluding pronouns)
        nouns = self._nlp.extract_keywords(combined_text)
        
        self.logger.debug(f"Keywords for Memblock {mem_block.mem_block_id}: {nouns}")
        
        # Return the expected dictionary structure
        return {
            "raw": nouns,
            "refined": nouns
        }
    
    def _generate_embedding_for_memory_block(self, mem_block: AbstractMemoryBlock) -> Dict:
        """
        Generate an embedding vector for the input string using the embedding model.

        Args:
            mem_block (AbstractMemoryBlock): The memory block containing the input string.

        Returns:
            List[Tensor] | Tensor: The generated embedding vector.
        """
        # Use the embedding model's encode method (assumes it returns List[float])
        refined_input_embedding = self._emb_model.encode(mem_block.refined_input_query)
        raw_input_embedding = self._emb_model.encode(mem_block.input_query)
        raw_output_embedding = self._emb_model.encode(mem_block.output_response)
        context_embedding = self._emb_model.encode(str(mem_block))
        self.logger.debug(f"Generated embeddings for Memblock {mem_block.mem_block_id}")
        return {
            'refined': {
                'refined_input_embedding': refined_input_embedding,
            },
            'raw': {
                'input_embedding': raw_input_embedding,
                'output_embedding': raw_output_embedding,
                'context_embedding': context_embedding
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
        assert mem_block.mem_block_state == MemoryBlockState.INPUT_AND_OUTPUT, "❌ Memory Block State should be INPUT_AND_OUTPUT."
        
        container_topic_ids = mem_block.topic_container_ids
        # Set the topic container ID for the memory block
        # mem_block.identifying_features["address_in_topic"] = AbstractMemoryTopic.get_memtopic_instance_by_id(container_topic_id).get_address_of_block_by_id(mem_block.mem_block_id)
        
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
        
        mem_block.mem_block_state = MemoryBlockState.FEATURE_ENGINEERED
    
    # Memory Topic Methods
    def feature_engineer_for_memory_topic(self, mem_topic: AbstractMemoryTopic, alpha: float = 0.1, w: int = 5) -> None:
        mem_blocks = mem_topic.chain_of_memblocks

        # Keyword Engineering
        block_keywords = [mem_block.identifying_features["feature_for_raw_context"]["keywords"] for mem_block in mem_blocks]
        if block_keywords:
            mem_topic.identifying_features["keywords"] = list(set.union(*(set(kw) for kw in block_keywords)))
        else:
            mem_topic.identifying_features["keywords"] = []
        self.logger.debug(f"Memtopic keywords: {mem_topic.identifying_features['keywords']}")
        # Embedding Engineering
        """
        Computes the topic embedding using a sigmoid-based temporal weighting function.

        This method implements the formula:
        e_t_i = sum_{j=1}^{m_i} sigma(alpha * (j - (m_i - w))) * e_t_i,j

        where:
        - e_t_i: The topic embedding vector.
        - m_i: The number of blocks in the topic.
        - j: The 1-based index of a block in the topic.
        - w: The window size parameter (default: 5).
        - alpha: The temporal decay factor (default: 0.1).
        - sigma(x) = 1 / (1 + e^(-x)) is the sigmoid function.
        - e_t_i,j: The embedding vector of the j-th block.

        :param mem_topic: The memory topic to compute the embedding for.
        :param alpha: The temporal decay factor.
        :param w: The window size parameter.
        """
        
        m_i = len(mem_blocks)

        if m_i == 0:
            self.logger.warning(f"Topic {mem_topic.mem_topic_id} has no memory blocks. Cannot compute topic embedding.")
            return None

        first_block_embedding = mem_blocks[0].identifying_features["feature_for_raw_context"]["context_embedding"]
        if first_block_embedding is None:
            self.logger.warning(f"The first block in topic {mem_topic.mem_topic_id} does not have a context embedding. Cannot compute topic embedding.")
            return None

        embedding_dim = len(first_block_embedding)
        topic_embedding = torch.zeros(embedding_dim, dtype=torch.float64)

        for j, mem_block in enumerate(mem_blocks):
            block_embedding = mem_block.identifying_features["feature_for_raw_context"]["context_embedding"]
            
            if block_embedding is None:
                self.logger.warning(f"Memory block {mem_block.mem_block_id} at index {j} is missing context embedding. Skipping.")
                continue

            x = alpha * ((j + 1) - (m_i - w))
            sigmoid_weight = 1 / (1 + torch.exp(torch.tensor(-x, dtype=torch.float64)))
            # self.logger.debug(f"Sigmoid weight: {sigmoid_weight}")
            
            topic_embedding += sigmoid_weight * torch.tensor(block_embedding, dtype=torch.float64)

        mem_topic.identifying_features["embedding"] = topic_embedding

    # Memory block retrieval method (inside topic)
    def __retrieval_temporal_score_weight(self, turn_number, current_turn) -> float:
        """
        Computes the temporal score weight for a memory block based on its turn number and the current turn.
        """
        diff = turn_number - current_turn
        x = (2/3) * (7 - diff)
        return torch.sigmoid(torch.tensor(x, dtype=torch.float32)).item()
    def __retrieval_access_score_weight(self, access_count: int) -> float:
        return (torch.log(torch.tensor(access_count + 1.0)) / torch.log(torch.tensor(30.0))).item()
    
    def calculate_keyword_matching_score(self, mem_block_1: AbstractMemoryBlock, mem_block_2: AbstractMemoryBlock, 
                                       similarity_type: str = "jaccard") -> float:
        """
        Calculate keyword matching score between two memory blocks.
        
        Formula options:
        1. Jaccard Similarity: |A ∩ B| / |A ∪ B|
        2. Cosine Similarity: |A ∩ B| / sqrt(|A| * |B|)
        3. Intersection over Union (IoU): |A ∩ B| / |A ∪ B|
        4. Dice Coefficient: 2 * |A ∩ B| / (|A| + |B|)
        5. Overlap Coefficient: |A ∩ B| / min(|A|, |B|)
        
        Args:
            mem_block_1: First memory block
            mem_block_2: Second memory block
            similarity_type: Type of similarity metric ("jaccard", "cosine", "iou", "dice", "overlap")
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Extract keywords from both memory blocks
        keywords_1 = set(mem_block_1.identifying_features["feature_for_raw_context"]["keywords"])
        keywords_2 = set(mem_block_2.identifying_features["feature_for_raw_context"]["keywords"])
        
        # Calculate set operations
        intersection = keywords_1 & keywords_2
        union = keywords_1 | keywords_2
        
        # Handle edge cases
        if not keywords_1 and not keywords_2:
            return 0.0  # Both blocks have no keywords
        if not keywords_1 or not keywords_2:
            return 0.0  # One block has no keywords
        
        # Calculate similarity based on type
        if similarity_type == "jaccard" or similarity_type == "iou":
            # Jaccard Similarity / IoU: |A ∩ B| / |A ∪ B|
            return len(intersection) / len(union)
        
        elif similarity_type == "cosine":
            # Cosine Similarity: |A ∩ B| / sqrt(|A| * |B|)
            return len(intersection) / (len(keywords_1) * len(keywords_2)) ** 0.5
        
        elif similarity_type == "dice":
            # Dice Coefficient: 2 * |A ∩ B| / (|A| + |B|)
            return (2 * len(intersection)) / (len(keywords_1) + len(keywords_2))
        
        elif similarity_type == "overlap":
            # Overlap Coefficient: |A ∩ B| / min(|A|, |B|)
            return len(intersection) / min(len(keywords_1), len(keywords_2))
        
        else:
            # Default to Jaccard similarity
            return len(intersection) / len(union)
    
    def memory_block_retrieval(self, mem_block: AbstractMemoryBlock, top_k: int = 5) -> List[AbstractMemoryBlock]:
        self.logger.debug(f"Retrieving memory blocks for mem_block {mem_block.mem_block_id}")
        
        container_topic_ids = mem_block.topic_container_ids
        container_topics = [AbstractMemoryTopic.get_memtopic_instance_by_id(container_topic_id) for container_topic_id in container_topic_ids]
        list_of_mem_blocks_in_stacks = AbstractMemoryStack.get_memstack_instance_by_id(container_topics[0].stack_container_id).sequence_of_mem_blocks
        current_turn = len(list_of_mem_blocks_in_stacks)
        score_list = [(_mem_block, 0.0) for _mem_block in list_of_mem_blocks_in_stacks]
        input_query = mem_block.input_query
        input_query_emb = self._emb_model.encode(input_query)

        block_in_stack_embs = [mem_block.identifying_features["feature_for_raw_context"]["context_embedding"] for mem_block in list_of_mem_blocks_in_stacks]
        
        # Convert to torch tensors and perform matrix multiplication for similarity
        query_tensor = torch.tensor(input_query_emb, dtype=torch.float32)
        
        # Handle potential None embeddings
        embedding_dim = len(input_query_emb)
        cleaned_block_embs = [emb if emb is not None else [0.0] * embedding_dim for emb in block_in_stack_embs]
        
        stack_matrix = torch.tensor(cleaned_block_embs, dtype=torch.float32)
        
        # Normalize for cosine similarity, which is generally better than raw dot product
        query_tensor_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=0)
        stack_matrix_norm = torch.nn.functional.normalize(stack_matrix, p=2, dim=1)
        
        block_pairwise_semantic_similarity_scores_matrix = stack_matrix_norm @ query_tensor_norm
        
        for mb_idx, _mem_block in list(enumerate(list_of_mem_blocks_in_stacks))[:-1]:
            shared_topic_score = 0.0
            block_pairwise_semantic_similarity_score = block_pairwise_semantic_similarity_scores_matrix[mb_idx].item()
            block_pairwise_keyword_matching_score = self.calculate_keyword_matching_score(_mem_block, mem_block)
            # Compute intersection over union (IoU) of container_topic_ids lists
            set1 = set(_mem_block.topic_container_ids)
            set2 = set(mem_block.topic_container_ids)
            intersection = set1 & set2
            union = set1 | set2
            shared_topic_score = len(intersection) / len(union) if union else 0.0
            
            # Calculate the temporal score
            temporal_score_weight = self.__retrieval_temporal_score_weight(mb_idx, current_turn)
            
            # Calculate the access score
            access_count = _mem_block.access_count
            access_score_weight = self.__retrieval_access_score_weight(access_count)
            
            retrieval_score = 1/3 * (access_score_weight * temporal_score_weight * (block_pairwise_semantic_similarity_score + shared_topic_score + block_pairwise_keyword_matching_score))

            score_list.append((_mem_block, retrieval_score))
            self.logger.debug(f"Retrieval score for block {_mem_block.mem_block_id}: {retrieval_score}")
        
        score_list.sort(key=lambda x: x[1], reverse=True)
        for mb, score in score_list:
            self.logger.debug(f"mb: {mb.mem_block_id} | retrieval score: {score}")
        return [mem_block for mem_block, score in score_list[:top_k]]
    
    def select_relevant_topics(self, new_mem_block: AbstractMemoryBlock, mem_stack: AbstractMemoryStack, top_n: int = 1) -> List[tuple[AbstractMemoryTopic, float]]:
        """
        Select relevant topics for a new memory block using rule-based and semantic similarity as described in the provided algorithm.
        Returns a list of (topic, score) tuples.
        """

        query = new_mem_block.input_query
        query_keywords = set(self._nlp.extract_keywords(query))
        query_word_count = len(self._nlp.tokenize(query))
        self.logger.debug(f"query_word_count: {query_word_count}")
        all_topics = mem_stack.list_of_mem_topics
        last_chat_block = mem_stack.sequence_of_mem_blocks[-1] if mem_stack.sequence_of_mem_blocks else None
        topic_scores = []

        for topic in all_topics:
            self.logger.debug(f"Processing topic {topic.mem_topic_id}")
            topic_keywords = set(topic.identifying_features.get("keywords", []))
            topic_blocks = topic.chain_of_memblocks
            if not topic_blocks:
                continue
            last_block = topic_blocks[-1]
            self.logger.debug(f"Last block of topic {topic.mem_topic_id}: {last_block.mem_block_id}")
            # Rule 1: |q| < 6 and last block is from last chat
            if query_word_count < 6 and last_chat_block and last_block.mem_block_id == last_chat_block.mem_block_id:
                topic_scores.append((topic, 1.0))
                self.logger.debug(f"Rule 1 - topic_scores: {topic_scores}")
                continue

            # Rule 2: Any keyword in query appears only in this topic
            keyword_match_count = len(query_keywords & topic_keywords)
            if keyword_match_count > 0:
                topic_scores.append((topic, float(keyword_match_count)))
                self.logger.debug(f"Rule 2 - topic_scores: {topic_scores}")
                continue

            # Rule 3: Semantic similarity (dot product)
            topic_emb = topic.identifying_features.get("embedding", None)
            if topic_emb is not None:
                query_emb = self._emb_model.encode(query)
                # Ensure both are tensors
                import torch
                topic_emb_tensor = torch.tensor(topic_emb, dtype=torch.float64)
                query_emb_tensor = torch.tensor(query_emb, dtype=torch.float64)
                sim = float(torch.dot(query_emb_tensor, topic_emb_tensor) / (torch.norm(query_emb_tensor) * torch.norm(topic_emb_tensor) + 1e-8))
                topic_scores.append((topic, sim))
            else:
                topic_scores.append((topic, 0.0))
            self.logger.debug(f"Rule 3 - topic_scores: {topic_scores}")

        # Sort by score descending and return top_n (topic, score) tuples
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        if top_n == -1:
            return topic_scores
        return topic_scores[:top_n]
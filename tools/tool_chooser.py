from typing import Any, Dict, List, Union
from torch import Tensor
import concurrent.futures
import torch
from collections import Counter

from base_classes.tool import AbstractTool
from base_classes.embedding import AbstractEmbeddingModel
from configuration.tool_configuration import ToolChooserToolConfiguration
from prompt.zero_shot import ZeroShotPrompt

class ToolChooserTool(AbstractTool):
    """
    This class is used to decide which tool(s) to use for the a given prompt.
    """
    def __init__(self, tool_config: ToolChooserToolConfiguration) -> None:
        super().__init__(tool_config = tool_config)
        self._embbedding_model_ids = getattr(tool_config, 'embbedding', [])
        self._embbedding_models = []
        list_of_initiated_embbedding_models = AbstractEmbeddingModel.get_emb_ids()
        for embbedding_model_id in self._embbedding_model_ids:
            embbedding_model_id = "EMBEDDING | " + embbedding_model_id
            if embbedding_model_id in list_of_initiated_embbedding_models:
                self._embbedding_models.append(AbstractEmbeddingModel.get_emb_instance_by_id(embbedding_model_id))
            else:
                raise ValueError(f"❌ Embedding model {embbedding_model_id} is not initiated.")
        
    def _set_tool_data(self, input_message: ZeroShotPrompt, tools_dict: Dict[str, Tensor], top_k: int = 5) -> None:
        self._data = {'message': input_message.prompt[0]['content'], 'tools_dict': tools_dict, 'top_k': top_k}
    
    def _embed_and_rank_tools(self, embedding_model: AbstractEmbeddingModel, input_message: str, tools_dict: Dict[str, Tensor], top_k: int) -> List[str]:
        """
        Embed the input message and rank tools based on similarity with tool descriptions.
        
        :param embedding_model: The embedding model to use
        :param input_message: The input message to embed
        :param tools_dict: Dictionary mapping tool names to their description embeddings
        :param top_k: Number of top tools to return
        :return: List of top_k tool names ranked by similarity
        """
        # Embed the input message
        input_embedding = embedding_model.encode(input_message)
        input_tensor = torch.tensor(input_embedding, dtype=torch.float32)
        
        # Calculate similarities with all tools
        similarities = {}
        for tool_name, tool_embedding in tools_dict.items():
            # Ensure tool_embedding is a tensor
            if not isinstance(tool_embedding, torch.Tensor):
                tool_embedding = torch.tensor(tool_embedding, dtype=torch.float32)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                input_tensor.unsqueeze(0), 
                tool_embedding.unsqueeze(0), 
                dim=1
            ).item()
            similarities[tool_name] = similarity
        
        # Sort tools by similarity and return top_k
        sorted_tools = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [tool_name for tool_name, _ in sorted_tools[:top_k]]
    
    def execute(self, input_message: ZeroShotPrompt, tools_dict: Dict[str, Tensor], top_k: int = 5) -> List[Dict]:
        """
        This method is used to execute the tool chooser tool, which returns the tool ID to be used for the given prompt.
        
        :param input_message: The input message to process
        :return: A list of dictionaries containing the top-k most voted tools
        :rtype: List[Dict]
        """
        self._set_tool_data(input_message = input_message, tools_dict = tools_dict, top_k = top_k)
        # Get data from _data if available, otherwise use default values
        if hasattr(self, '_data') and self._data:
            tools_dict = self._data.get('tools_dict', {})
            top_k = self._data.get('top_k', 5)
        else:
            tools_dict = {}
            top_k = 5
        
        if not tools_dict:
            self.logger.warning("No tools dictionary provided. Returning empty result.")
            return []
        
        if not self._embbedding_models:
            self.logger.warning("No embedding models available. Returning empty result.")
            return []
        
        # Process each embedding model in parallel
        all_rankings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each embedding model
            future_to_model = {
                executor.submit(self._embed_and_rank_tools, model, input_message, tools_dict, top_k): model 
                for model in self._embbedding_models
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_model):
                try:
                    ranking = future.result()
                    all_rankings.append(ranking)
                    self.logger.debug(f"Completed ranking from embedding model: {future_to_model[future].emb_id}")
                except Exception as exc:
                    self.logger.error(f"Embedding model {future_to_model[future].emb_id} generated an exception: {exc}")
        
        if not all_rankings:
            self.logger.error("No valid rankings generated from any embedding model.")
            return []
        
        # Count votes for each tool across all models
        vote_counts = Counter()
        for ranking in all_rankings:
            for i, tool_name in enumerate(ranking):
                # Weight votes by position (higher position = more votes)
                vote_weight = len(ranking) - i
                vote_counts[tool_name] += vote_weight
        
        # Get top-k tools by vote count
        top_tools = vote_counts.most_common(top_k)
        
        # Convert to the expected format
        result = []
        for tool_name, vote_count in top_tools:
            result.append({
                'tool_name': tool_name,
                'vote_count': vote_count,
                'confidence': vote_count / (len(self._embbedding_models) * top_k)  # Normalize confidence
            })
        
        self.logger.info(f"Selected top {len(result)} tools from {len(self._embbedding_models)} embedding models")
        return result
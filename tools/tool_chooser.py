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
        
    def _set_tool_data(self, input_message: ZeroShotPrompt, tools_dict: Dict[str, Dict[str, Tensor]], top_k: int = 5) -> None:
        self._data = {'message': input_message.prompt[0]['content'], 'tools_dict': tools_dict, 'top_k': top_k}
    
    def _embed_and_rank_tools(self, input_message: str, tools_dict: Dict[str, Dict[str, Tensor]], top_k: int) -> List[str]:
        """
        Embed the input message using all embedding models and rank tools based on similarity with tool descriptions.
        
        :param input_message: The input message to embed
        :param tools_dict: Dictionary mapping tool names to their embedding model embeddings
        :param top_k: Number of top tools to return
        :return: List of top_k tool names ranked by similarity
        """
        # Calculate similarities with all tools using all embedding models in parallel
        similarities = {}
        
        def process_embedding_model(embedding_model):
            """Process a single embedding model and return similarities for all tools."""
            model_similarities = {}
            
            # Embed the input message with this model
            input_embedding = embedding_model.encode(input_message)
            input_tensor = torch.tensor(input_embedding, dtype=torch.float32)
            
            for tool_name, tool_embeddings in tools_dict.items():
                # Get the embedding for this specific model
                if embedding_model.emb_id not in tool_embeddings:
                    self.logger.warning(f"Tool {tool_name} does not have embedding for model {embedding_model.emb_id}")
                    continue
                    
                tool_embedding = tool_embeddings[embedding_model.emb_id]
                
                # Ensure tool_embedding is a tensor
                if not isinstance(tool_embedding, torch.Tensor):
                    tool_embedding = torch.tensor(tool_embedding, dtype=torch.float32)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    input_tensor.unsqueeze(0), 
                    tool_embedding.unsqueeze(0), 
                    dim=1
                ).item()
                
                model_similarities[tool_name] = similarity
            
            return model_similarities
        
        # Process all embedding models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks for each embedding model
            future_to_model = {
                executor.submit(process_embedding_model, model): model 
                for model in self._embbedding_models
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_model):
                try:
                    model_similarities = future.result()
                    # Accumulate similarities across all models
                    for tool_name, similarity in model_similarities.items():
                        if tool_name not in similarities:
                            similarities[tool_name] = []
                        similarities[tool_name].append(similarity)
                    
                    self.logger.debug(f"Completed similarity calculation for embedding model: {future_to_model[future].emb_id}")
                except Exception as exc:
                    self.logger.error(f"Embedding model {future_to_model[future].emb_id} generated an exception: {exc}")
        
        # Average similarities across all models for each tool
        averaged_similarities = {}
        for tool_name, similarity_list in similarities.items():
            if similarity_list:  # Only average if we have at least one similarity score
                averaged_similarities[tool_name] = sum(similarity_list) / len(similarity_list)
        
        # Sort tools by averaged similarity and return top_k
        sorted_tools = sorted(averaged_similarities.items(), key=lambda x: x[1], reverse=True)
        return [tool_name for tool_name, _ in sorted_tools[:top_k]]
    
    def get_tool_emb_dict(self, tool_description_dict: Dict[str, str]) -> Dict[str, Dict[str, Tensor]]:
        """
        This method is used to get the tool embedding dictionary with embeddings from all models.
        
        :param tool_description_dict: Dictionary mapping tool names to their descriptions
        :return: Dictionary mapping tool names to dictionaries of embedding model IDs to tensors
        """
        tool_emb_dict = {}
        
        # Process each tool description in parallel across all embedding models
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create tasks for each tool and embedding model combination
            future_to_task = {}
            
            for tool_name, tool_description in tool_description_dict.items():
                tool_emb_dict[tool_name] = {}
                
                for embedding_model in self._embbedding_models:
                    future = executor.submit(embedding_model.encode, tool_description)
                    future_to_task[future] = (tool_name, embedding_model.emb_id)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    embedding = future.result()
                    tool_name, model_id = future_to_task[future]
                    tool_emb_dict[tool_name][model_id] = torch.tensor(embedding, dtype=torch.float32)
                    self.logger.debug(f"Generated embedding for tool '{tool_name}' using model '{model_id}'")
                except Exception as exc:
                    tool_name, model_id = future_to_task[future]
                    self.logger.error(f"Failed to generate embedding for tool '{tool_name}' using model '{model_id}': {exc}")
        self.logger.debug(f"Tool embedding dictionary: {tool_emb_dict}")
        return tool_emb_dict

    def execute(self, input_message: ZeroShotPrompt, tools_dict: Dict[str, Dict[str, Tensor]], top_k: int = 5) -> List[Dict]:
        """
        This method is used to execute the tool chooser tool, which returns the tool ID to be used for the given prompt.
        
        :param input_message: The input message to process
        :param tools_dict: Dictionary mapping tool names to dictionaries of embedding model IDs to tensors
        :param top_k: Number of top tools to return
        :return: A list of dictionaries containing the top-k most voted tools
        :rtype: List[Dict]
        """
        self._set_tool_data(input_message = input_message, tools_dict = tools_dict, top_k = top_k)
        # Get data from _data if available, otherwise use default values
        if hasattr(self, '_data') and self._data:
            tools_dict = self._data.get('tools_dict', {})
            top_k = self._data.get('top_k', 5)
            input_message_content = self._data.get('message', '')
        else:
            tools_dict = {}
            top_k = 5
            input_message_content = input_message.prompt[0]['content']
        
        if not tools_dict:
            self.logger.warning("No tools dictionary provided. Returning empty result.")
            return []
        
        if not self._embbedding_models:
            self.logger.warning("No embedding models available. Returning empty result.")
            return []
        
        # Get ranking using all embedding models
        try:
            ranking = self._embed_and_rank_tools(input_message_content, tools_dict, top_k)
            self.logger.debug(f"Completed ranking using {len(self._embbedding_models)} embedding models")
        except Exception as exc:
            self.logger.error(f"Failed to generate ranking: {exc}")
            return []
        
        if not ranking:
            self.logger.error("No valid ranking generated.")
            return []
        
        # Convert to the expected format
        result = []
        for i, tool_name in enumerate(ranking):
            # Calculate confidence based on position (higher position = higher confidence)
            confidence = (len(ranking) - i) / len(ranking)
            result.append({
                'tool_name': tool_name,
                'vote_count': len(ranking) - i,  # Weight by position
                'confidence': confidence
            })
        
        self.logger.info(f"Selected top {len(result)} tools using {len(self._embbedding_models)} embedding models")
        return result
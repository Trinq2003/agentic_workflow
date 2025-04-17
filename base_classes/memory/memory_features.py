from typing import Union, Iterable, Optional
from typing_extensions import Literal, Required, TypeAlias, TypedDict

class BlockFeatureForRefinedContent(TypedDict, total=False):
    keywords: Optional[Iterable[str]] = []
    refined_input_embedding: Optional[Iterable[float]]
    refined_output_embedding: Optional[Iterable[float]]
class BlockFeatureForRawContent(TypedDict, total=False):
    keywords: Optional[Iterable[str]] = []
    input_embedding: Optional[Iterable[float]]
    output_embedding: Optional[Iterable[float]]
    context_embedding: Optional[Iterable[float]]
class MemoryBlockFeature(TypedDict, total=False):
    address_in_topic: Optional[int]
    feature_for_refined_context: Optional[BlockFeatureForRefinedContent]
    feature_for_raw_context: Optional[BlockFeatureForRawContent]
    
class MemoryTopicFeature(TypedDict, total=False):
    raw_context_embedding: Optional[Iterable[float]]
    refined_context_embedding: Optional[Iterable[float]]
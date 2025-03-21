from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor

class DataOperationType(Enum):
    MODEL = "MODEL"
    CHAIN = "CHAIN" 
    STATIC_RECOMMENDATION = "STATIC_RECOMMENDATION"

class OperationType(Enum):
    FILTER = "FILTER"
    MERGE = "MERGE"
    RE_RANK = "RE_RANK"
    SWITCH = "SWITCH"

@dataclass
class FilterConfig:
    filter_id: str
    filter_model_id: Optional[str] = None

@dataclass
class Data:
    type: DataOperationType
    operation: Optional[OperationType] = None
    model_id: Optional[str] = None
    filter_config: Optional[FilterConfig] = None
    data: Optional['Data'] = None
    data_a: Optional['Data'] = None
    data_b: Optional['Data'] = None
    data_list: Optional[List['Data']] = None
    metadata_pass_through: Optional[Dict] = None
    audience: Optional[Dict] = None

@dataclass
class ChainResult:
    recommendations: List[Dict]
    metadata: Dict

class ChainNode(ABC):
    def __init__(self, node_id: int):
        self.node_id = node_id
        
    @abstractmethod
    def execute(self) -> ChainResult:
        pass

class AbstractChainService(ABC):
    def __init__(self, model_registry: Any, matcher_service: Any):
        self.log = logging.getLogger(self.__class__.__name__)
        self.model_registry = model_registry
        self.matcher_service = matcher_service

    @abstractmethod
    def create_model_fetch_node(self, request: Any, serving_model: Any, data: Data, node_id: int) -> ChainNode:
        pass

    @abstractmethod
    def create_filter_node(self, filter_id: str, filter_model_id: str, node_id: int, 
                          child_node: ChainNode, request: Any, context: Dict) -> ChainNode:
        pass

    @abstractmethod
    def create_re_rank_node(self, node_id: int, node_a: ChainNode, node_b: ChainNode, request: Any) -> ChainNode:
        pass

    def create_chain(self, request: Any, serving_model: Any, context: Dict) -> ChainNode:
        return self._create_chain(request, serving_model.data, context, 1)

    def _create_chain(self, request: Any, data: Data, context: Dict, current_id: int) -> ChainNode:
        if data.type == DataOperationType.MODEL:
            serving_model = self.model_registry.find_model(data.model_id)
            if serving_model.data:
                return self._create_chain(request, serving_model.data, context, current_id)
            return self.create_model_fetch_node(request, serving_model, data, current_id)

        if data.type == DataOperationType.CHAIN:
            if data.data:
                child_node = self._create_chain(request, data.data, context, current_id + 1)
                return self._create_chain_node_with_single_child(data, child_node, current_id, request, context)
            
            if data.data_a and data.data_b:
                child_a = self._create_chain(request, data.data_a, context, current_id + 1)
                child_b = self._create_chain(request, data.data_b, context, current_id + 2)
                return self._create_chain_node_with_multiple_children(data, child_a, child_b, request, current_id)

            if data.data_list:
                return self._create_chain_node_with_eligible_child(data, request, context, current_id)

            raise ValueError("Data type is CHAIN but data is null")

        raise ValueError(f"Unknown data type: {data.type}")

    def _create_chain_node_with_single_child(self, data: Data, child_node: ChainNode, 
                                           current_id: int, request: Any, context: Dict) -> ChainNode:
        if data.operation == OperationType.FILTER:
            return self.create_filter_node(
                data.filter_config.filter_id,
                data.filter_config.filter_model_id,
                current_id,
                child_node,
                request,
                context
            )
        raise ValueError(f"Unknown operation type: {data.operation}")

    def _check_audience(self, data: Data, context: Dict) -> bool:
        if not data.audience or not data.audience.get("condition"):
            return True
        return self.matcher_service.check_condition(data.audience["condition"], context)

    def _create_chain_node_with_eligible_child(self, data: Data, request: Any, 
                                             context: Dict, current_id: int) -> ChainNode:
        if data.operation == OperationType.SWITCH:
            if not context:
                raise ValueError("Context is null for SWITCH operation")
            for child in data.data_list:
                if self._check_audience(child, context):
                    if data.metadata_pass_through:
                        child.metadata_pass_through = data.metadata_pass_through
                    return self._create_chain(request, child, context, current_id + 1)
            return None
        raise ValueError(f"Unknown operation type: {data.operation}")

    def _create_chain_node_with_multiple_children(self, data: Data, child_a: ChainNode,
                                                child_b: ChainNode, request: Any, current_id: int) -> ChainNode:
        if data.operation == OperationType.RE_RANK:
            return self.create_re_rank_node(current_id, child_a, child_b, request)
        raise ValueError(f"Unknown operation type: {data.operation}")
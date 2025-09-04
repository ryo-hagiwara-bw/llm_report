"""Workflow node definitions for LangGraph integration."""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, List
from enum import Enum


class NodeType(Enum):
    """Types of workflow nodes."""
    LLM_GENERATION = "llm_generation"
    FUNCTION_CALLING = "function_calling"
    DATA_ANALYSIS = "data_analysis"
    CONDITIONAL = "conditional"
    RETRY = "retry"
    ERROR_HANDLING = "error_handling"
    AGGREGATION = "aggregation"


@dataclass
class WorkflowNode:
    """Definition of a workflow node."""
    
    node_id: str
    node_type: NodeType
    name: str
    description: str
    
    # Execution configuration
    handler: Callable[[Dict[str, Any]], Any]
    timeout: float = 30.0
    max_retries: int = 3
    
    # Input/Output configuration
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    
    # Conditional execution
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    # Error handling
    error_handler: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None
    
    # Dependencies
    depends_on: List[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.depends_on is None:
            self.depends_on = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowEdge:
    """Definition of a workflow edge (connection between nodes)."""
    
    from_node: str
    to_node: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    
    workflow_id: str
    name: str
    description: str
    version: str = "1.0.0"
    
    # Graph structure
    nodes: Dict[str, WorkflowNode] = None
    edges: List[WorkflowEdge] = None
    
    # Entry and exit points
    entry_node: str = None
    exit_nodes: List[str] = None
    
    # Global configuration
    max_execution_time: float = 300.0  # 5 minutes
    parallel_execution: bool = False
    
    # Error handling
    global_error_handler: Optional[Callable[[Exception, Dict[str, Any]], Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.nodes is None:
            self.nodes = {}
        if self.edges is None:
            self.edges = []
        if self.exit_nodes is None:
            self.exit_nodes = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the workflow."""
        self.edges.append(edge)
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get dependencies for a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return node.depends_on
    
    def get_next_nodes(self, node_id: str) -> List[str]:
        """Get next nodes from a specific node."""
        next_nodes = []
        for edge in self.edges:
            if edge.from_node == node_id:
                next_nodes.append(edge.to_node)
        return next_nodes

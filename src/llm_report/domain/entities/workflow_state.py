"""Workflow state entity for LangGraph integration."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class NodeStatus(Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeResult:
    """Result of a single node execution."""
    node_id: str
    status: NodeStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """State management for LangGraph workflows."""
    
    # Workflow identification
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.INITIALIZED
    
    # Current execution state
    current_node: Optional[str] = None
    completed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    
    # Data flow
    input_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Any] = None
    
    # Execution tracking
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node_result(self, node_result: NodeResult) -> None:
        """Add a node result to the state."""
        self.node_results[node_result.node_id] = node_result
        self.updated_at = node_result.execution_time
        
        if node_result.status == NodeStatus.COMPLETED:
            self.completed_nodes.append(node_result.node_id)
        elif node_result.status == NodeStatus.FAILED:
            self.failed_nodes.append(node_result.node_id)
    
    def get_node_result(self, node_id: str) -> Optional[NodeResult]:
        """Get a node result by ID."""
        return self.node_results.get(node_id)
    
    def is_workflow_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
    
    def can_retry_node(self, node_id: str) -> bool:
        """Check if a node can be retried."""
        node_result = self.get_node_result(node_id)
        if not node_result:
            return False
        return node_result.retry_count < self.max_retries

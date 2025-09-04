"""LangGraph workflow engine implementation."""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add

from ...domain.entities.workflow_state import WorkflowState, WorkflowStatus, NodeStatus, NodeResult
from ...domain.entities.workflow_node import WorkflowDefinition, WorkflowNode, NodeType

logger = logging.getLogger(__name__)


class WorkflowGraphState(TypedDict):
    """State type for LangGraph workflow."""
    # Input data
    file_path: str
    target_column: str
    chart_type: str
    x_column: str
    y_column: str
    
    # Data flow
    data: Optional[Any]
    validation_results: Optional[Dict[str, Any]]
    basic_statistics: Optional[Dict[str, Any]]
    correlation_analysis: Optional[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    report: Optional[str]
    
    # Execution tracking
    current_step: str
    completed_steps: Annotated[List[str], add]
    failed_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    
    # Metadata
    execution_start_time: float
    execution_end_time: Optional[float]


class LangGraphWorkflowEngine:
    """LangGraph-based workflow execution engine."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.checkpointer = MemorySaver()
        self.active_executions: Dict[str, WorkflowState] = {}
    
    def register_workflow(self, workflow_def: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self.workflows[workflow_def.workflow_id] = workflow_def
        logger.info(f"Registered workflow: {workflow_def.workflow_id}")
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowState:
        """Execute a workflow with the given input data."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_def = self.workflows[workflow_id]
        
        # Create execution ID if not provided
        if not execution_id:
            execution_id = f"{workflow_id}_{int(time.time())}"
        
        # Initialize workflow state
        state = WorkflowState(
            workflow_id=workflow_id,
            input_data=input_data,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.active_executions[execution_id] = state
        
        try:
            # Build LangGraph
            graph = await self._build_langgraph(workflow_def)
            
            # Execute workflow
            result = await graph.ainvoke(
                input_data,
                config={"configurable": {"thread_id": execution_id}}
            )
            
            # Update final state
            state.status = WorkflowStatus.COMPLETED
            state.output_data = result
            state.updated_at = time.time()
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            state.status = WorkflowStatus.FAILED
            state.error_message = str(e)
            state.updated_at = time.time()
        
        return state
    
    async def _build_langgraph(self, workflow_def: WorkflowDefinition) -> StateGraph:
        """Build LangGraph from workflow definition."""
        
        # Create state graph
        graph = StateGraph(Dict[str, Any])
        
        # Add nodes
        for node_id, node in workflow_def.nodes.items():
            graph.add_node(
                node_id,
                self._create_node_handler(node)
            )
        
        # Add edges
        for edge in workflow_def.edges:
            if edge.condition:
                graph.add_conditional_edges(
                    edge.from_node,
                    self._create_condition_handler(edge.condition),
                    {
                        "continue": edge.to_node,
                        "end": END
                    }
                )
            else:
                graph.add_edge(edge.from_node, edge.to_node)
        
        # Set entry point
        if workflow_def.entry_node:
            graph.set_entry_point(workflow_def.entry_node)
        
        # Compile graph
        return graph.compile(checkpointer=self.checkpointer)
    
    def _create_node_handler(self, node: WorkflowNode) -> Callable:
        """Create a handler function for a workflow node."""
        
        async def node_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single workflow node."""
            start_time = time.time()
            
            try:
                logger.info(f"Executing node: {node.node_id}")
                
                # Execute node handler
                if asyncio.iscoroutinefunction(node.handler):
                    result = await node.handler(state)
                else:
                    result = node.handler(state)
                
                # Create node result
                node_result = NodeResult(
                    node_id=node.node_id,
                    status=NodeStatus.COMPLETED,
                    result=result,
                    execution_time=time.time() - start_time
                )
                
                # Update state
                state[f"{node.node_id}_result"] = result
                state[f"{node.node_id}_status"] = "completed"
                
                logger.info(f"Node {node.node_id} completed successfully")
                
                return state
                
            except Exception as e:
                logger.error(f"Node {node.node_id} failed: {e}")
                
                # Create failed node result
                node_result = NodeResult(
                    node_id=node.node_id,
                    status=NodeStatus.FAILED,
                    error=str(e),
                    execution_time=time.time() - start_time
                )
                
                # Update state
                state[f"{node.node_id}_result"] = None
                state[f"{node.node_id}_status"] = "failed"
                state[f"{node.node_id}_error"] = str(e)
                
                # Handle error
                if node.error_handler:
                    try:
                        if asyncio.iscoroutinefunction(node.error_handler):
                            error_result = await node.error_handler(e, state)
                        else:
                            error_result = node.error_handler(e, state)
                        state[f"{node.node_id}_error_result"] = error_result
                    except Exception as error_handler_e:
                        logger.error(f"Error handler for {node.node_id} failed: {error_handler_e}")
                
                return state
        
        return node_handler
    
    def _create_condition_handler(self, condition: Callable) -> Callable:
        """Create a condition handler for conditional edges."""
        
        def condition_handler(state: Dict[str, Any]) -> str:
            """Evaluate condition and return next node."""
            try:
                if condition(state):
                    return "continue"
                else:
                    return "end"
            except Exception as e:
                logger.error(f"Condition evaluation failed: {e}")
                return "end"
        
        return condition_handler
    
    async def retry_failed_node(
        self, 
        execution_id: str, 
        node_id: str
    ) -> bool:
        """Retry a failed node."""
        
        if execution_id not in self.active_executions:
            return False
        
        state = self.active_executions[execution_id]
        node_result = state.get_node_result(node_id)
        
        if not node_result or not state.can_retry_node(node_id):
            return False
        
        # Increment retry count
        node_result.retry_count += 1
        node_result.status = NodeStatus.RETRYING
        
        # Find workflow definition
        workflow_def = self.workflows.get(state.workflow_id)
        if not workflow_def:
            return False
        
        node = workflow_def.nodes.get(node_id)
        if not node:
            return False
        
        try:
            # Retry node execution
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(node.handler):
                result = await node.handler(state.input_data)
            else:
                result = node.handler(state.input_data)
            
            # Update node result
            node_result.status = NodeStatus.COMPLETED
            node_result.result = result
            node_result.execution_time = time.time() - start_time
            
            logger.info(f"Node {node_id} retry successful")
            return True
            
        except Exception as e:
            logger.error(f"Node {node_id} retry failed: {e}")
            node_result.status = NodeStatus.FAILED
            node_result.error = str(e)
            return False
    
    def get_workflow_state(self, execution_id: str) -> Optional[WorkflowState]:
        """Get current state of a workflow execution."""
        return self.active_executions.get(execution_id)
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        if execution_id not in self.active_executions:
            return False
        
        state = self.active_executions[execution_id]
        state.status = WorkflowStatus.CANCELLED
        state.updated_at = time.time()
        
        logger.info(f"Workflow {execution_id} cancelled")
        return True

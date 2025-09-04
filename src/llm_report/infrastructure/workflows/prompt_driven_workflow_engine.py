"""Prompt-driven LangGraph workflow engine for automatic data analysis."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ...application.services.data_loader_service import DataLoaderService
from ...application.services.data_overview_service import DataOverviewService
from ...application.services.llm_function_selection_service import LLMFunctionSelectionService
from ...application.services.dynamic_function_execution_service import DynamicFunctionExecutionService
from ...application.services.advanced_data_analysis_service import AdvancedDataAnalysisService
from ...domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)


class PromptDrivenWorkflowState(TypedDict):
    """State for the prompt-driven workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_prompt: str
    data: Optional[Any]
    data_overview: Optional[Any]
    function_selection: Optional[Any]
    execution_result: Optional[Any]
    current_step: str
    error: Optional[str]
    file_paths: List[str]
    insights: List[str]
    completed_steps: List[str]
    failed_steps: List[str]


class PromptDrivenWorkflowEngine:
    """Prompt-driven workflow engine for automatic data analysis."""
    
    def __init__(
        self, 
        data_loader: DataLoaderService,
        data_overview_service: DataOverviewService,
        llm_function_selection_service: LLMFunctionSelectionService,
        dynamic_function_execution_service: DynamicFunctionExecutionService
    ):
        """Initialize the workflow engine.
        
        Args:
            data_loader: Service for loading data
            data_overview_service: Service for data overview analysis
            llm_function_selection_service: Service for LLM-based function selection
            dynamic_function_execution_service: Service for dynamic function execution
        """
        self.data_loader = data_loader
        self.data_overview_service = data_overview_service
        self.llm_function_selection_service = llm_function_selection_service
        self.dynamic_function_execution_service = dynamic_function_execution_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the prompt-driven workflow."""
        # Create the state graph
        workflow = StateGraph(PromptDrivenWorkflowState)
        
        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("analyze_data_overview", self._analyze_data_overview_node)
        workflow.add_node("select_function", self._select_function_node)
        workflow.add_node("execute_function", self._execute_function_node)
        workflow.add_node("display_results", self._display_results_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow edges
        workflow.set_entry_point("load_data")
        
        # Main flow
        workflow.add_edge("load_data", "analyze_data_overview")
        workflow.add_edge("analyze_data_overview", "select_function")
        workflow.add_edge("select_function", "execute_function")
        workflow.add_edge("execute_function", "display_results")
        workflow.add_edge("display_results", END)
        
        # Error handling
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, user_prompt: str, file_path: str) -> Dict[str, Any]:
        """Execute the prompt-driven workflow.
        
        Args:
            user_prompt: User's natural language prompt
            file_path: Path to data file
            
        Returns:
            Workflow execution result
        """
        try:
            logger.info(f"Starting prompt-driven workflow for: {user_prompt}")
            
            # Initialize state
            initial_state = PromptDrivenWorkflowState(
                messages=[HumanMessage(content=user_prompt)],
                user_prompt=user_prompt,
                data=None,
                data_overview=None,
                function_selection=None,
                execution_result=None,
                current_step="load_data",
                error=None,
                file_paths=[],
                insights=[],
                completed_steps=[],
                failed_steps=[]
            )
            
            # Store file path for use in nodes
            self._file_path = file_path
            
            # Execute the workflow
            result = await self.graph.ainvoke(initial_state)
            
            logger.info(f"Prompt-driven workflow completed successfully")
            
            return {
                "success": True,
                "user_prompt": result.get("user_prompt"),
                "data": result.get("data"),
                "data_overview": result.get("data_overview"),
                "function_selection": result.get("function_selection"),
                "execution_result": result.get("execution_result"),
                "insights": result.get("insights", []),
                "file_paths": result.get("file_paths", []),
                "completed_steps": result.get("completed_steps", []),
                "failed_steps": result.get("failed_steps", []),
                "messages": result.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Prompt-driven workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_prompt": user_prompt,
                "data": None,
                "data_overview": None,
                "function_selection": None,
                "execution_result": None,
                "insights": [],
                "file_paths": [],
                "completed_steps": [],
                "failed_steps": []
            }
    
    async def _load_data_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Load data from file."""
        try:
            logger.info(f"Loading data from {self._file_path}")
            
            data = self.data_loader.load_csv_data(self._file_path)
            
            state["data"] = data
            state["current_step"] = "analyze_data_overview"
            state["completed_steps"].append("load_data")
            state["messages"].append(AIMessage(content=f"データを読み込みました: {data.shape[0]}行, {data.shape[1]}列"))
            
            return state
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("load_data")
            state["current_step"] = "error_handler"
            return state
    
    async def _analyze_data_overview_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Analyze data overview."""
        try:
            data = state["data"]
            logger.info("Analyzing data overview")
            
            # Analyze data overview
            data_overview = self.data_overview_service.analyze_data_overview(data)
            
            state["data_overview"] = data_overview
            state["current_step"] = "select_function"
            state["completed_steps"].append("analyze_data_overview")
            state["messages"].append(AIMessage(content=f"データ概要を分析しました: {data_overview.shape[0]}行, {len(data_overview.columns)}列"))
            
            return state
            
        except Exception as e:
            logger.error(f"Data overview analysis failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("analyze_data_overview")
            state["current_step"] = "error_handler"
            return state
    
    async def _select_function_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Select appropriate function using LLM."""
        try:
            user_prompt = state["user_prompt"]
            data_overview = state["data_overview"]
            logger.info("Selecting function using LLM")
            
            # Format data overview for LLM
            overview_text = self.data_overview_service.format_overview_for_llm(data_overview)
            
            # Available functions
            available_functions = [
                "analyze_by_dimensions",
                "create_comprehensive_report", 
                "analyze_temporal_changes",
                "create_dashboard",
                "get_data_summary"
            ]
            
            # Select function using LLM
            function_selection = await self.llm_function_selection_service.select_function(
                user_prompt, overview_text, available_functions
            )
            
            state["function_selection"] = function_selection
            state["current_step"] = "execute_function"
            state["completed_steps"].append("select_function")
            state["messages"].append(AIMessage(content=f"関数を選択しました: {function_selection.function_name} (信頼度: {function_selection.confidence:.2f})"))
            
            return state
            
        except Exception as e:
            logger.error(f"Function selection failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("select_function")
            state["current_step"] = "error_handler"
            return state
    
    async def _execute_function_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Execute the selected function."""
        try:
            data = state["data"]
            function_selection = state["function_selection"]
            logger.info(f"Executing function: {function_selection.function_name}")
            
            # Execute selected function
            execution_result = self.dynamic_function_execution_service.execute_selected_function(
                data, function_selection
            )
            
            state["execution_result"] = execution_result
            state["current_step"] = "display_results"
            state["completed_steps"].append("execute_function")
            
            if execution_result.success:
                state["messages"].append(AIMessage(content=f"関数を実行しました: {function_selection.function_name} (実行時間: {execution_result.execution_time:.2f}秒)"))
            else:
                state["messages"].append(AIMessage(content=f"関数の実行に失敗しました: {execution_result.error}"))
            
            return state
            
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("execute_function")
            state["current_step"] = "error_handler"
            return state
    
    async def _display_results_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Display execution results."""
        try:
            execution_result = state["execution_result"]
            function_selection = state["function_selection"]
            logger.info("Displaying results")
            
            if execution_result.success:
                # Format results for display
                result_text = self.dynamic_function_execution_service.format_result_for_display(
                    execution_result.result, function_selection.function_name
                )
                
                state["insights"].append(result_text)
                state["current_step"] = "completed"
                state["completed_steps"].append("display_results")
                state["messages"].append(AIMessage(content="結果を表示しました"))
            else:
                state["current_step"] = "error_handler"
                state["failed_steps"].append("display_results")
                state["messages"].append(AIMessage(content=f"結果の表示に失敗しました: {execution_result.error}"))
            
            return state
            
        except Exception as e:
            logger.error(f"Result display failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("display_results")
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: PromptDrivenWorkflowState) -> PromptDrivenWorkflowState:
        """Handle errors in the workflow."""
        logger.error(f"Workflow error: {state.get('error', 'Unknown error')}")
        
        state["messages"].append(AIMessage(content=f"ワークフローが失敗しました: {state.get('error', 'Unknown error')}"))
        
        return state

"""Simple workflow engine with LangGraph integration."""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """Simple workflow state for LangGraph."""
    # Input parameters
    file_path: str
    target_column: str
    chart_type: str
    x_column: str
    y_column: str
    
    # Data flow
    data_shape: Optional[tuple]
    data_columns: Optional[List[str]]
    validation_passed: bool
    statistics_result: Optional[Dict[str, Any]]
    correlation_result: Optional[Dict[str, Any]]
    visualization_result: Optional[Dict[str, Any]]
    final_report: Optional[str]
    
    # Execution tracking
    current_step: str
    completed_steps: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    
    # Metadata
    start_time: float
    end_time: Optional[float]


class SimpleWorkflowEngine:
    """Simple workflow engine using LangGraph."""
    
    def __init__(self, data_analyzer, data_loader, llm_repository):
        self.data_analyzer = data_analyzer
        self.data_loader = data_loader
        self.llm_repository = llm_repository
        self.checkpointer = MemorySaver()
    
    async def execute_data_analysis_workflow(
        self, 
        file_path: str,
        target_column: str = "visitor_count",
        chart_type: str = "bar_chart",
        x_column: str = "area",
        y_column: str = "visitor_count"
    ) -> Dict[str, Any]:
        """Execute data analysis workflow."""
        
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("validate_data", self._validate_data_node)
        workflow.add_node("calculate_statistics", self._calculate_statistics_node)
        workflow.add_node("analyze_correlation", self._analyze_correlation_node)
        workflow.add_node("create_visualization", self._create_visualization_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Add edges
        workflow.add_edge("load_data", "validate_data")
        workflow.add_conditional_edges(
            "validate_data",
            self._should_continue_after_validation,
            {
                "continue": "calculate_statistics",
                "error": "handle_error"
            }
        )
        workflow.add_edge("calculate_statistics", "analyze_correlation")
        workflow.add_edge("analyze_correlation", "create_visualization")
        workflow.add_edge("create_visualization", "generate_report")
        workflow.add_edge("generate_report", END)
        workflow.add_edge("handle_error", END)
        
        # Set entry point
        workflow.set_entry_point("load_data")
        
        # Compile workflow
        compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
        
        # Prepare initial state
        initial_state = {
            "file_path": file_path,
            "target_column": target_column,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "data_shape": None,
            "data_columns": None,
            "validation_passed": False,
            "statistics_result": None,
            "correlation_result": None,
            "visualization_result": None,
            "final_report": None,
            "current_step": "load_data",
            "completed_steps": [],
            "errors": [],
            "start_time": time.time(),
            "end_time": None
        }
        
        try:
            # Execute workflow
            result = await compiled_workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"workflow_{int(time.time())}"}}
            )
            
            # Calculate execution time
            execution_time = time.time() - result["start_time"]
            
            return {
                "success": True,
                "execution_time": execution_time,
                "completed_steps": result["completed_steps"],
                "errors": result["errors"],
                "results": {
                    "statistics": result.get("statistics_result"),
                    "correlation": result.get("correlation_result"),
                    "visualization": result.get("visualization_result"),
                    "report": result.get("final_report")
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - initial_state["start_time"]
            }
    
    async def _load_data_node(self, state: WorkflowState) -> WorkflowState:
        """Load data from CSV file."""
        try:
            logger.info("Loading data...")
            data = self.data_loader.load_csv_data(state["file_path"])
            
            return {
                **state,
                "data_shape": data.shape,
                "data_columns": list(data.columns),
                "current_step": "validate_data",
                "completed_steps": state["completed_steps"] + ["load_data"]
            }
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return {
                **state,
                "current_step": "handle_error",
                "errors": state["errors"] + [f"Data loading failed: {str(e)}"]
            }
    
    async def _validate_data_node(self, state: WorkflowState) -> WorkflowState:
        """Validate loaded data."""
        try:
            logger.info("Validating data...")
            data = state["data"]
            
            # Basic validation
            if data is None or len(data) == 0:
                raise ValueError("No data loaded")
            
            if len(data) < 10:
                raise ValueError("Insufficient data: less than 10 rows")
            
            validation_passed = True
            
            return {
                **state,
                "validation_passed": validation_passed,
                "current_step": "calculate_statistics",
                "completed_steps": state["completed_steps"] + ["validate_data"]
            }
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                **state,
                "validation_passed": False,
                "current_step": "handle_error",
                "errors": state["errors"] + [f"Data validation failed: {str(e)}"]
            }
    
    async def _calculate_statistics_node(self, state: WorkflowState) -> WorkflowState:
        """Calculate basic statistics."""
        try:
            logger.info("Calculating statistics...")
            data = state["data"]
            target_column = state["target_column"]
            
            stats = await self.data_analyzer.calculate_basic_statistics(data, target_column)
            
            statistics_result = {
                "count": stats.count,
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "median": stats.median,
                "q25": stats.q25,
                "q75": stats.q75
            }
            
            return {
                **state,
                "statistics_result": statistics_result,
                "current_step": "analyze_correlation",
                "completed_steps": state["completed_steps"] + ["calculate_statistics"]
            }
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {
                **state,
                "current_step": "analyze_correlation",  # Continue even if statistics fail
                "errors": state["errors"] + [f"Statistics calculation failed: {str(e)}"]
            }
    
    async def _analyze_correlation_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze correlation between variables."""
        try:
            logger.info("Analyzing correlations...")
            data = state["data"]
            
            corr_result = await self.data_analyzer.calculate_correlation_matrix(data)
            
            correlation_result = {
                "correlation_matrix": corr_result.correlation_matrix.to_dict(),
                "significant_correlations": corr_result.significant_correlations
            }
            
            return {
                **state,
                "correlation_result": correlation_result,
                "current_step": "create_visualization",
                "completed_steps": state["completed_steps"] + ["analyze_correlation"]
            }
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {
                **state,
                "current_step": "create_visualization",  # Continue even if correlation fails
                "errors": state["errors"] + [f"Correlation analysis failed: {str(e)}"]
            }
    
    async def _create_visualization_node(self, state: WorkflowState) -> WorkflowState:
        """Create data visualization."""
        try:
            logger.info("Creating visualization...")
            data = state["data"]
            chart_type = state["chart_type"]
            x_column = state["x_column"]
            y_column = state["y_column"]
            
            viz_result = await self.data_analyzer.create_visualization(
                data, chart_type, x_column, y_column
            )
            
            visualization_result = {
                "chart_type": viz_result.chart_type,
                "description": viz_result.description,
                "file_path": viz_result.file_path,
                "data_summary": viz_result.data_summary
            }
            
            return {
                **state,
                "visualization_result": visualization_result,
                "current_step": "generate_report",
                "completed_steps": state["completed_steps"] + ["create_visualization"]
            }
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {
                **state,
                "current_step": "generate_report",  # Continue even if visualization fails
                "errors": state["errors"] + [f"Visualization creation failed: {str(e)}"]
            }
    
    async def _generate_report_node(self, state: WorkflowState) -> WorkflowState:
        """Generate final analysis report."""
        try:
            logger.info("Generating report...")
            
            # Collect results
            results = {
                "statistics": state.get("statistics_result"),
                "correlation": state.get("correlation_result"),
                "visualization": state.get("visualization_result")
            }
            
            # Generate report using LLM
            prompt = f"""
            以下のデータ分析結果を基に、包括的なレポートを生成してください：
            
            基本統計: {results.get('statistics', {})}
            相関分析: {results.get('correlation', {})}
            可視化: {results.get('visualization', {})}
            
            レポートには以下の要素を含めてください：
            1. データの概要
            2. 主要な発見事項
            3. 統計的洞察
            4. 推奨事項
            """
            
            response = await self.llm_repository.generate_content(prompt)
            final_report = response.content
            
            return {
                **state,
                "final_report": final_report,
                "current_step": "completed",
                "completed_steps": state["completed_steps"] + ["generate_report"],
                "end_time": time.time()
            }
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            # Fallback to simple summary
            final_report = f"Analysis completed with {len([r for r in state.values() if r is not None])} components"
            return {
                **state,
                "final_report": final_report,
                "current_step": "completed",
                "completed_steps": state["completed_steps"] + ["generate_report"],
                "end_time": time.time(),
                "errors": state["errors"] + [f"Report generation failed: {str(e)}"]
            }
    
    async def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors."""
        logger.error("Handling workflow error")
        return {
            **state,
            "current_step": "error",
            "end_time": time.time()
        }
    
    def _should_continue_after_validation(self, state: WorkflowState) -> str:
        """Determine if workflow should continue after validation."""
        if state.get("validation_passed", False):
            return "continue"
        else:
            return "error"

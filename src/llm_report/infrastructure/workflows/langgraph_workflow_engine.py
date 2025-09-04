"""LangGraph-based workflow engine for advanced data analysis."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from ...application.services.data_loader_service import DataLoaderService
from ...application.services.advanced_data_analysis_service import AdvancedDataAnalysisService

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    data: Optional[Any]
    analysis_results: Dict[str, Any]
    current_step: str
    error: Optional[str]
    file_paths: List[str]
    insights: List[str]
    completed_steps: List[str]
    failed_steps: List[str]


@dataclass
class WorkflowConfig:
    """Configuration for the workflow."""
    file_path: str
    target_metrics: List[str]
    key_dimensions: List[str]
    analysis_types: List[str]
    filters: Optional[Dict[str, Any]] = None
    report_type: str = "full"
    export_format: str = "excel"


class LangGraphWorkflowEngine:
    """LangGraph-based workflow engine for data analysis."""
    
    def __init__(self, data_loader: DataLoaderService, advanced_analyzer: AdvancedDataAnalysisService):
        """Initialize the workflow engine.
        
        Args:
            data_loader: Service for loading data
            advanced_analyzer: Service for advanced analysis
        """
        self.data_loader = data_loader
        self.advanced_analyzer = advanced_analyzer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("validate_data", self._validate_data_node)
        workflow.add_node("analyze_dimensions", self._analyze_dimensions_node)
        workflow.add_node("temporal_analysis", self._temporal_analysis_node)
        workflow.add_node("create_visualizations", self._create_visualizations_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("create_report", self._create_report_node)
        workflow.add_node("export_results", self._export_results_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow edges
        workflow.set_entry_point("load_data")
        
        # Main flow
        workflow.add_edge("load_data", "validate_data")
        workflow.add_edge("validate_data", "analyze_dimensions")
        workflow.add_edge("analyze_dimensions", "temporal_analysis")
        workflow.add_edge("temporal_analysis", "create_visualizations")
        workflow.add_edge("create_visualizations", "generate_insights")
        workflow.add_edge("generate_insights", "create_report")
        workflow.add_edge("create_report", "export_results")
        workflow.add_edge("export_results", END)
        
        # Error handling
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Execute the complete workflow.
        
        Args:
            config: Workflow configuration
            
        Returns:
            Workflow execution result
        """
        try:
            logger.info(f"Starting LangGraph workflow for {config.file_path}")
            
            # Store config for use in nodes
            self._current_config = config
            
            # Initialize state
            initial_state = WorkflowState(
                messages=[HumanMessage(content=f"Analyze data from {config.file_path}")],
                data=None,
                analysis_results={},
                current_step="load_data",
                error=None,
                file_paths=[],
                insights=[],
                completed_steps=[],
                failed_steps=[]
            )
            
            # Execute the workflow
            result = await self.graph.ainvoke(initial_state)
            
            logger.info(f"LangGraph workflow completed successfully")
            
            return {
                "success": True,
                "data": result.get("data"),
                "analysis_results": result.get("analysis_results", {}),
                "insights": result.get("insights", []),
                "file_paths": result.get("file_paths", []),
                "completed_steps": result.get("completed_steps", []),
                "failed_steps": result.get("failed_steps", []),
                "messages": result.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"LangGraph workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "analysis_results": {},
                "insights": [],
                "file_paths": [],
                "completed_steps": [],
                "failed_steps": []
            }
    
    async def _load_data_node(self, state: WorkflowState) -> WorkflowState:
        """Load data from file."""
        try:
            # Get config from state or use default
            workflow_config = getattr(self, '_current_config', None)
            if not workflow_config:
                logger.error("No workflow config available")
                state["error"] = "No workflow config available"
                state["current_step"] = "error_handler"
                return state
            
            logger.info(f"Loading data from {workflow_config.file_path}")
            
            data = self.data_loader.load_csv_data(workflow_config.file_path)
            
            state["data"] = data
            state["current_step"] = "validate_data"
            state["completed_steps"].append("load_data")
            state["messages"].append(AIMessage(content=f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"))
            
            return state
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("load_data")
            state["current_step"] = "error_handler"
            return state
    
    async def _validate_data_node(self, state: WorkflowState) -> WorkflowState:
        """Validate the loaded data."""
        try:
            data = state["data"]
            logger.info("Validating data")
            
            # Basic validation
            validation_results = {
                "shape": data.shape,
                "columns": list(data.columns),
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.to_dict()
            }
            
            state["analysis_results"]["validation"] = validation_results
            state["current_step"] = "analyze_dimensions"
            state["completed_steps"].append("validate_data")
            state["messages"].append(AIMessage(content=f"Data validation completed: {data.shape[0]} rows, {data.shape[1]} columns"))
            
            return state
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("validate_data")
            state["current_step"] = "error_handler"
            return state
    
    async def _analyze_dimensions_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze data by dimensions."""
        try:
            data = state["data"]
            workflow_config = self._current_config
            logger.info("Analyzing data by dimensions")
            
            # Apply filters to data
            filtered_data = data.copy()
            if workflow_config.filters:
                for key, value in workflow_config.filters.items():
                    if key in filtered_data.columns:
                        filtered_data = filtered_data[filtered_data[key] == value]
                        logger.info(f"Applied filter {key}={value}: {len(filtered_data)} rows remaining")
            
            dimension_results = []
            
            # Analyze each metric and dimension combination
            for metric in workflow_config.target_metrics:
                if metric in filtered_data.columns:
                    for dimension in workflow_config.key_dimensions:
                        if dimension in filtered_data.columns:
                            result = self.advanced_analyzer.analyze_by_dimensions(
                                data=filtered_data,
                                target_metric=metric,
                                group_by=[dimension],
                                filters=None  # Already filtered above
                            )
                            dimension_results.append(result)
            
            # Store filtered data for further analysis
            state["data"] = filtered_data
            state["analysis_results"]["dimension_analysis"] = dimension_results
            state["current_step"] = "temporal_analysis"
            state["completed_steps"].append("analyze_dimensions")
            state["messages"].append(AIMessage(content=f"Dimension analysis completed: {len(dimension_results)} analyses on {len(filtered_data)} filtered rows"))
            
            return state
            
        except Exception as e:
            logger.error(f"Dimension analysis failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("analyze_dimensions")
            state["current_step"] = "error_handler"
            return state
    
    async def _temporal_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Perform temporal analysis."""
        try:
            data = state["data"]
            workflow_config = self._current_config
            logger.info("Performing temporal analysis")
            
            temporal_results = []
            
            # Perform temporal analysis for each metric
            for metric in workflow_config.target_metrics:
                if metric in data.columns:
                    result = self.advanced_analyzer.analyze_temporal_changes(
                        data=data,
                        target_metric=metric,
                        group_by=workflow_config.key_dimensions,
                        time_dimension="period"
                    )
                    temporal_results.append(result)
            
            state["analysis_results"]["temporal_analysis"] = temporal_results
            state["current_step"] = "create_visualizations"
            state["completed_steps"].append("temporal_analysis")
            state["messages"].append(AIMessage(content=f"Temporal analysis completed: {len(temporal_results)} analyses"))
            
            return state
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("temporal_analysis")
            state["current_step"] = "error_handler"
            return state
    
    async def _create_visualizations_node(self, state: WorkflowState) -> WorkflowState:
        """Create visualizations."""
        try:
            data = state["data"]
            workflow_config = self._current_config
            logger.info("Creating visualizations")
            
            visualization_paths = []
            
            # Create visualizations for each metric and dimension
            for metric in workflow_config.target_metrics:
                if metric in data.columns:
                    for dimension in workflow_config.key_dimensions:
                        if dimension in data.columns:
                            viz_path = self.advanced_analyzer._create_dimension_visualization(
                                data, metric, dimension, f"langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )
                            if viz_path:
                                visualization_paths.append(viz_path)
            
            state["file_paths"].extend(visualization_paths)
            state["current_step"] = "generate_insights"
            state["completed_steps"].append("create_visualizations")
            state["messages"].append(AIMessage(content=f"Visualizations created: {len(visualization_paths)} files"))
            
            return state
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("create_visualizations")
            state["current_step"] = "error_handler"
            return state
    
    async def _generate_insights_node(self, state: WorkflowState) -> WorkflowState:
        """Generate insights from analysis results."""
        try:
            analysis_results = state["analysis_results"]
            logger.info("Generating insights")
            
            insights = []
            
            # Generate insights from dimension analysis
            if "dimension_analysis" in analysis_results:
                for result in analysis_results["dimension_analysis"]:
                    insights.extend(result.insights)
            
            # Generate insights from temporal analysis
            if "temporal_analysis" in analysis_results:
                for result in analysis_results["temporal_analysis"]:
                    insights.extend(result.insights)
            
            # Add general insights
            data = state["data"]
            if data is not None:
                insights.append(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns")
                insights.append(f"Analysis completed for {len(analysis_results.get('dimension_analysis', []))} dimension combinations")
            
            state["insights"] = insights
            state["current_step"] = "create_report"
            state["completed_steps"].append("generate_insights")
            state["messages"].append(AIMessage(content=f"Insights generated: {len(insights)} insights"))
            
            return state
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("generate_insights")
            state["current_step"] = "error_handler"
            return state
    
    async def _create_report_node(self, state: WorkflowState) -> WorkflowState:
        """Create comprehensive report."""
        try:
            data = state["data"]
            workflow_config = self._current_config
            logger.info("Creating comprehensive report")
            
            # Create comprehensive report
            report = self.advanced_analyzer.create_comprehensive_report(
                data=data,
                target_metrics=workflow_config.target_metrics,
                key_dimensions=workflow_config.key_dimensions
            )
            
            if report:
                state["file_paths"].extend(report.file_paths)
                state["analysis_results"]["comprehensive_report"] = report
                state["current_step"] = "export_results"
                state["completed_steps"].append("create_report")
                state["messages"].append(AIMessage(content=f"Comprehensive report created: {len(report.file_paths)} files"))
            
            return state
            
        except Exception as e:
            logger.error(f"Report creation failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("create_report")
            state["current_step"] = "error_handler"
            return state
    
    async def _export_results_node(self, state: WorkflowState) -> WorkflowState:
        """Export results."""
        try:
            workflow_config = self._current_config
            logger.info("Exporting results")
            
            # Export results if requested
            if workflow_config.export_format:
                export_path = self.advanced_analyzer.export_analysis_results(
                    state["analysis_results"].get("comprehensive_report"),
                    workflow_config.export_format
                )
                if export_path:
                    state["file_paths"].append(export_path)
            
            state["current_step"] = "completed"
            state["completed_steps"].append("export_results")
            state["messages"].append(AIMessage(content=f"Results exported: {workflow_config.export_format} format"))
            
            return state
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("export_results")
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        logger.error(f"Workflow error: {state.get('error', 'Unknown error')}")
        
        state["messages"].append(AIMessage(content=f"Workflow failed: {state.get('error', 'Unknown error')}"))
        
        return state

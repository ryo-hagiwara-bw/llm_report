"""Data analysis workflow implementation using LangGraph."""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from ...domain.entities.workflow_node import WorkflowDefinition, WorkflowNode, NodeType, WorkflowEdge
from ...domain.entities.workflow_state import WorkflowState
from ..workflows.langgraph_workflow_engine import LangGraphWorkflowEngine
from ..repositories.vertex_ai_llm_repository import VertexAILLMRepository
from ..services.data_analysis_service import DataAnalysisService
from ..services.data_loader_service import DataLoaderService

logger = logging.getLogger(__name__)


class DataAnalysisWorkflowBuilder:
    """Builder for data analysis workflows."""
    
    def __init__(
        self,
        llm_repository: VertexAILLMRepository,
        data_analyzer: DataAnalysisService,
        data_loader: DataLoaderService
    ):
        self.llm_repository = llm_repository
        self.data_analyzer = data_analyzer
        self.data_loader = data_loader
        self.workflow_engine = LangGraphWorkflowEngine()
    
    def create_comprehensive_analysis_workflow(self) -> WorkflowDefinition:
        """Create a comprehensive data analysis workflow."""
        
        workflow = WorkflowDefinition(
            workflow_id="comprehensive_data_analysis",
            name="Comprehensive Data Analysis Workflow",
            description="Complete data analysis workflow with retry and error handling",
            version="1.0.0",
            entry_node="load_data",
            exit_nodes=["generate_report", "error_handler"]
        )
        
        # Add nodes
        workflow.add_node(self._create_data_loading_node())
        workflow.add_node(self._create_data_validation_node())
        workflow.add_node(self._create_basic_statistics_node())
        workflow.add_node(self._create_correlation_analysis_node())
        workflow.add_node(self._create_visualization_node())
        workflow.add_node(self._create_report_generation_node())
        workflow.add_node(self._create_error_handler_node())
        workflow.add_node(self._create_retry_node())
        
        # Add edges
        workflow.add_edge(WorkflowEdge("load_data", "data_validation"))
        workflow.add_edge(WorkflowEdge("data_validation", "basic_statistics"))
        workflow.add_edge(WorkflowEdge("basic_statistics", "correlation_analysis"))
        workflow.add_edge(WorkflowEdge("correlation_analysis", "visualization"))
        workflow.add_edge(WorkflowEdge("visualization", "generate_report"))
        
        # Conditional edges for error handling
        workflow.add_edge(WorkflowEdge("data_validation", "error_handler", 
                                     condition=lambda state: state.get("data_validation_status") == "failed"))
        workflow.add_edge(WorkflowEdge("basic_statistics", "retry", 
                                     condition=lambda state: state.get("basic_statistics_status") == "failed"))
        workflow.add_edge(WorkflowEdge("correlation_analysis", "retry", 
                                     condition=lambda state: state.get("correlation_analysis_status") == "failed"))
        workflow.add_edge(WorkflowEdge("visualization", "retry", 
                                     condition=lambda state: state.get("visualization_status") == "failed"))
        
        # Retry edges
        workflow.add_edge(WorkflowEdge("retry", "basic_statistics"))
        workflow.add_edge(WorkflowEdge("retry", "correlation_analysis"))
        workflow.add_edge(WorkflowEdge("retry", "visualization"))
        
        return workflow
    
    def _create_data_loading_node(self) -> WorkflowNode:
        """Create data loading node."""
        
        async def load_data_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            file_path = state.get("file_path")
            if not file_path:
                raise ValueError("file_path is required")
            
            try:
                data = await self.data_loader.load_csv(file_path)
                return {
                    "data": data,
                    "data_loaded": True,
                    "data_shape": data.shape,
                    "data_columns": list(data.columns)
                }
            except Exception as e:
                logger.error(f"Data loading failed: {e}")
                raise
        
        return WorkflowNode(
            node_id="load_data",
            node_type=NodeType.DATA_ANALYSIS,
            name="Load Data",
            description="Load CSV data from file",
            handler=load_data_handler,
            timeout=60.0,
            max_retries=2
        )
    
    def _create_data_validation_node(self) -> WorkflowNode:
        """Create data validation node."""
        
        async def validate_data_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            data = state.get("data")
            if data is None:
                raise ValueError("No data loaded")
            
            # Basic validation
            validation_results = {
                "row_count": len(data),
                "column_count": len(data.columns),
                "has_nulls": data.isnull().any().any(),
                "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": data.select_dtypes(include=['object']).columns.tolist()
            }
            
            # Check for minimum data requirements
            if validation_results["row_count"] < 10:
                raise ValueError("Insufficient data: less than 10 rows")
            
            return {
                "validation_results": validation_results,
                "data_valid": True
            }
        
        return WorkflowNode(
            node_id="data_validation",
            node_type=NodeType.DATA_ANALYSIS,
            name="Validate Data",
            description="Validate loaded data",
            handler=validate_data_handler,
            timeout=30.0,
            max_retries=1
        )
    
    def _create_basic_statistics_node(self) -> WorkflowNode:
        """Create basic statistics node."""
        
        async def basic_statistics_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            data = state.get("data")
            target_column = state.get("target_column", "visitor_count")
            
            if target_column not in data.columns:
                raise ValueError(f"Target column {target_column} not found")
            
            try:
                stats = await self.data_analyzer.calculate_basic_statistics(data, target_column)
                return {
                    "basic_statistics": {
                        "count": stats.count,
                        "mean": stats.mean,
                        "std": stats.std,
                        "min": stats.min,
                        "max": stats.max,
                        "median": stats.median,
                        "q25": stats.q25,
                        "q75": stats.q75
                    }
                }
            except Exception as e:
                logger.error(f"Basic statistics calculation failed: {e}")
                raise
        
        return WorkflowNode(
            node_id="basic_statistics",
            node_type=NodeType.DATA_ANALYSIS,
            name="Basic Statistics",
            description="Calculate basic statistics",
            handler=basic_statistics_handler,
            timeout=30.0,
            max_retries=2
        )
    
    def _create_correlation_analysis_node(self) -> WorkflowNode:
        """Create correlation analysis node."""
        
        async def correlation_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            data = state.get("data")
            
            try:
                corr_result = await self.data_analyzer.calculate_correlation_matrix(data)
                return {
                    "correlation_matrix": corr_result.correlation_matrix.to_dict(),
                    "significant_correlations": corr_result.significant_correlations
                }
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
                raise
        
        return WorkflowNode(
            node_id="correlation_analysis",
            node_type=NodeType.DATA_ANALYSIS,
            name="Correlation Analysis",
            description="Calculate correlation matrix",
            handler=correlation_handler,
            timeout=45.0,
            max_retries=2
        )
    
    def _create_visualization_node(self) -> WorkflowNode:
        """Create visualization node."""
        
        async def visualization_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            data = state.get("data")
            chart_type = state.get("chart_type", "bar_chart")
            x_column = state.get("x_column", "area")
            y_column = state.get("y_column", "visitor_count")
            
            try:
                viz_result = await self.data_analyzer.create_visualization(
                    data, chart_type, x_column, y_column
                )
                return {
                    "visualization": {
                        "chart_type": viz_result.chart_type,
                        "description": viz_result.description,
                        "file_path": viz_result.file_path,
                        "data_summary": viz_result.data_summary
                    }
                }
            except Exception as e:
                logger.error(f"Visualization creation failed: {e}")
                raise
        
        return WorkflowNode(
            node_id="visualization",
            node_type=NodeType.DATA_ANALYSIS,
            name="Create Visualization",
            description="Create data visualization",
            handler=visualization_handler,
            timeout=60.0,
            max_retries=2
        )
    
    def _create_report_generation_node(self) -> WorkflowNode:
        """Create report generation node."""
        
        async def report_generation_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            # Collect all analysis results
            analysis_results = {
                "basic_statistics": state.get("basic_statistics"),
                "correlation_analysis": state.get("correlation_analysis"),
                "visualization": state.get("visualization"),
                "validation_results": state.get("validation_results")
            }
            
            # Generate summary report using LLM
            try:
                prompt = f"""
                以下のデータ分析結果を基に、包括的なレポートを生成してください：
                
                基本統計: {analysis_results.get('basic_statistics', {})}
                相関分析: {analysis_results.get('correlation_analysis', {})}
                可視化: {analysis_results.get('visualization', {})}
                データ検証: {analysis_results.get('validation_results', {})}
                
                レポートには以下の要素を含めてください：
                1. データの概要
                2. 主要な発見事項
                3. 統計的洞察
                4. 推奨事項
                """
                
                response = await self.llm_repository.generate_content(prompt)
                
                return {
                    "report": response.content,
                    "analysis_complete": True
                }
                
            except Exception as e:
                logger.error(f"Report generation failed: {e}")
                # Fallback to simple summary
                return {
                    "report": f"Analysis completed with {len(analysis_results)} components",
                    "analysis_complete": True,
                    "error": str(e)
                }
        
        return WorkflowNode(
            node_id="generate_report",
            node_type=NodeType.LLM_GENERATION,
            name="Generate Report",
            description="Generate comprehensive analysis report",
            handler=report_generation_handler,
            timeout=120.0,
            max_retries=1
        )
    
    def _create_error_handler_node(self) -> WorkflowNode:
        """Create error handler node."""
        
        async def error_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            error_message = state.get("error_message", "Unknown error")
            
            return {
                "error_handled": True,
                "error_summary": f"Workflow failed: {error_message}",
                "workflow_status": "failed"
            }
        
        return WorkflowNode(
            node_id="error_handler",
            node_type=NodeType.ERROR_HANDLING,
            name="Error Handler",
            description="Handle workflow errors",
            handler=error_handler,
            timeout=10.0,
            max_retries=0
        )
    
    def _create_retry_node(self) -> WorkflowNode:
        """Create retry node."""
        
        async def retry_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            # Determine which node to retry based on state
            failed_nodes = []
            for key, value in state.items():
                if key.endswith("_status") and value == "failed":
                    node_id = key.replace("_status", "")
                    failed_nodes.append(node_id)
            
            if failed_nodes:
                # Retry the first failed node
                node_to_retry = failed_nodes[0]
                return {
                    "retry_node": node_to_retry,
                    "retry_attempted": True
                }
            else:
                return {
                    "retry_node": None,
                    "retry_attempted": False
                }
        
        return WorkflowNode(
            node_id="retry",
            node_type=NodeType.RETRY,
            name="Retry Failed Node",
            description="Retry failed operations",
            handler=retry_handler,
            timeout=30.0,
            max_retries=0
        )
    
    async def execute_comprehensive_analysis(
        self, 
        file_path: str,
        target_column: str = "visitor_count",
        chart_type: str = "bar_chart",
        x_column: str = "area",
        y_column: str = "visitor_count"
    ) -> WorkflowState:
        """Execute comprehensive data analysis workflow."""
        
        # Create workflow
        workflow = self.create_comprehensive_analysis_workflow()
        
        # Register workflow
        self.workflow_engine.register_workflow(workflow)
        
        # Prepare input data
        input_data = {
            "file_path": file_path,
            "target_column": target_column,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column
        }
        
        # Execute workflow
        return await self.workflow_engine.execute_workflow(
            "comprehensive_data_analysis",
            input_data
        )

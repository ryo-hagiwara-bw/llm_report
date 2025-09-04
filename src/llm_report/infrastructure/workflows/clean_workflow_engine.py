"""Clean workflow engine with proper error handling and retry logic."""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of a workflow step."""
    step_name: str
    status: StepStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    execution_time: float
    completed_steps: List[str]
    failed_steps: List[str]
    results: Dict[str, Any]
    errors: List[str]


class CleanWorkflowEngine:
    """Clean workflow engine with retry and error handling."""
    
    def __init__(self, data_analyzer, data_loader, llm_repository):
        self.data_analyzer = data_analyzer
        self.data_loader = data_loader
        self.llm_repository = llm_repository
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def execute_data_analysis_workflow(
        self, 
        file_path: str,
        target_column: str = "visitor_count",
        chart_type: str = "bar_chart",
        x_column: str = "area",
        y_column: str = "visitor_count"
    ) -> WorkflowResult:
        """Execute data analysis workflow with proper error handling."""
        
        start_time = time.time()
        completed_steps = []
        failed_steps = []
        results = {}
        errors = []
        
        try:
            # Step 1: Load data
            data = await self._execute_step_with_retry(
                "load_data",
                self._load_data_step,
                file_path
            )
            completed_steps.append("load_data")
            results["data_info"] = {
                "shape": data.shape,
                "columns": list(data.columns)
            }
            
            # Step 2: Validate data
            validation_result = await self._execute_step_with_retry(
                "validate_data",
                self._validate_data_step,
                data
            )
            completed_steps.append("validate_data")
            results["validation"] = validation_result
            
            # Step 3: Calculate statistics
            try:
                stats = await self._execute_step_with_retry(
                    "calculate_statistics",
                    self._calculate_statistics_step,
                    data, target_column
                )
                completed_steps.append("calculate_statistics")
                results["statistics"] = {
                    "count": stats.count,
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "median": stats.median,
                    "q25": stats.q25,
                    "q75": stats.q75
                }
            except Exception as e:
                failed_steps.append("calculate_statistics")
                errors.append(f"Statistics calculation failed: {str(e)}")
            
            # Step 4: Analyze correlation
            try:
                corr_result = await self._execute_step_with_retry(
                    "analyze_correlation",
                    self._analyze_correlation_step,
                    data
                )
                completed_steps.append("analyze_correlation")
                results["correlation"] = {
                    "correlation_matrix": corr_result.correlation_matrix.to_dict(),
                    "significant_correlations": corr_result.significant_correlations
                }
            except Exception as e:
                failed_steps.append("analyze_correlation")
                errors.append(f"Correlation analysis failed: {str(e)}")
            
            # Step 5: Create visualization
            try:
                viz_result = await self._execute_step_with_retry(
                    "create_visualization",
                    self._create_visualization_step,
                    data, chart_type, x_column, y_column
                )
                completed_steps.append("create_visualization")
                results["visualization"] = {
                    "chart_type": viz_result.chart_type,
                    "description": viz_result.description,
                    "file_path": viz_result.file_path,
                    "data_summary": viz_result.data_summary
                }
            except Exception as e:
                failed_steps.append("create_visualization")
                errors.append(f"Visualization creation failed: {str(e)}")
            
            # Step 6: Generate report
            try:
                report = await self._execute_step_with_retry(
                    "generate_report",
                    self._generate_report_step,
                    results
                )
                completed_steps.append("generate_report")
                results["report"] = report
            except Exception as e:
                failed_steps.append("generate_report")
                errors.append(f"Report generation failed: {str(e)}")
            
            # Determine overall success
            success = len(failed_steps) == 0 or len(completed_steps) > len(failed_steps)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            errors.append(f"Workflow execution failed: {str(e)}")
            success = False
        
        execution_time = time.time() - start_time
        
        return WorkflowResult(
            success=success,
            execution_time=execution_time,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            results=results,
            errors=errors
        )
    
    async def _execute_step_with_retry(
        self, 
        step_name: str, 
        step_func: callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute a step with retry logic."""
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Executing {step_name} (attempt {attempt + 1})")
                result = await step_func(*args, **kwargs)
                logger.info(f"{step_name} completed successfully")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"{step_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying {step_name} in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"{step_name} failed after {self.max_retries + 1} attempts")
                    raise e
        
        raise last_error
    
    async def _load_data_step(self, file_path: str):
        """Load data from CSV file."""
        return self.data_loader.load_csv_data(file_path)
    
    async def _validate_data_step(self, data):
        """Validate loaded data."""
        if data is None or len(data) == 0:
            raise ValueError("No data loaded")
        
        if len(data) < 10:
            raise ValueError("Insufficient data: less than 10 rows")
        
        return {
            "row_count": len(data),
            "column_count": len(data.columns),
            "has_nulls": data.isnull().any().any(),
            "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist()
        }
    
    async def _calculate_statistics_step(self, data, target_column: str):
        """Calculate basic statistics."""
        return self.data_analyzer.calculate_basic_statistics(data, target_column)
    
    async def _analyze_correlation_step(self, data):
        """Analyze correlation between variables."""
        return self.data_analyzer.calculate_correlation_matrix(data)
    
    async def _create_visualization_step(self, data, chart_type: str, x_column: str, y_column: str):
        """Create data visualization."""
        if chart_type == "bar_chart":
            return self.data_analyzer.create_bar_chart(data, x_column, y_column)
        elif chart_type == "scatter_plot":
            return self.data_analyzer.create_scatter_plot(data, x_column, y_column)
        elif chart_type == "histogram":
            return self.data_analyzer.create_histogram(data, y_column)
        else:
            # Default to bar chart
            return self.data_analyzer.create_bar_chart(data, x_column, y_column)
    
    async def _generate_report_step(self, results: Dict[str, Any]) -> str:
        """Generate final analysis report."""
        # Create a simple report without LLM for now
        report_parts = []
        
        if "data_info" in results:
            data_info = results["data_info"]
            report_parts.append(f"データ概要: {data_info['shape'][0]}行, {data_info['shape'][1]}列")
        
        if "statistics" in results:
            stats = results["statistics"]
            report_parts.append(f"基本統計: 平均={stats['mean']:.2f}, 標準偏差={stats['std']:.2f}")
        
        if "correlation" in results:
            corr = results["correlation"]
            report_parts.append(f"相関分析: {len(corr['significant_correlations'])}個の有意な相関関係")
        
        if "visualization" in results:
            viz = results["visualization"]
            report_parts.append(f"可視化: {viz['chart_type']}を作成")
        
        return "分析レポート:\n" + "\n".join(f"- {part}" for part in report_parts)

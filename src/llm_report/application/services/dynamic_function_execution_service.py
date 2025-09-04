"""Service for dynamic function execution based on LLM selection."""

import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .advanced_data_analysis_service import AdvancedDataAnalysisService
from .llm_function_selection_service import FunctionSelection


@dataclass
class ExecutionResult:
    """Function execution result."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


class DynamicFunctionExecutionService:
    """Service for dynamic function execution."""
    
    def __init__(self, advanced_analyzer: AdvancedDataAnalysisService):
        """Initialize the service.
        
        Args:
            advanced_analyzer: Advanced data analysis service
        """
        self.advanced_analyzer = advanced_analyzer
    
    def execute_selected_function(
        self, 
        data: pd.DataFrame, 
        function_selection: FunctionSelection
    ) -> ExecutionResult:
        """Execute the selected function with given parameters.
        
        Args:
            data: DataFrame to analyze
            function_selection: Selected function and parameters
            
        Returns:
            ExecutionResult with execution details
        """
        try:
            import time
            start_time = time.time()
            
            # Apply filters to data
            filtered_data = self._apply_filters(data, function_selection.parameters.get("filters", {}))
            
            # Execute selected function
            result = self._execute_function(
                filtered_data, 
                function_selection.function_name, 
                function_selection.parameters
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=None
            )
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to data.
        
        Args:
            data: Original DataFrame
            filters: Filter conditions
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        for column, value in filters.items():
            if column in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data
    
    def _execute_function(
        self, 
        data: pd.DataFrame, 
        function_name: str, 
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute the selected function.
        
        Args:
            data: Filtered DataFrame
            function_name: Name of function to execute
            parameters: Function parameters
            
        Returns:
            Function execution result
        """
        if function_name == "analyze_by_dimensions":
            return self.advanced_analyzer.analyze_by_dimensions(
                data=data,
                target_metric=parameters.get("target_metric", "average_daily_visiting_seconds"),
                group_by=parameters.get("group_by", ["area"]),
                filters=None  # Already applied
            )
        
        elif function_name == "create_comprehensive_report":
            return self.advanced_analyzer.create_comprehensive_report(
                data=data,
                target_metrics=parameters.get("target_metrics", ["average_daily_visiting_seconds"]),
                key_dimensions=parameters.get("key_dimensions", ["area", "period"])
            )
        
        elif function_name == "analyze_temporal_changes":
            return self.advanced_analyzer.analyze_temporal_changes(
                data=data,
                target_metric=parameters.get("target_metric", "average_daily_visiting_seconds"),
                group_by=parameters.get("group_by", ["area"]),
                time_dimension=parameters.get("time_dimension", "period")
            )
        
        elif function_name == "create_dashboard":
            return self.advanced_analyzer.create_dashboard(
                data=data,
                target_metrics=parameters.get("target_metrics", ["average_daily_visiting_seconds"]),
                key_dimensions=parameters.get("key_dimensions", ["area", "period"])
            )
        
        elif function_name == "get_data_summary":
            return self.advanced_analyzer.get_data_summary(data)
        
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def format_result_for_display(self, result: Any, function_name: str) -> str:
        """Format execution result for display.
        
        Args:
            result: Function execution result
            function_name: Name of executed function
            
        Returns:
            Formatted string for display
        """
        try:
            if hasattr(result, 'insights') and result.insights:
                insights_text = "\n".join([f"- {insight}" for insight in result.insights])
                return f"""
{function_name} の実行結果:

洞察:
{insights_text}

統計情報:
- データ数: {getattr(result, 'data_count', 'N/A')}
- 平均値: {getattr(result, 'mean_value', 'N/A')}
- 標準偏差: {getattr(result, 'std_value', 'N/A')}
"""
            else:
                return f"""
{function_name} の実行結果:

{str(result)}
"""
                
        except Exception as e:
            return f"結果の表示に失敗しました: {e}"

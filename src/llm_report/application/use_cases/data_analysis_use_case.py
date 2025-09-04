"""Data analysis use case."""

import logging
import time
from typing import Dict, Any, List
import pandas as pd

from ...domain.entities.data_analysis_request import DataAnalysisRequest, AnalysisType, ChartType
from ...domain.entities.data_analysis_response import DataAnalysisResponse, StatisticalSummary, CrossTabulationResult, CorrelationResult, VisualizationResult
from ...infrastructure.services.data_loader_service import DataLoaderService
from ...infrastructure.services.data_analysis_service import DataAnalysisService

logger = logging.getLogger(__name__)


class DataAnalysisUseCase:
    """Use case for data analysis operations."""
    
    def __init__(self, data_loader: DataLoaderService, data_analyzer: DataAnalysisService):
        """Initialize the data analysis use case.
        
        Args:
            data_loader: Data loader service
            data_analyzer: Data analysis service
        """
        self.data_loader = data_loader
        self.data_analyzer = data_analyzer
    
    async def execute(self, request: DataAnalysisRequest) -> DataAnalysisResponse:
        """Execute data analysis request.
        
        Args:
            request: Data analysis request
            
        Returns:
            Data analysis response
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting data analysis: {request.analysis_type.value}")
            
            # Load data
            data = self.data_loader.load_csv_data(request.file_path)
            
            # Get data info
            data_info = self.data_loader.get_data_info(data)
            
            # Execute analysis based on type
            if request.analysis_type == AnalysisType.BASIC_STATISTICS:
                result = await self._execute_basic_statistics(data, request)
            elif request.analysis_type == AnalysisType.CROSS_TABULATION:
                result = await self._execute_cross_tabulation(data, request)
            elif request.analysis_type == AnalysisType.CORRELATION:
                result = await self._execute_correlation(data, request)
            elif request.analysis_type == AnalysisType.VISUALIZATION:
                result = await self._execute_visualization(data, request)
            else:
                raise ValueError(f"Unsupported analysis type: {request.analysis_type}")
            
            execution_time = time.time() - start_time
            
            return DataAnalysisResponse(
                success=True,
                analysis_type=request.analysis_type.value,
                result=result,
                execution_time=execution_time,
                data_info=data_info
            )
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            execution_time = time.time() - start_time
            
            return DataAnalysisResponse(
                success=False,
                analysis_type=request.analysis_type.value,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_basic_statistics(self, data: pd.DataFrame, request: DataAnalysisRequest) -> StatisticalSummary:
        """Execute basic statistics analysis."""
        if len(request.target_columns) != 1:
            raise ValueError("Basic statistics requires exactly one target column")
        
        return self.data_analyzer.calculate_basic_statistics(data, request.target_columns[0])
    
    async def _execute_cross_tabulation(self, data: pd.DataFrame, request: DataAnalysisRequest) -> CrossTabulationResult:
        """Execute cross tabulation analysis."""
        if len(request.target_columns) != 2:
            raise ValueError("Cross tabulation requires exactly two target columns")
        
        return self.data_analyzer.create_cross_tabulation(
            data, 
            request.target_columns[0], 
            request.target_columns[1]
        )
    
    async def _execute_correlation(self, data: pd.DataFrame, request: DataAnalysisRequest) -> CorrelationResult:
        """Execute correlation analysis."""
        # Pass target_columns directly to the analyzer (can be None for auto-detection)
        return self.data_analyzer.calculate_correlation_matrix(data, request.target_columns)
    
    async def _execute_visualization(self, data: pd.DataFrame, request: DataAnalysisRequest) -> VisualizationResult:
        """Execute visualization analysis."""
        if request.chart_type == ChartType.BAR_CHART:
            if len(request.target_columns) != 2:
                raise ValueError("Bar chart requires exactly two columns")
            return self.data_analyzer.create_bar_chart(
                data, 
                request.target_columns[0], 
                request.target_columns[1],
                f"Bar Chart: {request.target_columns[1]} by {request.target_columns[0]}"
            )
        elif request.chart_type == ChartType.HISTOGRAM:
            if len(request.target_columns) != 1:
                raise ValueError("Histogram requires exactly one column")
            return self.data_analyzer.create_histogram(
                data, 
                request.target_columns[0],
                f"Histogram: {request.target_columns[0]}"
            )
        elif request.chart_type == ChartType.HEATMAP:
            return self.data_analyzer.create_heatmap(data, "Correlation Heatmap")
        else:
            raise ValueError(f"Unsupported chart type: {request.chart_type}")

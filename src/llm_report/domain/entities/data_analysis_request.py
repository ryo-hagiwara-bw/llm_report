"""Data analysis request entity."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class AnalysisType(Enum):
    """Analysis type enumeration."""
    BASIC_STATISTICS = "basic_statistics"
    CROSS_TABULATION = "cross_tabulation"
    CORRELATION = "correlation"
    VISUALIZATION = "visualization"


class ChartType(Enum):
    """Chart type enumeration."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX_PLOT = "box_plot"


@dataclass
class DataAnalysisRequest:
    """Data analysis request entity."""
    
    file_path: str
    analysis_type: AnalysisType
    target_columns: Optional[List[str]] = None
    group_by_columns: Optional[List[str]] = None
    chart_type: Optional[ChartType] = None
    filters: Optional[Dict[str, Any]] = None
    output_format: str = "text"
    
    def __post_init__(self):
        """Validate the request after initialization."""
        if self.analysis_type == AnalysisType.BASIC_STATISTICS and (not self.target_columns or len(self.target_columns) != 1):
            raise ValueError("Basic statistics requires exactly one target column")
        
        if self.analysis_type == AnalysisType.CROSS_TABULATION and not self.group_by_columns:
            raise ValueError("group_by_columns is required for cross tabulation")
        
        if self.analysis_type == AnalysisType.VISUALIZATION and not self.chart_type:
            raise ValueError("chart_type is required for visualization")

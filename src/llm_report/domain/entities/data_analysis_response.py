"""Data analysis response entity."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class StatisticalSummary:
    """Statistical summary data."""
    
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float


@dataclass
class CrossTabulationResult:
    """Cross tabulation result data."""
    
    table: pd.DataFrame
    row_totals: pd.Series
    col_totals: pd.Series
    percentages: pd.DataFrame


@dataclass
class CorrelationResult:
    """Correlation analysis result data."""
    
    correlation_matrix: pd.DataFrame
    significant_correlations: List[Dict[str, Any]]


@dataclass
class VisualizationResult:
    """Visualization result data."""
    
    chart_type: str
    figure: Optional[plt.Figure] = None
    description: Optional[str] = None
    data_summary: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


@dataclass
class DataAnalysisResponse:
    """Data analysis response entity."""
    
    success: bool
    analysis_type: str
    result: Union[StatisticalSummary, CrossTabulationResult, CorrelationResult, VisualizationResult]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    data_info: Optional[Dict[str, Any]] = None

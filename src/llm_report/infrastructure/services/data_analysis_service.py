"""Data analysis service for statistical analysis and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats

from ...domain.entities.data_analysis_response import (
    StatisticalSummary, 
    CrossTabulationResult, 
    CorrelationResult, 
    VisualizationResult
)

logger = logging.getLogger(__name__)


class DataAnalysisService:
    """Service for data analysis operations."""
    
    def __init__(self):
        """Initialize the data analysis service."""
        # Set matplotlib backend for non-interactive use
        plt.switch_backend('Agg')
        
    def calculate_basic_statistics(self, data: pd.DataFrame, target_column: str) -> StatisticalSummary:
        """Calculate basic statistics for a numeric column.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of the column to analyze
            
        Returns:
            StatisticalSummary object with basic statistics
        """
        try:
            if target_column not in data.columns:
                raise ValueError(f"Column '{target_column}' not found in data")
            
            series = data[target_column].dropna()
            
            if series.empty:
                raise ValueError(f"Column '{target_column}' has no valid data")
            
            return StatisticalSummary(
                count=len(series),
                mean=float(series.mean()),
                std=float(series.std()),
                min=float(series.min()),
                max=float(series.max()),
                median=float(series.median()),
                q25=float(series.quantile(0.25)),
                q75=float(series.quantile(0.75))
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate basic statistics: {e}")
            raise
    
    def create_cross_tabulation(self, data: pd.DataFrame, row_col: str, col_col: str) -> CrossTabulationResult:
        """Create cross tabulation between two categorical columns.
        
        Args:
            data: DataFrame containing the data
            row_col: Name of the row variable
            col_col: Name of the column variable
            
        Returns:
            CrossTabulationResult object with cross tabulation data
        """
        try:
            if row_col not in data.columns:
                raise ValueError(f"Row column '{row_col}' not found in data")
            if col_col not in data.columns:
                raise ValueError(f"Column '{col_col}' not found in data")
            
            # Create cross tabulation
            crosstab = pd.crosstab(data[row_col], data[col_col])
            
            # Calculate percentages
            percentages = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            
            return CrossTabulationResult(
                table=crosstab,
                row_totals=crosstab.sum(axis=1),
                col_totals=crosstab.sum(axis=0),
                percentages=percentages
            )
            
        except Exception as e:
            logger.error(f"Failed to create cross tabulation: {e}")
            raise
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> CorrelationResult:
        """Calculate correlation matrix for numeric columns.
        
        Args:
            data: DataFrame containing the data
            numeric_columns: List of numeric column names (optional, if None, auto-detect)
            
        Returns:
            CorrelationResult object with correlation data
        """
        try:
            if numeric_columns is None:
                # Auto-detect numeric columns
                available_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Filter to only numeric columns that exist
                available_columns = [col for col in numeric_columns if col in data.columns]
            
            if not available_columns:
                raise ValueError("No valid numeric columns found")
            
            # Calculate correlation matrix
            corr_matrix = data[available_columns].corr()
            
            # Find significant correlations (|r| > 0.5)
            significant_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        significant_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            return CorrelationResult(
                correlation_matrix=corr_matrix,
                significant_correlations=significant_correlations
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            raise
    
    def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Chart") -> VisualizationResult:
        """Create a bar chart.
        
        Args:
            data: DataFrame containing the data
            x_col: Name of the x-axis column
            y_col: Name of the y-axis column
            title: Chart title
            
        Returns:
            VisualizationResult object with the chart
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            data.plot(x=x_col, y=y_col, kind='bar', ax=ax, legend=False)
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return VisualizationResult(
                chart_type="bar_chart",
                figure=fig,
                description=f"Bar chart showing {y_col} by {x_col}",
                data_summary={
                    "x_column": x_col,
                    "y_column": y_col,
                    "data_points": len(data)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            raise
    
    def create_histogram(self, data: pd.DataFrame, column: str, title: str = "Histogram") -> VisualizationResult:
        """Create a histogram.
        
        Args:
            data: DataFrame containing the data
            column: Name of the column to plot
            title: Chart title
            
        Returns:
            VisualizationResult object with the chart
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            data[column].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            plt.tight_layout()
            
            return VisualizationResult(
                chart_type="histogram",
                figure=fig,
                description=f"Histogram showing distribution of {column}",
                data_summary={
                    "column": column,
                    "data_points": len(data[column].dropna()),
                    "mean": float(data[column].mean()),
                    "std": float(data[column].std())
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create histogram: {e}")
            raise
    
    def create_heatmap(self, data: pd.DataFrame, title: str = "Heatmap") -> VisualizationResult:
        """Create a correlation heatmap.
        
        Args:
            data: DataFrame containing the data
            title: Chart title
            
        Returns:
            VisualizationResult object with the chart
        """
        try:
            # Get numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                raise ValueError("No numeric columns found for heatmap")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create correlation heatmap
            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            
            return VisualizationResult(
                chart_type="heatmap",
                figure=fig,
                description="Correlation heatmap of numeric variables",
                data_summary={
                    "numeric_columns": list(numeric_data.columns),
                    "correlation_matrix": corr_matrix.to_dict()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            raise

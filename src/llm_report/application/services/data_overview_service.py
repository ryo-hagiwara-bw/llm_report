"""Service for analyzing data overview and structure."""

import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class DataOverview:
    """Data overview information."""
    columns: List[str]
    dtypes: Dict[str, str]
    shape: tuple
    head: str
    basic_stats: str
    unique_values: Dict[str, List[str]]
    missing_values: Dict[str, int]
    sample_data: str


class DataOverviewService:
    """Service for analyzing data overview."""
    
    def analyze_data_overview(self, data: pd.DataFrame) -> DataOverview:
        """Analyze data overview and structure.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            DataOverview object with comprehensive data information
        """
        try:
            # Basic information
            columns = list(data.columns)
            dtypes = {col: str(data[col].dtype) for col in columns}
            shape = data.shape
            
            # Head data (first 5 rows)
            head = data.head().to_string()
            
            # Basic statistics
            basic_stats = data.describe().to_string()
            
            # Unique values for each column (limit to 10 per column)
            unique_values = {}
            for col in columns:
                unique_vals = data[col].unique()
                unique_values[col] = unique_vals[:10].tolist()
            
            # Missing values
            missing_values = data.isnull().sum().to_dict()
            
            # Sample data (random 3 rows)
            sample_data = data.sample(min(3, len(data))).to_string()
            
            return DataOverview(
                columns=columns,
                dtypes=dtypes,
                shape=shape,
                head=head,
                basic_stats=basic_stats,
                unique_values=unique_values,
                missing_values=missing_values,
                sample_data=sample_data
            )
            
        except Exception as e:
            raise Exception(f"Data overview analysis failed: {e}")
    
    def format_overview_for_llm(self, overview: DataOverview) -> str:
        """Format data overview for LLM consumption.
        
        Args:
            overview: DataOverview object
            
        Returns:
            Formatted string for LLM
        """
        try:
            formatted = f"""
データ概要:
- カラム数: {len(overview.columns)}
- 行数: {overview.shape[0]}
- 列名: {overview.columns}

データ型:
{overview.dtypes}

基本統計量:
{overview.basic_stats}

各カラムのユニーク値（上位10個）:
"""
            for col, values in overview.unique_values.items():
                formatted += f"- {col}: {values}\n"
            
            formatted += f"""
欠損値:
{overview.missing_values}

サンプルデータ（3行）:
{overview.sample_data}
"""
            return formatted
            
        except Exception as e:
            raise Exception(f"Overview formatting failed: {e}")

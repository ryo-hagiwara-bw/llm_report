"""Data loader service for CSV files."""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoaderService:
    """Service for loading and validating CSV data."""
    
    def __init__(self):
        """Initialize the data loader service."""
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_csv_data(self, file_path: str, cache: bool = True) -> pd.DataFrame:
        """Load CSV data from file.
        
        Args:
            file_path: Path to the CSV file
            cache: Whether to cache the loaded data
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be parsed
        """
        try:
            # Check cache first
            if cache and file_path in self._data_cache:
                logger.info(f"Loading data from cache: {file_path}")
                return self._data_cache[file_path]
            
            # Load the file
            logger.info(f"Loading CSV data from: {file_path}")
            data = pd.read_csv(file_path)
            
            # Cache if requested
            if cache:
                self._data_cache[file_path] = data
            
            logger.info(f"Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise ValueError(f"Failed to load CSV data: {e}")
    
    def validate_data_schema(self, data: pd.DataFrame, required_columns: list) -> bool:
        """Validate that the data contains required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns are present
        """
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with data information
        """
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "null_counts": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum()
        }
    
    def detect_numeric_columns(self, data: pd.DataFrame) -> list:
        """Detect numeric columns in the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of numeric column names
        """
        return data.select_dtypes(include=['number']).columns.tolist()
    
    def detect_categorical_columns(self, data: pd.DataFrame) -> list:
        """Detect categorical columns in the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            List of categorical column names
        """
        return data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
        logger.info("Data cache cleared")

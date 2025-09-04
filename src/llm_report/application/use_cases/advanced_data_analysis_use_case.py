"""Advanced data analysis use case for comprehensive CSV analysis."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...domain.entities.data_analysis_request import DataAnalysisRequest, AnalysisType, ChartType
from ...domain.entities.data_analysis_response import DataAnalysisResponse
from ...application.services.data_loader_service import DataLoaderService
from ...application.services.advanced_data_analysis_service import (
    AdvancedDataAnalysisService, 
    AnalysisType as AdvancedAnalysisType,
    ReportType,
    ComprehensiveAnalysisResult,
    ComprehensiveReport
)

logger = logging.getLogger(__name__)


@dataclass
class AdvancedAnalysisRequest:
    """Request for advanced data analysis."""
    file_path: str
    target_metrics: List[str]
    key_dimensions: List[str]
    analysis_types: List[str] = None
    filters: Optional[Dict[str, Any]] = None
    report_type: str = "full"
    export_format: str = "excel"


@dataclass
class AdvancedAnalysisResponse:
    """Response from advanced data analysis."""
    success: bool
    report: Optional[ComprehensiveReport] = None
    individual_analyses: List[ComprehensiveAnalysisResult] = []
    error: Optional[str] = None
    file_paths: List[str] = None


class AdvancedDataAnalysisUseCase:
    """Use case for advanced data analysis operations."""
    
    def __init__(self, data_loader: DataLoaderService, advanced_analyzer: AdvancedDataAnalysisService):
        """Initialize the use case.
        
        Args:
            data_loader: Service for loading data
            advanced_analyzer: Service for advanced analysis
        """
        self.data_loader = data_loader
        self.advanced_analyzer = advanced_analyzer
    
    def execute_comprehensive_analysis(self, request: AdvancedAnalysisRequest) -> AdvancedAnalysisResponse:
        """Execute comprehensive data analysis.
        
        Args:
            request: Advanced analysis request
            
        Returns:
            Advanced analysis response
        """
        try:
            logger.info(f"Starting comprehensive analysis for {request.file_path}")
            
            # Load data
            data = self.data_loader.load_csv_data(request.file_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            
            # Perform individual analyses
            individual_analyses = []
            
            # Convert string analysis types to enum
            analysis_types = request.analysis_types or ["summary"]
            analysis_type_map = {
                "summary": AdvancedAnalysisType.SUMMARY,
                "comparison": AdvancedAnalysisType.COMPARISON,
                "distribution": AdvancedAnalysisType.DISTRIBUTION,
                "trend": AdvancedAnalysisType.TREND,
                "correlation": AdvancedAnalysisType.CORRELATION,
                "cross_tab": AdvancedAnalysisType.CROSS_TAB
            }
            
            for metric in request.target_metrics:
                if metric in data.columns:
                    for dimension in request.key_dimensions:
                        if dimension in data.columns:
                            for analysis_type_str in analysis_types:
                                if analysis_type_str in analysis_type_map:
                                    analysis_type = analysis_type_map[analysis_type_str]
                                    
                                    result = self.advanced_analyzer.analyze_by_dimensions(
                                        data=data,
                                        target_metric=metric,
                                        group_by=[dimension],
                                        filters=request.filters,
                                        analysis_type=analysis_type
                                    )
                                    individual_analyses.append(result)
            
            # Create comprehensive report
            report_type_map = {
                "full": ReportType.FULL,
                "summary": ReportType.SUMMARY,
                "focused": ReportType.FOCUSED,
                "dashboard": ReportType.DASHBOARD
            }
            
            report_type = report_type_map.get(request.report_type, ReportType.FULL)
            
            report = self.advanced_analyzer.create_comprehensive_report(
                data=data,
                target_metrics=request.target_metrics,
                key_dimensions=request.key_dimensions,
                report_type=report_type
            )
            
            # Export results if requested
            file_paths = report.file_paths.copy()
            if request.export_format:
                export_path = self.advanced_analyzer.export_analysis_results(
                    report, request.export_format
                )
                file_paths.append(export_path)
            
            logger.info(f"Comprehensive analysis completed successfully")
            
            return AdvancedAnalysisResponse(
                success=True,
                report=report,
                individual_analyses=individual_analyses,
                file_paths=file_paths
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return AdvancedAnalysisResponse(
                success=False,
                error=str(e)
            )
    
    def execute_specific_analysis(
        self, 
        file_path: str, 
        target_metric: str, 
        group_by: List[str],
        analysis_type: str = "summary",
        filters: Optional[Dict[str, Any]] = None
    ) -> AdvancedAnalysisResponse:
        """Execute specific analysis for a single metric and dimension combination.
        
        Args:
            file_path: Path to the CSV file
            target_metric: Metric to analyze
            group_by: Dimensions to group by
            analysis_type: Type of analysis
            filters: Optional filters
            
        Returns:
            Analysis response
        """
        try:
            logger.info(f"Starting specific analysis: {target_metric} by {group_by}")
            
            # Load data
            data = self.data_loader.load_csv_data(file_path)
            
            # Convert analysis type
            analysis_type_map = {
                "summary": AdvancedAnalysisType.SUMMARY,
                "comparison": AdvancedAnalysisType.COMPARISON,
                "distribution": AdvancedAnalysisType.DISTRIBUTION,
                "trend": AdvancedAnalysisType.TREND,
                "correlation": AdvancedAnalysisType.CORRELATION,
                "cross_tab": AdvancedAnalysisType.CROSS_TAB
            }
            
            analysis_type_enum = analysis_type_map.get(analysis_type, AdvancedAnalysisType.SUMMARY)
            
            # Perform analysis
            result = self.advanced_analyzer.analyze_by_dimensions(
                data=data,
                target_metric=target_metric,
                group_by=group_by,
                filters=filters,
                analysis_type=analysis_type_enum
            )
            
            # Create visualization
            viz_paths = []
            for dimension in group_by:
                viz_path = self.advanced_analyzer._create_dimension_visualization(
                    data, target_metric, dimension, f"specific_analysis_{target_metric}"
                )
                if viz_path:
                    viz_paths.append(viz_path)
            
            logger.info(f"Specific analysis completed successfully")
            
            return AdvancedAnalysisResponse(
                success=True,
                individual_analyses=[result],
                file_paths=viz_paths
            )
            
        except Exception as e:
            logger.error(f"Specific analysis failed: {e}")
            return AdvancedAnalysisResponse(
                success=False,
                error=str(e)
            )
    
    def execute_temporal_analysis(
        self,
        file_path: str,
        target_metrics: List[str],
        key_dimensions: List[str],
        time_dimension: str = "period"
    ) -> AdvancedAnalysisResponse:
        """Execute temporal analysis to understand changes over time.
        
        Args:
            file_path: Path to the CSV file
            target_metrics: Metrics to analyze
            key_dimensions: Dimensions to analyze
            time_dimension: Time dimension column
            
        Returns:
            Analysis response
        """
        try:
            logger.info(f"Starting temporal analysis for {time_dimension}")
            
            # Load data
            data = self.data_loader.load_csv_data(file_path)
            
            # Perform temporal analysis for each metric
            temporal_results = []
            for metric in target_metrics:
                if metric in data.columns:
                    result = self.advanced_analyzer.analyze_temporal_changes(
                        data=data,
                        target_metric=metric,
                        group_by=key_dimensions,
                        time_dimension=time_dimension
                    )
                    temporal_results.append(result)
            
            logger.info(f"Temporal analysis completed successfully")
            
            return AdvancedAnalysisResponse(
                success=True,
                individual_analyses=[],  # Temporal results are handled separately
                file_paths=[]
            )
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return AdvancedAnalysisResponse(
                success=False,
                error=str(e)
            )
    
    def create_dashboard(
        self,
        file_path: str,
        config: Dict[str, Any]
    ) -> AdvancedAnalysisResponse:
        """Create an interactive dashboard.
        
        Args:
            file_path: Path to the CSV file
            config: Dashboard configuration
            
        Returns:
            Analysis response
        """
        try:
            logger.info(f"Creating dashboard for {file_path}")
            
            # Load data
            data = self.data_loader.load_csv_data(file_path)
            
            # Create dashboard
            dashboard_path = self.advanced_analyzer.create_interactive_dashboard(
                data=data,
                config=config
            )
            
            logger.info(f"Dashboard created successfully: {dashboard_path}")
            
            return AdvancedAnalysisResponse(
                success=True,
                file_paths=[dashboard_path]
            )
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return AdvancedAnalysisResponse(
                success=False,
                error=str(e)
            )
    
    def get_data_summary(self, file_path: str) -> Dict[str, Any]:
        """Get a summary of the data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Data summary
        """
        try:
            data = self.data_loader.load_csv_data(file_path)
            
            summary = {
                "shape": data.shape,
                "columns": list(data.columns),
                "data_types": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "numeric_columns": list(data.select_dtypes(include=['number']).columns),
                "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
                "sample_data": data.head().to_dict('records')
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Data summary failed: {e}")
            return {"error": str(e)}

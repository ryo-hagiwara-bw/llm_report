"""Advanced data analysis service for comprehensive CSV analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    SUMMARY = "summary"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    TREND = "trend"
    CORRELATION = "correlation"
    CROSS_TAB = "cross_tab"


class ReportType(Enum):
    """Types of reports that can be generated."""
    FULL = "full"
    SUMMARY = "summary"
    FOCUSED = "focused"
    DASHBOARD = "dashboard"


@dataclass
class DimensionAnalysisResult:
    """Result of dimension-based analysis."""
    dimension: str
    values: List[str]
    statistics: Dict[str, float]
    count: int
    percentage: float
    insights: List[str]


@dataclass
class ComprehensiveAnalysisResult:
    """Result of comprehensive analysis."""
    target_metric: str
    group_by: List[str]
    analysis_type: str
    results: List[DimensionAnalysisResult]
    summary_statistics: Dict[str, float]
    visualizations: List[str]
    insights: List[str]
    recommendations: List[str]


@dataclass
class TemporalAnalysisResult:
    """Result of temporal analysis."""
    time_dimension: str
    periods: List[str]
    changes: Dict[str, float]
    trend_direction: str
    significant_changes: List[str]
    insights: List[str]


@dataclass
class ComprehensiveReport:
    """Comprehensive analysis report."""
    report_id: str
    generated_at: datetime
    data_shape: Tuple[int, int]
    target_metrics: List[str]
    key_dimensions: List[str]
    analysis_results: List[ComprehensiveAnalysisResult]
    temporal_analysis: List[TemporalAnalysisResult]
    overall_insights: List[str]
    recommendations: List[str]
    file_paths: List[str]


class AdvancedDataAnalysisService:
    """Advanced service for comprehensive data analysis."""
    
    def __init__(self):
        """Initialize the advanced data analysis service."""
        plt.switch_backend('Agg')
        sns.set_style("whitegrid")
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def analyze_by_dimensions(
        self, 
        data: pd.DataFrame, 
        target_metric: str,
        group_by: List[str],
        filters: Optional[Dict[str, Any]] = None,
        analysis_type: AnalysisType = AnalysisType.SUMMARY
    ) -> ComprehensiveAnalysisResult:
        """Analyze data by multiple dimensions.
        
        Args:
            data: DataFrame containing the data
            target_metric: Column to analyze
            group_by: Columns to group by
            filters: Optional filters to apply
            analysis_type: Type of analysis to perform
            
        Returns:
            Comprehensive analysis result
        """
        try:
            # Apply filters if provided
            filtered_data = data.copy()
            if filters:
                for key, value in filters.items():
                    if key in filtered_data.columns:
                        filtered_data = filtered_data[filtered_data[key] == value]
            
            # Validate inputs
            if target_metric not in filtered_data.columns:
                raise ValueError(f"Target metric '{target_metric}' not found in data")
            
            for col in group_by:
                if col not in filtered_data.columns:
                    raise ValueError(f"Group by column '{col}' not found in data")
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.SUMMARY:
                return self._perform_summary_analysis(filtered_data, target_metric, group_by)
            elif analysis_type == AnalysisType.COMPARISON:
                return self._perform_comparison_analysis(filtered_data, target_metric, group_by)
            elif analysis_type == AnalysisType.DISTRIBUTION:
                return self._perform_distribution_analysis(filtered_data, target_metric, group_by)
            elif analysis_type == AnalysisType.TREND:
                return self._perform_trend_analysis(filtered_data, target_metric, group_by)
            elif analysis_type == AnalysisType.CORRELATION:
                return self._perform_correlation_analysis(filtered_data, target_metric, group_by)
            elif analysis_type == AnalysisType.CROSS_TAB:
                return self._perform_cross_tab_analysis(filtered_data, target_metric, group_by)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Dimension analysis failed: {e}")
            raise
    
    def create_comprehensive_report(
        self,
        data: pd.DataFrame,
        target_metrics: List[str],
        key_dimensions: List[str],
        report_type: ReportType = ReportType.FULL
    ) -> ComprehensiveReport:
        """Create a comprehensive analysis report.
        
        Args:
            data: DataFrame containing the data
            target_metrics: Metrics to analyze
            key_dimensions: Dimensions to analyze
            report_type: Type of report to generate
            
        Returns:
            Comprehensive report
        """
        try:
            report_id = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            analysis_results = []
            temporal_analysis = []
            file_paths = []
            
            # Perform analysis for each metric and dimension combination
            for metric in target_metrics:
                if metric in data.columns:
                    for dimension in key_dimensions:
                        if dimension in data.columns:
                            # Summary analysis
                            result = self.analyze_by_dimensions(
                                data, metric, [dimension], 
                                analysis_type=AnalysisType.SUMMARY
                            )
                            analysis_results.append(result)
                            
                            # Create visualization
                            viz_path = self._create_dimension_visualization(
                                data, metric, dimension, report_id
                            )
                            if viz_path:
                                file_paths.append(viz_path)
            
            # Temporal analysis if period is available
            if 'period' in data.columns:
                for metric in target_metrics:
                    if metric in data.columns:
                        temporal_result = self.analyze_temporal_changes(
                            data, metric, key_dimensions, 'period'
                        )
                        temporal_analysis.append(temporal_result)
            
            # Generate insights and recommendations
            overall_insights = self._generate_overall_insights(analysis_results, temporal_analysis)
            recommendations = self._generate_recommendations(analysis_results, temporal_analysis)
            
            # Create report file
            report_path = self._create_report_file(
                report_id, analysis_results, temporal_analysis, 
                overall_insights, recommendations, file_paths
            )
            file_paths.append(report_path)
            
            return ComprehensiveReport(
                report_id=report_id,
                generated_at=datetime.now(),
                data_shape=data.shape,
                target_metrics=target_metrics,
                key_dimensions=key_dimensions,
                analysis_results=analysis_results,
                temporal_analysis=temporal_analysis,
                overall_insights=overall_insights,
                recommendations=recommendations,
                file_paths=file_paths
            )
            
        except Exception as e:
            logger.error(f"Comprehensive report creation failed: {e}")
            raise
    
    def analyze_temporal_changes(
        self,
        data: pd.DataFrame,
        target_metric: str,
        group_by: List[str],
        time_dimension: str = "period"
    ) -> TemporalAnalysisResult:
        """Analyze temporal changes in data.
        
        Args:
            data: DataFrame containing the data
            target_metric: Metric to analyze
            group_by: Dimensions to group by
            time_dimension: Time dimension column
            
        Returns:
            Temporal analysis result
        """
        try:
            if time_dimension not in data.columns:
                raise ValueError(f"Time dimension '{time_dimension}' not found in data")
            
            periods = data[time_dimension].unique().tolist()
            changes = {}
            significant_changes = []
            
            # Calculate changes between periods
            for i in range(len(periods) - 1):
                period1 = periods[i]
                period2 = periods[i + 1]
                
                data1 = data[data[time_dimension] == period1][target_metric].mean()
                data2 = data[data[time_dimension] == period2][target_metric].mean()
                
                if not pd.isna(data1) and not pd.isna(data2) and data1 != 0:
                    change_pct = ((data2 - data1) / data1) * 100
                    changes[f"{period1}_to_{period2}"] = change_pct
                    
                    if abs(change_pct) > 10:  # Significant change threshold
                        significant_changes.append(f"{period1} to {period2}: {change_pct:.1f}%")
            
            # Determine trend direction
            if changes:
                avg_change = np.mean(list(changes.values()))
                if avg_change > 5:
                    trend_direction = "increasing"
                elif avg_change < -5:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "unknown"
            
            # Generate insights
            insights = self._generate_temporal_insights(changes, significant_changes, trend_direction)
            
            return TemporalAnalysisResult(
                time_dimension=time_dimension,
                periods=periods,
                changes=changes,
                trend_direction=trend_direction,
                significant_changes=significant_changes,
                insights=insights
            )
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            raise
    
    def create_interactive_dashboard(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> str:
        """Create an interactive dashboard.
        
        Args:
            data: DataFrame containing the data
            config: Dashboard configuration
            
        Returns:
            Path to the dashboard file
        """
        try:
            dashboard_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create multiple visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Data Analysis Dashboard - {dashboard_id}', fontsize=16)
            
            # Plot 1: Distribution of target metric
            if 'target_metric' in config:
                metric = config['target_metric']
                if metric in data.columns:
                    axes[0, 0].hist(data[metric].dropna(), bins=30, alpha=0.7)
                    axes[0, 0].set_title(f'Distribution of {metric}')
                    axes[0, 0].set_xlabel(metric)
                    axes[0, 0].set_ylabel('Frequency')
            
            # Plot 2: Category analysis
            if 'category_column' in config:
                cat_col = config['category_column']
                if cat_col in data.columns:
                    category_counts = data[cat_col].value_counts()
                    axes[0, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
                    axes[0, 1].set_title(f'Distribution by {cat_col}')
            
            # Plot 3: Correlation heatmap
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0])
                axes[1, 0].set_title('Correlation Matrix')
            
            # Plot 4: Time series if available
            if 'time_column' in config and config['time_column'] in data.columns:
                time_col = config['time_column']
                if 'value_column' in config and config['value_column'] in data.columns:
                    value_col = config['value_column']
                    time_series = data.groupby(time_col)[value_col].mean()
                    axes[1, 1].plot(time_series.index, time_series.values, marker='o')
                    axes[1, 1].set_title(f'{value_col} over {time_col}')
                    axes[1, 1].set_xlabel(time_col)
                    axes[1, 1].set_ylabel(value_col)
                    axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_path = os.path.join(self.output_dir, f"{dashboard_id}.png")
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return dashboard_path
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            raise
    
    def export_analysis_results(
        self, 
        results: Union[ComprehensiveAnalysisResult, ComprehensiveReport], 
        format: str = "excel"
    ) -> str:
        """Export analysis results to file.
        
        Args:
            results: Analysis results to export
            format: Export format (excel, csv, json)
            
        Returns:
            Path to the exported file
        """
        try:
            export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format.lower() == "excel":
                return self._export_to_excel(results, export_id)
            elif format.lower() == "csv":
                return self._export_to_csv(results, export_id)
            elif format.lower() == "json":
                return self._export_to_json(results, export_id)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    # Private helper methods
    def _perform_summary_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform summary analysis."""
        results = []
        
        for dimension in group_by:
            dimension_data = data.groupby(dimension)[target_metric].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).reset_index()
            
            for _, row in dimension_data.iterrows():
                value = row[dimension]
                stats = {
                    'count': row['count'],
                    'mean': row['mean'],
                    'std': row['std'],
                    'min': row['min'],
                    'max': row['max'],
                    'median': row['median']
                }
                
                percentage = (row['count'] / len(data)) * 100
                insights = self._generate_dimension_insights(dimension, value, stats, percentage)
                
                results.append(DimensionAnalysisResult(
                    dimension=dimension,
                    values=[str(value)],
                    statistics=stats,
                    count=int(row['count']),
                    percentage=percentage,
                    insights=insights
                ))
        
        summary_stats = data[target_metric].describe().to_dict()
        overall_insights = self._generate_overall_insights_for_metric(data, target_metric)
        
        return ComprehensiveAnalysisResult(
            target_metric=target_metric,
            group_by=group_by,
            analysis_type="summary",
            results=results,
            summary_statistics=summary_stats,
            visualizations=[],
            insights=overall_insights,
            recommendations=[]
        )
    
    def _perform_comparison_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform comparison analysis."""
        # Implementation for comparison analysis
        return self._perform_summary_analysis(data, target_metric, group_by)
    
    def _perform_distribution_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform distribution analysis."""
        # Implementation for distribution analysis
        return self._perform_summary_analysis(data, target_metric, group_by)
    
    def _perform_trend_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform trend analysis."""
        # Implementation for trend analysis
        return self._perform_summary_analysis(data, target_metric, group_by)
    
    def _perform_correlation_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform correlation analysis."""
        # Implementation for correlation analysis
        return self._perform_summary_analysis(data, target_metric, group_by)
    
    def _perform_cross_tab_analysis(self, data: pd.DataFrame, target_metric: str, group_by: List[str]) -> ComprehensiveAnalysisResult:
        """Perform cross-tabulation analysis."""
        # Implementation for cross-tabulation analysis
        return self._perform_summary_analysis(data, target_metric, group_by)
    
    def _create_dimension_visualization(self, data: pd.DataFrame, metric: str, dimension: str, report_id: str) -> Optional[str]:
        """Create visualization for a dimension."""
        try:
            plt.figure(figsize=(10, 6))
            
            if data[dimension].dtype in ['object', 'category']:
                # Categorical data - create bar chart
                dimension_stats = data.groupby(dimension)[metric].mean().sort_values(ascending=False)
                dimension_stats.plot(kind='bar')
                plt.title(f'{metric} by {dimension}')
                plt.xlabel(dimension)
                plt.ylabel(metric)
                plt.xticks(rotation=45)
            else:
                # Numerical data - create scatter plot
                plt.scatter(data[dimension], data[metric], alpha=0.6)
                plt.title(f'{metric} vs {dimension}')
                plt.xlabel(dimension)
                plt.ylabel(metric)
            
            plt.tight_layout()
            
            viz_path = os.path.join(self.output_dir, f"{report_id}_{metric}_{dimension}.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    def _generate_dimension_insights(self, dimension: str, value: str, stats: Dict[str, float], percentage: float) -> List[str]:
        """Generate insights for a dimension."""
        insights = []
        
        if percentage > 50:
            insights.append(f"{dimension} '{value}' represents {percentage:.1f}% of the data (majority)")
        elif percentage > 20:
            insights.append(f"{dimension} '{value}' represents {percentage:.1f}% of the data (significant)")
        else:
            insights.append(f"{dimension} '{value}' represents {percentage:.1f}% of the data (minority)")
        
        if stats['std'] > stats['mean']:
            insights.append(f"High variability in {dimension} '{value}' (std > mean)")
        
        return insights
    
    def _generate_overall_insights_for_metric(self, data: pd.DataFrame, metric: str) -> List[str]:
        """Generate overall insights for a metric."""
        insights = []
        
        metric_data = data[metric].dropna()
        if len(metric_data) > 0:
            cv = metric_data.std() / metric_data.mean() if metric_data.mean() != 0 else 0
            if cv > 1:
                insights.append(f"High variability in {metric} (coefficient of variation: {cv:.2f})")
            elif cv < 0.3:
                insights.append(f"Low variability in {metric} (coefficient of variation: {cv:.2f})")
        
        return insights
    
    def _generate_overall_insights(self, analysis_results: List[ComprehensiveAnalysisResult], temporal_analysis: List[TemporalAnalysisResult]) -> List[str]:
        """Generate overall insights from all analyses."""
        insights = []
        
        # Add insights from analysis results
        for result in analysis_results:
            insights.extend(result.insights)
        
        # Add insights from temporal analysis
        for temporal in temporal_analysis:
            insights.extend(temporal.insights)
        
        return list(set(insights))  # Remove duplicates
    
    def _generate_recommendations(self, analysis_results: List[ComprehensiveAnalysisResult], temporal_analysis: List[TemporalAnalysisResult]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Add recommendations based on insights
        recommendations.append("Consider further analysis of high-variability metrics")
        recommendations.append("Monitor temporal trends for significant changes")
        recommendations.append("Focus on dimensions with highest impact on target metrics")
        
        return recommendations
    
    def _generate_temporal_insights(self, changes: Dict[str, float], significant_changes: List[str], trend_direction: str) -> List[str]:
        """Generate insights for temporal analysis."""
        insights = []
        
        if trend_direction == "increasing":
            insights.append("Overall upward trend observed")
        elif trend_direction == "decreasing":
            insights.append("Overall downward trend observed")
        else:
            insights.append("Stable trend observed")
        
        if significant_changes:
            insights.append(f"Significant changes detected: {len(significant_changes)} periods")
        
        return insights
    
    def _create_report_file(self, report_id: str, analysis_results: List[ComprehensiveAnalysisResult], 
                          temporal_analysis: List[TemporalAnalysisResult], overall_insights: List[str], 
                          recommendations: List[str], file_paths: List[str]) -> str:
        """Create a comprehensive report file."""
        report_path = os.path.join(self.output_dir, f"{report_id}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Comprehensive Data Analysis Report\n")
            f.write(f"Report ID: {report_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL INSIGHTS:\n")
            for i, insight in enumerate(overall_insights, 1):
                f.write(f"{i}. {insight}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            f.write("ANALYSIS RESULTS:\n")
            for result in analysis_results:
                f.write(f"\nMetric: {result.target_metric}\n")
                f.write(f"Dimensions: {', '.join(result.group_by)}\n")
                f.write(f"Analysis Type: {result.analysis_type}\n")
                f.write("-" * 30 + "\n")
            
            f.write("\nTEMPORAL ANALYSIS:\n")
            for temporal in temporal_analysis:
                f.write(f"Time Dimension: {temporal.time_dimension}\n")
                f.write(f"Trend Direction: {temporal.trend_direction}\n")
                f.write("-" * 30 + "\n")
            
            f.write(f"\nGENERATED FILES:\n")
            for file_path in file_paths:
                f.write(f"- {file_path}\n")
        
        return report_path
    
    def _export_to_excel(self, results: Union[ComprehensiveAnalysisResult, ComprehensiveReport], export_id: str) -> str:
        """Export results to Excel file."""
        # Implementation for Excel export
        excel_path = os.path.join(self.output_dir, f"{export_id}.xlsx")
        # Add Excel export logic here
        return excel_path
    
    def _export_to_csv(self, results: Union[ComprehensiveAnalysisResult, ComprehensiveReport], export_id: str) -> str:
        """Export results to CSV file."""
        # Implementation for CSV export
        csv_path = os.path.join(self.output_dir, f"{export_id}.csv")
        # Add CSV export logic here
        return csv_path
    
    def _export_to_json(self, results: Union[ComprehensiveAnalysisResult, ComprehensiveReport], export_id: str) -> str:
        """Export results to JSON file."""
        # Implementation for JSON export
        json_path = os.path.join(self.output_dir, f"{export_id}.json")
        # Add JSON export logic here
        return json_path

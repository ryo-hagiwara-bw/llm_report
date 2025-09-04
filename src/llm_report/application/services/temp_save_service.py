"""Service for temporarily saving analysis results and content."""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class TempSaveResult:
    """Result of temporary save operation."""
    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None


class TempSaveService:
    """Service for temporarily saving analysis results."""
    
    def __init__(self, temp_dir: str = "tmp"):
        """Initialize the service.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Ensure temporary directory exists."""
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def save_analysis_result(
        self, 
        analysis_type: str,
        prompt: str,
        result: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> TempSaveResult:
        """Save analysis result to temporary file.
        
        Args:
            analysis_type: Type of analysis (e.g., 'langgraph', 'prompt_driven')
            prompt: User prompt
            result: Analysis result
            additional_info: Additional information to save
            
        Returns:
            TempSaveResult with save operation details
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{analysis_type}_analysis_{timestamp}.txt"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Prepare content to save
            content = self._format_analysis_content(
                analysis_type, prompt, result, additional_info
            )
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return TempSaveResult(
                success=True,
                file_path=file_path
            )
            
        except Exception as e:
            return TempSaveResult(
                success=False,
                error=str(e)
            )
    
    def save_workflow_messages(
        self,
        workflow_type: str,
        messages: List[Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> TempSaveResult:
        """Save workflow messages to temporary file.
        
        Args:
            workflow_type: Type of workflow
            messages: List of workflow messages
            additional_data: Additional data to save
            
        Returns:
            TempSaveResult with save operation details
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{workflow_type}_messages_{timestamp}.txt"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Format messages
            content = f"=== {workflow_type.upper()} WORKFLOW MESSAGES ===\n"
            content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"Total messages: {len(messages)}\n\n"
            
            for i, msg in enumerate(messages, 1):
                content += f"Message {i}:\n"
                if hasattr(msg, 'content'):
                    content += f"Content: {msg.content}\n"
                else:
                    content += f"Content: {str(msg)}\n"
                content += f"Type: {type(msg).__name__}\n"
                content += "-" * 50 + "\n"
            
            # Add additional data if provided
            if additional_data:
                content += f"\n=== ADDITIONAL DATA ===\n"
                for key, value in additional_data.items():
                    content += f"{key}: {value}\n"
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return TempSaveResult(
                success=True,
                file_path=file_path
            )
            
        except Exception as e:
            return TempSaveResult(
                success=False,
                error=str(e)
            )
    
    def save_function_selection(
        self,
        prompt: str,
        function_selection: Any,
        execution_result: Any
    ) -> TempSaveResult:
        """Save function selection and execution result.
        
        Args:
            prompt: User prompt
            function_selection: Function selection result
            execution_result: Execution result
            
        Returns:
            TempSaveResult with save operation details
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"function_selection_{timestamp}.txt"
            file_path = os.path.join(self.temp_dir, filename)
            
            # Format content
            content = f"=== FUNCTION SELECTION AND EXECUTION ===\n"
            content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"User Prompt: {prompt}\n\n"
            
            # Function selection details
            content += "=== FUNCTION SELECTION ===\n"
            if hasattr(function_selection, 'function_name'):
                content += f"Selected Function: {function_selection.function_name}\n"
            if hasattr(function_selection, 'confidence'):
                content += f"Confidence: {function_selection.confidence}\n"
            if hasattr(function_selection, 'reasoning'):
                content += f"Reasoning: {function_selection.reasoning}\n"
            if hasattr(function_selection, 'parameters'):
                content += f"Parameters: {json.dumps(function_selection.parameters, indent=2, ensure_ascii=False)}\n"
            
            # Execution result details
            content += "\n=== EXECUTION RESULT ===\n"
            if hasattr(execution_result, 'success'):
                content += f"Success: {execution_result.success}\n"
            if hasattr(execution_result, 'execution_time'):
                content += f"Execution Time: {execution_result.execution_time}s\n"
            if hasattr(execution_result, 'error'):
                content += f"Error: {execution_result.error}\n"
            if hasattr(execution_result, 'result'):
                content += f"Result: {str(execution_result.result)}\n"
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return TempSaveResult(
                success=True,
                file_path=file_path
            )
            
        except Exception as e:
            return TempSaveResult(
                success=False,
                error=str(e)
            )
    
    def _format_analysis_content(
        self,
        analysis_type: str,
        prompt: str,
        result: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format analysis content for saving.
        
        Args:
            analysis_type: Type of analysis
            prompt: User prompt
            result: Analysis result
            additional_info: Additional information
            
        Returns:
            Formatted content string
        """
        content = f"=== {analysis_type.upper()} ANALYSIS RESULT ===\n"
        content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"User Prompt: {prompt}\n\n"
        
        # Basic result information
        content += "=== BASIC INFORMATION ===\n"
        content += f"Success: {result.get('success', 'N/A')}\n"
        content += f"Data Shape: {result.get('data', {}).shape if hasattr(result.get('data'), 'shape') else 'N/A'}\n"
        content += f"Completed Steps: {', '.join(result.get('completed_steps', []))}\n"
        content += f"Failed Steps: {', '.join(result.get('failed_steps', []))}\n"
        content += f"Insights Count: {len(result.get('insights', []))}\n"
        content += f"Generated Files: {len(result.get('file_paths', []))}\n\n"
        
        # Insights
        if result.get('insights'):
            content += "=== INSIGHTS ===\n"
            for i, insight in enumerate(result['insights'], 1):
                content += f"{i}. {insight}\n"
            content += "\n"
        
        # Detailed analysis results
        if result.get('analysis_results', {}).get('dimension_analysis'):
            content += "=== DETAILED ANALYSIS RESULTS ===\n"
            for analysis in result['analysis_results']['dimension_analysis']:
                if hasattr(analysis, 'dimension') and hasattr(analysis, 'statistics'):
                    content += f"\n{analysis.dimension}:\n"
                    stats = analysis.statistics
                    content += f"  - Count: {stats.get('count', 'N/A')}\n"
                    content += f"  - Mean: {stats.get('mean', 'N/A'):.2f} seconds ({stats.get('mean', 0)/3600:.2f} hours)\n"
                    content += f"  - Std: {stats.get('std', 'N/A'):.2f} seconds\n"
                    content += f"  - Min: {stats.get('min', 'N/A'):.2f} seconds ({stats.get('min', 0)/3600:.2f} hours)\n"
                    content += f"  - Max: {stats.get('max', 'N/A'):.2f} seconds ({stats.get('max', 0)/3600:.2f} hours)\n"
                    content += f"  - Median: {stats.get('median', 'N/A'):.2f} seconds ({stats.get('median', 0)/3600:.2f} hours)\n"
            content += "\n"
        
        # Summary of key findings
        content += "=== KEY FINDINGS SUMMARY ===\n"
        if result.get('analysis_results', {}).get('dimension_analysis'):
            for analysis in result['analysis_results']['dimension_analysis']:
                if hasattr(analysis, 'dimension') and hasattr(analysis, 'statistics'):
                    stats = analysis.statistics
                    if analysis.dimension == 'home_area':
                        for i, value in enumerate(analysis.values):
                            if value == 'エリア内':
                                content += f"🏠 エリア内居住者: {stats['count']}件, 平均{stats['mean']:.0f}秒 ({stats['mean']/3600:.1f}時間)\n"
                            elif value == 'エリア外':
                                content += f"🏢 エリア外居住者: {stats['count']}件, 平均{stats['mean']:.0f}秒 ({stats['mean']/3600:.1f}時間)\n"
                    elif analysis.dimension == 'day_type':
                        for i, value in enumerate(analysis.values):
                            if value == '土日祝日':
                                content += f"📅 土日祝日: {stats['count']}件, 平均{stats['mean']:.0f}秒 ({stats['mean']/3600:.1f}時間)\n"
                            elif value == '平日':
                                content += f"📅 平日: {stats['count']}件, 平均{stats['mean']:.0f}秒 ({stats['mean']/3600:.1f}時間)\n"
        content += "\n"
        
        # Generated files
        if result.get('file_paths'):
            content += "=== GENERATED FILES ===\n"
            for i, file_path in enumerate(result['file_paths'], 1):
                content += f"{i}. {file_path}\n"
            content += "\n"
        
        # Analysis results details
        if result.get('analysis_results'):
            content += "=== ANALYSIS RESULTS ===\n"
            for key, value in result['analysis_results'].items():
                content += f"{key}: {type(value).__name__}\n"
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        content += f"  {sub_key}: {sub_value}\n"
            content += "\n"
        
        # Function selection details (for prompt-driven workflow)
        if result.get('function_selection'):
            fs = result['function_selection']
            content += "=== FUNCTION SELECTION ===\n"
            if hasattr(fs, 'function_name'):
                content += f"Selected Function: {fs.function_name}\n"
            if hasattr(fs, 'confidence'):
                content += f"Confidence: {fs.confidence}\n"
            if hasattr(fs, 'reasoning'):
                content += f"Reasoning: {fs.reasoning}\n"
            if hasattr(fs, 'parameters'):
                content += f"Parameters: {json.dumps(fs.parameters, indent=2, ensure_ascii=False)}\n"
            content += "\n"
        
        # Execution result details (for prompt-driven workflow)
        if result.get('execution_result'):
            er = result['execution_result']
            content += "=== EXECUTION RESULT ===\n"
            if hasattr(er, 'success'):
                content += f"Success: {er.success}\n"
            if hasattr(er, 'execution_time'):
                content += f"Execution Time: {er.execution_time}s\n"
            if hasattr(er, 'error'):
                content += f"Error: {er.error}\n"
            content += "\n"
        
        # Additional information
        if additional_info:
            content += "=== ADDITIONAL INFORMATION ===\n"
            for key, value in additional_info.items():
                content += f"{key}: {value}\n"
            content += "\n"
        
        # Error information
        if result.get('error'):
            content += "=== ERROR INFORMATION ===\n"
            content += f"Error: {result['error']}\n"
        
        return content
    
    def list_temp_files(self) -> List[str]:
        """List all temporary files.
        
        Returns:
            List of temporary file paths
        """
        try:
            if not os.path.exists(self.temp_dir):
                return []
            
            files = []
            for filename in os.listdir(self.temp_dir):
                if filename.endswith('.txt'):
                    files.append(os.path.join(self.temp_dir, filename))
            
            return sorted(files, key=os.path.getmtime, reverse=True)
            
        except Exception as e:
            return []
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """Clean up temporary files older than specified days.
        
        Args:
            days: Number of days to keep files
            
        Returns:
            Number of files deleted
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_count = 0
            for file_path in self.list_temp_files():
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            return 0

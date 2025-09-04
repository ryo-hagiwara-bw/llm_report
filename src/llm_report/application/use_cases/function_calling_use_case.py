"""Simple function calling use case for LLM Report application."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a function call from the LLM."""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class FunctionResult:
    """Represents the result of a function call."""
    call_id: Optional[str]
    name: str
    response: Any


@dataclass
class FunctionCallingRequest:
    """Request for simple function calling generation."""
    prompt: str
    model: ModelConfig = None
    max_iterations: int = 3
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()


@dataclass
class FunctionCallingResponse:
    """Response from simple function calling generation."""
    content: str
    function_calls: List[FunctionCall]
    function_results: List[FunctionResult]
    iterations: int
    success: bool
    error: Optional[str] = None


class FunctionCallingUseCase:
    """Simple use case for handling function calling with LLM."""
    
    def __init__(self, llm_repository, data_analysis_use_case=None):
        """Initialize the use case.
        
        Args:
            llm_repository: Repository for LLM operations
            data_analysis_use_case: Data analysis use case (optional)
        """
        self.llm_repository = llm_repository
        self.data_analysis_use_case = data_analysis_use_case
        self.function_handlers = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default function handlers."""
        self.function_handlers = {
            "get_weather": self._weather_handler,
            "calculate": self._calculator_handler,
            "get_time": self._time_handler,
            "analyze_data_basic_statistics": self._analyze_data_basic_statistics_handler,
            "analyze_data_cross_tabulation": self._analyze_data_cross_tabulation_handler,
            "analyze_data_correlation": self._analyze_data_correlation_handler,
            "create_visualization": self._create_visualization_handler,
        }
    
    async def execute(self, request: FunctionCallingRequest) -> FunctionCallingResponse:
        """Execute simple function calling generation.
        
        Args:
            request: Function calling request
            
        Returns:
            Function calling response
        """
        try:
            logger.info(f"Starting simple function calling for prompt: {request.prompt[:50]}...")
            
            # Create prompt object
            prompt_obj = Prompt(content=request.prompt)
            
            # Create generation request
            generation_request = GenerationRequest(
                prompt=prompt_obj,
                model=request.model
            )
            
            # Generate content
            response = await self.llm_repository.generate_content(generation_request)
            
            # Process the response
            function_calls = []
            function_results = []
            iterations = 0
            
            # Check if the response contains function call requests
            content = response.content if response.content else ""
            
            # Simple function detection based on keywords
            if "天気" in request.prompt or "weather" in request.prompt.lower():
                function_calls.append(FunctionCall(
                    name="get_weather",
                    args={"location": "東京"},
                    call_id="weather_call_1"
                ))
                
                # Execute the function
                result = await self._weather_handler({"location": "東京"})
                function_results.append(FunctionResult(
                    call_id="weather_call_1",
                    name="get_weather",
                    response=result
                ))
                iterations += 1
                
                # Update content with function result
                content = f"東京の天気情報をお答えします。{result}"
            
            elif "計算" in request.prompt or "calculate" in request.prompt.lower():
                function_calls.append(FunctionCall(
                    name="calculate",
                    args={"expression": "2+2"},
                    call_id="calc_call_1"
                ))
                
                # Execute the function
                result = await self._calculator_handler({"expression": "2+2"})
                function_results.append(FunctionResult(
                    call_id="calc_call_1",
                    name="calculate",
                    response=result
                ))
                iterations += 1
                
                # Update content with function result
                content = f"計算結果をお答えします。{result}"
            
            elif "時間" in request.prompt or "time" in request.prompt.lower():
                function_calls.append(FunctionCall(
                    name="get_time",
                    args={},
                    call_id="time_call_1"
                ))
                
                # Execute the function
                result = await self._time_handler({})
                function_results.append(FunctionResult(
                    call_id="time_call_1",
                    name="get_time",
                    response=result
                ))
                iterations += 1
                
                # Update content with function result
                content = f"現在の時間をお答えします。{result}"
            
            # If no specific function was detected, return the original content
            if not function_calls:
                content = content or f"「{request.prompt}」についてお答えします。"
            
            return FunctionCallingResponse(
                content=content,
                function_calls=function_calls,
                function_results=function_results,
                iterations=iterations,
                success=True,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Simple function calling failed: {e}")
            return FunctionCallingResponse(
                content=f"「{request.prompt}」についてお答えします。",
                function_calls=[],
                function_results=[],
                iterations=0,
                success=False,
                error=str(e)
            )
    
    async def _weather_handler(self, args: Dict[str, Any]) -> str:
        """Handle weather function call."""
        location = args.get("location", "東京")
        return f"{location}の天気は晴れです。気温は25度、湿度は60%です。"
    
    async def _calculator_handler(self, args: Dict[str, Any]) -> str:
        """Handle calculator function call."""
        expression = args.get("expression", "2+2")
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"計算エラー: {str(e)}"
    
    async def _time_handler(self, args: Dict[str, Any]) -> str:
        """Handle time function call."""
        import datetime
        now = datetime.datetime.now()
        return f"現在の時刻は {now.strftime('%Y年%m月%d日 %H:%M:%S')} です。"
    
    async def _analyze_data_basic_statistics_handler(self, args: Dict[str, Any]) -> str:
        """Handle basic statistics analysis."""
        if not self.data_analysis_use_case:
            return "データ分析機能が利用できません。"
        
        try:
            file_path = args.get("file_path", "dataset/result_15_osakabanpaku_stay.csv")
            target_column = args.get("target_column", "visitor_count")
            
            # Load data
            data = self.data_analysis_use_case.data_loader.load_csv_data(file_path)
            
            # Calculate statistics
            stats = self.data_analysis_use_case.data_analyzer.calculate_basic_statistics(data, target_column)
            
            return f"基本統計: 平均={stats.mean:.2f}, 標準偏差={stats.std:.2f}, 最小値={stats.min:.2f}, 最大値={stats.max:.2f}"
            
        except Exception as e:
            return f"統計分析エラー: {str(e)}"
    
    async def _analyze_data_cross_tabulation_handler(self, args: Dict[str, Any]) -> str:
        """Handle cross tabulation analysis."""
        if not self.data_analysis_use_case:
            return "データ分析機能が利用できません。"
        
        try:
            file_path = args.get("file_path", "dataset/result_15_osakabanpaku_stay.csv")
            row_column = args.get("row_column", "area")
            column_column = args.get("column_column", "day_type")
            
            # Load data
            data = self.data_analysis_use_case.data_loader.load_csv_data(file_path)
            
            # Create cross tabulation
            result = self.data_analysis_use_case.data_analyzer.create_cross_tabulation(data, row_column, column_column)
            
            return f"クロス集計完了: {row_column} × {column_column}"
            
        except Exception as e:
            return f"クロス集計エラー: {str(e)}"
    
    async def _analyze_data_correlation_handler(self, args: Dict[str, Any]) -> str:
        """Handle correlation analysis."""
        if not self.data_analysis_use_case:
            return "データ分析機能が利用できません。"
        
        try:
            file_path = args.get("file_path", "dataset/result_15_osakabanpaku_stay.csv")
            
            # Load data
            data = self.data_analysis_use_case.data_loader.load_csv_data(file_path)
            
            # Calculate correlation
            result = self.data_analysis_use_case.data_analyzer.calculate_correlation_matrix(data)
            
            return f"相関分析完了: {len(result.correlation_matrix.columns)}変数の相関行列を計算"
            
        except Exception as e:
            return f"相関分析エラー: {str(e)}"
    
    async def _create_visualization_handler(self, args: Dict[str, Any]) -> str:
        """Handle visualization creation."""
        if not self.data_analysis_use_case:
            return "データ分析機能が利用できません。"
        
        try:
            file_path = args.get("file_path", "dataset/result_15_osakabanpaku_stay.csv")
            chart_type = args.get("chart_type", "bar_chart")
            x_column = args.get("x_column", "area")
            y_column = args.get("y_column", "visitor_count")
            
            # Load data
            data = self.data_analysis_use_case.data_loader.load_csv_data(file_path)
            
            # Create visualization
            if chart_type == "bar_chart":
                result = self.data_analysis_use_case.data_analyzer.create_bar_chart(data, x_column, y_column)
            elif chart_type == "histogram":
                result = self.data_analysis_use_case.data_analyzer.create_histogram(data, y_column)
            else:
                result = self.data_analysis_use_case.data_analyzer.create_bar_chart(data, x_column, y_column)
            
            return f"可視化完了: {chart_type}を作成し、{result.file_path}に保存"
            
        except Exception as e:
            return f"可視化エラー: {str(e)}"

"""Main entry point for the LLM Report application."""

import asyncio
import logging
import warnings
from typing import Dict, Any

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest
from src.llm_report.domain.entities.data_analysis_request import DataAnalysisRequest, AnalysisType, ChartType
from src.llm_report.infrastructure.config.dependency_container import DependencyContainer, ContainerConfig
from src.llm_report.application.use_cases.generate_content_use_case import GenerateContentUseCase
from src.llm_report.application.use_cases.function_calling_use_case import FunctionCallingUseCase, FunctionCallingRequest
from src.llm_report.application.use_cases.data_analysis_use_case import DataAnalysisUseCase
from src.llm_report.infrastructure.services.data_loader_service import DataLoaderService
from src.llm_report.infrastructure.services.data_analysis_service import DataAnalysisService

# Configure logging and suppress warnings
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Suppress all logging from specific modules
logging.getLogger("src.llm_report").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("vertexai").setLevel(logging.CRITICAL)


class SimpleLLMApp:
    """Simple LLM application with basic generation and function calling simulation."""
    
    def __init__(self, project_id: str = "stg-ai-421505", location: str = "global"):
        """Initialize the application.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize DDD components
        self.container = DependencyContainer(
            ContainerConfig.VERTEX_AI,
            project_id=project_id,
            location=location
        )
        self.generate_content_use_case = self.container.get_generate_content_use_case()
        
        # Initialize data analysis services
        self.data_loader = DataLoaderService()
        self.data_analyzer = DataAnalysisService()
        self.data_analysis_use_case = DataAnalysisUseCase(self.data_loader, self.data_analyzer)
        
        # Initialize function calling with data analysis
        self.function_calling_use_case = FunctionCallingUseCase(
            self.container.get_llm_repository(),
            self.data_analysis_use_case
        )
        
        # Function definitions for actual function calling
        self.functions = self._get_function_definitions()
    
    def _get_function_definitions(self):
        """Get function definitions for Vertex AI function calling."""
        return [
            {
                "name": "get_weather",
                "description": "指定された場所の現在の天気を取得します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "都市名（例：東京、大阪、福岡）"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度の単位"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "数学的な計算を実行します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "計算式（例：15 * 8, 100 / 4）"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_time",
                "description": "指定されたタイムゾーンの現在時刻を取得します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "タイムゾーン（例：Asia/Tokyo, UTC）"
                        }
                    },
                    "required": ["timezone"]
                }
            },
            {
                "name": "analyze_data_basic_statistics",
                "description": "CSVデータの基本統計量を計算します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "CSVファイルのパス"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "分析対象のカラム名"
                        }
                    },
                    "required": ["file_path", "target_column"]
                }
            },
            {
                "name": "analyze_data_cross_tabulation",
                "description": "CSVデータのクロス集計を実行します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "CSVファイルのパス"
                        },
                        "row_column": {
                            "type": "string",
                            "description": "行のカラム名"
                        },
                        "column_column": {
                            "type": "string",
                            "description": "列のカラム名"
                        }
                    },
                    "required": ["file_path", "row_column", "column_column"]
                }
            },
            {
                "name": "analyze_data_correlation",
                "description": "CSVデータの相関分析を実行します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "CSVファイルのパス"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "分析対象のカラム名のリスト（空の場合は全数値カラム）"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "create_data_visualization",
                "description": "CSVデータの可視化を作成します。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "CSVファイルのパス"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar_chart", "histogram", "heatmap"],
                            "description": "グラフの種類"
                        },
                        "x_column": {
                            "type": "string",
                            "description": "X軸のカラム名（bar_chartの場合）"
                        },
                        "y_column": {
                            "type": "string",
                            "description": "Y軸のカラム名（bar_chartの場合）"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "対象カラム名（histogramの場合）"
                        }
                    },
                    "required": ["file_path", "chart_type"]
                }
            }
        ]
    
    async def generate_content(self, prompt_text: str, model: str = "gemini-2.5-pro") -> str:
        """Generate content from a prompt using DDD architecture.
        
        Args:
            prompt_text: Input prompt
            model: Model name
            
        Returns:
            Generated content
        """
        try:
            # Create domain objects
            prompt = Prompt(content=prompt_text)
            model_config = ModelConfig(name=model, temperature=0.7)
            request = GenerationRequest(prompt=prompt, model=model_config)
            
            # Execute use case
            response = await self.generate_content_use_case.execute(request)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise
    
    async def generate_with_functions(self, prompt_text: str, model: str = "gemini-2.5-pro") -> Dict[str, Any]:
        """Generate content with actual function calling using Vertex AI.
        
        Args:
            prompt_text: User prompt
            model: Model name
            
        Returns:
            Generation result with function calls
        """
        try:
            # Create domain objects
            prompt = Prompt(content=prompt_text)
            model_config = ModelConfig(name=model, temperature=0.7)
            
            # Create function calling request
            request = FunctionCallingRequest(
                prompt=prompt,
                model=model_config,
                functions=self.functions
            )
            
            # Execute function calling use case
            response = await self.function_calling_use_case.execute(request)
            
            return {
                "content": response.content,
                "function_calls": [
                    {
                        "name": call.name,
                        "args": call.args,
                        "call_id": call.call_id
                    } for call in response.function_calls
                ],
                "function_results": [
                    {
                        "name": result.name,
                        "call_id": result.call_id,
                        "result": result.response
                    } for result in response.function_results
                ],
                "has_function_calls": len(response.function_calls) > 0,
                "success": response.success,
                "error": response.error
            }
            
        except Exception as e:
            logger.error(f"Failed to generate with functions: {e}")
            return {
                "content": None,
                "function_calls": [],
                "function_results": [],
                "has_function_calls": False,
                "success": False,
                "error": str(e)
            }
    


async def main():
    """Main function for testing both basic generation and function calling."""
    try:
        # Initialize application
        app = SimpleLLMApp()
        
        # Test 1: Basic content generation
        basic_prompt = "こんにちは！今日はいい天気ですね。"
        basic_response = await app.generate_content(basic_prompt)
        print(f"Prompt: {basic_prompt}")
        print(f"Response: {basic_response}")
        print()
        
        # Test 2: Function calling
        function_prompt = "東京の天気を教えて"
        function_result = await app.generate_with_functions(function_prompt)
        print(f"Prompt: {function_prompt}")
        print(f"Response: {function_result['content']}")
        print()
        
        # Test 3: Calculator
        calc_prompt = "15 * 8 を計算して"
        calc_result = await app.generate_with_functions(calc_prompt)
        print(f"Prompt: {calc_prompt}")
        print(f"Response: {calc_result['content']}")
        print()
        
        # Test 4: Complex query
        complex_prompt = "大阪の天気と、100 / 4 を計算して"
        complex_result = await app.generate_with_functions(complex_prompt)
        print(f"Prompt: {complex_prompt}")
        print(f"Response: {complex_result['content']}")
        print()
        
        # Test 5: Data Analysis - Basic Statistics
        stats_prompt = "dataset/result_15_osakabanpaku_stay.csvのvisitor_countの基本統計量を計算して"
        stats_result = await app.generate_with_functions(stats_prompt)
        print(f"Prompt: {stats_prompt}")
        print(f"Response: {stats_result['content']}")
        print()
        
        # Test 6: Data Analysis - Cross Tabulation
        crosstab_prompt = "dataset/result_15_osakabanpaku_stay.csvでareaとperiodのクロス集計を作成して"
        crosstab_result = await app.generate_with_functions(crosstab_prompt)
        print(f"Prompt: {crosstab_prompt}")
        print(f"Response: {crosstab_result['content']}")
        print()
        
        # Test 7: Data Analysis - Correlation
        corr_prompt = "dataset/result_15_osakabanpaku_stay.csvの数値カラムの相関分析を実行して"
        corr_result = await app.generate_with_functions(corr_prompt)
        print(f"Prompt: {corr_prompt}")
        print(f"Response: {corr_result['content']}")
        print()
        
        # Test 8: Data Analysis - Visualization
        viz_prompt = "dataset/result_15_osakabanpaku_stay.csvでareaとvisitor_countの棒グラフを作成して"
        viz_result = await app.generate_with_functions(viz_prompt)
        print(f"Prompt: {viz_prompt}")
        print(f"Response: {viz_result['content']}")
        print()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
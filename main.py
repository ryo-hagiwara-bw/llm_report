"""Main entry point for the LLM Report application."""

import asyncio
import logging
from typing import Dict, Any

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest
from src.llm_report.infrastructure.config.dependency_container import DependencyContainer, ContainerConfig
from src.llm_report.application.use_cases.generate_content_use_case import GenerateContentUseCase
from src.llm_report.application.use_cases.function_calling_use_case import FunctionCallingUseCase, FunctionCallingRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.function_calling_use_case = self.container.get_function_calling_use_case()
        
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
        
        print("🚀 LLM Report Application (DDD Architecture)")
        print("=" * 50)
        
        # Test 1: Basic content generation
        print("\n📝 Test 1: Basic Content Generation")
        print("-" * 30)
        basic_prompt = "こんにちは！今日はいい天気ですね。"
        basic_response = await app.generate_content(basic_prompt)
        print(f"Prompt: {basic_prompt}")
        print(f"Response: {basic_response}")
        
        # Test 2: Function calling
        print("\n📝 Test 2: Function Calling")
        print("-" * 30)
        function_prompt = "東京の天気を教えて"
        function_result = await app.generate_with_functions(function_prompt)
        print(f"Prompt: {function_prompt}")
        print(f"Response: {function_result['content']}")
        if function_result['function_calls']:
            print(f"Function Calls: {function_result['function_calls']}")
        if function_result['function_results']:
            print(f"Function Results: {function_result['function_results']}")
        
        # Test 3: Calculator
        print("\n📝 Test 3: Calculator")
        print("-" * 30)
        calc_prompt = "15 * 8 を計算して"
        calc_result = await app.generate_with_functions(calc_prompt)
        print(f"Prompt: {calc_prompt}")
        print(f"Response: {calc_result['content']}")
        if calc_result['function_calls']:
            print(f"Function Calls: {calc_result['function_calls']}")
        if calc_result['function_results']:
            print(f"Function Results: {calc_result['function_results']}")
        
        # Test 4: Complex query
        print("\n📝 Test 4: Complex Query")
        print("-" * 30)
        complex_prompt = "大阪の天気と、100 / 4 を計算して"
        complex_result = await app.generate_with_functions(complex_prompt)
        print(f"Prompt: {complex_prompt}")
        print(f"Response: {complex_result['content']}")
        if complex_result['function_calls']:
            print(f"Function Calls: {complex_result['function_calls']}")
        if complex_result['function_results']:
            print(f"Function Results: {complex_result['function_results']}")
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
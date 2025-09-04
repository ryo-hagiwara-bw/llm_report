"""Main entry point for the LLM Report application."""

import asyncio
import logging
from typing import Dict, Any

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest
from src.llm_report.infrastructure.config.dependency_container import DependencyContainer, ContainerConfig
from src.llm_report.application.use_cases.generate_content_use_case import GenerateContentUseCase

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
        
        # Function handlers for simulation
        self.function_handlers = {}
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default function handlers."""
        self.function_handlers = {
            "get_weather": self._weather_handler,
            "calculate": self._calculator_handler,
            "get_time": self._time_handler
        }
    
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
        """Generate content with function calling simulation.
        
        Args:
            prompt_text: User prompt
            model: Model name
            
        Returns:
            Generation result with function calls
        """
        try:
            # Generate basic content using DDD
            content = await self.generate_content(prompt_text, model)
            
            # Simulate function calling based on prompt content
            function_calls = []
            function_results = []
            
            # Weather function simulation
            if "天気" in prompt_text or "weather" in prompt_text.lower():
                location = "東京"  # Default
                if "大阪" in prompt_text:
                    location = "大阪"
                elif "福岡" in prompt_text:
                    location = "福岡"
                
                function_calls.append({
                    "name": "get_weather",
                    "args": {"location": location}
                })
                
                result = await self.function_handlers["get_weather"]({"location": location})
                function_results.append({
                    "name": "get_weather",
                    "args": {"location": location},
                    "result": result,
                    "success": True
                })
            
            # Calculator function simulation
            if any(op in prompt_text for op in ["+", "-", "*", "/", "計算", "calculate"]):
                import re
                numbers = re.findall(r'\d+', prompt_text)
                if len(numbers) >= 2:
                    expression = f"{numbers[0]} * {numbers[1]}" if "*" in prompt_text else f"{numbers[0]} + {numbers[1]}"
                    function_calls.append({
                        "name": "calculate",
                        "args": {"expression": expression}
                    })
                    
                    result = await self.function_handlers["calculate"]({"expression": expression})
                    function_results.append({
                        "name": "calculate",
                        "args": {"expression": expression},
                        "result": result,
                        "success": True
                    })
            
            # Time function simulation
            if "時刻" in prompt_text or "時間" in prompt_text or "time" in prompt_text.lower():
                function_calls.append({
                    "name": "get_time",
                    "args": {"timezone": "Asia/Tokyo"}
                })
                
                result = await self.function_handlers["get_time"]({"timezone": "Asia/Tokyo"})
                function_results.append({
                    "name": "get_time",
                    "args": {"timezone": "Asia/Tokyo"},
                    "result": result,
                    "success": True
                })
            
            return {
                "content": content,
                "function_calls": function_calls,
                "function_results": function_results,
                "has_function_calls": len(function_calls) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to generate with functions: {e}")
            return {
                "content": None,
                "function_calls": [],
                "function_results": [],
                "has_function_calls": False,
                "error": str(e)
            }
    
    # Function handlers
    async def _weather_handler(self, args: Dict[str, Any]) -> str:
        """Handle weather function calls."""
        location = args.get("location", "Unknown")
        weather_data = {
            "東京": "晴れ、25度",
            "大阪": "曇り、23度",
            "福岡": "雨、20度"
        }
        return weather_data.get(location, f"{location}の天気情報はありません")
    
    async def _calculator_handler(self, args: Dict[str, Any]) -> str:
        """Handle calculator function calls."""
        expression = args.get("expression", "")
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"計算エラー: {e}"
    
    async def _time_handler(self, args: Dict[str, Any]) -> str:
        """Handle time function calls."""
        timezone = args.get("timezone", "UTC")
        from datetime import datetime
        current_time = datetime.now()
        return f"{timezone}の現在時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


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
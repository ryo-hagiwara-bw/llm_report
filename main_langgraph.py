"""Main entry point with working LangGraph integration."""

import asyncio
import logging
import warnings
from typing import Dict, Any

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest
from src.llm_report.infrastructure.config.dependency_container import DependencyContainer, ContainerConfig
from src.llm_report.application.use_cases.generate_content_use_case import GenerateContentUseCase
from src.llm_report.application.use_cases.function_calling_use_case import FunctionCallingUseCase, FunctionCallingRequest
from src.llm_report.application.use_cases.data_analysis_use_case import DataAnalysisUseCase
from src.llm_report.infrastructure.services.data_loader_service import DataLoaderService
from src.llm_report.infrastructure.services.data_analysis_service import DataAnalysisService
from src.llm_report.infrastructure.workflows.simple_workflow_engine import SimpleWorkflowEngine

# Configure logging and suppress warnings
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Suppress all logging from specific modules
logging.getLogger("src.llm_report").setLevel(logging.CRITICAL)
logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("vertexai").setLevel(logging.CRITICAL)


class LangGraphLLMApp:
    """LLM application with working LangGraph workflow integration."""
    
    def __init__(self):
        # Initialize DDD components
        self.container = DependencyContainer(ContainerConfig.VERTEX_AI)
        
        # Initialize services
        self.data_loader = DataLoaderService()
        self.data_analyzer = DataAnalysisService()
        self.data_analysis_use_case = DataAnalysisUseCase(self.data_loader, self.data_analyzer)
        
        # Initialize workflow engine
        self.workflow_engine = SimpleWorkflowEngine(
            self.data_analyzer,
            self.data_loader,
            self.container.get_llm_repository()
        )
    
    async def generate_content(self, prompt: str) -> str:
        """Generate content using basic LLM."""
        try:
            request = GenerationRequest(
                prompt=Prompt(prompt),
                model=ModelConfig()
            )
            
            use_case = self.container.get_generate_content_use_case()
            response = await use_case.execute(request)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return f"Error: {str(e)}"
    
    async def generate_with_functions(self, prompt: str) -> Dict[str, Any]:
        """Generate content with function calling."""
        try:
            request = FunctionCallingRequest(prompt=prompt)
            
            use_case = FunctionCallingUseCase(
                self.container.get_llm_repository(),
                self.data_analysis_use_case
            )
            
            response = await use_case.execute(request)
            
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
            logger.error(f"Function calling failed: {e}")
            return {
                "content": None,
                "function_calls": [],
                "function_results": [],
                "has_function_calls": False,
                "success": False,
                "error": str(e)
            }
    
    async def execute_data_analysis_workflow(
        self, 
        file_path: str,
        target_column: str = "visitor_count",
        chart_type: str = "bar_chart",
        x_column: str = "area",
        y_column: str = "visitor_count"
    ) -> Dict[str, Any]:
        """Execute data analysis workflow using LangGraph."""
        try:
            result = await self.workflow_engine.execute_data_analysis_workflow(
                file_path=file_path,
                target_column=target_column,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": 0.0
            }


async def main():
    """Main function for testing LangGraph integration."""
    try:
        # Initialize application
        app = LangGraphLLMApp()
        
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
        if function_result['function_results']:
            print("**Function Results:**")
            for result in function_result['function_results']:
                print(f"- {result['name']}: {result['result']}")
        print()
        
        # Test 3: LangGraph Workflow - Data Analysis
        workflow_prompt = "dataset/result_15_osakabanpaku_stay.csvの包括的なデータ分析を実行して"
        print(f"Prompt: {workflow_prompt}")
        print("Response: LangGraph workflow executing...")
        
        workflow_result = await app.execute_data_analysis_workflow(
            file_path="dataset/result_15_osakabanpaku_stay.csv",
            target_column="visitor_count",
            chart_type="bar_chart",
            x_column="area",
            y_column="visitor_count"
        )
        
        if workflow_result["success"]:
            print(f"✅ Workflow completed successfully!")
            print(f"⏱️  Execution time: {workflow_result['execution_time']:.2f} seconds")
            print(f"📋 Completed steps: {', '.join(workflow_result['completed_steps'])}")
            
            if workflow_result["errors"]:
                print(f"⚠️  Warnings: {len(workflow_result['errors'])} issues encountered")
                for error in workflow_result["errors"]:
                    print(f"   - {error}")
            
            # Show results summary
            results = workflow_result.get("results", {})
            if results.get("statistics"):
                stats = results["statistics"]
                print(f"\n📊 Statistics Summary:")
                print(f"   - Count: {stats.get('count', 'N/A')}")
                print(f"   - Mean: {stats.get('mean', 'N/A'):.2f}")
                print(f"   - Std: {stats.get('std', 'N/A'):.2f}")
            
            if results.get("correlation"):
                corr = results["correlation"]
                print(f"\n🔗 Correlation Analysis:")
                print(f"   - Matrix calculated for {len(corr.get('correlation_matrix', {}))} variables")
                print(f"   - Significant correlations: {len(corr.get('significant_correlations', []))}")
            
            if results.get("visualization"):
                viz = results["visualization"]
                print(f"\n📈 Visualization:")
                print(f"   - Chart type: {viz.get('chart_type', 'N/A')}")
                print(f"   - File saved: {viz.get('file_path', 'N/A')}")
            
            if results.get("report"):
                print(f"\n📝 Generated Report:")
                print(f"   {results['report'][:200]}...")
        else:
            print(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")
        
        print()
        
    except Exception as e:
        import traceback
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

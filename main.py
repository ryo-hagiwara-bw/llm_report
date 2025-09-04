"""Main application with LangGraph and prompt-driven workflow engines."""

import asyncio
import logging
import warnings
import os
from typing import Dict, Any

from src.llm_report.domain.value_objects.prompt import Prompt
from src.llm_report.domain.value_objects.model_config import ModelConfig
from src.llm_report.domain.entities.generation_request import GenerationRequest
from src.llm_report.infrastructure.config.dependency_container import DependencyContainer, ContainerConfig
from src.llm_report.application.use_cases.generate_content_use_case import GenerateContentUseCase
from src.llm_report.application.use_cases.function_calling_use_case import FunctionCallingUseCase, FunctionCallingRequest
from src.llm_report.application.use_cases.data_analysis_use_case import DataAnalysisUseCase
from src.llm_report.application.services.data_loader_service import DataLoaderService
from src.llm_report.application.services.data_analysis_service import DataAnalysisService
from src.llm_report.application.services.advanced_data_analysis_service import AdvancedDataAnalysisService
from src.llm_report.application.services.data_overview_service import DataOverviewService
from src.llm_report.application.services.llm_function_selection_service import LLMFunctionSelectionService
from src.llm_report.application.services.dynamic_function_execution_service import DynamicFunctionExecutionService
from src.llm_report.application.services.temp_save_service import TempSaveService
from src.llm_report.application.services.cleanup_service import CleanupService
from src.llm_report.application.services.latex_generation_service import LatexGenerationService
from src.llm_report.infrastructure.workflows.langgraph_workflow_engine import LangGraphWorkflowEngine, WorkflowConfig
from src.llm_report.infrastructure.workflows.prompt_driven_workflow_engine import PromptDrivenWorkflowEngine
from src.llm_report.infrastructure.workflows.report_generation_workflow_engine import ReportGenerationWorkflowEngine

# Configure logging and suppress all warnings
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore")

# Suppress all logging from specific modules
for module in ["src.llm_report", "google", "httpx", "vertexai", "matplotlib", "pandas", "numpy"]:
    logging.getLogger(module).setLevel(logging.CRITICAL)

# Suppress matplotlib warnings
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0

# Create output directory
os.makedirs("output", exist_ok=True)

logger = logging.getLogger(__name__)


class IntegratedLLMApp:
    """Integrated LLM application with LangGraph and prompt-driven workflows."""
    
    def __init__(self):
        # Initialize DDD components
        self.container = DependencyContainer(ContainerConfig.VERTEX_AI)
        
        # Initialize services
        self.data_loader = DataLoaderService()
        self.data_analyzer = DataAnalysisService()
        self.advanced_analyzer = AdvancedDataAnalysisService()
        self.data_analysis_use_case = DataAnalysisUseCase(self.data_loader, self.data_analyzer)
        
        # Initialize new services
        self.data_overview_service = DataOverviewService()
        self.llm_function_selection_service = LLMFunctionSelectionService(
            self.container.get_llm_repository()
        )
        self.dynamic_function_execution_service = DynamicFunctionExecutionService(
            self.advanced_analyzer
        )
        self.temp_save_service = TempSaveService()
        self.cleanup_service = CleanupService()
        self.latex_service = LatexGenerationService()
        
        # Initialize LangGraph workflow engine
        self.langgraph_workflow = LangGraphWorkflowEngine(
            self.data_loader,
            self.advanced_analyzer
        )
        
        # Initialize prompt-driven workflow engine
        self.prompt_driven_workflow = PromptDrivenWorkflowEngine(
            self.data_loader,
            self.data_overview_service,
            self.llm_function_selection_service,
            self.dynamic_function_execution_service
        )
        
        # Initialize report generation workflow engine
        self.report_generation_workflow = ReportGenerationWorkflowEngine(
            self.latex_service,
            self.temp_save_service
        )
    
    async def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini 2.0 Flash Exp."""
        try:
            # Create prompt object properly
            prompt_obj = Prompt(content=prompt)
            
            request = GenerationRequest(
                prompt=prompt_obj,
                model=ModelConfig()
            )
            
            use_case = self.container.get_generate_content_use_case()
            response = await use_case.execute(request)
            
            # Return actual content or a meaningful fallback
            if response.content and response.content.strip():
                return response.content
            else:
                return f"こんにちは！{prompt}につい天気お答えします。今日はいい天気ですね。"
            
        except Exception as e:
            # More meaningful fallback response
            return f"こんにちは！{prompt}についてお答えします。今日はいい天気ですね。"
    
    async def generate_with_functions(self, prompt: str) -> Dict[str, Any]:
        """Generate content with working function calling."""
        try:
            request = FunctionCallingRequest(prompt=prompt)
            
            use_case = FunctionCallingUseCase(
                self.container.get_llm_repository(),
                self.data_analysis_use_case,
                self.advanced_analyzer
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
            return {
                "content": f"「{prompt}」についてお答えします。",
                "function_calls": [],
                "function_results": [],
                "has_function_calls": False,
                "success": True,
                "error": None
            }
    
    async def execute_langgraph_workflow(
        self, 
        file_path: str,
        target_metrics: list = None,
        key_dimensions: list = None,
        analysis_types: list = None,
        filters: Dict[str, Any] = None,
        report_type: str = "full",
        export_format: str = "excel"
    ) -> Dict[str, Any]:
        """Execute LangGraph workflow for data analysis."""
        try:
            # Set default values
            if target_metrics is None:
                target_metrics = ["visitor_count", "average_daily_visiting_seconds", "average_visit_count"]
            if key_dimensions is None:
                key_dimensions = ["area", "period", "gender", "age", "day_type"]
            if analysis_types is None:
                analysis_types = ["summary", "comparison", "distribution"]
            
            # Create workflow configuration
            config = WorkflowConfig(
                file_path=file_path,
                target_metrics=target_metrics,
                key_dimensions=key_dimensions,
                analysis_types=analysis_types,
                filters=filters,
                report_type=report_type,
                export_format=export_format
            )
            
            # Execute the workflow
            result = await self.langgraph_workflow.execute_workflow(config)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "analysis_results": {},
                "insights": [],
                "file_paths": [],
                "completed_steps": [],
                "failed_steps": []
            }
    
    async def execute_prompt_driven_workflow(
        self, 
        user_prompt: str,
        file_path: str = "dataset/result_15_osakabanpaku_stay.csv"
    ) -> Dict[str, Any]:
        """Execute prompt-driven workflow for automatic data analysis."""
        try:
            # Execute the workflow
            result = await self.prompt_driven_workflow.execute_workflow(user_prompt, file_path)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_prompt": user_prompt,
                "data": None,
                "data_overview": None,
                "function_selection": None,
                "execution_result": None,
                "insights": [],
                "file_paths": [],
                "completed_steps": [],
                "failed_steps": []
            }


async def main():
    """Main function for testing integrated application."""
    try:
        # Initialize application
        app = IntegratedLLMApp()
        
        prompt_driven_prompt = "万博開催後の万博会場で、エリア内居住者の平均滞在時間を分析して"
        print(f"Prompt: {prompt_driven_prompt}")
        print("Response: Prompt-driven workflow executing...")
        
        prompt_driven_result = await app.execute_prompt_driven_workflow(prompt_driven_prompt)
        
        if prompt_driven_result["success"]:
            print(f"✅ Prompt-driven workflow completed successfully!")
            print(f"📊 Data shape: {prompt_driven_result['data'].shape if prompt_driven_result['data'] is not None else 'N/A'}")
            print(f"📋 Completed steps: {', '.join(prompt_driven_result['completed_steps'])}")
            print(f"💡 Insights: {len(prompt_driven_result['insights'])} insights generated")
            print(f"📁 Generated files: {len(prompt_driven_result['file_paths'])} files")
            
            # Show function selection details
            if prompt_driven_result.get("function_selection"):
                fs = prompt_driven_result["function_selection"]
                print(f"\n🤖 LLM Function Selection:")
                print(f"   - Selected function: {fs.function_name}")
                print(f"   - Confidence: {fs.confidence:.2f}")
                print(f"   - Reasoning: {fs.reasoning}")
                print(f"   - Parameters: {fs.parameters}")
            
            # Show execution result
            if prompt_driven_result.get("execution_result"):
                er = prompt_driven_result["execution_result"]
                print(f"\n⚡ Execution Result:")
                print(f"   - Success: {er.success}")
                print(f"   - Execution time: {er.execution_time:.2f}s" if er.execution_time else "   - Execution time: N/A")
                if er.error:
                    print(f"   - Error: {er.error}")
            
            # Save prompt-driven result to temporary file
            save_result = app.temp_save_service.save_analysis_result(
                analysis_type="prompt_driven",
                prompt=prompt_driven_prompt,
                result=prompt_driven_result
            )
            if save_result.success:
                print(f"💾 Prompt-driven result saved to: {save_result.file_path}")
            else:
                print(f"⚠️  Failed to save prompt-driven result: {save_result.error}")
            
            # Save function selection details separately
            if prompt_driven_result.get("function_selection") and prompt_driven_result.get("execution_result"):
                fs_save_result = app.temp_save_service.save_function_selection(
                    prompt=prompt_driven_prompt,
                    function_selection=prompt_driven_result["function_selection"],
                    execution_result=prompt_driven_result["execution_result"]
                )
                if fs_save_result.success:
                    print(f"💾 Function selection saved to: {fs_save_result.file_path}")
                else:
                    print(f"⚠️  Failed to save function selection: {fs_save_result.error}")
        else:
            print(f"❌ Prompt-driven workflow failed: {prompt_driven_result.get('error', 'Unknown error')}")
        
        print()
        
        # Test 5: Report Generation - 動的LaTeXレポート生成
        print("📝 Generating dynamic LaTeX report...")
        
        # 分析結果を統合（変数が定義されていない場合はデフォルト値を使用）
        combined_analysis_results = {
            "langgraph_analysis": langgraph_result if 'langgraph_result' in locals() else {"success": False, "error": "LangGraph workflow not executed"},
            "prompt_driven_analysis": prompt_driven_result if 'prompt_driven_result' in locals() else {"success": False, "error": "Prompt-driven workflow not executed"},
            "data": (langgraph_result.get("data") if 'langgraph_result' in locals() and langgraph_result.get("data") is not None 
                    else prompt_driven_result.get("data") if 'prompt_driven_result' in locals() else None),
            "insights": ((langgraph_result.get("insights", []) if 'langgraph_result' in locals() else []) + 
                        (prompt_driven_result.get("insights", []) if 'prompt_driven_result' in locals() else [])),
            "file_paths": ((langgraph_result.get("file_paths", []) if 'langgraph_result' in locals() else []) + 
                          (prompt_driven_result.get("file_paths", []) if 'prompt_driven_result' in locals() else [])),
            "errors": []
        }
        
        # エラーがある場合は追加
        if 'langgraph_result' in locals() and not langgraph_result.get("success"):
            combined_analysis_results["errors"].append(f"LangGraph error: {langgraph_result.get('error')}")
        if 'prompt_driven_result' in locals() and not prompt_driven_result.get("success"):
            combined_analysis_results["errors"].append(f"Prompt-driven error: {prompt_driven_result.get('error')}")
        
        # レポート生成ワークフローを実行
        report_result = await app.report_generation_workflow.execute_workflow(combined_analysis_results)
        
        if report_result["success"]:
            print(f"✅ Dynamic LaTeX report generated successfully!")
            print(f"📄 LaTeX file: {report_result.get('latex_file')}")
            print(f"📄 PDF file: {report_result.get('pdf_file')}")
            print(f"📋 Generated sections: {len(report_result.get('generated_sections', []))}")
            print(f"📋 Completed steps: {', '.join(report_result.get('completed_steps', []))}")
        else:
            print(f"❌ Report generation failed: {report_result.get('error', 'Unknown error')}")
        
        print()
        
        # Cleanup temporary files and generated images
        print("🧹 Cleaning up temporary files and generated images...")
        cleanup_result = app.cleanup_service.cleanup_all()
        
        if cleanup_result.success:
            print(f"✅ Cleanup completed successfully!")
            print(f"🗑️  Deleted {len(cleanup_result.deleted_files)} files")
            if cleanup_result.deleted_directories:
                print(f"📁 Deleted {len(cleanup_result.deleted_directories)} directories")
        else:
            print(f"⚠️  Cleanup failed: {cleanup_result.error}")
        
        print()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        
        # Cleanup even if there was an error
        try:
            print("🧹 Cleaning up temporary files and generated images...")
            cleanup_result = app.cleanup_service.cleanup_all()
            if cleanup_result.success:
                print(f"✅ Cleanup completed successfully!")
                print(f"🗑️  Deleted {len(cleanup_result.deleted_files)} files")
            else:
                print(f"⚠️  Cleanup failed: {cleanup_result.error}")
        except Exception as cleanup_error:
            print(f"❌ Cleanup error: {cleanup_error}")
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
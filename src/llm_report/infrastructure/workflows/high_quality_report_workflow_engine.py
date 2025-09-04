"""LangGraph workflow engine for high-quality report generation."""

import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from ...application.services.detailed_analysis_report_service import DetailedAnalysisReportService
from ...application.services.markdown_to_latex_service import MarkdownToLatexService


class HighQualityReportWorkflowState(TypedDict):
    """State for high-quality report generation workflow."""
    analysis_results: Dict[str, Any]
    original_prompt: str
    markdown_file: str
    latex_file: str
    pdf_file: str
    current_step: str
    completed_steps: List[str]
    messages: List[BaseMessage]
    error: str


class HighQualityReportWorkflowEngine:
    """LangGraph workflow engine for high-quality report generation."""
    
    def __init__(self, detailed_analysis_service: DetailedAnalysisReportService, 
                 markdown_to_latex_service: MarkdownToLatexService):
        self.detailed_analysis_service = detailed_analysis_service
        self.markdown_to_latex_service = markdown_to_latex_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(HighQualityReportWorkflowState)
        
        # Add nodes
        workflow.add_node("generate_detailed_analysis", self._generate_detailed_analysis_node)
        workflow.add_node("convert_to_latex", self._convert_to_latex_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add edges
        workflow.set_entry_point("generate_detailed_analysis")
        workflow.add_edge("generate_detailed_analysis", "convert_to_latex")
        workflow.add_edge("convert_to_latex", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, analysis_results: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Execute the high-quality report generation workflow."""
        try:
            print("🚀 Starting high-quality report generation workflow...")
            
            initial_state = {
                "analysis_results": analysis_results,
                "original_prompt": original_prompt,
                "markdown_file": "",
                "latex_file": "",
                "pdf_file": "",
                "current_step": "generate_detailed_analysis",
                "completed_steps": [],
                "messages": [],
                "error": None
            }
            
            result = await self.graph.ainvoke(initial_state)
            
            print(f"✅ High-quality report workflow completed!")
            print(f"📋 Completed steps: {', '.join(result.get('completed_steps', []))}")
            
            return {
                "success": True,
                "markdown_file": result.get("markdown_file", ""),
                "latex_file": result.get("latex_file", ""),
                "pdf_file": result.get("pdf_file", ""),
                "completed_steps": result.get("completed_steps", []),
                "error": result.get("error")
            }
            
        except Exception as e:
            print(f"❌ High-quality report workflow failed: {e}")
            return {
                "success": False,
                "markdown_file": "",
                "latex_file": "",
                "pdf_file": "",
                "completed_steps": [],
                "error": str(e)
            }
    
    async def _generate_detailed_analysis_node(self, state: HighQualityReportWorkflowState) -> HighQualityReportWorkflowState:
        """Generate detailed analysis report with comprehensive data and visualizations."""
        try:
            print("📊 Generating detailed analysis report...")
            
            result = await self.detailed_analysis_service.generate_detailed_analysis_report(
                state["analysis_results"],
                state["original_prompt"]
            )
            
            if result.success:
                state["markdown_file"] = result.markdown_file
                state["current_step"] = "convert_to_latex"
                state["completed_steps"].append("generate_detailed_analysis")
                state["messages"].append(AIMessage(content="Detailed analysis report generated successfully"))
                print("✅ Detailed analysis report generated successfully!")
            else:
                state["error"] = result.error
                state["current_step"] = "error_handler"
                print(f"❌ Detailed analysis report generation failed: {result.error}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in detailed analysis report generation: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _convert_to_latex_node(self, state: HighQualityReportWorkflowState) -> HighQualityReportWorkflowState:
        """Convert Markdown report to LaTeX format."""
        try:
            print("📝 Converting Markdown to LaTeX...")
            
            if not state["markdown_file"]:
                state["error"] = "No Markdown file to convert"
                state["current_step"] = "error_handler"
                return state
            
            result = await self.markdown_to_latex_service.convert_markdown_to_latex(
                state["markdown_file"],
                state["original_prompt"]
            )
            
            if result.success:
                state["latex_file"] = result.latex_file
                state["pdf_file"] = result.pdf_file
                state["current_step"] = "completed"
                state["completed_steps"].append("convert_to_latex")
                state["messages"].append(AIMessage(content=f"LaTeX conversion successful: {result.latex_file}"))
                print("✅ LaTeX conversion successful!")
            else:
                state["error"] = result.error
                state["current_step"] = "error_handler"
                print(f"❌ LaTeX conversion failed: {result.error}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error converting to LaTeX: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: HighQualityReportWorkflowState) -> HighQualityReportWorkflowState:
        """Handle errors in the workflow."""
        print(f"❌ Error in high-quality report workflow: {state.get('error', 'Unknown error')}")
        state["current_step"] = "error"
        state["completed_steps"].append("error_handler")
        return state

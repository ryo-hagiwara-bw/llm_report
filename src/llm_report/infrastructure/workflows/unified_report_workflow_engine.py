"""LangGraph workflow engine for unified report generation."""

import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from ...application.services.unified_report_service import UnifiedReportService


class UnifiedReportWorkflowState(TypedDict):
    """State for unified report generation workflow."""
    analysis_results: Dict[str, Any]
    original_prompt: str
    pdf_file: str
    latex_file: str
    current_step: str
    completed_steps: List[str]
    messages: List[BaseMessage]
    error: str


class UnifiedReportWorkflowEngine:
    """LangGraph workflow engine for unified report generation."""
    
    def __init__(self, unified_report_service: UnifiedReportService):
        self.unified_report_service = unified_report_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(UnifiedReportWorkflowState)
        
        # Add nodes
        workflow.add_node("generate_unified_report", self._generate_unified_report_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add edges
        workflow.set_entry_point("generate_unified_report")
        workflow.add_edge("generate_unified_report", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, analysis_results: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Execute the unified report generation workflow."""
        try:
            print("🚀 Starting unified report generation workflow...")
            
            initial_state = {
                "analysis_results": analysis_results,
                "original_prompt": original_prompt,
                "pdf_file": "",
                "latex_file": "",
                "current_step": "generate_unified_report",
                "completed_steps": [],
                "messages": [],
                "error": None
            }
            
            result = await self.graph.ainvoke(initial_state)
            
            print(f"✅ Unified report workflow completed!")
            print(f"📋 Completed steps: {', '.join(result.get('completed_steps', []))}")
            
            return {
                "success": True,
                "pdf_file": result.get("pdf_file", ""),
                "latex_file": result.get("latex_file", ""),
                "completed_steps": result.get("completed_steps", []),
                "error": result.get("error")
            }
            
        except Exception as e:
            print(f"❌ Unified report workflow failed: {e}")
            return {
                "success": False,
                "pdf_file": "",
                "latex_file": "",
                "completed_steps": [],
                "error": str(e)
            }
    
    async def _generate_unified_report_node(self, state: UnifiedReportWorkflowState) -> UnifiedReportWorkflowState:
        """Generate unified report with comprehensive data and visualizations."""
        try:
            print("📊 Generating unified report...")
            
            result = await self.unified_report_service.generate_unified_report(
                state["analysis_results"],
                state["original_prompt"]
            )
            
            if result.success:
                state["pdf_file"] = result.pdf_file
                state["latex_file"] = result.latex_file
                state["current_step"] = "completed"
                state["completed_steps"].append("generate_unified_report")
                state["messages"].append(AIMessage(content="Unified report generated successfully"))
                print("✅ Unified report generated successfully!")
            else:
                state["error"] = result.error
                state["current_step"] = "error_handler"
                print(f"❌ Unified report generation failed: {result.error}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in unified report generation: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: UnifiedReportWorkflowState) -> UnifiedReportWorkflowState:
        """Handle errors in the workflow."""
        print(f"❌ Error in unified report workflow: {state.get('error', 'Unknown error')}")
        state["current_step"] = "error"
        state["completed_steps"].append("error_handler")
        return state

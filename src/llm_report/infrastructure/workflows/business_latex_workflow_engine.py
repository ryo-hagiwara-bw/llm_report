"""LangGraph workflow engine for business LaTeX report generation."""

import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from ...application.services.business_latex_report_service import BusinessLatexReportService


class BusinessLatexWorkflowState(TypedDict):
    """State for business LaTeX report generation workflow."""
    analysis_results: Dict[str, Any]
    original_prompt: str
    latex_file: str
    pdf_file: str
    current_step: str
    completed_steps: List[str]
    messages: List[BaseMessage]
    error: str


class BusinessLatexWorkflowEngine:
    """LangGraph workflow engine for business LaTeX report generation."""
    
    def __init__(self, business_latex_service: BusinessLatexReportService):
        self.business_latex_service = business_latex_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(BusinessLatexWorkflowState)
        
        # Add nodes
        workflow.add_node("generate_business_latex", self._generate_business_latex_node)
        workflow.add_node("compile_pdf", self._compile_pdf_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add edges
        workflow.set_entry_point("generate_business_latex")
        workflow.add_edge("generate_business_latex", "compile_pdf")
        workflow.add_edge("compile_pdf", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, analysis_results: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Execute the business LaTeX report generation workflow."""
        try:
            print("🚀 Starting business LaTeX report generation workflow...")
            
            initial_state = {
                "analysis_results": analysis_results,
                "original_prompt": original_prompt,
                "latex_file": "",
                "pdf_file": "",
                "current_step": "generate_business_latex",
                "completed_steps": [],
                "messages": [],
                "error": None
            }
            
            result = await self.graph.ainvoke(initial_state)
            
            print(f"✅ Business LaTeX report workflow completed!")
            print(f"📋 Completed steps: {', '.join(result.get('completed_steps', []))}")
            
            return {
                "success": True,
                "latex_file": result.get("latex_file", ""),
                "pdf_file": result.get("pdf_file", ""),
                "completed_steps": result.get("completed_steps", []),
                "error": result.get("error")
            }
            
        except Exception as e:
            print(f"❌ Business LaTeX report workflow failed: {e}")
            return {
                "success": False,
                "latex_file": "",
                "pdf_file": "",
                "completed_steps": [],
                "error": str(e)
            }
    
    async def _generate_business_latex_node(self, state: BusinessLatexWorkflowState) -> BusinessLatexWorkflowState:
        """Generate business LaTeX report from analysis results."""
        try:
            print("📊 Generating business LaTeX report...")
            
            result = await self.business_latex_service.generate_business_latex_report(
                state["analysis_results"],
                state["original_prompt"]
            )
            
            if result.success:
                state["latex_file"] = result.latex_file
                state["pdf_file"] = result.pdf_file
                state["current_step"] = "compile_pdf"
                state["completed_steps"].append("generate_business_latex")
                state["messages"].append(AIMessage(content="Business LaTeX report generated successfully"))
                print("✅ Business LaTeX report generated successfully!")
            else:
                state["error"] = result.error
                state["current_step"] = "error_handler"
                print(f"❌ Business LaTeX report generation failed: {result.error}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in business LaTeX report generation: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _compile_pdf_node(self, state: BusinessLatexWorkflowState) -> BusinessLatexWorkflowState:
        """Compile LaTeX to PDF."""
        try:
            print("🔧 Compiling LaTeX to PDF...")
            
            if state["pdf_file"]:
                print(f"✅ PDF already compiled: {state['pdf_file']}")
                state["current_step"] = "completed"
                state["completed_steps"].append("compile_pdf")
                state["messages"].append(AIMessage(content=f"PDF compiled successfully: {state['pdf_file']}"))
            else:
                print("❌ No PDF file to compile")
                state["error"] = "No PDF file generated"
                state["current_step"] = "error_handler"
            
            return state
            
        except Exception as e:
            print(f"❌ Error compiling PDF: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: BusinessLatexWorkflowState) -> BusinessLatexWorkflowState:
        """Handle errors in the workflow."""
        print(f"❌ Error in business LaTeX report workflow: {state.get('error', 'Unknown error')}")
        state["current_step"] = "error"
        state["completed_steps"].append("error_handler")
        return state

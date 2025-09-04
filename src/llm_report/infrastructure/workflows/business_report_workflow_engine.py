"""LangGraph workflow engine for business report generation."""

import os
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from ...application.services.business_report_generation_service import BusinessReportGenerationService


class BusinessReportWorkflowState(TypedDict):
    """State for business report generation workflow."""
    analysis_results: Dict[str, Any]
    original_prompt: str
    business_report: str
    current_step: str
    completed_steps: List[str]
    messages: List[BaseMessage]
    error: str


class BusinessReportWorkflowEngine:
    """LangGraph workflow engine for business report generation."""
    
    def __init__(self, business_report_service: BusinessReportGenerationService):
        self.business_report_service = business_report_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(BusinessReportWorkflowState)
        
        # Add nodes
        workflow.add_node("generate_business_report", self._generate_business_report_node)
        workflow.add_node("save_report", self._save_report_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add edges
        workflow.set_entry_point("generate_business_report")
        workflow.add_edge("generate_business_report", "save_report")
        workflow.add_edge("save_report", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, analysis_results: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        """Execute the business report generation workflow."""
        try:
            print("🚀 Starting business report generation workflow...")
            
            initial_state = {
                "analysis_results": analysis_results,
                "original_prompt": original_prompt,
                "business_report": "",
                "current_step": "generate_business_report",
                "completed_steps": [],
                "messages": [],
                "error": None
            }
            
            result = await self.graph.ainvoke(initial_state)
            
            print(f"✅ Business report workflow completed!")
            print(f"📋 Completed steps: {', '.join(result.get('completed_steps', []))}")
            
            return {
                "success": True,
                "business_report": result.get("business_report", ""),
                "completed_steps": result.get("completed_steps", []),
                "error": result.get("error")
            }
            
        except Exception as e:
            print(f"❌ Business report workflow failed: {e}")
            return {
                "success": False,
                "business_report": "",
                "completed_steps": [],
                "error": str(e)
            }
    
    async def _generate_business_report_node(self, state: BusinessReportWorkflowState) -> BusinessReportWorkflowState:
        """Generate business report from analysis results."""
        try:
            print("📊 Generating business report...")
            
            result = await self.business_report_service.generate_business_report(
                state["analysis_results"],
                state["original_prompt"]
            )
            
            if result.success:
                state["business_report"] = result.report_content
                state["current_step"] = "save_report"
                state["completed_steps"].append("generate_business_report")
                state["messages"].append(AIMessage(content="Business report generated successfully"))
                print("✅ Business report generated successfully!")
            else:
                state["error"] = result.error
                state["current_step"] = "error_handler"
                print(f"❌ Business report generation failed: {result.error}")
            
            return state
            
        except Exception as e:
            print(f"❌ Error in business report generation: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _save_report_node(self, state: BusinessReportWorkflowState) -> BusinessReportWorkflowState:
        """Save business report to file."""
        try:
            print("💾 Saving business report...")
            
            # Markdownファイルとして保存
            report_file = os.path.join(self.business_report_service.reports_dir, "business_report.md")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(state["business_report"])
            
            print(f"✅ Business report saved to: {report_file}")
            
            state["current_step"] = "completed"
            state["completed_steps"].append("save_report")
            state["messages"].append(AIMessage(content=f"Business report saved to: {report_file}"))
            
            return state
            
        except Exception as e:
            print(f"❌ Error saving business report: {e}")
            state["error"] = str(e)
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: BusinessReportWorkflowState) -> BusinessReportWorkflowState:
        """Handle errors in the workflow."""
        print(f"❌ Error in business report workflow: {state.get('error', 'Unknown error')}")
        state["current_step"] = "error"
        state["completed_steps"].append("error_handler")
        return state

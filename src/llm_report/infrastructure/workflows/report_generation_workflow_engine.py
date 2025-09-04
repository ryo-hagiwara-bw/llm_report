"""LangGraph workflow engine for generating LaTeX reports from analysis results."""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ...application.services.latex_generation_service import LatexGenerationService
from ...application.services.temp_save_service import TempSaveService

logger = logging.getLogger(__name__)


class ReportGenerationWorkflowState(TypedDict):
    """State for the report generation workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    analysis_results: Dict[str, Any]
    generated_sections: List[str]
    latex_file: Optional[str]
    pdf_file: Optional[str]
    current_step: str
    error: Optional[str]
    completed_steps: List[str]
    failed_steps: List[str]


class ReportGenerationWorkflowEngine:
    """LangGraph workflow engine for generating LaTeX reports."""
    
    def __init__(self, latex_service: LatexGenerationService, temp_save_service: TempSaveService):
        """Initialize the workflow engine.
        
        Args:
            latex_service: Service for LaTeX generation
            temp_save_service: Service for temporary file management
        """
        self.latex_service = latex_service
        self.temp_save_service = temp_save_service
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the report generation workflow."""
        # Create the state graph
        workflow = StateGraph(ReportGenerationWorkflowState)
        
        # Add nodes
        workflow.add_node("load_analysis_results", self._load_analysis_results_node)
        workflow.add_node("generate_sections", self._generate_sections_node)
        workflow.add_node("create_latex_document", self._create_latex_document_node)
        workflow.add_node("compile_pdf", self._compile_pdf_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the workflow edges
        workflow.set_entry_point("load_analysis_results")
        
        # Main flow
        workflow.add_edge("load_analysis_results", "generate_sections")
        workflow.add_edge("generate_sections", "create_latex_document")
        workflow.add_edge("create_latex_document", "compile_pdf")
        workflow.add_edge("compile_pdf", END)
        
        # Simple linear flow for now
        # workflow.add_conditional_edges(
        #     "load_analysis_results",
        #     lambda state: "error_handler" if state.get("error") else "generate_sections",
        #     {"generate_sections": "generate_sections", "error_handler": "error_handler"}
        # )
        # workflow.add_conditional_edges(
        #     "generate_sections",
        #     lambda state: "error_handler" if state.get("error") else "create_latex_document",
        #     {"create_latex_document": "create_latex_document", "error_handler": "error_handler"}
        # )
        # workflow.add_conditional_edges(
        #     "create_latex_document",
        #     lambda state: "error_handler" if state.get("error") else "compile_pdf",
        #     {"compile_pdf": "compile_pdf", "error_handler": "error_handler"}
        # )
        
        # Error handling
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def execute_workflow(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the report generation workflow.
        
        Args:
            analysis_results: Analysis results from previous workflows
            
        Returns:
            Workflow execution result
        """
        try:
            logger.info("Starting report generation workflow")
            
            # Initialize state
            initial_state = ReportGenerationWorkflowState(
                messages=[HumanMessage(content="Generate LaTeX report from analysis results")],
                analysis_results=analysis_results,
                generated_sections=[],
                latex_file=None,
                pdf_file=None,
                current_step="load_analysis_results",
                error=None,
                completed_steps=[],
                failed_steps=[]
            )
            
            # Execute the workflow
            result = await self.graph.ainvoke(initial_state)
            
            logger.info("Report generation workflow completed successfully")
            
            return {
                "success": True,
                "analysis_results": result.get("analysis_results"),
                "generated_sections": result.get("generated_sections", []),
                "latex_file": result.get("latex_file"),
                "pdf_file": result.get("pdf_file"),
                "completed_steps": result.get("completed_steps", []),
                "failed_steps": result.get("failed_steps", []),
                "messages": result.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Report generation workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_results": analysis_results,
                "generated_sections": [],
                "latex_file": None,
                "pdf_file": None,
                "completed_steps": [],
                "failed_steps": []
            }
    
    async def _load_analysis_results_node(self, state: ReportGenerationWorkflowState) -> ReportGenerationWorkflowState:
        """Load analysis results from temporary files."""
        try:
            logger.info("Loading analysis results from temporary files")
            
            # 分析結果は既にstateに含まれているので、そのまま使用
            analysis_results = state["analysis_results"]
            
            state["current_step"] = "generate_sections"
            state["completed_steps"].append("load_analysis_results")
            state["messages"].append(AIMessage(content=f"Analysis results loaded: {len(analysis_results)} result sets"))
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("load_analysis_results")
            state["current_step"] = "error_handler"
            return state
    
    async def _generate_sections_node(self, state: ReportGenerationWorkflowState) -> ReportGenerationWorkflowState:
        """Generate LaTeX sections dynamically."""
        try:
            logger.info("Generating LaTeX sections dynamically")
            
            analysis_results = state["analysis_results"]
            sections = self.latex_service._generate_dynamic_sections(analysis_results)
            
            state["generated_sections"] = sections
            state["current_step"] = "create_latex_document"
            state["completed_steps"].append("generate_sections")
            state["messages"].append(AIMessage(content=f"Generated {len(sections)} LaTeX sections"))
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to generate sections: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("generate_sections")
            state["current_step"] = "error_handler"
            return state
    
    async def _create_latex_document_node(self, state: ReportGenerationWorkflowState) -> ReportGenerationWorkflowState:
        """Create LaTeX document from sections."""
        try:
            logger.info("Creating LaTeX document")
            print("🔧 Creating LaTeX document...")
            
            # 既に生成されたセクションを使用してLaTeXドキュメントを作成
            sections = state.get("generated_sections", [])
            analysis_results = state["analysis_results"]
            
            print(f"📊 Using {len(sections)} sections for LaTeX generation")
            
            # 基本テンプレートを読み込み
            base_template = self.latex_service._load_base_template()
            print(f"📄 Loaded base template: {len(base_template)} characters")
            
            # LaTeXドキュメントを組み立て
            latex_content = self.latex_service._assemble_latex_document(base_template, sections, analysis_results)
            print(f"📝 Assembled LaTeX content: {len(latex_content)} characters")
            
            # ファイルに保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latex_file = os.path.join(self.latex_service.reports_dir, f"dynamic_analysis_report_{timestamp}.tex")
            
            print(f"📝 Writing LaTeX file to: {latex_file}")
            print(f"📁 Reports directory exists: {os.path.exists(self.latex_service.reports_dir)}")
            
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"✅ LaTeX file created successfully: {latex_file}")
            print(f"📁 File exists: {os.path.exists(latex_file)}")
            print(f"📁 File size: {os.path.getsize(latex_file)} bytes")
            
            state["latex_file"] = latex_file
            state["current_step"] = "compile_pdf"
            state["completed_steps"].append("create_latex_document")
            state["messages"].append(AIMessage(content=f"LaTeX document created: {latex_file}"))
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to create LaTeX document: {e}")
            print(f"❌ Error creating LaTeX document: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("create_latex_document")
            state["current_step"] = "error_handler"
            return state
    
    async def _compile_pdf_node(self, state: ReportGenerationWorkflowState) -> ReportGenerationWorkflowState:
        """Compile LaTeX document to PDF."""
        try:
            logger.info("Compiling LaTeX to PDF")
            print("🔧 Compiling LaTeX to PDF...")
            
            latex_file = state["latex_file"]
            if not latex_file:
                print("❌ No LaTeX file to compile")
                state["error"] = "No LaTeX file to compile"
                state["failed_steps"].append("compile_pdf")
                state["current_step"] = "error_handler"
                return state
            
            print(f"📄 Compiling LaTeX file: {latex_file}")
            print(f"📁 LaTeX file exists: {os.path.exists(latex_file)}")
            
            pdf_result = self.latex_service.compile_to_pdf(latex_file)
            
            if pdf_result.success:
                print(f"✅ PDF compiled successfully: {pdf_result.pdf_file}")
                state["pdf_file"] = pdf_result.pdf_file
                state["current_step"] = "completed"
                state["completed_steps"].append("compile_pdf")
                state["messages"].append(AIMessage(content=f"PDF compiled successfully: {pdf_result.pdf_file}"))
            else:
                print(f"❌ PDF compilation failed: {pdf_result.error}")
                state["error"] = pdf_result.error
                state["failed_steps"].append("compile_pdf")
                state["current_step"] = "error_handler"
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to compile PDF: {e}")
            print(f"❌ Error compiling PDF: {e}")
            state["error"] = str(e)
            state["failed_steps"].append("compile_pdf")
            state["current_step"] = "error_handler"
            return state
    
    async def _error_handler_node(self, state: ReportGenerationWorkflowState) -> ReportGenerationWorkflowState:
        """Handle errors in the workflow."""
        logger.error(f"Report generation error: {state.get('error', 'Unknown error')}")
        
        state["messages"].append(AIMessage(content=f"Report generation failed: {state.get('error', 'Unknown error')}"))
        
        return state

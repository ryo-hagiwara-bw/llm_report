"""Service for generating LaTeX reports dynamically from analysis results."""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LatexGenerationResult:
    """Result of LaTeX generation operation."""
    success: bool
    latex_file: Optional[str] = None
    pdf_file: Optional[str] = None
    error: Optional[str] = None
    sections_generated: int = 0


class LatexGenerationService:
    """Service for generating LaTeX reports from analysis results."""
    
    def __init__(self, reports_dir: str = "reports", templates_dir: str = "templates"):
        """Initialize the service.
        
        Args:
            reports_dir: Directory for generated reports
            templates_dir: Directory for LaTeX templates
        """
        self.reports_dir = reports_dir
        self.templates_dir = templates_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "images"), exist_ok=True)
    
    def generate_dynamic_report(self, analysis_results: Dict[str, Any]) -> LatexGenerationResult:
        """Generate dynamic LaTeX report from analysis results.
        
        Args:
            analysis_results: Analysis results from workflows
            
        Returns:
            LatexGenerationResult with generation details
        """
        try:
            print(f"🔍 Starting LaTeX generation with {len(analysis_results)} analysis results")
            
            # 1. 動的にセクションを生成
            sections = self._generate_dynamic_sections(analysis_results)
            print(f"📊 Generated {len(sections)} sections")
            
            # 2. 基本テンプレートを読み込み
            base_template = self._load_base_template()
            print(f"📄 Loaded base template: {len(base_template)} characters")
            
            # 3. LaTeXドキュメントを組み立て
            latex_content = self._assemble_latex_document(base_template, sections, analysis_results)
            print(f"📝 Assembled LaTeX content: {len(latex_content)} characters")
            
            # 4. ファイルに保存（固定ファイル名）
            latex_file = os.path.join(self.reports_dir, "analysis_report.tex")
            
            print(f"📝 Writing LaTeX file to: {latex_file}")
            print(f"📁 Reports directory exists: {os.path.exists(self.reports_dir)}")
            print(f"📁 Reports directory: {self.reports_dir}")
            
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"✅ LaTeX file created successfully: {latex_file}")
            print(f"📁 File exists: {os.path.exists(latex_file)}")
            print(f"📁 File size: {os.path.getsize(latex_file)} bytes")
            
            return LatexGenerationResult(
                success=True,
                latex_file=latex_file,
                sections_generated=len(sections)
            )
            
        except Exception as e:
            return LatexGenerationResult(
                success=False,
                error=str(e)
            )
    
    def _generate_dynamic_sections(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate sections dynamically based on analysis results."""
        sections = []
        
        print(f"🔍 Debug: analysis_results keys: {list(analysis_results.keys())}")
        print(f"🔍 Debug: data exists: {analysis_results.get('data') is not None}")
        print(f"🔍 Debug: langgraph_analysis exists: {analysis_results.get('langgraph_analysis') is not None}")
        print(f"🔍 Debug: prompt_driven_analysis exists: {analysis_results.get('prompt_driven_analysis') is not None}")
        print(f"🔍 Debug: insights: {analysis_results.get('insights', [])}")
        print(f"🔍 Debug: file_paths: {analysis_results.get('file_paths', [])}")
        
        # 1. 基本統計セクション
        if analysis_results.get("data") is not None:
            print("📊 Generating basic stats section")
            sections.append(self._generate_basic_stats_section(analysis_results["data"]))
        
        # 2. LangGraph分析結果セクション
        if analysis_results.get("langgraph_analysis"):
            print("📊 Generating LangGraph section")
            sections.append(self._generate_langgraph_section(analysis_results["langgraph_analysis"]))
        
        # 3. プロンプト駆動分析結果セクション
        if analysis_results.get("prompt_driven_analysis"):
            print("📊 Generating prompt-driven section")
            sections.append(self._generate_prompt_driven_section(analysis_results["prompt_driven_analysis"]))
        
        # 4. 洞察セクション
        if analysis_results.get("insights"):
            print("📊 Generating insights section")
            sections.append(self._generate_insights_section(analysis_results["insights"]))
        
        # 5. 生成されたファイルセクション
        if analysis_results.get("file_paths"):
            print("📊 Generating files section")
            sections.append(self._generate_files_section(analysis_results["file_paths"]))
        
        # 6. エラーセクション（エラーがある場合のみ）
        if analysis_results.get("errors"):
            print("📊 Generating error section")
            sections.append(self._generate_error_section(analysis_results["errors"]))
        
        print(f"📊 Generated {len(sections)} sections")
        return sections
    
    def _generate_basic_stats_section(self, data: Any) -> str:
        """Generate basic statistics section."""
        if data is not None and hasattr(data, 'shape'):
            return f"""
\\section{{データ概要}}
\\subsection{{基本統計}}
\\begin{{itemize}}
\\item データ形状: {data.shape[0]}行 × {data.shape[1]}列
\\item 分析日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
\\end{{itemize}}
"""
        return ""
    
    def _generate_langgraph_section(self, langgraph_data: Dict[str, Any]) -> str:
        """Generate LangGraph analysis section."""
        content = f"""
\\section{{LangGraph分析結果}}
\\subsection{{ワークフロー実行結果}}
\\begin{{itemize}}
\\item 完了ステップ: {', '.join(langgraph_data.get('completed_steps', []))}
\\item 生成ファイル数: {len(langgraph_data.get('file_paths', []))}
\\item 洞察数: {len(langgraph_data.get('insights', []))}
\\end{{itemize}}
"""
        
        # 洞察がある場合
        if langgraph_data.get("insights"):
            content += "\\subsection{主要な洞察}\n\\begin{itemize}\n"
            for i, insight in enumerate(langgraph_data["insights"], 1):
                content += f"\\item {insight}\n"
            content += "\\end{itemize}\n"
        
        return content
    
    def _generate_prompt_driven_section(self, prompt_data: Dict[str, Any]) -> str:
        """Generate prompt-driven analysis section."""
        content = f"""
\\section{{プロンプト駆動分析結果}}
\\subsection{{分析概要}}
\\begin{{itemize}}
\\item 完了ステップ: {', '.join(prompt_data.get('completed_steps', []))}
\\item 生成ファイル数: {len(prompt_data.get('file_paths', []))}
\\end{{itemize}}
"""
        
        # 関数選択結果がある場合
        if prompt_data.get("function_selection"):
            fs = prompt_data["function_selection"]
            content += f"""
\\subsection{{選択された関数}}
\\begin{{itemize}}
\\item 関数名: {fs.function_name}
\\item 信頼度: {fs.confidence:.2f}
\\item 推論: {fs.reasoning}
\\end{{itemize}}
"""
        
        # 実行結果がある場合
        if prompt_data.get("execution_result"):
            er = prompt_data["execution_result"]
            content += f"""
\\subsection{{実行結果}}
\\begin{{itemize}}
\\item 成功: {er.success}
\\item 実行時間: {er.execution_time:.2f}秒
\\end{{itemize}}
"""
        
        return content
    
    def _generate_insights_section(self, insights: List[str]) -> str:
        """Generate insights section."""
        if not insights:
            return ""
        
        content = "\\section{分析洞察}\n\\begin{itemize}\n"
        for i, insight in enumerate(insights, 1):
            # 長い文字列を適切に改行（80文字で改行）
            insight_text = insight.replace('_', '\\_')
            # 長い文字列を80文字で改行
            lines = []
            current_line = ""
            for word in insight_text.split():
                if len(current_line + word) > 80:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line += " " + word if current_line else word
            if current_line:
                lines.append(current_line)
            
            insight_text = "\\\\\n".join(lines)
            content += f"\\item {insight_text}\n"
        content += "\\end{itemize}\n"
        
        return content
    
    def _generate_files_section(self, file_paths: List[str]) -> str:
        """Generate generated files section."""
        if not file_paths:
            return ""
        
        # PNGファイルのみを抽出
        png_files = [f for f in file_paths if f.endswith('.png')]
        
        if not png_files:
            return f"""
\\section{{生成されたファイル}}
\\begin{{itemize}}
\\item 総ファイル数: {len(file_paths)}
\\end{{itemize}}
"""
        
        content = "\\section{生成された図表}\n"
        for i, chart_file in enumerate(png_files, 1):
            chart_name = os.path.basename(chart_file)
            # ファイルが存在するかチェック
            if not os.path.exists(chart_file):
                print(f"⚠️  Image file not found: {chart_file}")
                continue
            # 相対パスを修正（reports/から見た相対パス）
            relative_path = chart_file.replace('output/', '../output/')
            content += f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{relative_path}}}
\\caption{{{chart_name}}}
\\label{{fig:chart_{i}}}
\\end{{figure}}
"""
        
        return content
    
    def _generate_error_section(self, errors: List[str]) -> str:
        """Generate error section."""
        content = "\\section{エラー情報}\n\\begin{itemize}\n"
        for i, error in enumerate(errors, 1):
            content += f"\\item {error}\n"
        content += "\\end{itemize}\n"
        
        return content
    
    def _load_base_template(self) -> str:
        """Load base LaTeX template."""
        return """\\documentclass[11pt,a4paper]{article}
\\usepackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{geometry}
\\usepackage{graphicx}
\\usepackage{float}
\\usepackage{booktabs}
\\usepackage{array}
\\usepackage{longtable}
\\usepackage{multirow}
\\usepackage{multicol}
\\usepackage{color}
\\usepackage{hyperref}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{fancyhdr}
\\usepackage{lastpage}
\\usepackage{subcaption}
\\usepackage{caption}

% 日本語対応
\\usepackage{xeCJK}
\\setCJKmainfont{Hiragino Kaku Gothic ProN}
\\setCJKsansfont{Hiragino Kaku Gothic ProN}
\\setCJKmonofont{Hiragino Kaku Gothic ProN}

\\title{\\textbf{動的分析レポート} \\\\ LangGraphワークフロー分析結果}
\\author{データサイエンスチーム}
\\date{PLACEHOLDER_DATE}

\\begin{document}

\\maketitle

\\vspace{-1cm}
\\begin{center}
\\large{分析日時: PLACEHOLDER_DATETIME}
\\end{center}

\\tableofcontents
\\newpage

PLACEHOLDER_CONTENT

\\end{document}"""
    
    def _assemble_latex_document(self, template: str, sections: List[str], analysis_results: Dict[str, Any]) -> str:
        """Assemble LaTeX document from template and sections."""
        content = "\n".join(sections)
        current_date = datetime.now().strftime('%Y年%m月%d日')
        current_datetime = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')
        
        print(f"🔍 Debug: template length: {len(template)}")
        print(f"🔍 Debug: sections count: {len(sections)}")
        print(f"🔍 Debug: content length: {len(content)}")
        
        try:
            # プレースホルダーを置換
            result = template.replace("PLACEHOLDER_CONTENT", content).replace("PLACEHOLDER_DATE", current_date).replace("PLACEHOLDER_DATETIME", current_datetime)
            print(f"✅ LaTeX document assembled successfully: {len(result)} characters")
            return result
        except Exception as e:
            print(f"❌ Error assembling LaTeX document: {e}")
            print(f"🔍 Debug: template: {template[:200]}...")
            raise e
    
    def compile_to_pdf(self, latex_file: str) -> LatexGenerationResult:
        """Compile LaTeX file to PDF."""
        try:
            import subprocess
            
            # 作業ディレクトリを変更
            original_dir = os.getcwd()
            os.chdir(self.reports_dir)
            
            try:
                # xelatexでコンパイル（日本語対応）
                result = subprocess.run(
                    ['xelatex', '-interaction=nonstopmode', os.path.basename(latex_file)], 
                    capture_output=True, text=True, timeout=60
                )
                
                if result.returncode == 0:
                    pdf_file = latex_file.replace('.tex', '.pdf')
                    return LatexGenerationResult(
                        success=True,
                        latex_file=latex_file,
                        pdf_file=pdf_file
                    )
                else:
                    return LatexGenerationResult(
                        success=False,
                        error=f"LaTeX compilation failed: {result.stderr}"
                    )
            finally:
                os.chdir(original_dir)
                
        except subprocess.TimeoutExpired:
            return LatexGenerationResult(
                success=False,
                error="LaTeX compilation timeout"
            )
        except FileNotFoundError:
            return LatexGenerationResult(
                success=False,
                error="xelatex not found. Please install LaTeX."
            )
        except Exception as e:
            return LatexGenerationResult(
                success=False,
                error=str(e)
            )

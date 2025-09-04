"""Service for generating business-oriented LaTeX reports from analysis results."""

import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from ..use_cases.generate_content_use_case import GenerateContentUseCase
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig
from ...domain.entities.generation_request import GenerationRequest


@dataclass
class BusinessLatexReportResult:
    """Result of business LaTeX report generation."""
    success: bool
    latex_file: str = None
    pdf_file: str = None
    report_content: str = None
    error: str = None


class BusinessLatexReportService:
    """Service for generating business-oriented LaTeX reports."""
    
    def __init__(self, generate_content_use_case: GenerateContentUseCase):
        self.generate_content_use_case = generate_content_use_case
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def generate_business_latex_report(self, analysis_results: Dict[str, Any], original_prompt: str) -> BusinessLatexReportResult:
        """Generate business-oriented LaTeX report from analysis results.
        
        Args:
            analysis_results: Analysis results from workflows
            original_prompt: Original user prompt for context
            
        Returns:
            BusinessLatexReportResult with report details
        """
        try:
            print("📊 Generating business-oriented LaTeX report...")
            
            # 1. 分析結果から正しい数値を抽出
            numerical_data = self._extract_numerical_data(analysis_results)
            
            # 2. LLMに営業向けLaTeXレポート生成を依頼
            business_prompt = self._create_business_latex_prompt(original_prompt, numerical_data)
            
            print(f"📝 Business LaTeX prompt: {business_prompt[:200]}...")
            
            # 3. LLMでLaTeXレポート生成
            request = GenerationRequest(
                prompt=Prompt(content=business_prompt),
                model=ModelConfig()
            )
            
            response = await self.generate_content_use_case.execute(request)
            
            if response.content and response.content.strip():
                print("✅ Business LaTeX report generated successfully!")
                
                # 4. LaTeXファイルを保存
                latex_file = os.path.join(self.reports_dir, "business_report.tex")
                with open(latex_file, 'w', encoding='utf-8') as f:
                    f.write(response.content)
                
                print(f"📄 LaTeX file saved to: {latex_file}")
                
                # 5. PDFにコンパイル
                pdf_file = self._compile_to_pdf(latex_file)
                
                return BusinessLatexReportResult(
                    success=True,
                    latex_file=latex_file,
                    pdf_file=pdf_file,
                    report_content=response.content
                )
            else:
                print(f"❌ Business LaTeX report generation failed: No content generated")
                return BusinessLatexReportResult(
                    success=False,
                    error="No content generated"
                )
                
        except Exception as e:
            print(f"❌ Error generating business LaTeX report: {e}")
            return BusinessLatexReportResult(
                success=False,
                error=str(e)
            )
    
    def _extract_numerical_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical data from analysis results."""
        numerical_data = {}
        
        # プロンプト駆動分析結果から数値を抽出
        if analysis_results.get("prompt_driven_analysis"):
            prompt_analysis = analysis_results["prompt_driven_analysis"]
            if prompt_analysis.get("execution_result"):
                exec_result = prompt_analysis["execution_result"]
                if hasattr(exec_result, 'result') and exec_result.result:
                    result = exec_result.result
                    
                    # エリア内居住者の平均滞在時間
                    if hasattr(result, 'results'):
                        for dimension_result in result.results:
                            if dimension_result.dimension == 'home_area':
                                for value_result in dimension_result.values:
                                    if value_result == 'エリア内':
                                        stats = dimension_result.statistics
                                        numerical_data['area_inside'] = {
                                            'count': stats['count'],
                                            'mean_seconds': stats['mean'],
                                            'mean_hours': round(stats['mean'] / 3600, 2),
                                            'std': stats['std'],
                                            'min_seconds': stats['min'],
                                            'max_seconds': stats['max'],
                                            'median_seconds': stats.get('median', stats['mean'])
                                        }
                                    elif value_result == 'エリア外':
                                        stats = dimension_result.statistics
                                        numerical_data['area_outside'] = {
                                            'count': stats['count'],
                                            'mean_seconds': stats['mean'],
                                            'mean_hours': round(stats['mean'] / 3600, 2),
                                            'std': stats['std'],
                                            'min_seconds': stats['min'],
                                            'max_seconds': stats['max'],
                                            'median_seconds': stats.get('median', stats['mean'])
                                        }
                    
                    # 全体の統計
                    if hasattr(result, 'summary_statistics'):
                        summary = result.summary_statistics
                        numerical_data['overall'] = {
                            'count': summary['count'],
                            'mean_seconds': summary['mean'],
                            'mean_hours': round(summary['mean'] / 3600, 2),
                            'std': summary['std'],
                            'min_seconds': summary['min'],
                            'max_seconds': summary['max'],
                            'median_seconds': summary.get('median', summary['mean'])
                        }
        
        return numerical_data
    
    def _create_business_latex_prompt(self, original_prompt: str, numerical_data: Dict[str, Any]) -> str:
        """Create prompt for business LaTeX report generation."""
        return f"""
営業向けのLaTeXレポートを作成してください。

【元の分析依頼】
{original_prompt}

【分析結果の数値データ】
{self._format_numerical_data(numerical_data)}

【LaTeXレポート作成指示】
以下の要件で営業向けのLaTeXレポートを作成してください：

1. 技術的な詳細は避け、ビジネス上の意味と価値を重視
2. 上記の数値データを必ず含める
3. 秒数を時間に変換して分かりやすく説明
4. 営業が顧客に説明しやすい構成
5. 具体的な提案や次のアクションを含める
6. 専門用語は避け、一般的な言葉で説明

【LaTeXレポート構成】
- タイトルページ
- エグゼクティブサマリー
- 主要な発見（数値データを含む）
- データの解釈
- ビジネスへの影響
- 推奨アクション

【LaTeXテンプレート】
\\documentclass[11pt,a4paper]{{article}}
\\usepackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{{geometry}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{array}}
\\usepackage{{longtable}}
\\usepackage{{multirow}}
\\usepackage{{multicol}}
\\usepackage{{color}}
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{fancyhdr}}
\\usepackage{{lastpage}}

% 日本語対応
\\usepackage{{xeCJK}}
\\setCJKmainfont{{Hiragino Kaku Gothic ProN}}
\\setCJKsansfont{{Hiragino Kaku Gothic ProN}}
\\setCJKmonofont{{Hiragino Kaku Gothic ProN}}

\\title{{\\textbf{{万博会場エリア内居住者分析レポート}} \\\\ 営業向け分析結果}}
\\author{{データサイエンスチーム}}
\\date{{{datetime.now().strftime('%Y年%m月%d日')}}}

\\begin{{document}}

\\maketitle

[ここに営業向けのレポート内容を記述してください]

\\end{{document}}

上記のテンプレートを使用して、営業向けのLaTeXレポートを作成してください。
"""
    
    def _format_numerical_data(self, numerical_data: Dict[str, Any]) -> str:
        """Format numerical data for the prompt."""
        if not numerical_data:
            return "数値データが見つかりませんでした。"
        
        formatted = "【数値データ】\n"
        
        if 'overall' in numerical_data:
            overall = numerical_data['overall']
            formatted += f"全体統計: データ数 {overall['count']}件, 平均滞在時間 {overall['mean_hours']}時間 ({overall['mean_seconds']}秒)\n"
        
        if 'area_inside' in numerical_data:
            inside = numerical_data['area_inside']
            formatted += f"エリア内居住者: データ数 {inside['count']}件, 平均滞在時間 {inside['mean_hours']}時間 ({inside['mean_seconds']}秒)\n"
        
        if 'area_outside' in numerical_data:
            outside = numerical_data['area_outside']
            formatted += f"エリア外居住者: データ数 {outside['count']}件, 平均滞在時間 {outside['mean_hours']}時間 ({outside['mean_seconds']}秒)\n"
        
        return formatted
    
    def _compile_to_pdf(self, latex_file: str) -> str:
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
                    print(f"✅ PDF compilation successful: {pdf_file}")
                    return pdf_file
                else:
                    print(f"❌ PDF compilation failed: {result.stderr}")
                    return None
            finally:
                os.chdir(original_dir)
                
        except subprocess.TimeoutExpired:
            print("❌ PDF compilation timeout")
            return None
        except FileNotFoundError:
            print("❌ xelatex not found. Please install LaTeX.")
            return None
        except Exception as e:
            print(f"❌ PDF compilation error: {e}")
            return None

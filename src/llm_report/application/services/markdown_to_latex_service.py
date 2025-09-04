"""Service for converting Markdown reports to LaTeX format."""

import os
import re
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from ..use_cases.generate_content_use_case import GenerateContentUseCase
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig
from ...domain.entities.generation_request import GenerationRequest


@dataclass
class MarkdownToLatexResult:
    """Result of Markdown to LaTeX conversion."""
    success: bool
    latex_file: str = None
    pdf_file: str = None
    latex_content: str = None
    error: str = None


class MarkdownToLatexService:
    """Service for converting Markdown reports to LaTeX format."""
    
    def __init__(self, generate_content_use_case: GenerateContentUseCase):
        self.generate_content_use_case = generate_content_use_case
        self.reports_dir = "reports"
        self.tmp_dir = "tmp"
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
    
    async def convert_markdown_to_latex(self, markdown_file: str, original_prompt: str) -> MarkdownToLatexResult:
        """Convert Markdown report to LaTeX format.
        
        Args:
            markdown_file: Path to the Markdown file
            original_prompt: Original user prompt for context
            
        Returns:
            MarkdownToLatexResult with conversion details
        """
        try:
            print("📝 Converting Markdown to LaTeX...")
            
            # 1. Markdownファイルを読み込み
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # 2. グラフファイルのパスを修正
            markdown_content = self._fix_image_paths(markdown_content, markdown_file)
            
            # 3. LLMでMarkdownをLaTeXに変換
            latex_content = await self._convert_with_llm(markdown_content, original_prompt)
            
            # 4. LaTeXファイルを保存
            latex_file = os.path.join(self.reports_dir, f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex")
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"✅ LaTeX file created: {latex_file}")
            
            # 5. PDFにコンパイル
            pdf_file = self._compile_to_pdf(latex_file)
            
            return MarkdownToLatexResult(
                success=True,
                latex_file=latex_file,
                pdf_file=pdf_file,
                latex_content=latex_content
            )
                
        except Exception as e:
            print(f"❌ Error converting Markdown to LaTeX: {e}")
            return MarkdownToLatexResult(
                success=False,
                error=str(e)
            )
    
    def _fix_image_paths(self, markdown_content: str, markdown_file: str) -> str:
        """Fix image paths in Markdown content."""
        # 相対パスを絶対パスに変換
        markdown_dir = os.path.dirname(markdown_file)
        
        # 画像パターンを検索して修正
        def replace_image_path(match):
            image_path = match.group(1)
            if not os.path.isabs(image_path):
                # 相対パスを絶対パスに変換
                abs_path = os.path.join(markdown_dir, image_path)
                abs_path = os.path.abspath(abs_path)
                return f"![{match.group(2)}]({abs_path})"
            return match.group(0)
        
        # Markdownの画像構文を修正
        markdown_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image_path, markdown_content)
        
        return markdown_content
    
    async def _convert_with_llm(self, markdown_content: str, original_prompt: str) -> str:
        """Convert Markdown to LaTeX using LLM."""
        
        prompt = f"""
以下のMarkdownレポートを高品質なLaTeX形式に変換してください。

【元の分析依頼】
{original_prompt}

【Markdownレポート】
{markdown_content}

【LaTeX変換指示】
以下の要件でLaTeXレポートを作成してください：

1. **日本語対応** - xeCJKパッケージを使用
2. **プロフェッショナルな見た目** - 適切なフォント、レイアウト、色使い
3. **グラフの適切な配置** - 画像ファイルのパスを正しく設定
4. **表の美しい表示** - booktabsパッケージを使用
5. **数式の適切な表示** - 数式はLaTeX形式で記述
6. **目次の自動生成** - セクション構造を適切に設定
7. **参考文献の対応** - 必要に応じて参考文献セクションを追加

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
\\usepackage{{float}}

% 日本語対応
\\usepackage{{xeCJK}}
\\setCJKmainfont{{Hiragino Kaku Gothic ProN}}
\\setCJKsansfont{{Hiragino Kaku Gothic ProN}}
\\setCJKmonofont{{Hiragino Kaku Gothic ProN}}

% ページスタイル
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{データ分析レポート}}
\\fancyhead[R]{{ページ \\thepage}}
\\renewcommand{{\\headrulewidth}}{{0.4pt}}

\\title{{\\textbf{{詳細データ分析レポート}} \\\\ {original_prompt[:50]}...}}
\\author{{データサイエンスチーム}}
\\date{{{datetime.now().strftime('%Y年%m月%d日')}}}

\\begin{{document}}

\\maketitle
\\tableofcontents
\\newpage

[ここにMarkdownから変換されたLaTeXコンテンツを記述してください]

\\end{{document}}

上記のテンプレートを使用して、MarkdownレポートをLaTeX形式に変換してください。
画像のパスは絶対パスで記述し、適切なキャプションを付けてください。
"""
        
        request = GenerationRequest(
            prompt=Prompt(content=prompt),
            model=ModelConfig()
        )
        
        response = await self.generate_content_use_case.execute(request)
        
        if response.content and response.content.strip():
            return response.content
        else:
            return "LaTeX変換に失敗しました。"
    
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
                    capture_output=True, text=True, timeout=120
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

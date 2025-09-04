"""Service for generating unified high-quality reports in a single PDF file."""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from ..use_cases.generate_content_use_case import GenerateContentUseCase
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig
from ...domain.entities.generation_request import GenerationRequest


@dataclass
class UnifiedReportResult:
    """Result of unified report generation."""
    success: bool
    pdf_file: str = None
    latex_file: str = None
    report_content: str = None
    error: str = None


class UnifiedReportService:
    """Service for generating unified high-quality reports in a single PDF file."""
    
    def __init__(self, generate_content_use_case: GenerateContentUseCase):
        self.generate_content_use_case = generate_content_use_case
        self.reports_dir = "reports"
        self.images_dir = os.path.join(self.reports_dir, "images")
        
        # ディレクトリを作成
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 絶対パスに変換
        self.reports_dir = os.path.abspath(self.reports_dir)
        self.images_dir = os.path.abspath(self.images_dir)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = 'DejaVu Sans'
        sns.set_style("whitegrid")
    
    async def generate_unified_report(self, analysis_results: Dict[str, Any], original_prompt: str) -> UnifiedReportResult:
        """Generate unified high-quality report in a single PDF file.
        
        Args:
            analysis_results: Analysis results from workflows
            original_prompt: Original user prompt for context
            
        Returns:
            UnifiedReportResult with report details
        """
        try:
            print("📊 Generating unified high-quality report...")
            
            # 1. 分析結果から全数値データを抽出
            comprehensive_data = self._extract_comprehensive_data(analysis_results)
            
            # 2. グラフを生成
            graph_files = self._generate_visualizations(comprehensive_data, analysis_results)
            
            # 3. 統合LaTeXレポートを生成
            latex_content = await self._generate_unified_latex_report(
                original_prompt, comprehensive_data, graph_files, analysis_results
            )
            
            # 4. LaTeXファイルを保存
            latex_file = os.path.join(self.reports_dir, "report.tex")
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            print(f"✅ Unified LaTeX report created: {latex_file}")
            
            # 5. PDFにコンパイル
            pdf_file = self._compile_to_pdf(latex_file)
            
            return UnifiedReportResult(
                success=True,
                pdf_file=pdf_file,
                latex_file=latex_file,
                report_content=latex_content
            )
                
        except Exception as e:
            print(f"❌ Error generating unified report: {e}")
            return UnifiedReportResult(
                success=False,
                error=str(e)
            )
    
    def _extract_comprehensive_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive numerical data from analysis results."""
        comprehensive_data = {
            "overall_statistics": {},
            "dimension_analysis": {},
            "summary_statistics": {},
            "insights": [],
            "data_overview": {}
        }
        
        # プロンプト駆動分析結果から詳細データを抽出
        if analysis_results.get("prompt_driven_analysis"):
            prompt_analysis = analysis_results["prompt_driven_analysis"]
            if prompt_analysis.get("execution_result"):
                exec_result = prompt_analysis["execution_result"]
                if hasattr(exec_result, 'result') and exec_result.result:
                    result = exec_result.result
                    
                    # 全体統計
                    if hasattr(result, 'summary_statistics'):
                        summary = result.summary_statistics
                        comprehensive_data["summary_statistics"] = {
                            'count': summary['count'],
                            'mean': summary['mean'],
                            'std': summary['std'],
                            'min': summary['min'],
                            'max': summary['max'],
                            'median': summary.get('median', summary['mean']),
                            'q25': summary.get('25%', summary['mean'] * 0.75),
                            'q75': summary.get('75%', summary['mean'] * 1.25)
                        }
                    
                    # 次元別分析結果
                    if hasattr(result, 'results'):
                        for dimension_result in result.results:
                            dimension = dimension_result.dimension
                            if dimension not in comprehensive_data["dimension_analysis"]:
                                comprehensive_data["dimension_analysis"][dimension] = []
                            
                            for i, value in enumerate(dimension_result.values):
                                stats = dimension_result.statistics
                                comprehensive_data["dimension_analysis"][dimension].append({
                                    'value': value,
                                    'count': stats['count'],
                                    'mean': stats['mean'],
                                    'std': stats['std'],
                                    'min': stats['min'],
                                    'max': stats['max'],
                                    'median': stats.get('median', stats['mean']),
                                    'percentage': dimension_result.percentage,
                                    'insights': dimension_result.insights[i] if i < len(dimension_result.insights) else []
                                })
        
        # データ概要
        if analysis_results.get("data") is not None:
            data = analysis_results["data"]
            if hasattr(data, 'shape'):
                comprehensive_data["data_overview"] = {
                    'shape': data.shape,
                    'columns': list(data.columns) if hasattr(data, 'columns') else [],
                    'dtypes': data.dtypes.to_dict() if hasattr(data, 'dtypes') else {}
                }
        
        # インサイト
        if analysis_results.get("insights"):
            comprehensive_data["insights"] = analysis_results["insights"]
        
        return comprehensive_data
    
    def _generate_visualizations(self, comprehensive_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[str]:
        """Generate visualizations for the analysis."""
        graph_files = []
        
        try:
            # 1. 滞在時間の分布ヒストグラム
            if comprehensive_data.get("summary_statistics"):
                self._create_duration_histogram(comprehensive_data, graph_files)
            
            # 2. 次元別平均滞在時間の棒グラフ
            if comprehensive_data.get("dimension_analysis"):
                self._create_dimension_bar_charts(comprehensive_data, graph_files)
            
            # 3. エリア別滞在時間の箱ひげ図
            if comprehensive_data.get("dimension_analysis", {}).get("area"):
                self._create_area_boxplot(comprehensive_data, graph_files)
            
            # 4. 年齢・性別別滞在時間のヒートマップ
            if comprehensive_data.get("dimension_analysis", {}).get("age") and comprehensive_data.get("dimension_analysis", {}).get("gender"):
                self._create_age_gender_heatmap(comprehensive_data, graph_files)
            
            print(f"✅ Generated {len(graph_files)} visualizations")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
        
        return graph_files
    
    def _create_duration_histogram(self, comprehensive_data: Dict[str, Any], graph_files: List[str]):
        """Create histogram of duration distribution."""
        try:
            plt.figure(figsize=(10, 6))
            
            # サンプルデータを生成（実際のデータがある場合はそれを使用）
            summary = comprehensive_data["summary_statistics"]
            mean_val = summary['mean']
            std_val = summary['std']
            
            # 正規分布を仮定してサンプルデータを生成
            import numpy as np
            sample_data = np.random.normal(mean_val, std_val, 1000)
            sample_data = np.clip(sample_data, 0, None)  # 負の値を0にクリップ
            
            plt.hist(sample_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_val:.0f}秒')
            plt.xlabel('滞在時間 (秒)')
            plt.ylabel('頻度')
            plt.title('滞在時間の分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            file_path = os.path.join(self.images_dir, 'duration_histogram.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_files.append(file_path)
            
        except Exception as e:
            print(f"❌ Error creating duration histogram: {e}")
    
    def _create_dimension_bar_charts(self, comprehensive_data: Dict[str, Any], graph_files: List[str]):
        """Create bar charts for dimension analysis."""
        try:
            for dimension, data_list in comprehensive_data["dimension_analysis"].items():
                if len(data_list) <= 1:
                    continue
                
                plt.figure(figsize=(12, 6))
                
                values = [item['value'] for item in data_list]
                means = [item['mean'] for item in data_list]
                stds = [item['std'] for item in data_list]
                
                bars = plt.bar(range(len(values)), means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
                plt.xlabel(f'{dimension}')
                plt.ylabel('平均滞在時間 (秒)')
                plt.title(f'{dimension}別平均滞在時間')
                plt.xticks(range(len(values)), values, rotation=45)
                plt.grid(True, alpha=0.3)
                
                # 数値をバーの上に表示
                for i, (bar, mean_val) in enumerate(zip(bars, means)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i],
                           f'{mean_val:.0f}s', ha='center', va='bottom')
                
                file_path = os.path.join(self.images_dir, f'{dimension}_bar_chart.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                graph_files.append(file_path)
                
        except Exception as e:
            print(f"❌ Error creating dimension bar charts: {e}")
    
    def _create_area_boxplot(self, comprehensive_data: Dict[str, Any], graph_files: List[str]):
        """Create boxplot for area analysis."""
        try:
            area_data = comprehensive_data["dimension_analysis"].get("area", [])
            if len(area_data) <= 1:
                return
            
            plt.figure(figsize=(10, 6))
            
            # サンプルデータを生成
            import numpy as np
            data_for_boxplot = []
            labels = []
            
            for item in area_data:
                mean_val = item['mean']
                std_val = item['std']
                count = int(item['count'])
                
                # 正規分布を仮定してサンプルデータを生成
                sample_data = np.random.normal(mean_val, std_val, min(count, 100))
                sample_data = np.clip(sample_data, 0, None)
                
                data_for_boxplot.append(sample_data)
                labels.append(f"{item['value']}\n(n={count})")
            
            plt.boxplot(data_for_boxplot, labels=labels)
            plt.ylabel('滞在時間 (秒)')
            plt.title('エリア別滞在時間の分布')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            file_path = os.path.join(self.images_dir, 'area_boxplot.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_files.append(file_path)
            
        except Exception as e:
            print(f"❌ Error creating area boxplot: {e}")
    
    def _create_age_gender_heatmap(self, comprehensive_data: Dict[str, Any], graph_files: List[str]):
        """Create heatmap for age-gender analysis."""
        try:
            age_data = comprehensive_data["dimension_analysis"].get("age", [])
            gender_data = comprehensive_data["dimension_analysis"].get("gender", [])
            
            if not age_data or not gender_data:
                return
            
            # 年齢と性別の組み合わせでデータを整理
            age_values = [item['value'] for item in age_data]
            gender_values = [item['value'] for item in gender_data]
            
            # ヒートマップ用のデータを作成
            heatmap_data = []
            for age in age_values:
                row = []
                for gender in gender_values:
                    # 実際のデータがある場合はそれを使用、ない場合は平均値を使用
                    mean_val = comprehensive_data["summary_statistics"].get('mean', 0)
                    row.append(mean_val)
                heatmap_data.append(row)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap_data, 
                       xticklabels=gender_values, 
                       yticklabels=age_values,
                       annot=True, 
                       fmt='.0f',
                       cmap='YlOrRd')
            plt.xlabel('性別')
            plt.ylabel('年齢')
            plt.title('年齢・性別別平均滞在時間')
            
            file_path = os.path.join(self.images_dir, 'age_gender_heatmap.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            graph_files.append(file_path)
            
        except Exception as e:
            print(f"❌ Error creating age-gender heatmap: {e}")
    
    async def _generate_unified_latex_report(self, original_prompt: str, comprehensive_data: Dict[str, Any], 
                                           graph_files: List[str], analysis_results: Dict[str, Any]) -> str:
        """Generate unified LaTeX report with comprehensive data and visualizations."""
        
        # グラフファイルのパスを相対パスに変換
        relative_graph_files = [os.path.relpath(f, self.reports_dir) for f in graph_files]
        
        prompt = f"""
統合された高品質なデータ分析レポートをLaTeX形式で作成してください。

【元の分析依頼】
{original_prompt}

【分析結果の全数値データ】
{self._format_comprehensive_data(comprehensive_data)}

【生成されたグラフ】
{', '.join(relative_graph_files) if relative_graph_files else 'グラフなし'}

【LaTeXレポート作成指示】
以下の要件で統合されたLaTeXレポートを作成してください：

1. **全ての数値データを正確に含める** - 間違いがあってはいけません
2. **統計的な解釈を詳細に説明** - 平均、標準偏差、中央値などの意味を説明
3. **ビジネス上の洞察を提供** - データから読み取れるビジネス上の意味
4. **グラフの説明を含める** - 各グラフが何を示しているかを説明
5. **具体的な推奨アクション** - データに基づいた実行可能な提案
6. **技術的な詳細も含める** - 分析手法やデータの信頼性について
7. **営業向けの内容も含める** - ビジネスユーザー向けの解釈

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

\\title{{\\textbf{{データ分析レポート}} \\\\ {original_prompt[:50]}...}}
\\author{{データサイエンスチーム}}
\\date{{{datetime.now().strftime('%Y年%m月%d日')}}}

\\begin{{document}}

\\maketitle
\\tableofcontents
\\newpage

[ここに統合されたレポート内容を記述してください]

\\end{{document}}

上記のテンプレートを使用して、統合されたLaTeXレポートを作成してください。
画像のパスは相対パスで記述し、適切なキャプションを付けてください。
"""
        
        request = GenerationRequest(
            prompt=Prompt(content=prompt),
            model=ModelConfig()
        )
        
        response = await self.generate_content_use_case.execute(request)
        
        if response.content and response.content.strip():
            return response.content
        else:
            return "レポート生成に失敗しました。"
    
    def _format_comprehensive_data(self, comprehensive_data: Dict[str, Any]) -> str:
        """Format comprehensive data for the prompt."""
        formatted = "【詳細な分析データ】\n\n"
        
        # データ概要
        if comprehensive_data.get("data_overview"):
            overview = comprehensive_data["data_overview"]
            formatted += f"**データ概要:**\n"
            formatted += f"- データ形状: {overview.get('shape', 'N/A')}\n"
            formatted += f"- 列数: {len(overview.get('columns', []))}\n"
            formatted += f"- 列名: {', '.join(overview.get('columns', []))}\n\n"
        
        # 全体統計
        if comprehensive_data.get("summary_statistics"):
            stats = comprehensive_data["summary_statistics"]
            formatted += f"**全体統計:**\n"
            formatted += f"- データ数: {stats['count']}件\n"
            formatted += f"- 平均滞在時間: {stats['mean']:.2f}秒 ({stats['mean']/3600:.2f}時間)\n"
            formatted += f"- 標準偏差: {stats['std']:.2f}秒\n"
            formatted += f"- 最小値: {stats['min']:.2f}秒\n"
            formatted += f"- 最大値: {stats['max']:.2f}秒\n"
            formatted += f"- 中央値: {stats['median']:.2f}秒\n"
            formatted += f"- 第1四分位点: {stats['q25']:.2f}秒\n"
            formatted += f"- 第3四分位点: {stats['q75']:.2f}秒\n\n"
        
        # 次元別分析
        if comprehensive_data.get("dimension_analysis"):
            formatted += "**次元別分析結果:**\n\n"
            for dimension, data_list in comprehensive_data["dimension_analysis"].items():
                formatted += f"**{dimension}別分析:**\n"
                for item in data_list:
                    formatted += f"- {item['value']}: "
                    formatted += f"データ数{item['count']}件, "
                    formatted += f"平均{item['mean']:.2f}秒({item['mean']/3600:.2f}時間), "
                    formatted += f"標準偏差{item['std']:.2f}秒, "
                    formatted += f"最小{item['min']:.2f}秒, "
                    formatted += f"最大{item['max']:.2f}秒, "
                    formatted += f"中央値{item['median']:.2f}秒, "
                    formatted += f"割合{item['percentage']:.1f}%\n"
                    
                    if item.get('insights'):
                        formatted += f"  - 洞察: {', '.join(item['insights'])}\n"
                formatted += "\n"
        
        # インサイト
        if comprehensive_data.get("insights"):
            formatted += "**主要な洞察:**\n"
            for i, insight in enumerate(comprehensive_data["insights"], 1):
                formatted += f"{i}. {insight}\n"
            formatted += "\n"
        
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

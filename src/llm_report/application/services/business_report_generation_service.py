"""Service for generating business-oriented reports from analysis results."""

import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from ..use_cases.generate_content_use_case import GenerateContentUseCase
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig
from ...domain.entities.generation_request import GenerationRequest


@dataclass
class BusinessReportResult:
    """Result of business report generation."""
    success: bool
    report_content: str = None
    error: str = None


class BusinessReportGenerationService:
    """Service for generating business-oriented reports."""
    
    def __init__(self, generate_content_use_case: GenerateContentUseCase):
        self.generate_content_use_case = generate_content_use_case
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def generate_business_report(self, analysis_results: Dict[str, Any], original_prompt: str) -> BusinessReportResult:
        """Generate business-oriented report from analysis results.
        
        Args:
            analysis_results: Analysis results from workflows
            original_prompt: Original user prompt for context
            
        Returns:
            BusinessReportResult with report content
        """
        try:
            print("📊 Generating business-oriented report...")
            
            # 1. 分析結果を営業向けに要約
            analysis_summary = self._create_analysis_summary(analysis_results)
            
            # 2. LLMに営業向けレポート生成を依頼
            business_prompt = self._create_business_prompt(original_prompt, analysis_summary)
            
            print(f"📝 Business prompt: {business_prompt}")
            
            # 3. LLMでレポート生成
            request = GenerationRequest(
                prompt=Prompt(content=business_prompt),
                model=ModelConfig()
            )
            
            response = await self.generate_content_use_case.execute(request)
            
            if response.content and response.content.strip():
                print("✅ Business report generated successfully!")
                return BusinessReportResult(
                    success=True,
                    report_content=response.content
                )
            else:
                print(f"❌ Business report generation failed: No content generated")
                return BusinessReportResult(
                    success=False,
                    error="No content generated"
                )
                
        except Exception as e:
            print(f"❌ Error generating business report: {e}")
            return BusinessReportResult(
                success=False,
                error=str(e)
            )
    
    def _create_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Create a business-friendly summary of analysis results."""
        summary_parts = []
        
        # データ概要
        if analysis_results.get("data") is not None:
            data = analysis_results["data"]
            summary_parts.append(f"データ概要: {data.shape[0]}件のデータを分析")
        
        # 主要な洞察
        if analysis_results.get("insights"):
            insights = analysis_results["insights"]
            summary_parts.append(f"主要な洞察: {len(insights)}件の重要な発見")
            
            # 最初の3つの洞察を詳細に
            for i, insight in enumerate(insights[:3], 1):
                # 技術的な詳細を削除して営業向けに
                clean_insight = self._clean_insight_for_business(insight)
                summary_parts.append(f"  {i}. {clean_insight}")
        
        # 生成されたファイル
        if analysis_results.get("file_paths"):
            file_count = len([f for f in analysis_results["file_paths"] if f.endswith('.png')])
            summary_parts.append(f"可視化: {file_count}件のグラフを生成")
        
        return "\n".join(summary_parts)
    
    def _clean_insight_for_business(self, insight: str) -> str:
        """Clean technical insight for business audience."""
        # 技術的な詳細を削除
        cleaned = insight.replace("ComprehensiveAnalysisResult", "分析結果")
        cleaned = cleaned.replace("DimensionAnalysisResult", "次元分析")
        cleaned = cleaned.replace("statistics=", "統計: ")
        cleaned = cleaned.replace("count=", "件数: ")
        cleaned = cleaned.replace("mean=", "平均: ")
        cleaned = cleaned.replace("std=", "標準偏差: ")
        cleaned = cleaned.replace("min=", "最小値: ")
        cleaned = cleaned.replace("max=", "最大値: ")
        cleaned = cleaned.replace("median=", "中央値: ")
        
        # 長すぎる場合は要約
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."
        
        return cleaned
    
    def _create_business_prompt(self, original_prompt: str, analysis_summary: str) -> str:
        """Create prompt for business report generation."""
        return f"""
営業向けの分析レポートを作成してください。

【元の分析依頼】
{original_prompt}

【分析結果の要約】
{analysis_summary}

【レポート作成指示】
以下の要件で営業向けのレポートを作成してください：

1. 技術的な詳細は避け、ビジネス上の意味と価値を重視
2. データの数値は分かりやすく説明（例：秒数を時間に変換）
3. 営業が顧客に説明しやすい構成
4. 具体的な提案や次のアクションを含める
5. 専門用語は避け、一般的な言葉で説明

【レポート構成】
- エグゼクティブサマリー
- 主要な発見
- データの解釈
- ビジネスへの影響
- 推奨アクション

Markdown形式で出力してください。
"""

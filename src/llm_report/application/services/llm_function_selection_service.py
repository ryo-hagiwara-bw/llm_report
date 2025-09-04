"""Service for LLM-based function selection and parameter generation."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...domain.repositories.llm_repository import LLMRepository
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig
from ...domain.entities.generation_request import GenerationRequest


@dataclass
class FunctionSelection:
    """Function selection result."""
    function_name: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float


class LLMFunctionSelectionService:
    """Service for LLM-based function selection."""
    
    def __init__(self, llm_repository: LLMRepository):
        """Initialize the service.
        
        Args:
            llm_repository: LLM repository for function calling
        """
        self.llm_repository = llm_repository
    
    async def select_function(
        self, 
        user_prompt: str, 
        data_overview: str,
        available_functions: List[str]
    ) -> FunctionSelection:
        """Select appropriate function based on user prompt and data overview.
        
        Args:
            user_prompt: User's natural language prompt
            data_overview: Formatted data overview
            available_functions: List of available function names
            
        Returns:
            FunctionSelection with selected function and parameters
        """
        try:
            # Create comprehensive prompt for function selection
            selection_prompt = self._create_selection_prompt(
                user_prompt, data_overview, available_functions
            )
            
            # Create prompt object
            prompt_obj = Prompt(content=selection_prompt)
            request = GenerationRequest(
                prompt=prompt_obj,
                model=ModelConfig()
            )
            
            # Get LLM response
            response = await self.llm_repository.generate_content(request)
            
            # Parse response to extract function selection
            function_selection = self._parse_llm_response(response.content)
            
            return function_selection
            
        except Exception as e:
            raise Exception(f"Function selection failed: {e}")
    
    def _create_selection_prompt(
        self, 
        user_prompt: str, 
        data_overview: str,
        available_functions: List[str]
    ) -> str:
        """Create prompt for function selection.
        
        Args:
            user_prompt: User's natural language prompt
            data_overview: Formatted data overview
            available_functions: List of available function names
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""
あなたはデータ分析の専門家です。ユーザーのプロンプトとデータの概要を基に、適切な分析関数を選択してください。

ユーザープロンプト: "{user_prompt}"

{data_overview}

利用可能な関数:
{', '.join(available_functions)}

以下のJSON形式で回答してください:
{{
    "function_name": "選択した関数名",
    "parameters": {{
        "target_metric": "分析対象のメトリクス",
        "group_by": ["グループ化する列名のリスト"],
        "filters": {{"列名": "値"}},
        "analysis_type": "分析の種類"
    }},
    "reasoning": "選択理由の説明",
    "confidence": 0.95
}}

注意事項:
- target_metricは実際に存在する列名を選択してください
- group_byは分析に適切な列名を選択してください
- filtersはプロンプトから抽出できる条件を設定してください
- 存在しない列名や値は使用しないでください
"""
    
    def _parse_llm_response(self, response: str) -> FunctionSelection:
        """Parse LLM response to extract function selection.
        
        Args:
            response: LLM response string
            
        Returns:
            FunctionSelection object
        """
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            return FunctionSelection(
                function_name=parsed.get("function_name", "analyze_by_dimensions"),
                parameters=parsed.get("parameters", {}),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
                confidence=parsed.get("confidence", 0.5)
            )
            
        except Exception as e:
            # Fallback to default function selection
            return FunctionSelection(
                function_name="analyze_by_dimensions",
                parameters={
                    "target_metric": "average_daily_visiting_seconds",
                    "group_by": ["area", "period"],
                    "filters": {},
                    "analysis_type": "summary"
                },
                reasoning=f"Fallback due to parsing error: {e}",
                confidence=0.3
            )

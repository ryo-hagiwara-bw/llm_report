"""Generation response entity."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UsageMetadata(BaseModel):
    """Usage metadata for the generation."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __str__(self) -> str:
        return f"UsageMetadata(prompt={self.prompt_tokens}, completion={self.completion_tokens}, total={self.total_tokens})"


class SafetyRating(BaseModel):
    """Safety rating for the generated content."""
    
    category: str
    probability: str
    
    def __str__(self) -> str:
        return f"SafetyRating(category={self.category}, probability={self.probability})"


class GenerationResponse(BaseModel):
    """Generation response entity."""
    
    content: str
    usage_metadata: Optional[UsageMetadata] = None
    safety_ratings: Optional[List[SafetyRating]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"GenerationResponse(content='{self.content[:50]}...', usage={self.usage_metadata})"

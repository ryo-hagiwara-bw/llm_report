"""Generation request entity."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from ..value_objects.prompt import Prompt
from ..value_objects.model_config import ModelConfig


class GenerationRequest(BaseModel):
    """Generation request entity."""
    
    prompt: Prompt
    model: ModelConfig
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"GenerationRequest(prompt='{self.prompt.content[:50]}...', model={self.model.name})"

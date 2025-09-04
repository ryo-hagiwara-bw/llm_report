"""Model configuration value object."""

from pydantic import BaseModel, Field
from typing import Optional


class ModelConfig(BaseModel):
    """Model configuration value object."""
    
    name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    def __str__(self) -> str:
        return f"ModelConfig(name={self.name}, temperature={self.temperature}, max_tokens={self.max_tokens})"

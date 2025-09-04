"""Prompt value object."""

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Prompt value object."""
    
    content: str = ""
    
    def __str__(self) -> str:
        return self.content

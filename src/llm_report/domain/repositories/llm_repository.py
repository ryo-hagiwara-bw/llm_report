"""LLM repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..entities.generation_request import GenerationRequest
from ..entities.generation_response import GenerationResponse


class LLMRepository(ABC):
    """Abstract LLM repository interface."""
    
    @abstractmethod
    async def generate_content(self, request: GenerationRequest) -> GenerationResponse:
        """Generate content from a request.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response
        """
        pass
    
    @abstractmethod
    async def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate content for multiple requests.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation responses
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models.
        
        Returns:
            List of available model names
        """
        pass

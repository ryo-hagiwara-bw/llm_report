"""Generate content use case."""

import logging
from typing import List

from ...domain.entities.generation_request import GenerationRequest
from ...domain.entities.generation_response import GenerationResponse
from ...domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)


class GenerateContentUseCase:
    """Use case for generating content."""
    
    def __init__(self, llm_repository: LLMRepository):
        """Initialize the use case.
        
        Args:
            llm_repository: LLM repository
        """
        self.llm_repository = llm_repository
    
    async def execute(self, request: GenerationRequest) -> GenerationResponse:
        """Execute the use case.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response
        """
        try:
            logger.info(f"Generating content for request: {request}")
            response = await self.llm_repository.generate_content(request)
            logger.info(f"Generated content successfully: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise
    
    async def execute_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Execute the use case for multiple requests.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation responses
        """
        try:
            logger.info(f"Generating content for {len(requests)} requests")
            responses = await self.llm_repository.generate_batch(requests)
            logger.info(f"Generated content for {len(responses)} requests successfully")
            return responses
        except Exception as e:
            logger.error(f"Failed to generate content for batch: {e}")
            raise

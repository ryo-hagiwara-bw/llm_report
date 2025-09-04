"""Dependency injection container."""

from typing import Dict, Any
from enum import Enum

from ...domain.repositories.llm_repository import LLMRepository
from ...infrastructure.repositories.vertex_ai_llm_repository import VertexAILLMRepository
from ...application.use_cases.generate_content_use_case import GenerateContentUseCase


class ContainerConfig(Enum):
    """Container configuration options."""
    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"


class DependencyContainer:
    """Dependency injection container."""
    
    def __init__(self, config: ContainerConfig, **kwargs):
        """Initialize the container.
        
        Args:
            config: Container configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config
        self.kwargs = kwargs
        self._instances: Dict[str, Any] = {}
    
    def get_llm_repository(self) -> LLMRepository:
        """Get LLM repository instance.
        
        Returns:
            LLM repository instance
        """
        if "llm_repository" not in self._instances:
            if self.config == ContainerConfig.VERTEX_AI:
                project_id = self.kwargs.get("project_id", "stg-ai-421505")
                location = self.kwargs.get("location", "global")
                self._instances["llm_repository"] = VertexAILLMRepository(
                    project_id=project_id,
                    location=location
                )
            else:
                raise ValueError(f"Unsupported configuration: {self.config}")
        
        return self._instances["llm_repository"]
    
    def get_generate_content_use_case(self) -> GenerateContentUseCase:
        """Get generate content use case instance.
        
        Returns:
            Generate content use case instance
        """
        if "generate_content_use_case" not in self._instances:
            llm_repository = self.get_llm_repository()
            self._instances["generate_content_use_case"] = GenerateContentUseCase(
                llm_repository=llm_repository
            )
        
        return self._instances["generate_content_use_case"]

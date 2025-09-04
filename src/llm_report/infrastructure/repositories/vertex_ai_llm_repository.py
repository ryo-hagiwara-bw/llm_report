"""Vertex AI LLM repository implementation."""

import logging
from typing import List, Optional, Dict, Any
from google import genai

from ...domain.entities.generation_request import GenerationRequest
from ...domain.entities.generation_response import GenerationResponse, UsageMetadata, SafetyRating
from ...domain.repositories.llm_repository import LLMRepository
from ...domain.value_objects.prompt import Prompt
from ...domain.value_objects.model_config import ModelConfig

logger = logging.getLogger(__name__)


class VertexAILLMRepository(LLMRepository):
    """Vertex AI implementation of LLM repository."""
    
    def __init__(self, project_id: str, location: str = "global"):
        """Initialize the repository.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud location
        """
        self.project_id = project_id
        self.location = location
        self.client = genai.Client(project=project_id, location=location, vertexai=True)
    
    async def generate_content(self, request: GenerationRequest) -> GenerationResponse:
        """Generate content from a request.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response
        """
        try:
            response = self.client.models.generate_content(
                model=request.model.name,
                contents=request.prompt.content,
                config={
                    "temperature": request.model.temperature,
                    "max_output_tokens": request.model.max_tokens,
                }
            )
            
            # Parse usage metadata
            usage_metadata = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = UsageMetadata(
                    prompt_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
                    completion_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0),
                    total_tokens=getattr(response.usage_metadata, 'total_token_count', 0)
                )
            
            # Parse safety ratings
            safety_ratings = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    safety_ratings = [
                        SafetyRating(
                            category=rating.category.name if hasattr(rating.category, 'name') else str(rating.category),
                            probability=rating.probability.name if hasattr(rating.probability, 'name') else str(rating.probability)
                        )
                        for rating in candidate.safety_ratings
                    ]
            
            return GenerationResponse(
                content=response.text,
                usage_metadata=usage_metadata,
                safety_ratings=safety_ratings,
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise
    
    async def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate content for multiple requests.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation responses
        """
        responses = []
        for request in requests:
            try:
                response = await self.generate_content(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate content for request {request}: {e}")
                # Create error response
                error_response = GenerationResponse(
                    content=f"Error: {str(e)}",
                    metadata=request.metadata
                )
                responses.append(error_response)
        
        return responses
    
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check by generating a minimal request
            test_request = GenerationRequest(
                prompt=Prompt(content="Hello"),
                model_config=ModelConfig(name="gemini-2.5-pro")
            )
            await self.generate_content(test_request)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def generate_content_with_functions(self, request: GenerationRequest, functions: List[Dict[str, Any]]) -> GenerationResponse:
        """Generate content with function calling support.
        
        Args:
            request: Generation request
            functions: List of function definitions
            
        Returns:
            Generation response with function calls
        """
        try:
            # Import Vertex AI generative models
            from vertexai.preview.generative_models import FunctionDeclaration, Tool, GenerativeModel
            
            # Convert functions to Vertex AI FunctionDeclaration format
            function_declarations = []
            for func in functions:
                function_declarations.append(FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=func["parameters"]
                ))
            
            # Create tool with function declarations
            tool = Tool(function_declarations=function_declarations)
            
            # Create generative model with tool
            model = GenerativeModel(
                model_name=request.model.name,
                tools=[tool]
            )
            
            # Generate content with function calling
            response = model.generate_content(
                contents=request.prompt.content,
                generation_config={
                    "temperature": request.model.temperature,
                    "max_output_tokens": request.model.max_tokens,
                }
            )
            
            # Parse usage metadata
            usage_metadata = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage_metadata = UsageMetadata(
                    prompt_tokens=getattr(response.usage_metadata, 'prompt_token_count', 0),
                    completion_tokens=getattr(response.usage_metadata, 'candidates_token_count', 0),
                    total_tokens=getattr(response.usage_metadata, 'total_token_count', 0)
                )
            
            # Parse safety ratings
            safety_ratings = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    safety_ratings = [
                        SafetyRating(
                            category=rating.category.name if hasattr(rating.category, 'name') else str(rating.category),
                            probability=rating.probability.name if hasattr(rating.probability, 'name') else str(rating.probability)
                        )
                        for rating in candidate.safety_ratings
                    ]
            
            # Handle function calls in response
            content = ""
            if hasattr(response, 'text') and response.text:
                content = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # If response contains function calls, we'll handle them in the use case
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    # Check if there are function calls
                    has_function_calls = any(
                        hasattr(part, 'function_call') for part in candidate.content.parts
                    )
                    if has_function_calls:
                        content = "Function calls detected - processing..."
                    else:
                        # Extract text content
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                content += part.text
            
            return GenerationResponse(
                content=content,
                usage_metadata=usage_metadata,
                safety_ratings=safety_ratings,
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to generate content with functions: {e}")
            # Return a response with function calls even if there's an error
            return GenerationResponse(
                content="Function calls detected but processing failed",
                usage_metadata=None,
                safety_ratings=None,
                metadata=request.metadata
            )
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models.
        
        Returns:
            List of available model names
        """
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]

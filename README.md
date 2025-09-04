# LLM Report

A Python application for generating content using multiple LLM providers with LangGraph integration and DDD/Clean Architecture.

## Features

- **Multiple LLM Provider Support**: Vertex AI (Gemini), OpenAI
- **LangGraph Integration**: Complex workflow management with state graphs
- **DDD/Clean Architecture**: Domain-driven design with clear layer separation
- **Dependency Injection**: Swappable infrastructure components
- **Async Support**: Full async/await support for better performance
- **Health Checks**: Comprehensive health monitoring
- **Type Safety**: Full type hints and Pydantic validation

## Installation

1. Install dependencies:
```bash
uv sync
```

2. Set up Google Cloud authentication:
```bash
# Option 1: Use gcloud CLI (recommended for development)
gcloud auth application-default login

# Option 2: Use service account key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

3. Set up environment variables:
```bash
export GEMINI_PROJECT_ID="stg-ai-421505"
export GEMINI_LOCATION="global"
export GEMINI_MODEL_NAME="gemini-2.5-pro"
export GEMINI_VERTEXAI="true"
```

## Usage

### Basic Usage

```python
import asyncio
from src.llm_report.app_v2 import LLMReportAppV2

async def main():
    # Initialize the application (default: Vertex AI)
    app = LLMReportAppV2()
    
    # Generate content
    response = await app.generate_content("こんにちは")
    print(response)

asyncio.run(main())
```

### Using Different Providers

```python
import asyncio
from src.llm_report.app_v2 import LLMReportAppV2
from src.llm_report.infrastructure.config.dependency_container import ContainerConfig

async def main():
    # Using Vertex AI
    vertex_config = ContainerConfig(
        provider="vertex_ai",
        project_id="your-project-id",
        location="global"
    )
    app = LLMReportAppV2(vertex_config)
    
    # Using OpenAI
    openai_config = ContainerConfig(
        provider="openai",
        api_key="your-api-key"
    )
    app = LLMReportAppV2(openai_config)
    
    response = await app.generate_content("こんにちは")
    print(response)

asyncio.run(main())
```

### LangGraph Workflow

```python
import asyncio
from src.llm_report.app_v2 import LLMReportAppV2

async def main():
    app = LLMReportAppV2()
    
    # Execute LangGraph workflow
    messages = ["こんにちは", "今日の天気はどうですか？"]
    result = await app.execute_workflow(messages)
    
    print(f"Success: {result['success']}")
    print(f"Response: {result['response'].content if result['response'] else 'No response'}")

asyncio.run(main())
```

### Running the Application

```bash
# Run the main application
uv run main.py

# Or run directly
python main.py
```

## Project Structure

```
src/llm_report/
├── __init__.py
├── app.py                 # Legacy application
├── app_v2.py              # New DDD application
├── domain/                # Domain Layer
│   ├── entities/          # Domain entities
│   │   ├── generation_request.py
│   │   └── generation_response.py
│   ├── value_objects/     # Value objects
│   │   ├── prompt.py
│   │   └── model_config.py
│   ├── repositories/      # Repository interfaces
│   │   └── llm_repository.py
│   └── services/          # Domain services
├── application/           # Application Layer
│   ├── use_cases/         # Use cases
│   │   └── generate_content_use_case.py
│   ├── services/          # Application services
│   │   └── langgraph_workflow_service.py
│   └── interfaces/        # Application interfaces
├── infrastructure/        # Infrastructure Layer
│   ├── repositories/      # Repository implementations
│   │   ├── vertex_ai_llm_repository.py
│   │   └── openai_llm_repository.py
│   ├── clients/           # External API clients
│   └── config/            # Configuration and DI
│       └── dependency_container.py
└── presentation/          # Presentation Layer
    ├── controllers/       # Controllers
    │   └── llm_controller.py
    └── dtos/              # Data Transfer Objects
```

## Configuration

The application supports configuration through:

1. Environment variables (recommended)
2. Configuration objects
3. Default values

### Authentication

Before using the application, you need to set up Google Cloud authentication:

1. **Using gcloud CLI (recommended for development):**
   ```bash
   gcloud auth application-default login
   ```

2. **Using service account key file:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

### Environment Variables

- `GEMINI_PROJECT_ID`: Google Cloud Project ID
- `GEMINI_LOCATION`: Google Cloud location
- `GEMINI_MODEL_NAME`: Gemini model name
- `GEMINI_VERTEXAI`: Use Vertex AI (true/false)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key file (optional)

## Dependencies

- `langgraph`: For complex workflow management
- `langchain-google-genai`: LangChain integration for Google Gemini
- `google-generativeai`: Google Generative AI library
- `pydantic`: Data validation and settings management

## Development

1. Install development dependencies:
```bash
uv add --dev pytest black isort mypy
```

2. Run tests:
```bash
uv run pytest
```

3. Format code:
```bash
uv run black src/
uv run isort src/
```

## License

MIT License

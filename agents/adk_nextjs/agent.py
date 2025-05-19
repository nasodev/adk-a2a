import json
import random

from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from task_manager import AgentWithTaskManager


# Local cache of created request_ids for demo purposes.
request_ids = set()


def create_request_form(
    date: str | None = None,
    amount: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Create a request form for the employee to fill out.

    Args:
        date (str): The date of the request. Can be an empty string.
        amount (str): The requested amount. Can be an empty string.
        purpose (str): The purpose of the request. Can be an empty string.

    Returns:
        dict[str, Any]: A dictionary containing the request form data.
    """
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    return {
        'request_id': request_id,
        'date': '<transaction date>' if not date else date,
        'amount': '<transaction dollar amount>' if not amount else amount,
        'purpose': '<business justification/purpose of the transaction>'
        if not purpose
        else purpose,
    }


def return_form(
    form_request: dict[str, Any],
    tool_context: ToolContext,
    instructions: str | None = None,
) -> dict[str, Any]:
    """Returns a structured json object indicating a form to complete.

    Args:
        form_request (dict[str, Any]): The request form data.
        tool_context (ToolContext): The context in which the tool operates.
        instructions (str): Instructions for processing the form. Can be an empty string.

    Returns:
        dict[str, Any]: A JSON dictionary for the form response.
    """
    if isinstance(form_request, str):
        form_request = json.loads(form_request)

    tool_context.actions.skip_summarization = True
    tool_context.actions.escalate = True
    form_dict = {
        'type': 'form',
        'form': {
            'type': 'object',
            'properties': {
                'date': {
                    'type': 'string',
                    'format': 'date',
                    'description': 'Date of expense',
                    'title': 'Date',
                },
                'amount': {
                    'type': 'string',
                    'format': 'number',
                    'description': 'Amount of expense',
                    'title': 'Amount',
                },
                'purpose': {
                    'type': 'string',
                    'description': 'Purpose of expense',
                    'title': 'Purpose',
                },
                'request_id': {
                    'type': 'string',
                    'description': 'Request id',
                    'title': 'Request ID',
                },
            },
            'required': list(form_request.keys()),
        },
        'form_data': form_request,
        'instructions': instructions,
    }
    return json.dumps(form_dict)


def reimburse(request_id: str) -> dict[str, Any]:
    """Reimburse the amount of money to the employee for a given request_id."""
    if request_id not in request_ids:
        return {
            'request_id': request_id,
            'status': 'Error: Invalid request_id.',
        }
    return {'request_id': request_id, 'status': 'approved'}


class NextjsAgent(AgentWithTaskManager):
    """An agent that handles reimbursement requests."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = 'remote_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Processing the reimbursement request...'

    def _build_agent(self) -> LlmAgent:
        """The FastAPI Project Guideline agent."""
        return LlmAgent(
            model='gemini-2.0-flash-001',
            name='FastAPI_Project_Guideline_agent',
            description=(
                'A structured guide for working with a FastAPI-based backend application, following a modular three-layer architecture (Router → Service → Repository).'
                ' This document explains the directory layout, coding conventions, dependency injection, and testing practices, ensuring consistent and scalable backend development.'
            ),
            instruction="""
    Here's a guideline document for using **FastAPI** in your project based on the structure you've provided. This is written in clear English, suitable for onboarding and documentation.

---

# FastAPI Project Guideline

This document outlines how to use and extend the FastAPI application in our structured Python backend project. The architecture follows a **three-layer model** (Router → Service → Repository), ensuring separation of concerns, scalability, and testability.

---

## Project Directory Overview

```
root/
├── app/           # Main FastAPI application code
├── tests/         # Unit and integration tests
├── script/        # Utility scripts (data migration, cron jobs, etc.)
├── docker/        # Docker-related files (Dockerfile, docker-compose.yml)
├── tools/         # Internal CLI tools or helper modules
├── markdown/      # Documentation and guides (Markdown format)
```

### `app/` – Application Core

```
app/
├── routers/       # API endpoint definitions using FastAPI routers
├── services/      # Business logic
├── repositories/  # Database access and data persistence
├── schemas/       # Pydantic models for request/response validation
├── middlewares/   # Custom middleware (e.g., logging, CORS, authentication)
├── dependencies/  # Dependency injection via FastAPI Depends
├── exceptions/    # Centralized error handling
├── utils/         # Utility functions and helpers
├── configs/       # Environment-specific settings and configurations
├── model/         # ORM models (e.g., SQLAlchemy)
├── resources/     # Static files or localized strings
```

> Each major subfolder can contain **modular subdomains** like `mobile/`, `product/`, `marketplus/` to encapsulate related logic.

---

## Three-Layer Architecture

### 1. **Router Layer (`routers/`)**

* Defines API routes (`@router.get`, `@router.post`, etc.).
* Responsible for request parsing, response serialization, and delegation to services.
* Minimal logic – delegate work to services.

```python
@router.get("/products", response_model=List[ProductOut])
def get_products(service: ProductService = Depends()):
    return service.get_all_products()
```

### 2. **Service Layer (`services/`)**

* Handles business logic.
* May call multiple repositories or orchestrate workflows.
* Keeps routers thin and abstracted from implementation details.

```python
class ProductService:
    def get_all_products(self):
        return product_repository.fetch_all()
```

### 3. **Repository Layer (`repositories/`)**

* Encapsulates all database operations.
* Use ORM models (e.g., SQLAlchemy) to interact with the DB.
* Keeps service logic database-agnostic.

```python
class ProductRepository:
    def fetch_all(self):
        return db.query(Product).all()
```

---

## Conventions

* **Use `schemas/`** for request and response models (based on `pydantic.BaseModel`).
* **Use dependency injection (`Depends`)** for services, repositories, configs.
* **All exceptions** should be handled centrally via `exceptions/`.
* **Business domains** should have isolated folders (e.g., `product/routers`, `product/services`, etc.).
* **Avoid business logic in routers** – services should be the only place with logic.

---

## Testing (`tests/`)

* Mirror the application structure inside `tests/`.
* Separate unit tests and integration tests.
* Use `pytest`, `httpx`, and `pytest-asyncio` for testing async routes.

```bash
pytest tests/
```

---

## Docker

Docker-related files live in `docker/`:

* `Dockerfile`: Defines the app image.
* `docker-compose.yml`: For local development with services like DB, Redis, etc.

---

## Tools

* Place helper CLI scripts in `tools/`.
* Use scripts in `script/` for one-time jobs, batch processing, or scheduled tasks.

---

## Documentation

* Store Markdown-based documentation inside `markdown/`.
* Include API docs, setup guides, and module overviews.

---

## Summary

| Layer      | Location           | Responsibility           |
| ---------- | ------------------ | ------------------------ |
| Router     | `app/routers`      | Handle HTTP requests     |
| Service    | `app/services`     | Implement business logic |
| Repository | `app/repositories` | Handle DB interactions   |

Maintain modularity by keeping each business domain self-contained. Follow this guide to ensure consistency, readability, and maintainability across the team.

---

Let me know if you'd like this in Markdown format for documentation, or if you'd like to add sample templates.
    """,
            tools=[
                create_request_form,
                reimburse,
                return_form,
            ],
        )

# Invoice Processing Agent

An agentic AI system that automates invoice processing by combining specialized data extraction with a structured storage mechanism. The agent leverages LLM capabilities to understand and parse invoice content while maintaining strict data consistency through a fixed schema.

## Features

- **Intelligent Extraction** -- Uses LLM-powered agents to extract key fields (vendor, date, line items, totals, tax) from invoices in various formats.
- **Schema Enforcement** -- Validates extracted data against a fixed JSON schema to ensure consistency and correctness.
- **Agentic Workflow** -- Employs an agent loop that can reason about ambiguous entries, request clarification, and self-correct extraction errors.
- **Persistent Storage** -- Stores processed invoices indexed by invoice number with update-on-conflict semantics.
- **REST API** -- FastAPI backend with endpoints for processing, listing, retrieving, and deleting invoices, ready for a Streamlit or any web frontend.

## Project Structure

```
Invoice-Processing-Agent/
├── invoice_agent/
│   ├── __init__.py        # Package exports
│   ├── framework.py       # Core framework (Agent, ActionContext, registry, LLM helpers)
│   ├── tools.py           # Invoice extraction and storage tools
│   ├── agent.py           # Agent factory (create_invoice_agent)
│   ├── models.py          # Pydantic request/response models
│   └── api.py             # FastAPI backend API
├── tests/
│   ├── __init__.py
│   ├── test_framework.py  # Framework unit tests
│   ├── test_tools.py      # Tool unit tests
│   ├── test_agent.py      # Agent creation tests
│   └── test_api.py        # API endpoint tests
├── main.py                # CLI entry point with sample invoice
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Invoice-Processing-Agent
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Anthropic API key:**

   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

### Usage

#### CLI

**Run with the built-in sample invoice:**

```bash
python main.py
```

**Run with a custom invoice file:**

```bash
python main.py path/to/invoice.txt
```

#### Backend API

**Start the API server:**

```bash
uvicorn invoice_agent.api:app --reload --port 8000
```

Once running, interactive API docs are available at `http://localhost:8000/docs`.

**API Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/invoices/process` | Extract structured data from invoice text and store it |
| `GET` | `/invoices` | List all stored invoices |
| `GET` | `/invoices/{invoice_number}` | Retrieve a specific invoice |
| `DELETE` | `/invoices/{invoice_number}` | Delete an invoice |
| `POST` | `/agent/process` | Full agentic pipeline (LLM reasons and calls tools in a loop) |

**Example -- process an invoice via the API:**

```bash
curl -X POST http://localhost:8000/invoices/process \
  -H "Content-Type: application/json" \
  -d '{"document_text": "Invoice #INV-2024-001\nDate: Jan 15, 2024\nTotal: $9,396.00"}'
```

**Example -- use from a Streamlit frontend:**

```python
import requests

resp = requests.post(
    "http://localhost:8000/invoices/process",
    json={"document_text": invoice_text},
)
data = resp.json()
print(data["invoice_data"])
```

#### Programmatic

```python
from invoice_agent.agent import create_invoice_agent

agent = create_invoice_agent()
response = agent.process("Please process this invoice:\n\nInvoice #123 ...")
print(response)
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

### Framework (`invoice_agent/framework.py`)

The core framework provides:

| Component | Description |
|---|---|
| `ActionContext` | Key-value store shared across tool calls within an agent session |
| `@register_tool` | Decorator that registers functions as agent-callable tools |
| `PythonActionRegistry` | Collects registered tools and supports tag-based lookup |
| `prompt_llm_for_json` | Calls the Anthropic API with tool-use to extract structured JSON matching a schema |
| `generate_response` | Thin wrapper around the Anthropic Messages API |
| `Agent` | Orchestrates an agentic loop: prompt the LLM, execute tool calls, repeat until done |
| `Goal` | Named objective that shapes the agent's system prompt |
| `AgentFunctionCallingActionLanguage` | Converts Python function signatures into Anthropic tool definitions |

### Tools (`invoice_agent/tools.py`)

| Tool | Tags | Description |
|---|---|---|
| `extract_invoice_data` | `document_processing`, `invoices` | Sends invoice text to the LLM with a specialized prompt and fixed schema to extract structured data (invoice number, date, amounts, vendor, line items) |
| `store_invoice` | `storage`, `invoices` | Persists extracted invoice data in the `ActionContext`, indexed by invoice number; supports insert and update |

### Backend API (`invoice_agent/api.py`)

A FastAPI application that exposes the invoice processing capabilities over HTTP. Key design points:

- **Direct processing** (`POST /invoices/process`) -- calls the extraction and storage tools directly for fast, predictable responses.
- **Agent processing** (`POST /agent/process`) -- runs the full agentic loop, allowing the LLM to reason over the invoice, self-correct, and call tools as needed.
- **CRUD operations** -- list, get, and delete endpoints for managing stored invoices.
- **CORS enabled** -- allows requests from any origin, making it straightforward to connect a Streamlit or other web frontend.
- **Pydantic models** (`invoice_agent/models.py`) -- typed request/response schemas with validation.

## License

This project is provided as-is for educational and internal use.

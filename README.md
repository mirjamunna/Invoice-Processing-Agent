# Invoice Processing Agent

An agentic AI system that automates invoice processing by combining specialized data extraction with a structured storage mechanism. The agent leverages the Google Gemini API to understand and parse invoice content while maintaining strict data consistency through a fixed schema.

## Features

- **Intelligent Extraction** -- Uses Gemini-powered agents to extract key fields (vendor, date, line items, totals, tax) from invoices in various formats.
- **Schema Enforcement** -- Validates extracted data against a fixed JSON schema to ensure consistency and correctness.
- **Agentic Workflow** -- Employs an agent loop that can reason about ambiguous entries, request clarification, and self-correct extraction errors.
- **Persistent Storage** -- Stores processed invoices indexed by invoice number with update-on-conflict semantics.

## Project Structure

```
Invoice-Processing-Agent/
├── invoice_agent/
│   ├── __init__.py        # Package exports
│   ├── framework.py       # Core framework (Agent, ActionContext, registry, LLM helpers)
│   ├── tools.py           # Invoice extraction and storage tools
│   └── agent.py           # Agent factory (create_invoice_agent)
├── tests/
│   ├── __init__.py
│   ├── test_framework.py  # Framework unit tests
│   ├── test_tools.py      # Tool unit tests
│   └── test_agent.py      # Agent creation tests
├── main.py                # Entry point with sample invoice
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

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

4. **Set your Google API key:**

   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

### Usage

**Run with the built-in sample invoice:**

```bash
python main.py
```

**Run with a custom invoice file:**

```bash
python main.py path/to/invoice.txt
```

**Use programmatically:**

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
| `prompt_llm_for_json` | Calls the Gemini API with function calling to extract structured JSON matching a schema |
| `generate_response` | Thin wrapper around the Gemini `generateContent` API |
| `Agent` | Orchestrates an agentic loop: prompt the LLM, execute tool calls, repeat until done |
| `Goal` | Named objective that shapes the agent's system prompt |
| `AgentFunctionCallingActionLanguage` | Converts Python function signatures into Gemini tool definitions |

### Tools (`invoice_agent/tools.py`)

| Tool | Tags | Description |
|---|---|---|
| `extract_invoice_data` | `document_processing`, `invoices` | Sends invoice text to the LLM with a specialized prompt and fixed schema to extract structured data (invoice number, date, amounts, vendor, line items) |
| `store_invoice` | `storage`, `invoices` | Persists extracted invoice data in the `ActionContext`, indexed by invoice number; supports insert and update |

## License

This project is provided as-is for educational and internal use.

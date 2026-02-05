# Invoice Processing Agent

An agentic AI system that automates invoice processing by combining specialized data extraction with a structured storage mechanism. The agent leverages LLM capabilities to understand and parse invoice content while maintaining strict data consistency through a fixed schema.

## Features

- **Intelligent Extraction** -- Uses LLM-powered agents to extract key fields (vendor, date, line items, totals, tax) from invoices in various formats.
- **Schema Enforcement** -- Validates extracted data against a fixed schema to ensure consistency and correctness.
- **Multi-Format Support** -- Handles PDF, image, and text-based invoice inputs.
- **Agentic Workflow** -- Employs an agent loop that can reason about ambiguous entries, request clarification, and self-correct extraction errors.

## Project Structure

```
Invoice-Processing-Agent/
├── venv/               # Python virtual environment (not tracked in git)
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.11+

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Invoice-Processing-Agent
   ```

2. **Create and activate the virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## License

This project is provided as-is for educational and internal use.

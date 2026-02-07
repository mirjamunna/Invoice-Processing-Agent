"""FastAPI backend API for the Invoice Processing Agent.

Provides REST endpoints for processing, storing, retrieving,
and deleting invoices. Designed for use with a Streamlit frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from invoice_agent.agent import create_invoice_agent
from invoice_agent.framework import ActionContext
from invoice_agent.models import (
    DeleteInvoiceResponse,
    ErrorResponse,
    HealthResponse,
    InvoiceData,
    InvoiceListResponse,
    ProcessInvoiceRequest,
    ProcessInvoiceResponse,
)
from invoice_agent.tools import extract_invoice_data, store_invoice

app = FastAPI(
    title="Invoice Processing Agent API",
    description="Backend API for processing invoices using an AI-powered agent. "
    "Extracts structured data from invoice text and manages invoice storage.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared ActionContext acts as the in-memory invoice store for the API lifetime.
_context = ActionContext()


def get_context() -> ActionContext:
    """Return the shared ActionContext. Useful for testing overrides."""
    return _context


def _get_storage() -> dict:
    """Return the invoice storage dict from the shared context."""
    storage = get_context().get("invoice_storage")
    if storage is None:
        storage = {}
        get_context().set("invoice_storage", storage)
    return storage


# ---------- Health ----------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
)
def health_check():
    """Check if the API service is running."""
    return HealthResponse(status="healthy", service="invoice-processing-agent")


# ---------- Invoice Processing ----------

@app.post(
    "/invoices/process",
    response_model=ProcessInvoiceResponse,
    tags=["invoices"],
    summary="Process an invoice",
    responses={422: {"model": ErrorResponse}},
)
def process_invoice(request: ProcessInvoiceRequest):
    """Extract structured data from invoice text and store it.

    This endpoint sends the invoice text through the AI extraction pipeline
    to pull out invoice number, date, total, vendor, and line items, then
    persists the result in memory.
    """
    ctx = get_context()

    try:
        extracted = extract_invoice_data(ctx, request.document_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    invoice_number = extracted.get("invoice_number")
    if not invoice_number:
        raise HTTPException(
            status_code=422,
            detail="Could not extract an invoice number from the provided text",
        )

    try:
        store_result = store_invoice(ctx, extracted)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ProcessInvoiceResponse(
        status="success",
        message=store_result["message"],
        invoice_number=invoice_number,
        invoice_data=InvoiceData(**extracted),
    )


# ---------- Invoice CRUD ----------

@app.get(
    "/invoices",
    response_model=InvoiceListResponse,
    tags=["invoices"],
    summary="List all invoices",
)
def list_invoices():
    """Return every invoice currently stored in memory."""
    storage = _get_storage()
    invoices = [InvoiceData(**data) for data in storage.values()]
    return InvoiceListResponse(invoices=invoices, count=len(invoices))


@app.get(
    "/invoices/{invoice_number}",
    response_model=InvoiceData,
    tags=["invoices"],
    summary="Get a single invoice",
    responses={404: {"model": ErrorResponse}},
)
def get_invoice(invoice_number: str):
    """Retrieve a specific invoice by its invoice number."""
    storage = _get_storage()
    if invoice_number not in storage:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_number} not found")
    return InvoiceData(**storage[invoice_number])


@app.delete(
    "/invoices/{invoice_number}",
    response_model=DeleteInvoiceResponse,
    tags=["invoices"],
    summary="Delete an invoice",
    responses={404: {"model": ErrorResponse}},
)
def delete_invoice(invoice_number: str):
    """Remove an invoice from storage."""
    storage = _get_storage()
    if invoice_number not in storage:
        raise HTTPException(status_code=404, detail=f"Invoice {invoice_number} not found")
    del storage[invoice_number]
    return DeleteInvoiceResponse(
        status="success",
        message=f"Deleted invoice {invoice_number}",
        invoice_number=invoice_number,
    )


# ---------- Agent Chat ----------

@app.post(
    "/agent/process",
    tags=["agent"],
    summary="Process invoice via agent chat",
    responses={500: {"model": ErrorResponse}},
)
def agent_process(request: ProcessInvoiceRequest):
    """Send invoice text through the full agentic pipeline.

    Unlike ``/invoices/process`` which calls the extraction and storage
    tools directly, this endpoint spins up the full Agent loop so the LLM
    can reason, call tools, and self-correct before returning.
    """
    agent = create_invoice_agent()

    prompt = f"Please process the following invoice and store it:\n\n{request.document_text}"

    try:
        response = agent.process(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {e}")

    # Retrieve whatever the agent stored
    stored = agent.environment.context.get("invoice_storage", {})

    return {
        "status": "success",
        "agent_response": response,
        "stored_invoices": stored,
    }

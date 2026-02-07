"""Pydantic models for the Invoice Processing API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class Vendor(BaseModel):
    """Vendor information extracted from an invoice."""

    name: Optional[str] = None
    address: Optional[str] = None


class LineItem(BaseModel):
    """A single line item from an invoice."""

    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceData(BaseModel):
    """Structured invoice data returned from extraction."""

    invoice_number: str
    date: str
    total_amount: float
    vendor: Optional[Vendor] = None
    line_items: Optional[list[LineItem]] = None


class ProcessInvoiceRequest(BaseModel):
    """Request body for processing an invoice."""

    document_text: str = Field(
        ..., min_length=1, description="The raw text content of the invoice to process"
    )


class ProcessInvoiceResponse(BaseModel):
    """Response from processing an invoice."""

    status: str
    message: str
    invoice_number: str
    invoice_data: InvoiceData


class InvoiceListResponse(BaseModel):
    """Response containing a list of all stored invoices."""

    invoices: list[InvoiceData]
    count: int


class DeleteInvoiceResponse(BaseModel):
    """Response from deleting an invoice."""

    status: str
    message: str
    invoice_number: str


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""

    status: str
    service: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str

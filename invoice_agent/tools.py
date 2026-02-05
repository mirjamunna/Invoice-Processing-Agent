"""Invoice processing tools for extraction and storage."""

from invoice_agent.framework import ActionContext, prompt_llm_for_json, register_tool


@register_tool(tags=["document_processing", "invoices"])
def extract_invoice_data(action_context: ActionContext, document_text: str) -> dict:
    """
    Extract standardized invoice data from document text.

    This tool ensures consistent extraction of invoice information by using a fixed schema
    and specialized prompting for invoice understanding. It will identify key fields like
    invoice numbers, dates, amounts, and line items from any invoice format.

    Args:
        document_text: The text content of the invoice to process

    Returns:
        A dictionary containing the extracted invoice data in a standardized format
    """
    invoice_schema = {
        "type": "object",
        "required": ["invoice_number", "date", "total_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "date": {"type": "string"},
            "total_amount": {"type": "number"},
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                },
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"},
                    },
                },
            },
        },
    }

    # Create a focused prompt for invoice extraction
    extraction_prompt = f"""
            You are an expert invoice analyzer. Extract invoice information accurately and
            thoroughly. Pay special attention to:
            - Invoice numbers (look for 'Invoice #', 'No.', 'Reference', etc.)
            - Dates (focus on invoice date or issue date)
            - Amounts (ensure you capture the total amount correctly)
            - Line items (capture all individual charges)

            Stop and think step by step. Then, extract the invoice data from:

            <invoice>
            {document_text}
            </invoice>
    """

    # Use prompt_llm_for_json with our specialized prompt
    return prompt_llm_for_json(
        action_context=action_context,
        schema=invoice_schema,
        prompt=extraction_prompt,
    )


@register_tool(tags=["storage", "invoices"])
def store_invoice(action_context: ActionContext, invoice_data: dict) -> dict:
    """
    Store an invoice in our invoice database. If an invoice with the same number
    already exists, it will be updated.

    Args:
        invoice_data: The processed invoice data to store

    Returns:
        A dictionary containing the storage result and invoice number
    """
    # Get our invoice storage from context
    storage = action_context.get("invoice_storage")
    if storage is None:
        storage = {}
        action_context.set("invoice_storage", storage)

    # Extract invoice number for reference
    invoice_number = invoice_data.get("invoice_number")
    if not invoice_number:
        raise ValueError("Invoice data must contain an invoice number")

    # Store the invoice
    storage[invoice_number] = invoice_data

    return {
        "status": "success",
        "message": f"Stored invoice {invoice_number}",
        "invoice_number": invoice_number,
    }

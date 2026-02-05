"""Tests for invoice processing tools."""

from unittest.mock import MagicMock, patch

import pytest

from invoice_agent.framework import ActionContext, clear_registered_tools

# Import tools to trigger registration
import invoice_agent.tools
from invoice_agent.tools import extract_invoice_data, store_invoice


class TestExtractInvoiceData:
    def test_calls_prompt_llm_for_json(self):
        """Verify extract_invoice_data calls prompt_llm_for_json with correct args."""
        mock_result = {
            "invoice_number": "INV-001",
            "date": "2024-01-15",
            "total_amount": 1500.00,
        }

        with patch("invoice_agent.tools.prompt_llm_for_json", return_value=mock_result) as mock_fn:
            ctx = ActionContext()
            result = extract_invoice_data(ctx, "Invoice #INV-001\nTotal: $1500")

            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args

            # Check action_context was passed
            assert call_kwargs.kwargs["action_context"] is ctx

            # Check schema has required fields
            schema = call_kwargs.kwargs["schema"]
            assert "invoice_number" in schema["properties"]
            assert "date" in schema["properties"]
            assert "total_amount" in schema["properties"]
            assert "vendor" in schema["properties"]
            assert "line_items" in schema["properties"]
            assert set(schema["required"]) == {"invoice_number", "date", "total_amount"}

            # Check prompt contains the document text
            prompt = call_kwargs.kwargs["prompt"]
            assert "Invoice #INV-001" in prompt
            assert "Total: $1500" in prompt

            # Check result is returned
            assert result == mock_result

    def test_prompt_contains_instructions(self):
        """Verify the extraction prompt contains key instructions."""
        with patch("invoice_agent.tools.prompt_llm_for_json", return_value={}) as mock_fn:
            ctx = ActionContext()
            extract_invoice_data(ctx, "test")
            prompt = mock_fn.call_args.kwargs["prompt"]

            assert "Invoice" in prompt or "invoice" in prompt
            assert "<invoice>" in prompt
            assert "</invoice>" in prompt

    def test_schema_structure(self):
        """Verify the schema defines proper types for all fields."""
        with patch("invoice_agent.tools.prompt_llm_for_json", return_value={}) as mock_fn:
            ctx = ActionContext()
            extract_invoice_data(ctx, "test")
            schema = mock_fn.call_args.kwargs["schema"]

            assert schema["properties"]["invoice_number"]["type"] == "string"
            assert schema["properties"]["date"]["type"] == "string"
            assert schema["properties"]["total_amount"]["type"] == "number"
            assert schema["properties"]["vendor"]["type"] == "object"
            assert schema["properties"]["line_items"]["type"] == "array"

            # Check line_items item schema
            item_props = schema["properties"]["line_items"]["items"]["properties"]
            assert "description" in item_props
            assert "quantity" in item_props
            assert "unit_price" in item_props
            assert "total" in item_props


class TestStoreInvoice:
    def test_store_new_invoice(self):
        ctx = ActionContext()
        invoice = {"invoice_number": "INV-001", "total_amount": 100}
        result = store_invoice(ctx, invoice)

        assert result["status"] == "success"
        assert result["invoice_number"] == "INV-001"
        assert "INV-001" in result["message"]

        # Verify it's in storage
        storage = ctx.get("invoice_storage")
        assert storage["INV-001"] == invoice

    def test_store_multiple_invoices(self):
        ctx = ActionContext()
        inv1 = {"invoice_number": "INV-001", "total_amount": 100}
        inv2 = {"invoice_number": "INV-002", "total_amount": 200}

        store_invoice(ctx, inv1)
        store_invoice(ctx, inv2)

        storage = ctx.get("invoice_storage")
        assert len(storage) == 2
        assert storage["INV-001"] == inv1
        assert storage["INV-002"] == inv2

    def test_update_existing_invoice(self):
        ctx = ActionContext()
        inv_v1 = {"invoice_number": "INV-001", "total_amount": 100}
        inv_v2 = {"invoice_number": "INV-001", "total_amount": 150}

        store_invoice(ctx, inv_v1)
        store_invoice(ctx, inv_v2)

        storage = ctx.get("invoice_storage")
        assert len(storage) == 1
        assert storage["INV-001"]["total_amount"] == 150

    def test_missing_invoice_number_raises(self):
        ctx = ActionContext()
        with pytest.raises(ValueError, match="invoice number"):
            store_invoice(ctx, {"total_amount": 100})

    def test_empty_invoice_number_raises(self):
        ctx = ActionContext()
        with pytest.raises(ValueError, match="invoice number"):
            store_invoice(ctx, {"invoice_number": "", "total_amount": 100})

    def test_none_invoice_number_raises(self):
        ctx = ActionContext()
        with pytest.raises(ValueError, match="invoice number"):
            store_invoice(ctx, {"invoice_number": None, "total_amount": 100})

    def test_storage_persists_across_calls(self):
        """Verify that storage initialized on first call persists for subsequent calls."""
        ctx = ActionContext()
        # First call creates storage
        store_invoice(ctx, {"invoice_number": "A", "total_amount": 1})
        # Second call should use existing storage
        store_invoice(ctx, {"invoice_number": "B", "total_amount": 2})

        storage = ctx.get("invoice_storage")
        assert "A" in storage
        assert "B" in storage

    def test_pre_existing_storage(self):
        """Verify it works when invoice_storage already exists in context."""
        existing = {"EXISTING-001": {"invoice_number": "EXISTING-001", "total_amount": 50}}
        ctx = ActionContext({"invoice_storage": existing})

        store_invoice(ctx, {"invoice_number": "NEW-001", "total_amount": 100})

        storage = ctx.get("invoice_storage")
        assert "EXISTING-001" in storage
        assert "NEW-001" in storage

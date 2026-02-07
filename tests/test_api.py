"""Tests for the Invoice Processing API endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from invoice_agent.api import app, get_context
from invoice_agent.framework import ActionContext


@pytest.fixture(autouse=True)
def _reset_context(monkeypatch):
    """Give each test a fresh ActionContext so tests don't share state."""
    fresh = ActionContext()
    monkeypatch.setattr("invoice_agent.api._context", fresh)


@pytest.fixture()
def client():
    return TestClient(app)


# ---- Helpers ----

SAMPLE_EXTRACTED = {
    "invoice_number": "INV-001",
    "date": "2024-01-15",
    "total_amount": 1500.00,
    "vendor": {"name": "Acme Corp", "address": "123 Main St"},
    "line_items": [
        {
            "description": "Widget",
            "quantity": 10,
            "unit_price": 150.0,
            "total": 1500.0,
        }
    ],
}


def _seed_invoice(client, invoice_data=None):
    """Insert an invoice directly into storage via the shared context."""
    data = invoice_data or SAMPLE_EXTRACTED
    ctx = get_context()
    storage = ctx.get("invoice_storage")
    if storage is None:
        storage = {}
        ctx.set("invoice_storage", storage)
    storage[data["invoice_number"]] = data


# ---- Health ----


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "invoice-processing-agent"


# ---- POST /invoices/process ----


class TestProcessInvoice:
    def test_process_invoice_success(self, client):
        with patch(
            "invoice_agent.api.extract_invoice_data", return_value=SAMPLE_EXTRACTED
        ):
            resp = client.post(
                "/invoices/process",
                json={"document_text": "Invoice #INV-001\nTotal: $1500"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["invoice_number"] == "INV-001"
        assert body["invoice_data"]["total_amount"] == 1500.0

    def test_process_invoice_empty_text_rejected(self, client):
        resp = client.post("/invoices/process", json={"document_text": ""})
        assert resp.status_code == 422

    def test_process_invoice_missing_body(self, client):
        resp = client.post("/invoices/process")
        assert resp.status_code == 422

    def test_process_invoice_extraction_failure(self, client):
        with patch(
            "invoice_agent.api.extract_invoice_data",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            resp = client.post(
                "/invoices/process",
                json={"document_text": "some text"},
            )
        assert resp.status_code == 500
        assert "Extraction failed" in resp.json()["detail"]

    def test_process_invoice_no_invoice_number(self, client):
        bad_extracted = {"invoice_number": "", "date": "2024-01-01", "total_amount": 0}
        with patch(
            "invoice_agent.api.extract_invoice_data", return_value=bad_extracted
        ):
            resp = client.post(
                "/invoices/process",
                json={"document_text": "garbage"},
            )
        assert resp.status_code == 422


# ---- GET /invoices ----


class TestListInvoices:
    def test_list_empty(self, client):
        resp = client.get("/invoices")
        assert resp.status_code == 200
        body = resp.json()
        assert body["invoices"] == []
        assert body["count"] == 0

    def test_list_with_invoices(self, client):
        _seed_invoice(client)
        resp = client.get("/invoices")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["invoices"][0]["invoice_number"] == "INV-001"

    def test_list_multiple_invoices(self, client):
        _seed_invoice(client, SAMPLE_EXTRACTED)
        second = {**SAMPLE_EXTRACTED, "invoice_number": "INV-002"}
        _seed_invoice(client, second)
        resp = client.get("/invoices")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2


# ---- GET /invoices/{invoice_number} ----


class TestGetInvoice:
    def test_get_existing(self, client):
        _seed_invoice(client)
        resp = client.get("/invoices/INV-001")
        assert resp.status_code == 200
        assert resp.json()["invoice_number"] == "INV-001"
        assert resp.json()["total_amount"] == 1500.0

    def test_get_not_found(self, client):
        resp = client.get("/invoices/MISSING-999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ---- DELETE /invoices/{invoice_number} ----


class TestDeleteInvoice:
    def test_delete_existing(self, client):
        _seed_invoice(client)
        resp = client.delete("/invoices/INV-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["invoice_number"] == "INV-001"

        # Confirm it's gone
        resp = client.get("/invoices/INV-001")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/invoices/MISSING-999")
        assert resp.status_code == 404


# ---- POST /agent/process ----


class TestAgentProcess:
    def test_agent_process_success(self, client):
        mock_agent = type("MockAgent", (), {
            "process": lambda self, prompt: "Processed invoice INV-001 successfully.",
            "environment": type("Env", (), {
                "context": ActionContext(
                    {"invoice_storage": {"INV-001": SAMPLE_EXTRACTED}}
                ),
            })(),
        })()

        with patch("invoice_agent.api.create_invoice_agent", return_value=mock_agent):
            resp = client.post(
                "/agent/process",
                json={"document_text": "Invoice #INV-001\nTotal: $1500"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "INV-001" in body["agent_response"]
        assert "INV-001" in body["stored_invoices"]

    def test_agent_process_failure(self, client):
        mock_agent = type("MockAgent", (), {
            "process": lambda self, prompt: (_ for _ in ()).throw(
                RuntimeError("LLM down")
            ),
        })()

        with patch("invoice_agent.api.create_invoice_agent", return_value=mock_agent):
            resp = client.post(
                "/agent/process",
                json={"document_text": "some text"},
            )

        assert resp.status_code == 500
        assert "Agent processing failed" in resp.json()["detail"]


# ---- Models validation ----


class TestModels:
    def test_invoice_data_model(self):
        from invoice_agent.models import InvoiceData

        inv = InvoiceData(invoice_number="X", date="2024-01-01", total_amount=100.0)
        assert inv.invoice_number == "X"
        assert inv.vendor is None
        assert inv.line_items is None

    def test_process_request_requires_text(self):
        from invoice_agent.models import ProcessInvoiceRequest

        with pytest.raises(Exception):
            ProcessInvoiceRequest(document_text="")

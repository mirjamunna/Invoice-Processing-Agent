"""Tests for the invoice agent creation."""

from unittest.mock import patch

import pytest

from invoice_agent.framework import (
    Agent,
    AgentFunctionCallingActionLanguage,
    PythonActionRegistry,
    PythonEnvironment,
    clear_registered_tools,
)

# Must import tools before creating agent so they get registered
import invoice_agent.tools
from invoice_agent.agent import create_invoice_agent


class TestCreateInvoiceAgent:
    def test_returns_agent_instance(self):
        agent = create_invoice_agent()
        assert isinstance(agent, Agent)

    def test_has_two_goals(self):
        agent = create_invoice_agent()
        assert len(agent.goals) == 2

    def test_persona_goal(self):
        agent = create_invoice_agent()
        persona = agent.goals[0]
        assert persona.name == "Persona"
        assert "Invoice Processing Agent" in persona.description

    def test_process_invoices_goal(self):
        agent = create_invoice_agent()
        process_goal = agent.goals[1]
        assert process_goal.name == "Process Invoices"
        assert "extract" in process_goal.description.lower() or "Extract" in process_goal.description

    def test_has_agent_language(self):
        agent = create_invoice_agent()
        assert isinstance(agent.agent_language, AgentFunctionCallingActionLanguage)

    def test_has_action_registry(self):
        agent = create_invoice_agent()
        assert isinstance(agent.action_registry, PythonActionRegistry)

    def test_has_environment(self):
        agent = create_invoice_agent()
        assert isinstance(agent.environment, PythonEnvironment)

    def test_registry_contains_invoice_tools(self):
        agent = create_invoice_agent()
        tools = agent.action_registry.get_tools()
        assert "extract_invoice_data" in tools
        assert "store_invoice" in tools

    def test_tools_have_correct_tags(self):
        agent = create_invoice_agent()
        tools = agent.action_registry.get_tools()
        assert "invoices" in tools["extract_invoice_data"]["tags"]
        assert "document_processing" in tools["extract_invoice_data"]["tags"]
        assert "invoices" in tools["store_invoice"]["tags"]
        assert "storage" in tools["store_invoice"]["tags"]

    def test_generate_response_is_callable(self):
        agent = create_invoice_agent()
        assert callable(agent._generate_response)

"""Invoice processing agent creation and entry point."""

from invoice_agent.framework import (
    Agent,
    AgentFunctionCallingActionLanguage,
    Goal,
    PythonActionRegistry,
    PythonEnvironment,
    generate_response,
)

# Import tools so they get registered via @register_tool
import invoice_agent.tools  # noqa: F401


def create_invoice_agent() -> Agent:
    """Create and return a configured invoice processing agent.

    Returns:
        An Agent instance configured with invoice processing tools and goals.
    """
    # Create action registry with our invoice tools
    action_registry = PythonActionRegistry()

    # Create our base environment
    environment = PythonEnvironment()

    # Define our invoice processing goals
    goals = [
        Goal(
            name="Persona",
            description="You are an Invoice Processing Agent, specialized in handling and storing invoice data.",
        ),
        Goal(
            name="Process Invoices",
            description="""
            Your goal is to process invoices by extracting their data and storing it properly.
            For each invoice:
            1. Extract all important information including numbers, dates, amounts, and line items
            2. Store the extracted data indexed by invoice number
            3. Provide confirmation of successful processing
            4. Handle any errors appropriately
            """,
        ),
    ]

    # Create the agent
    return Agent(
        goals=goals,
        agent_language=AgentFunctionCallingActionLanguage(),
        action_registry=action_registry,
        generate_response=generate_response,
        environment=environment,
    )

"""Core framework providing the agent infrastructure for invoice processing."""

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import anthropic


class ActionContext:
    """Context object for actions, providing key-value storage and retrieval."""

    def __init__(self, initial_data: Optional[dict] = None):
        self._data = initial_data or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the context."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the context."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ActionContext({self._data})"


# Global registry for tools registered via the @register_tool decorator
_registered_tools: dict[str, dict] = {}


def register_tool(tags: Optional[list[str]] = None):
    """Decorator to register a function as an agent tool.

    Args:
        tags: Optional list of tags for categorizing the tool.

    Returns:
        The decorated function, now registered in the global tool registry.
    """

    def decorator(func: Callable) -> Callable:
        tool_info = {
            "function": func,
            "name": func.__name__,
            "tags": tags or [],
            "doc": func.__doc__ or "",
        }
        _registered_tools[func.__name__] = tool_info
        return func

    return decorator


def get_registered_tools() -> dict[str, dict]:
    """Return a copy of all registered tools."""
    return _registered_tools.copy()


def clear_registered_tools() -> None:
    """Clear all registered tools. Useful for testing."""
    _registered_tools.clear()


def prompt_llm_for_json(
    action_context: ActionContext,
    schema: dict,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Prompt an LLM to extract structured JSON data according to a schema.

    Uses Anthropic's tool use feature to guarantee the response conforms
    to the provided JSON schema.

    Args:
        action_context: The current action context.
        schema: A JSON Schema describing the expected output structure.
        prompt: The prompt to send to the LLM.
        model: The Anthropic model to use.

    Returns:
        A dictionary matching the provided schema.
    """
    client = anthropic.Anthropic()

    tool_definition = {
        "name": "extract_data",
        "description": "Extract structured data from the provided content",
        "input_schema": schema,
    }

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        tools=[tool_definition],
        tool_choice={"type": "tool", "name": "extract_data"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError("LLM did not return structured data")


def generate_response(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    system: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> anthropic.types.Message:
    """Generate a response from the Anthropic LLM.

    Args:
        messages: The conversation messages.
        tools: Optional tool definitions for function calling.
        system: Optional system prompt.
        model: The Anthropic model to use.

    Returns:
        The Anthropic API Message response.
    """
    client = anthropic.Anthropic()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system

    return client.messages.create(**kwargs)


@dataclass
class Goal:
    """Defines a named goal for an agent."""

    name: str
    description: str


class AgentFunctionCallingActionLanguage:
    """Translates registered tools into the Anthropic function-calling format."""

    def format_tools(self, tools: dict[str, dict]) -> list[dict]:
        """Convert registered tool info dicts into Anthropic tool definitions.

        Args:
            tools: A mapping of tool name to tool info dict.

        Returns:
            A list of tool definitions suitable for the Anthropic API.
        """
        formatted = []
        for name, info in tools.items():
            sig = inspect.signature(info["function"])
            properties: dict[str, dict] = {}
            required: list[str] = []

            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                dict: "object",
                list: "array",
            }

            for param_name, param in sig.parameters.items():
                if param_name == "action_context":
                    continue
                json_type = type_map.get(param.annotation, "string")
                properties[param_name] = {"type": json_type}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            formatted.append(
                {
                    "name": name,
                    "description": info["doc"],
                    "input_schema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return formatted


class PythonActionRegistry:
    """Collects and manages Python-based action tools."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def load_tools(self) -> None:
        """Load all tools that have been registered via @register_tool."""
        self._tools = get_registered_tools()

    def get_tools(self) -> dict[str, dict]:
        """Return all loaded tools, auto-loading if necessary."""
        if not self._tools:
            self.load_tools()
        return self._tools

    def get_tool(self, name: str) -> Optional[dict]:
        """Get a specific tool by name."""
        return self.get_tools().get(name)

    def get_tools_by_tag(self, tag: str) -> dict[str, dict]:
        """Get all tools matching a given tag."""
        return {
            name: info
            for name, info in self.get_tools().items()
            if tag in info.get("tags", [])
        }


class PythonEnvironment:
    """Provides the execution environment and shared context for the agent."""

    def __init__(self):
        self._context = ActionContext()

    @property
    def context(self) -> ActionContext:
        return self._context


class Agent:
    """Orchestrates tools and LLM interaction to accomplish defined goals."""

    def __init__(
        self,
        goals: list[Goal],
        agent_language: AgentFunctionCallingActionLanguage,
        action_registry: PythonActionRegistry,
        generate_response: Callable,
        environment: PythonEnvironment,
    ):
        self.goals = goals
        self.agent_language = agent_language
        self.action_registry = action_registry
        self._generate_response = generate_response
        self.environment = environment
        self._messages: list[dict] = []

    def _build_system_prompt(self) -> str:
        """Build a system prompt from the agent's goals."""
        parts = []
        for goal in self.goals:
            parts.append(f"## {goal.name}\n{goal.description}")
        return "\n\n".join(parts)

    def process(self, user_input: str) -> str:
        """Process user input through the agent loop.

        The agent will use its tools as needed to fulfill the request,
        continuing in a loop until it produces a final text response.

        Args:
            user_input: The user's message or request.

        Returns:
            The agent's final text response.
        """
        tools = self.action_registry.get_tools()
        formatted_tools = self.agent_language.format_tools(tools)
        system_prompt = self._build_system_prompt()

        self._messages.append({"role": "user", "content": user_input})

        max_iterations = 10
        for _ in range(max_iterations):
            response = self._generate_response(
                messages=self._messages,
                tools=formatted_tools if formatted_tools else None,
                system=system_prompt,
            )

            assistant_content = response.content
            self._messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_uses:
                text_parts = [b.text for b in assistant_content if b.type == "text"]
                return "\n".join(text_parts)

            tool_results = []
            for tool_use in tool_uses:
                tool_info = tools.get(tool_use.name)
                if tool_info:
                    func = tool_info["function"]
                    kwargs = dict(tool_use.input)
                    sig = inspect.signature(func)
                    if "action_context" in sig.parameters:
                        kwargs["action_context"] = self.environment.context

                    try:
                        result = func(**kwargs)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": json.dumps(result)
                                if isinstance(result, (dict, list))
                                else str(result),
                            }
                        )
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": f"Error: {e!s}",
                                "is_error": True,
                            }
                        )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: Unknown tool {tool_use.name}",
                            "is_error": True,
                        }
                    )

            self._messages.append({"role": "user", "content": tool_results})

        return "Maximum iterations reached without a final response."

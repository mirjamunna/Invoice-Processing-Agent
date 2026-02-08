"""Core framework providing the agent infrastructure for invoice processing."""

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from google import genai
from google.genai import types


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
    model: str = "gemini-2.0-flash",
) -> dict:
    """Prompt an LLM to extract structured JSON data according to a schema.

    Uses Gemini's function calling feature to guarantee the response conforms
    to the provided JSON schema.

    Args:
        action_context: The current action context.
        schema: A JSON Schema describing the expected output structure.
        prompt: The prompt to send to the LLM.
        model: The Gemini model to use.

    Returns:
        A dictionary matching the provided schema.
    """
    client = genai.Client()

    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="extract_data",
                description="Extract structured data from the provided content",
                parameters=schema,
            )
        ]
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                )
            ),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.function_call:
            return dict(part.function_call.args)

    raise ValueError("LLM did not return structured data")


def generate_response(
    messages: list,
    tools: Optional[list[dict]] = None,
    system: Optional[str] = None,
    model: str = "gemini-2.0-flash",
):
    """Generate a response from the Gemini LLM.

    Args:
        messages: The conversation messages as a list of Gemini Content objects.
        tools: Optional tool definitions for function calling.
        system: Optional system prompt.
        model: The Gemini model to use.

    Returns:
        The Gemini API GenerateContentResponse.
    """
    client = genai.Client()

    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system
    if tools:
        gemini_tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t["parameters"],
                    )
                    for t in tools
                ]
            )
        ]
        config_kwargs["tools"] = gemini_tools

    return client.models.generate_content(
        model=model,
        contents=messages,
        config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
    )


@dataclass
class Goal:
    """Defines a named goal for an agent."""

    name: str
    description: str


class AgentFunctionCallingActionLanguage:
    """Translates registered tools into the Gemini function-calling format."""

    def format_tools(self, tools: dict[str, dict]) -> list[dict]:
        """Convert registered tool info dicts into Gemini tool definitions.

        Args:
            tools: A mapping of tool name to tool info dict.

        Returns:
            A list of tool definitions suitable for the Gemini API.
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
                    "parameters": {
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
        self._messages: list = []

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

        self._messages.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        max_iterations = 10
        for _ in range(max_iterations):
            response = self._generate_response(
                messages=self._messages,
                tools=formatted_tools if formatted_tools else None,
                system=system_prompt,
            )

            response_content = response.candidates[0].content
            self._messages.append(response_content)

            function_calls = [
                p for p in response_content.parts if p.function_call
            ]

            if not function_calls:
                text_parts = [
                    p.text for p in response_content.parts if p.text
                ]
                return "\n".join(text_parts)

            function_responses = []
            for fc_part in function_calls:
                fc = fc_part.function_call
                tool_info = tools.get(fc.name)
                if tool_info:
                    func = tool_info["function"]
                    kwargs = dict(fc.args)
                    sig = inspect.signature(func)
                    if "action_context" in sig.parameters:
                        kwargs["action_context"] = self.environment.context

                    try:
                        result = func(**kwargs)
                        result_data = (
                            result
                            if isinstance(result, dict)
                            else {"result": json.dumps(result) if isinstance(result, list) else str(result)}
                        )
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=fc.name,
                                    response=result_data,
                                )
                            )
                        )
                    except Exception as e:
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=fc.name,
                                    response={"error": str(e)},
                                )
                            )
                        )
                else:
                    function_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"error": f"Unknown tool {fc.name}"},
                            )
                        )
                    )

            self._messages.append(
                types.Content(role="user", parts=function_responses)
            )

        return "Maximum iterations reached without a final response."

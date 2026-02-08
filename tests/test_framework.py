"""Tests for the core framework module."""

import pytest

from invoice_agent.framework import (
    ActionContext,
    AgentFunctionCallingActionLanguage,
    Goal,
    PythonActionRegistry,
    PythonEnvironment,
    clear_registered_tools,
    get_registered_tools,
    register_tool,
)


class TestActionContext:
    def test_empty_context(self):
        ctx = ActionContext()
        assert ctx.get("missing") is None

    def test_get_default(self):
        ctx = ActionContext()
        assert ctx.get("missing", 42) == 42

    def test_initial_data(self):
        ctx = ActionContext({"key": "value"})
        assert ctx.get("key") == "value"

    def test_set_and_get(self):
        ctx = ActionContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_overwrite(self):
        ctx = ActionContext({"key": "old"})
        ctx.set("key", "new")
        assert ctx.get("key") == "new"

    def test_contains(self):
        ctx = ActionContext({"exists": True})
        assert "exists" in ctx
        assert "missing" not in ctx

    def test_repr(self):
        ctx = ActionContext({"a": 1})
        assert "ActionContext" in repr(ctx)
        assert "'a'" in repr(ctx)

    def test_mutable_reference(self):
        """Verify that modifying a stored mutable object persists."""
        ctx = ActionContext()
        data = {"inner": []}
        ctx.set("data", data)
        data["inner"].append(1)
        assert ctx.get("data")["inner"] == [1]


class TestRegisterTool:
    def setup_method(self):
        clear_registered_tools()

    def test_register_simple_tool(self):
        @register_tool(tags=["test"])
        def my_tool(x: str) -> str:
            """A test tool."""
            return x

        tools = get_registered_tools()
        assert "my_tool" in tools
        assert tools["my_tool"]["name"] == "my_tool"
        assert tools["my_tool"]["tags"] == ["test"]
        assert "A test tool." in tools["my_tool"]["doc"]

    def test_register_tool_no_tags(self):
        @register_tool()
        def another_tool() -> None:
            pass

        tools = get_registered_tools()
        assert tools["another_tool"]["tags"] == []

    def test_registered_function_still_callable(self):
        @register_tool(tags=["test"])
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_multiple_tools_registered(self):
        @register_tool(tags=["a"])
        def tool_a():
            pass

        @register_tool(tags=["b"])
        def tool_b():
            pass

        tools = get_registered_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_get_registered_tools_returns_copy(self):
        @register_tool()
        def some_tool():
            pass

        tools1 = get_registered_tools()
        tools2 = get_registered_tools()
        assert tools1 is not tools2


class TestGoal:
    def test_goal_creation(self):
        goal = Goal(name="Test", description="A test goal")
        assert goal.name == "Test"
        assert goal.description == "A test goal"


class TestPythonEnvironment:
    def test_has_context(self):
        env = PythonEnvironment()
        assert isinstance(env.context, ActionContext)

    def test_context_is_persistent(self):
        env = PythonEnvironment()
        env.context.set("key", "val")
        assert env.context.get("key") == "val"


class TestPythonActionRegistry:
    def setup_method(self):
        clear_registered_tools()

    def test_load_tools(self):
        @register_tool(tags=["test"])
        def reg_tool():
            """Registered tool."""
            pass

        registry = PythonActionRegistry()
        registry.load_tools()
        assert "reg_tool" in registry.get_tools()

    def test_auto_load_on_get_tools(self):
        @register_tool()
        def auto_tool():
            pass

        registry = PythonActionRegistry()
        # Should auto-load without explicit load_tools()
        assert "auto_tool" in registry.get_tools()

    def test_get_tool_by_name(self):
        @register_tool(tags=["x"])
        def named_tool():
            pass

        registry = PythonActionRegistry()
        assert registry.get_tool("named_tool") is not None
        assert registry.get_tool("nonexistent") is None

    def test_get_tools_by_tag(self):
        @register_tool(tags=["alpha", "beta"])
        def tagged_a():
            pass

        @register_tool(tags=["beta"])
        def tagged_b():
            pass

        @register_tool(tags=["gamma"])
        def tagged_c():
            pass

        registry = PythonActionRegistry()
        beta_tools = registry.get_tools_by_tag("beta")
        assert "tagged_a" in beta_tools
        assert "tagged_b" in beta_tools
        assert "tagged_c" not in beta_tools


class TestAgentFunctionCallingActionLanguage:
    def setup_method(self):
        clear_registered_tools()

    def test_format_tools_basic(self):
        @register_tool()
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        lang = AgentFunctionCallingActionLanguage()
        tools = get_registered_tools()
        formatted = lang.format_tools(tools)

        assert len(formatted) == 1
        tool_def = formatted[0]
        assert tool_def["name"] == "greet"
        assert "Say hello." in tool_def["description"]
        assert tool_def["parameters"]["properties"]["name"]["type"] == "string"
        assert "name" in tool_def["parameters"]["required"]

    def test_format_tools_skips_action_context(self):
        @register_tool()
        def ctx_tool(action_context: ActionContext, value: str) -> str:
            """Tool with context."""
            return value

        lang = AgentFunctionCallingActionLanguage()
        formatted = lang.format_tools(get_registered_tools())
        props = formatted[0]["parameters"]["properties"]

        assert "action_context" not in props
        assert "value" in props

    def test_format_tools_type_mapping(self):
        @register_tool()
        def typed_tool(s: str, i: int, f: float, b: bool, d: dict, l: list) -> None:
            """Typed tool."""
            pass

        lang = AgentFunctionCallingActionLanguage()
        formatted = lang.format_tools(get_registered_tools())
        props = formatted[0]["parameters"]["properties"]

        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["d"]["type"] == "object"
        assert props["l"]["type"] == "array"

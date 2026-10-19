"""Per-task routes: every route is resolved before any child exists, and each child runs on its own route."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool
from tools.delegation_live_log import live_transcript_root
from tools.registry import registry

OPUS = "anthropic/claude-opus-5-5@xhigh"
ASTRA = "openai-codex/gpt-6-astra@high"
GROK = "xai-oauth/grok-4.7"
RUNTIMES = {
    "anthropic": {"provider": "anthropic", "api_mode": "anthropic_messages", "base_url": "https://anthropic.invalid",
                  "api_key": "sk-anthropic", "request_overrides": {"anthropic_beta": "route-own"}},
    "openai-codex": {"provider": "openai-codex", "api_mode": "codex_responses", "base_url": "https://codex.invalid",
                     "api_key": "sk-codex"},
    "xai-oauth": {"provider": "xai-oauth", "api_mode": "codex_responses", "base_url": "https://xai.invalid",
                  "api_key": "sk-xai"},
    # Same provider as the parent, and the runtime leaves the wire unset.
    "openrouter": {"provider": "openrouter", "api_mode": None, "base_url": "https://openrouter.invalid/v1",
                   "api_key": "sk-openrouter"},
}
PARENT_CHAIN = [{"provider": "deepseek", "model": "deepseek-chat"}]


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 1  # depth > 0 runs the batch synchronously through the registry handler
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._session_db = None
    parent._credential_pool = None
    parent._client_kwargs = {}
    parent.client = None
    parent._fallback_chain = list(PARENT_CHAIN)
    parent._oneshot_children_spawned = 0
    parent.enabled_toolsets = ["terminal", "file"]
    parent.disabled_toolsets = []
    parent.request_overrides = {"parent": "overrides"}
    parent.capabilities = {"vision": True}
    parent.provider = "openrouter"
    parent.requested_provider = "openrouter"
    parent.base_url = "https://parent.invalid/v1"
    parent.api_key = "sk-parent"
    parent.api_mode = "codex_responses"
    parent.model = "parent/model"
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent.platform = "cli"
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _child(**kwargs):
    child = MagicMock()
    child._credential_pool = None
    child._delegate_saved_tool_names = []
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = kwargs["model"]
    child.provider = kwargs["provider"]
    child.run_conversation.return_value = {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}
    return child


class _Harness:
    def __init__(self, monkeypatch, tmp_path, config, runtimes=RUNTIMES):
        self.config = {"max_spawn_depth": 3, **config}
        self.resolved: list[str] = []
        self.parent = _parent()
        self._monkeypatch = monkeypatch
        self._tmp_path = tmp_path
        self._runtimes = runtimes

    def _resolve(self, requested=None, target_model=None, **_kw):
        self.resolved.append(requested)
        runtime = self._runtimes[requested]
        if isinstance(runtime, Exception):
            raise runtime
        return dict(runtime)

    def call(self, tasks):
        with (
            patch.object(delegate_tool, "_load_config", side_effect=lambda: self.config),
            patch.object(delegate_tool, "_resolve_workspace_hint", return_value=str(self._tmp_path)),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=self._resolve),
            patch("agent.oneshot_footprint.is_single_query_session", return_value=True),
            patch("run_agent.AIAgent", side_effect=_child) as agent_class,
        ):
            result = json.loads(registry.get_entry("delegate_task").handler({"tasks": tasks}, parent_agent=self.parent))
        self.built = [c.kwargs for c in agent_class.call_args_list]
        return result


def _assert_nothing_happened(harness, result):
    assert "error" in result
    assert harness.built == []
    assert harness.parent._oneshot_children_spawned == 0
    root = live_transcript_root()
    assert not root.exists() or not any(root.iterdir())


def test_mixed_batch_builds_each_child_on_its_own_route(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
    result = harness.call([
        {"goal": "Review with opus", "route": OPUS},
        {"goal": "Review with astra", "route": ASTRA},
        {"goal": "Review on the default model"},
        {"goal": "Review with opus again", "route": OPUS},
    ])

    assert [r["status"] for r in result["results"]] == ["completed"] * 4
    assert harness.resolved == ["anthropic", "openai-codex"]
    picked = [(k["provider"], k["model"], k["base_url"], k["api_key"], k["api_mode"], k["reasoning_config"])
              for k in harness.built]
    assert picked == [
        ("anthropic", "claude-opus-5-5", "https://anthropic.invalid", "sk-anthropic", "anthropic_messages",
         {"enabled": True, "effort": "xhigh"}),
        ("openai-codex", "gpt-6-astra", "https://codex.invalid", "sk-codex", "codex_responses",
         {"enabled": True, "effort": "high"}),
        ("openrouter", "parent/model", "https://parent.invalid/v1", "sk-parent", "codex_responses",
         {"enabled": True, "effort": "medium"}),
        ("anthropic", "claude-opus-5-5", "https://anthropic.invalid", "sk-anthropic", "anthropic_messages",
         {"enabled": True, "effort": "xhigh"}),
    ]
    assert harness.parent._oneshot_children_spawned == 4


def test_route_child_never_inherits_delegation_endpoint_overrides_or_fallback(monkeypatch, tmp_path):
    delegation_chain = [{"provider": "zai", "model": "glm-5"}]
    harness = _Harness(monkeypatch, tmp_path, {
        "routes": [OPUS], "base_url": "https://delegation.invalid/v1", "api_key": "sk-delegation",
        "api_mode": "chat_completions", "request_overrides": {"service_tier": "flex"},
        "fallback_providers": delegation_chain, "oneshot_max_children": 10,
    })
    harness.call([{"goal": "Review with opus", "route": OPUS}, {"goal": "Review on the default route"}])

    routed, unrouted = harness.built
    assert (routed["base_url"], routed["api_key"]) == ("https://anthropic.invalid", "sk-anthropic")
    assert routed["request_overrides"] == {"anthropic_beta": "route-own"}
    assert routed["fallback_model"] is None
    assert routed["capabilities"] is None
    assert (unrouted["base_url"], unrouted["api_key"]) == ("https://delegation.invalid/v1", "sk-delegation")
    assert unrouted["request_overrides"] == {"service_tier": "flex"}
    assert unrouted["fallback_model"] == delegation_chain


def test_task_fields_cannot_supply_endpoint_key_or_transport(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, {"routes": [ASTRA], "oneshot_max_children": 10})
    harness.call([{
        "goal": "Review with astra", "route": ASTRA, "provider": "openrouter", "model": "attacker/model",
        "base_url": "https://attacker.invalid", "api_key": "sk-attacker", "api_mode": "chat_completions",
        "acp_command": "/bin/sh", "acp_args": ["-c", "id"], "command": "/bin/sh", "args": ["-c", "id"],
    }])

    (child,) = harness.built
    assert (child["provider"], child["model"]) == ("openai-codex", "gpt-6-astra")
    assert (child["base_url"], child["api_key"], child["api_mode"]) == ("https://codex.invalid", "sk-codex", "codex_responses")
    assert child["acp_command"] is None and child["acp_args"] == []


def test_same_provider_route_derives_its_own_wire_instead_of_the_parents(monkeypatch, tmp_path):
    route = "openrouter/qwen/qwen3-coder"
    harness = _Harness(monkeypatch, tmp_path, {"routes": [route], "oneshot_max_children": 10})
    harness.call([{"goal": "Review with qwen", "route": route}])

    (child,) = harness.built
    assert (child["provider"], child["model"]) == ("openrouter", "qwen/qwen3-coder")
    assert child["api_mode"] == "chat_completions"  # the parent runs codex_responses on the same provider


@pytest.mark.parametrize(
    ("delegation_effort", "route", "expected"),
    [
        ("low", OPUS, {"enabled": True, "effort": "xhigh"}),
        ("low", "anthropic/claude-opus-5-5", {"enabled": True, "effort": "low"}),
        ("low", "anthropic/claude-opus-5-5@none", {"enabled": False}),
        ("", "anthropic/claude-opus-5-5", {"enabled": True, "effort": "medium"}),  # the parent's
    ],
)
def test_route_effort_beats_delegation_effort_which_beats_the_parent(
    monkeypatch, tmp_path, delegation_effort, route, expected,
):
    harness = _Harness(monkeypatch, tmp_path, {
        "routes": [route], "reasoning_effort": delegation_effort, "oneshot_max_children": 10,
    })
    harness.call([{"goal": "Review with opus", "route": route}])

    assert harness.built[0]["reasoning_config"] == expected


@pytest.mark.parametrize(
    ("routes", "task_route", "expected"),
    [
        ([OPUS], "anthropic/claude-opus-5-5@max", ["Task 0", OPUS]),
        ([OPUS], None, ["Task 0", "non-empty string"]),
        ([OPUS], "", ["Task 0", "non-empty string"]),
        ([OPUS], 7, ["Task 0", "non-empty string"]),
        ([], OPUS, ["Task 0", "delegation.routes is empty"]),
    ],
)
def test_unusable_route_refuses_the_call_before_any_side_effect(monkeypatch, tmp_path, routes, task_route, expected):
    harness = _Harness(monkeypatch, tmp_path, {"routes": routes, "oneshot_max_children": 10})
    result = harness.call([{"goal": "Review the module", "route": task_route}])

    _assert_nothing_happened(harness, result)
    assert harness.resolved == []
    for text in expected:
        assert text in result["error"]


def test_every_failing_task_is_named_in_one_error(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
    result = harness.call([
        {"goal": "Review with a typo", "route": "anthropic/claude-opus-5-5@max"},
        {"goal": "Review with astra", "route": ASTRA},
        {"goal": "Review with nothing", "route": ""},
    ])

    _assert_nothing_happened(harness, result)
    assert harness.resolved == []
    assert "Task 0" in result["error"] and "Task 2" in result["error"] and "Task 1" not in result["error"]
    assert f"Configured: {OPUS}, {ASTRA}" in result["error"]


def test_resolution_failure_on_a_later_route_builds_nothing(monkeypatch, tmp_path):
    runtimes = {**RUNTIMES, "xai-oauth": RuntimeError("xai-oauth is not logged in")}
    harness = _Harness(monkeypatch, tmp_path, {"routes": [OPUS, GROK], "oneshot_max_children": 10}, runtimes)
    result = harness.call([
        {"goal": "Review with opus", "route": OPUS},
        {"goal": "Review with grok", "route": GROK},
        {"goal": "Review with grok again", "route": GROK},
    ])

    _assert_nothing_happened(harness, result)
    assert harness.resolved == ["anthropic", "xai-oauth"]
    assert result["error"].startswith(f"Task(s) 1, 2: route '{GROK}' (delegation.routes) cannot be used:")
    assert "not logged in" in result["error"]


def test_fully_routed_call_does_not_resolve_the_default_route(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, {
        "routes": [OPUS], "provider": "unconfigured-provider", "oneshot_max_children": 10,
    })
    result = harness.call([{"goal": "Review with opus", "route": OPUS}])

    assert result["results"][0]["status"] == "completed"
    assert harness.resolved == ["anthropic"]


def test_revoked_route_is_refused_on_the_next_call(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
    assert harness.call([{"goal": "Review with opus", "route": OPUS}])["results"][0]["status"] == "completed"

    harness.config["routes"] = [ASTRA]
    harness.built, harness.parent._oneshot_children_spawned = [], 0
    result = harness.call([{"goal": "Review with opus", "route": OPUS}])

    assert harness.built == []
    assert f"route '{OPUS}' is not in delegation.routes" in result["error"]

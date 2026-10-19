"""Per-task routes: every route is resolved before any child exists, and each child runs on its own route."""

import pytest

from tests.tools.delegate_routes_support import ASTRA, GROK, OPUS, RUNTIMES, Harness
from tools.delegation_live_log import live_transcript_root


def _assert_nothing_happened(harness, result):
    assert "error" in result
    assert harness.built == []
    assert harness.parent._oneshot_children_spawned == 0
    root = live_transcript_root()
    assert not root.exists() or not any(root.iterdir())


def test_mixed_batch_builds_each_child_on_its_own_route(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
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
    harness = Harness(monkeypatch, tmp_path, {
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
    harness = Harness(monkeypatch, tmp_path, {"routes": [ASTRA], "oneshot_max_children": 10})
    harness.call([{
        "goal": "Review with astra", "route": ASTRA, "provider": "openrouter", "model": "attacker/model",
        "base_url": "https://attacker.invalid", "api_key": "sk-attacker", "api_mode": "chat_completions",
        "acp_command": "/bin/sh", "acp_args": ["-c", "id"], "command": "/bin/sh", "args": ["-c", "id"],
    }])

    (child,) = harness.built
    assert (child["provider"], child["model"]) == ("openai-codex", "gpt-6-astra")
    assert (child["base_url"], child["api_key"], child["api_mode"]) == ("https://codex.invalid", "sk-codex", "codex_responses")
    assert child["acp_command"] is None and child["acp_args"] == []


@pytest.mark.parametrize(
    ("config", "task"),
    [
        ({"routes": ["openrouter/qwen/qwen3-coder"]}, {"goal": "Review with qwen", "route": "openrouter/qwen/qwen3-coder"}),
        ({"provider": "openrouter", "model": "qwen/qwen3-coder"}, {"goal": "Review with qwen"}),
    ],
    ids=["route", "delegation.provider"],
)
def test_pinned_same_provider_child_derives_its_own_wire_never_the_parents(monkeypatch, tmp_path, config, task):
    # The parent runs codex_responses on openrouter; this runtime names no wire (azure-foundry resolves that way).
    harness = Harness(monkeypatch, tmp_path, {**config, "oneshot_max_children": 10})
    harness.call([task])

    (child,) = harness.built
    assert (child["provider"], child["model"]) == ("openrouter", "qwen/qwen3-coder")
    assert child["api_mode"] is None  # AIAgent derives it from the provider, endpoint and model


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
    harness = Harness(monkeypatch, tmp_path, {
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
    harness = Harness(monkeypatch, tmp_path, {"routes": routes, "oneshot_max_children": 10})
    result = harness.call([{"goal": "Review the module", "route": task_route}])

    _assert_nothing_happened(harness, result)
    assert harness.resolved == []
    for text in expected:
        assert text in result["error"]


def test_every_failing_task_is_named_in_one_error(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
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
    harness = Harness(monkeypatch, tmp_path, {"routes": [OPUS, GROK], "oneshot_max_children": 10}, runtimes)
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
    harness = Harness(monkeypatch, tmp_path, {
        "routes": [OPUS], "provider": "unconfigured-provider", "oneshot_max_children": 10,
    })
    result = harness.call([{"goal": "Review with opus", "route": OPUS}])

    assert result["results"][0]["status"] == "completed"
    assert harness.resolved == ["anthropic"]


def test_revoked_route_is_refused_on_the_next_call(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
    assert harness.call([{"goal": "Review with opus", "route": OPUS}])["results"][0]["status"] == "completed"

    harness.config["routes"] = [ASTRA]
    harness.built, harness.parent._oneshot_children_spawned = [], 0
    result = harness.call([{"goal": "Review with opus", "route": OPUS}])

    assert harness.built == []
    assert f"route '{OPUS}' is not in delegation.routes" in result["error"]

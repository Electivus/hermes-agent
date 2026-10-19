import json

import pytest

from hermes_constants import get_hermes_home
from tools.delegate_tool import DELEGATE_TASK_SCHEMA
from tools.delegate_tool_routes import RouteError, load_route_catalog, parse_route_id
from tools.registry import registry

OPUS = "anthropic/claude-opus-5-5@xhigh"
ASTRA = "openai-codex/gpt-6-astra@xhigh"


def _definition():
    return registry.get_definitions({"delegate_task"})[0]


def _task_properties(definition):
    return definition["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]


@pytest.mark.parametrize(
    ("text", "provider", "model", "effort"),
    [
        (OPUS, "anthropic", "claude-opus-5-5", "xhigh"),
        ("openrouter/anthropic/claude-sonnet-4", "openrouter", "anthropic/claude-sonnet-4", None),
        ("vertex/claude-opus-4@20250514", "vertex", "claude-opus-4@20250514", None),
        ("vertex/claude-opus-4@20250514@high", "vertex", "claude-opus-4@20250514", "high"),
        ("custom:lab/qwen3@none", "custom:lab", "qwen3", "none"),
    ],
)
def test_route_id_splits_provider_at_first_slash_and_effort_at_last_letters_only_suffix(text, provider, model, effort):
    route = parse_route_id(text)
    assert (route.provider, route.model, route.effort) == (provider, model, effort)
    assert route.id == text


@pytest.mark.parametrize(
    "text",
    ["anthropic/claude-opus-5-5@hgih", "no-slash", "/model-only", "anthropic/", "bad provider/m", "a/m@", "a/m @high", 7],
)
def test_malformed_route_ids_are_errors(text):
    with pytest.raises(RouteError):
        parse_route_id(text)


def test_route_field_tracks_catalog_without_mutating_previous_definitions(monkeypatch):
    from tools import delegate_tool

    original = json.dumps(DELEGATE_TASK_SCHEMA)
    snapshots = []
    for routes in ([], [OPUS], [ASTRA, OPUS], None):
        config = {} if routes is None else {"routes": routes}
        monkeypatch.setattr(delegate_tool, "_load_config", lambda: config)
        definition = _definition()
        properties = _task_properties(definition)
        enabled = bool(routes)
        assert ("route" in properties) == enabled
        assert ("`route`" in definition["function"]["description"]) == enabled
        if enabled:
            assert properties["route"]["type"] == "string"
            assert properties["route"]["enum"] == routes
        assert json.dumps(_definition()) == json.dumps(definition)
        for previous, serialized in snapshots:
            assert json.dumps(previous) == serialized
        snapshots.append((definition, json.dumps(definition)))
    assert json.dumps(DELEGATE_TASK_SCHEMA) == original
    assert "route" not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]


def test_malformed_catalog_entries_are_never_advertised(monkeypatch):
    from tools import delegate_tool

    vertex = "vertex/claude-opus-4@20250514"
    routes = [OPUS, "no-slash", "/model-only", "bad provider/m", 7, "anthropic/claude-opus-5-5@hgih", OPUS, vertex]
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"routes": routes})
    assert _task_properties(_definition())["route"]["enum"] == [OPUS, vertex]


def test_non_list_catalog_advertises_nothing():
    assert load_route_catalog({"routes": OPUS}).ids == ()


def test_routes_reach_the_schema_through_the_real_config_loader(monkeypatch):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (get_hermes_home() / "config.yaml").write_text(
        f"delegation:\n  routes:\n    - {ASTRA}\n    - {OPUS}\n", encoding="utf-8",
    )
    assert _task_properties(_definition())["route"]["enum"] == [ASTRA, OPUS]

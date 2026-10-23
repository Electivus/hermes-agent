"""delegation.routes meets delegation.backend: a task that names a catalog route bypasses the backend, the backend
answers for the rest, and a call that names routes is all-or-nothing.

The backend is a real plugin found through PluginManager; children and runtime-provider resolution are stubbed.
"""

from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from tests.tools.delegate_routes_support import OPUS, Harness
from tools import process_registry_notifications as notifications
from tools.process_registry_notifications import format_process_notification
from tools.registry import registry

REPO_PLUGINS = Path(__file__).resolve().parents[2] / "plugins"

BACKEND_PLUGIN = '''
from agent.delegation_provider import DelegationProvider, DelegationRouteError

class Options(DelegationProvider):
    @property
    def name(self):
        return "probe-options"

    def task_parameters(self, config):
        return {"option": {"type": "string", "enum": ["fast", "refused"]},
                "route": {"type": "string", "enum": ["backend-owned"]}}

    def route_task(self, task, config):
        if task.get("option") == "refused":
            raise DelegationRouteError("that option is reserved for humans")
        if task.get("option") == "fast":
            return {"base_url": "http://127.0.0.1:9/v1", "api_key": "k-fast", "model": "fast-model", "label": "fast"}
        return dict(config)

def register(ctx):
    ctx.register_delegation_provider(Options())
'''
CONFIG = {"backend": "probe-options", "routes": [OPUS]}


@pytest.fixture
def backend(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "probe-options"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "probe-options", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(BACKEND_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["probe-options"]}, "delegation": CONFIG}), encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "delegation").symlink_to(REPO_PLUGINS / "delegation", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    manager = PluginManager()
    manager.discover_and_load()
    yield Harness(tmp_path, CONFIG)
    manager.unload()


def test_a_named_route_bypasses_the_backend_which_routes_the_rest(backend):
    result = backend.call([
        {"goal": "Review with opus", "route": OPUS},
        {"goal": "Review on the fast option", "option": "fast"},
        {"goal": "Review on the backend's default route"},
    ])

    assert [r["status"] for r in result["results"]] == ["completed"] * 3
    opus, fast, plain = backend.built
    assert (opus["provider"], opus["model"], opus["reasoning_config"]) == (
        "anthropic", "claude-opus-5-5", {"enabled": True, "effort": "xhigh"})
    assert (fast["model"], fast["base_url"], fast["api_key"]) == ("fast-model", "http://127.0.0.1:9/v1", "k-fast")
    assert plain["model"] == "parent/model"
    assert backend.resolved == ["anthropic"]
    by_task = {r["task_index"]: r for r in result["results"]}
    assert by_task[0]["route"] == OPUS
    assert by_task[1]["route"]["backend"] == "probe-options" and by_task[1]["route"]["label"] == "fast"
    assert by_task[2]["route"]["backend"] == "probe-options" and "label" not in by_task[2]["route"]


def test_route_is_core_owned_so_a_backend_cannot_declare_it(backend):
    from tools import delegate_tool

    properties = registry.get_definitions({"delegate_task"})[0]["function"]["parameters"]["properties"]["tasks"]["items"][
        "properties"]
    assert properties["route"]["enum"] == [OPUS]
    assert properties["option"]["enum"] == ["fast", "refused"]
    assert delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"].keys() == {
        "goal", "context", "output_schema", "images", "group"}


def test_a_call_that_names_routes_is_all_or_nothing_but_a_call_without_routes_refuses_per_task(backend):
    refused = {"goal": "Review on the refused option", "option": "refused"}

    result = backend.call([{"goal": "Review with opus", "route": OPUS}, refused])
    assert result["error"].startswith("Task(s) 1 (default route): Task 1: that option is reserved for humans")
    assert backend.built == [] and backend.parent._oneshot_children_spawned == 0

    result = backend.call([{"goal": "Review on the default route"}, refused])
    statuses = {r["task_index"]: r["status"] for r in result["results"]}
    assert statuses == {0: "completed", 1: "error"}
    assert len(backend.built) == 1


def test_completion_notices_ignore_a_backend_routed_tasks_route_object(monkeypatch):
    route = {"backend": "model_map", "label": "deep", "provider": "anthropic", "model": "claude-opus-5-5"}
    rejected = {"task_index": 0, "status": "failed", "summary": None, "model": "claude-opus-5-5", "route": route,
                "error": "HTTP 400: claude-opus-5-5 is not a valid model ID"}
    monkeypatch.setattr(notifications, "_delegation_config", lambda: {"model": "claude-opus-5-5", "provider": "anthropic"})
    completion = {
        "type": "async_delegation", "delegation_id": "deleg_backend", "is_batch": True, "role": "leaf",
        "model": "claude-opus-5-5", "goals": ["task 0"], "results": [rejected], "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }

    text = format_process_notification(completion)
    assert "TASK 1/1: task 0  (status=failed" in text
    assert "route=" not in text and "ROUTE MODEL REJECTED" not in text and "configured Subagent Model" not in text
    early = format_process_notification({**completion, "task_failure_notice": True, "n_tasks": 2})
    assert "Route:" not in early

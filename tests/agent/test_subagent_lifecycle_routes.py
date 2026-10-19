"""SubagentLifecycleService.routes(): plugins read the operator's delegation.routes, in process or across the host."""

import json
import os

import hermes_yaml as yaml
import pytest

from agent.subagent_lifecycle import SubagentLifecycleService
from hermes_constants import get_hermes_home

OPUS = "anthropic/claude-opus-5-5@xhigh"
ASTRA = "openai-codex/gpt-6-astra@xhigh"


def _write_config(home, config):
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_routes_lists_the_live_catalog_in_operator_order_without_a_parent(monkeypatch):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    service = SubagentLifecycleService(lambda: None)
    assert service.routes() == ()

    _write_config(get_hermes_home(), {"delegation": {"routes": [ASTRA, "no-slash", OPUS, ASTRA]}})
    assert service.routes() == (ASTRA, OPUS)

    _write_config(get_hermes_home(), {"delegation": {"routes": [OPUS]}})
    assert service.routes() == (OPUS,)


ROUTES_PLUGIN = '''
import json, os
def register(ctx):
    ctx.register_tool(name="host_routes", toolset="hostroutes",
        schema={"name": "host_routes", "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: json.dumps({"pid": os.getpid(), "routes": list(ctx.subagent_lifecycle.routes())}))
'''


@pytest.mark.platforms("any")
def test_host_isolated_plugin_reads_routes_through_the_remote_facade(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager
    from tests.hermes_cli.test_plugin_host import _home_with_plugins
    from tools.registry import registry

    home = _home_with_plugins(tmp_path, monkeypatch, {"hostroutes": ROUTES_PLUGIN})
    _write_config(home, {"plugins": {"enabled": ["hostroutes"], "isolation": "host"},
                         "delegation": {"routes": [OPUS, ASTRA]}})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        result = json.loads(registry.dispatch("host_routes", {}, scope=manager.scope_key))
    finally:
        manager.unload()
        manager._plugin_host().shutdown()

    assert result["pid"] != os.getpid()
    assert result["routes"] == [OPUS, ASTRA]

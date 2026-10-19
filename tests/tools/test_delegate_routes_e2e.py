import json
from unittest.mock import patch

from hermes_constants import get_hermes_home
from tools import delegate_tool
from tools.registry import registry
from tests.tools.delegate_routes_support import child_agent, parent_agent

LAB_A = "custom:lab-a/model-a@high"
LAB_B = "custom:lab-b/model-b"
CONFIG = f"""\
custom_providers:
  - name: lab-a
    base_url: https://lab-a.invalid/v1
    api_key: sk-lab-a
  - name: lab-b
    base_url: https://lab-b.invalid/v1
    api_key: sk-lab-b
delegation:
  base_url: https://delegation.invalid/v1
  api_key: sk-delegation
  max_spawn_depth: 3
  oneshot_max_children: 10
  routes:
    - {LAB_A}
    - {LAB_B}
"""


def _run_routed(tasks, parent, tmp_path):
    with (
        patch.object(delegate_tool, "_resolve_workspace_hint", return_value=str(tmp_path)),
        patch("run_agent.AIAgent", side_effect=child_agent) as agent_class,
    ):
        result = json.loads(registry.get_entry("delegate_task").handler({"tasks": tasks}, parent_agent=parent))
    assert [r["status"] for r in result["results"]] == ["completed"] * len(tasks)
    return [c.kwargs for c in agent_class.call_args_list]


def test_each_route_gets_its_own_providers_endpoint_and_key_never_the_delegation_url(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (get_hermes_home() / "config.yaml").write_text(CONFIG, encoding="utf-8")
    tasks = [
        {"goal": "Review on lab a", "route": LAB_A},
        {"goal": "Review on lab b", "route": LAB_B},
        {"goal": "Review on the default route"},
    ]

    lab_a, lab_b, unrouted = _run_routed(tasks, parent_agent(), tmp_path)

    assert (lab_a["base_url"], lab_a["api_key"], lab_a["model"]) == ("https://lab-a.invalid/v1", "sk-lab-a", "model-a")
    assert lab_a["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert (lab_b["base_url"], lab_b["api_key"], lab_b["model"]) == ("https://lab-b.invalid/v1", "sk-lab-b", "model-b")
    assert lab_a["requested_provider"] == "custom:lab-a" and lab_b["requested_provider"] == "custom:lab-b"
    assert (unrouted["base_url"], unrouted["api_key"]) == ("https://delegation.invalid/v1", "sk-delegation")


def test_openrouter_route_never_takes_the_parents_persisted_wire(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    (get_hermes_home() / "config.yaml").write_text("""\
model:
  provider: anthropic
  default: claude-opus-5-5
  api_mode: anthropic_messages
delegation:
  max_spawn_depth: 3
  routes:
    - openrouter/qwen/qwen3-coder
""", encoding="utf-8")
    parent = parent_agent(provider="anthropic", requested_provider="anthropic", api_mode="anthropic_messages")

    (child,) = _run_routed([{"goal": "Review with qwen", "route": "openrouter/qwen/qwen3-coder"}], parent, tmp_path)

    assert (child["provider"], child["model"], child["base_url"]) == (
        "openrouter", "qwen/qwen3-coder", "https://openrouter.ai/api/v1")
    assert child["api_mode"] == "chat_completions"


def test_named_route_resolves_its_provider_under_an_auto_config_with_a_local_endpoint(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    (get_hermes_home() / "config.yaml").write_text("""\
model:
  provider: auto
  base_url: http://localhost:11434
  default: llama3
delegation:
  max_spawn_depth: 3
  routes:
    - anthropic/claude-opus-5-5
""", encoding="utf-8")

    (child,) = _run_routed([{"goal": "Review with opus", "route": "anthropic/claude-opus-5-5"}], parent_agent(), tmp_path)

    assert (child["provider"], child["model"]) == ("anthropic", "claude-opus-5-5")
    assert child["base_url"].rstrip("/") == "https://api.anthropic.com"

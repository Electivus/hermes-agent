import json
import threading
from unittest.mock import MagicMock, patch

from tools import delegate_tool
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
    "openrouter": {"provider": "openrouter", "api_mode": None, "base_url": "https://openrouter.invalid/v1",
                   "api_key": "sk-openrouter"},
}
PARENT_CHAIN = [{"provider": "deepseek", "model": "deepseek-chat"}]


def parent_agent():
    parent = MagicMock()
    parent._delegate_depth = 1
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


def child_agent(**kwargs):
    child = MagicMock()
    child._credential_pool = None
    child._delegate_saved_tool_names = []
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = kwargs["model"]
    child.provider = kwargs["provider"]
    child.run_conversation.return_value = {"final_response": "done", "completed": True, "api_calls": 1, "messages": []}
    return child


class Harness:
    def __init__(self, monkeypatch, tmp_path, config, runtimes=RUNTIMES):
        self.config = {"max_spawn_depth": 3, **config}
        self.resolved: list[str] = []
        self.parent = parent_agent()
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
            patch("run_agent.AIAgent", side_effect=child_agent) as agent_class,
        ):
            result = json.loads(registry.get_entry("delegate_task").handler({"tasks": tasks}, parent_agent=self.parent))
        self.built = [c.kwargs for c in agent_class.call_args_list]
        return result

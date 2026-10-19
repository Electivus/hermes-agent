"""The persisted ``model`` block belongs to the provider it names: a request for another provider never takes its
api_mode. Real resolver against a temp HERMES_HOME config."""

import pytest

from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home


def _write_config(monkeypatch, text):
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (get_hermes_home() / "config.yaml").write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("model_block", "requested", "expected"),
    [
        ("provider: anthropic", "openrouter", ("openrouter", "https://openrouter.ai/api/v1", "chat_completions")),
        ("provider: openrouter", "openrouter", ("openrouter", "https://openrouter.ai/api/v1", "anthropic_messages")),
        ("provider: auto\n  base_url: https://proxy.example.test/v1", None,
         ("openrouter", "https://proxy.example.test/v1", "anthropic_messages")),
    ],
    ids=["another-providers-mode", "openrouters-own-mode", "the-configs-own-endpoint"],
)
def test_persisted_api_mode_applies_only_where_the_config_recorded_it(monkeypatch, model_block, requested, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    _write_config(monkeypatch, f"model:\n  {model_block}\n  default: some-model\n  api_mode: anthropic_messages\n")

    runtime = resolve_runtime_provider(requested=requested, target_model="qwen/qwen3-coder")

    assert (runtime["provider"], runtime["base_url"], runtime["api_mode"]) == expected

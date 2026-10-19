"""The async completion header names the model the children actually run on."""

from dataclasses import fields
from types import SimpleNamespace

from tools import async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_unit


def _unit(*models):
    batch = _Batch(**{field.name: None for field in fields(_Batch)})
    tasks = [{"goal": f"probe {i}"} for i in range(len(models))]
    batch.task_list, batch.live_writers, batch.top_role = tasks, [], "leaf"
    batch.children = [(i, task, SimpleNamespace(model=model)) for i, (task, model) in enumerate(zip(tasks, models))]
    return batch


def test_completion_model_is_the_model_the_children_run_on(monkeypatch):
    seen = []
    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", lambda **kw: seen.append(kw["model"]))
    _dispatch_unit(_unit("inherited/model"), None, None, {})  # delegation.model blank: children inherit the parent's
    _dispatch_unit(_unit("configured/model", "configured/model"), None, None, {})
    assert seen == ["inherited/model", "configured/model"]

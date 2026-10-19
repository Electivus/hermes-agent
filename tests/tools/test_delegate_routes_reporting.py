"""A batch whose children run different models reports each child's own model and route."""

import json
import threading
from dataclasses import fields
from types import SimpleNamespace

from tools import async_delegation, delegate_tool
from tools import process_registry_notifications as notifications
from tools.delegate_tool_child_run import _fabricated_entry
from tools.delegate_tool_dispatch import _Batch, _dispatch_unit
from tools.delegation_live_log import live_transcript_root
from tools.process_registry_notifications import format_process_notification
from tests.tools.delegate_routes_support import ASTRA, OPUS, Harness

REJECTED = "HTTP 400: claude-opus-5-5 is not a valid model ID"


def _completion(model, results):
    return {
        "type": "async_delegation", "delegation_id": "deleg_mixed", "is_batch": True, "role": "leaf", "model": model,
        "goals": [f"task {r['task_index']}" for r in results], "results": results,
        "dispatched_at": 1000.0, "completed_at": 1012.0,
    }


def _result(index, model, **extra):
    return {"task_index": index, "status": "completed", "summary": f"done {index}", "model": model, **extra}


def _unit_model(monkeypatch, *models):
    seen = []
    batch = _Batch(**{field.name: None for field in fields(_Batch)})
    tasks = [{"goal": f"probe {i}"} for i in range(len(models))]
    batch.task_list, batch.live_writers, batch.top_role = tasks, [], "leaf"
    batch.children = [(i, task, SimpleNamespace(model=model)) for i, (task, model) in enumerate(zip(tasks, models))]
    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", lambda **kw: seen.append(kw["model"]))
    _dispatch_unit(batch, None, None, {})
    return seen[0]


def test_mixed_unit_completion_says_mixed_and_names_each_childs_model(monkeypatch):
    assert _unit_model(monkeypatch, "claude-opus-5-5", "gpt-6-astra") == "mixed"
    assert _unit_model(monkeypatch, "claude-opus-5-5", "claude-opus-5-5") == "claude-opus-5-5"

    mixed = format_process_notification(_completion("mixed", [
        _result(0, "claude-opus-5-5", route=OPUS), _result(1, "gpt-6-astra", route=ASTRA),
    ]))
    assert "Model: mixed" in mixed
    assert "TASK 1/2: task 0  (status=completed, model=claude-opus-5-5" in mixed
    assert "TASK 2/2: task 1  (status=completed, model=gpt-6-astra" in mixed

    uniform = format_process_notification(_completion("claude-opus-5-5", [
        _result(0, "claude-opus-5-5"), _result(1, "claude-opus-5-5"),
    ]))
    assert "Model: claude-opus-5-5" in uniform
    assert "model=" not in uniform


def test_background_completion_names_each_tasks_route_and_any_differing_model():
    high, xhigh = "anthropic/claude-opus-5-5@high", "anthropic/claude-opus-5-5@xhigh"
    same_model = format_process_notification(_completion("claude-opus-5-5", [
        _result(0, "claude-opus-5-5", route=high), _result(1, "claude-opus-5-5", route=xhigh),
    ]))
    assert f"TASK 1/2: task 0  (status=completed, route={high}" in same_model
    assert f"TASK 2/2: task 1  (status=completed, route={xhigh}" in same_model

    # The record was written before a child's fallback moved it to another model; the results say what ran.
    fell_back = format_process_notification(_completion("claude-opus-5-5", [
        _result(0, "claude-opus-5-5"), _result(1, "deepseek-chat"),
    ]))
    assert "TASK 1/2: task 0  (status=completed, model=claude-opus-5-5" in fell_back
    assert "TASK 2/2: task 1  (status=completed, model=deepseek-chat" in fell_back


def test_early_task_failure_notice_names_the_childs_model_and_route():
    failed = _result(1, "gpt-6-astra", route=ASTRA, status="error", summary=None, error="boom")
    text = format_process_notification({**_completion("mixed", [failed]), "task_failure_notice": True, "n_tasks": 2})
    assert "task 2/2" in text
    assert "Model: gpt-6-astra" in text and f"Route: {ASTRA}" in text


def test_fabricated_entries_carry_model_and_route():
    routed = SimpleNamespace(model="claude-opus-5-5", _delegate_route=OPUS, _delegate_role="leaf")
    unrouted = SimpleNamespace(model="parent/model", _delegate_route=None, _delegate_role="leaf")

    entry = _fabricated_entry(0, "error", "boom", routed)
    assert (entry["model"], entry["route"]) == ("claude-opus-5-5", OPUS)
    entry = _fabricated_entry(1, "interrupted", "stopped", unrouted)
    assert entry["model"] == "parent/model" and "route" not in entry


class _HangingChild:
    def __init__(self):
        self.tool_progress_callback = None
        self._credential_pool = None
        self._delegate_saved_tool_names = []
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = None
        self._delegate_route = ASTRA
        self.model = "gpt-6-astra"
        self.session_prompt_tokens = self.session_completion_tokens = 0
        self.session_estimated_cost_usd, self.session_cost_status = 0.0, "unknown"
        self.started, self.interrupted = threading.Event(), threading.Event()

    def run_conversation(self, **_kwargs):
        self.started.set()
        self.interrupted.wait(timeout=30)
        return {"final_response": "", "completed": False, "interrupted": True, "api_calls": 0, "messages": []}

    def hard_interrupt(self, _reason=None):
        self.interrupted.set()

    def get_activity_summary(self):
        return {"api_call_count": 0}

    def close(self):
        self.interrupted.set()


def test_timeout_entry_carries_model_and_route(monkeypatch):
    from tools.daemon_pool import DaemonThreadPoolExecutor

    child = _HangingChild()
    parent = SimpleNamespace(session_id="parent", _current_task_id=None, _active_children=[child],
                             _active_children_lock=threading.Lock())
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.5)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    submit = DaemonThreadPoolExecutor.submit

    def submit_started(executor, *args, **kwargs):
        future = submit(executor, *args, **kwargs)
        assert child.started.wait(timeout=10)
        return future

    monkeypatch.setattr(DaemonThreadPoolExecutor, "submit", submit_started)
    entry = delegate_tool._run_single_child(task_index=0, goal="hang until timed out", child=child, parent_agent=parent)

    assert entry["status"] == "timeout"
    assert (entry["model"], entry["route"]) == ("gpt-6-astra", ASTRA)


def test_routed_results_progress_and_manifest_name_each_childs_own_route(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path, {"routes": [OPUS, ASTRA], "oneshot_max_children": 10})
    events = []
    harness.parent.tool_progress_callback = lambda event, *_a, **kw: events.append((event, kw))
    result = harness.call([
        {"goal": "Review with opus", "route": OPUS},
        {"goal": "Review with astra", "route": ASTRA},
        {"goal": "Review on the default model"},
    ])

    entries = result["results"]
    assert [(e["model"], e.get("route")) for e in entries] == [
        ("claude-opus-5-5", OPUS), ("gpt-6-astra", ASTRA), ("parent/model", None),
    ]
    spawned = {kw["task_index"]: kw["model"] for event, kw in events if event == "subagent.spawn_requested"}
    assert spawned == {0: "claude-opus-5-5", 1: "gpt-6-astra", 2: "parent/model"}
    (manifest_path,) = live_transcript_root().glob("*/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [(t["model"], t["provider"], t["route"]) for t in manifest["tasks"]] == [
        ("claude-opus-5-5", "anthropic", OPUS), ("gpt-6-astra", "openai-codex", ASTRA),
        ("parent/model", "openrouter", None),
    ]
    assert manifest["model"] is None and manifest["provider"] is None


def test_route_model_rejection_is_attributed_to_the_route_not_delegation_model(monkeypatch):
    rejected = _result(0, "claude-opus-5-5", route=OPUS, status="failed", summary=None, error=REJECTED)
    config = {"model": "claude-opus-5-5", "provider": "anthropic", "routes": [OPUS]}
    monkeypatch.setattr(notifications, "_delegation_config", lambda: config)

    text = format_process_notification(_completion("claude-opus-5-5", [rejected]))
    assert f'SUBAGENT ROUTE MODEL REJECTED: route "{OPUS}" (delegation.routes)' in text
    assert "configured Subagent Model" not in text

    unrouted = _result(1, "claude-opus-5-5", status="failed", summary=None, error=REJECTED)
    assert "configured Subagent Model" in format_process_notification(_completion("claude-opus-5-5", [unrouted]))


def test_route_rejection_notice_names_the_model_the_child_sent_whatever_the_live_catalog_says(monkeypatch):
    route = "anthropic/claude-sonnet-4.5"  # the provider normalizes it to claude-sonnet-4-5 before sending
    rejected = _result(0, "claude-sonnet-4-5", route=route, status="failed", summary=None,
                       error="HTTP 400: claude-sonnet-4-5 is not a valid model ID")
    monkeypatch.setattr(notifications, "_delegation_config", lambda: {"routes": []})  # revoked since dispatch

    text = format_process_notification(_completion("claude-sonnet-4-5", [rejected]))
    assert (f'SUBAGENT ROUTE MODEL REJECTED: route "{route}" (delegation.routes): provider "anthropic" rejected '
            'model "claude-sonnet-4-5"') in text


def test_delegation_model_notice_claims_the_whole_batch_only_when_every_task_was_rejected(monkeypatch):
    monkeypatch.setattr(notifications, "_delegation_config", lambda: {"model": "claude-opus-5-5", "provider": "anthropic"})
    rejected = _result(1, "claude-opus-5-5", status="failed", summary=None, error=REJECTED)

    mixed = format_process_notification(_completion("mixed", [_result(0, "gpt-6-astra", route=ASTRA), rejected]))
    assert "configured Subagent Model" in mixed
    assert "Every task" not in mixed
    assert "TASK 2 failed for this reason before doing any work." in mixed

    whole = format_process_notification(_completion("claude-opus-5-5", [{**rejected, "task_index": 0}, rejected]))
    assert "Every task in this batch failed for this reason before doing any work." in whole

"""The operator's route catalog for delegate_task: ``delegation.routes`` ids, the gated per-task ``route`` field, and
the all-or-nothing planning of a call whose tasks name routes.

A catalog route is the operator's choice of provider, model and reasoning effort for one task; the delegation backend
(``delegation.backend``, ``tools/delegate_tool_routes.py``) answers for every task that names none.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Optional, Sequence

from agent.reasoning_effort import EFFORT_LADDER
from tools.delegate_tool_config import _warn_once
from tools.delegate_tool_routes import TaskRoute, resolve_task_routes

_PROVIDER_RE = re.compile(r"[A-Za-z0-9_.:-]+")
_LETTERS_RE = re.compile(r"[A-Za-z]+")


class RouteError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Route:
    provider: str
    model: str
    effort: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}" + (f"@{self.effort}" if self.effort else "")

    def routing_block(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}


def parse_route_id(text: Any) -> Route:
    """Parse ``provider/model[@effort]``; ``Route.id`` round-trips to ``text``.

    The provider ends at the first ``/`` (models may contain ``/``). A suffix after the last ``@`` is the effort only
    when it is letters-only; a letters-only suffix outside the effort vocabulary is an error, so a typo never becomes
    part of a model id, while a suffix with digits (``claude-opus-4@20250514``) stays in the model.
    """
    if not isinstance(text, str) or not text or any(ch.isspace() for ch in text):
        raise RouteError("expected a provider/model[@effort] string without whitespace")
    provider, slash, rest = text.partition("/")
    if not slash or not _PROVIDER_RE.fullmatch(provider):
        raise RouteError("expected provider/model[@effort]")
    head, at, suffix = rest.rpartition("@")
    model, effort = rest, None
    if at and _LETTERS_RE.fullmatch(suffix):
        if suffix not in EFFORT_LADDER:
            raise RouteError(f"unknown effort {suffix!r} (one of {', '.join(EFFORT_LADDER)})")
        model, effort = head, suffix
    elif at and not suffix:
        raise RouteError("empty effort after '@'")
    if not model:
        raise RouteError("empty model")
    return Route(provider, model, effort)


@dataclasses.dataclass(frozen=True)
class RouteCatalog:
    routes: Mapping[str, Route] = dataclasses.field(default_factory=dict)
    malformed: int = 0

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.routes)


def load_route_catalog(delegation_cfg: Mapping[str, Any]) -> RouteCatalog:
    raw = delegation_cfg.get("routes")
    if not isinstance(raw, list):
        if raw is None:
            return RouteCatalog()
        _warn_once(f"routes:{raw!r}", "delegation.routes must be a list of provider/model[@effort] ids; got %r", raw)
        return RouteCatalog(malformed=1)
    routes: dict[str, Route] = {}
    malformed = 0
    for entry in raw:
        try:
            route = parse_route_id(entry)
        except RouteError as exc:
            _warn_once(f"entry:{entry!r}", "Ignoring delegation.routes entry %r: %s", entry, exc)
            malformed += 1
            continue
        routes.setdefault(route.id, route)
    return RouteCatalog(routes, malformed)


@dataclasses.dataclass
class CatalogRoute(TaskRoute):
    """A task routed by name through ``delegation.routes``. Its child and result carry the route id, a string; a
    backend-routed child carries ``TaskRoute.report()``'s dict instead."""

    route_id: str = ""

    def report(self) -> Optional[str]:  # ty: ignore[invalid-method-override]
        return self.route_id


def resolve_catalog_route(route: Route, parent_agent: Any) -> CatalogRoute:
    """Credentials from the provider's own configuration, never from ``delegation.base_url`` / ``api_key`` / ``api_mode``
    / ``request_overrides`` / ``fallback_providers``: the routing block names only provider and model."""
    from tools.delegate_tool import _resolve_delegation_credentials  # the facade is the patch seam
    block = route.routing_block()
    creds = _resolve_delegation_credentials(block, parent_agent, owner="delegation.routes")
    return CatalogRoute(creds, block, route.effort, route_id=route.id)


def _route_problem(index: int, raw: Any, catalog: RouteCatalog) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return f"Task {index} 'route' must be a non-empty string naming one of delegation.routes."
    if not catalog.routes:
        state = "has no valid entries" if catalog.malformed else "is empty"
        return (f"Task {index} names route {raw!r}, but delegation.routes {state}. Omit 'route' to use the "
                "default subagent model.")
    if raw not in catalog.routes:
        return f"Task {index} route {raw!r} is not in delegation.routes."
    return None


def _failure_text(route_id: Optional[str], error: str, indexes: Sequence[int]) -> str:
    tasks = ", ".join(map(str, indexes))
    if route_id is None:
        return f"Task(s) {tasks} (default route): {error}"
    return f"Task(s) {tasks}: route {route_id!r} (delegation.routes) cannot be used: {error}"


def plan_task_routes(
    task_list: list[dict[str, Any]], cfg: Mapping[str, Any], parent_agent: Any,
) -> tuple[list[TaskRoute], Optional[str]]:
    """One route per task. A call that names no ``route`` is the delegation backend's alone, task by task. A call
    that names routes is all-or-nothing: every ``route`` is checked against ``delegation.routes``, then each distinct
    route is resolved (in task order, the backend answering for the unrouted tasks), before anything is charged or
    built; any failure refuses the call with one error naming every failing task."""
    named = {i: task["route"] for i, task in enumerate(task_list) if "route" in task}
    if not named:
        return resolve_task_routes(task_list, cfg, parent_agent)
    catalog = load_route_catalog(cfg)
    problems = [problem for i, raw in named.items() if (problem := _route_problem(i, raw, catalog))]
    if problems:
        configured = f" Configured: {', '.join(catalog.ids)}." if catalog.routes else ""
        return [], " ".join(problems) + configured
    resolved: dict[str, TaskRoute] = {}

    def claim(index: int, _task: Mapping[str, Any]) -> Optional[TaskRoute]:
        route_id = named.get(index)
        if route_id is None:
            return None
        if route_id not in resolved:
            try:
                resolved[route_id] = resolve_catalog_route(catalog.routes[route_id], parent_agent)
            except ValueError as exc:
                resolved[route_id] = TaskRoute({}, {}, error=str(exc))
        return resolved[route_id]

    routes, error = resolve_task_routes(task_list, cfg, parent_agent, claim=claim)
    if error is not None:
        return [], error
    failures: dict[tuple[Optional[str], str], list[int]] = {}
    for i, route in enumerate(routes):
        if route.error:
            failures.setdefault((named.get(i), route.error), []).append(i)
    if failures:
        return [], " ".join(
            _failure_text(route_id, message, indexes) for (route_id, message), indexes in failures.items()
        )
    return routes, None


ROUTE_DESCRIPTION = (
    "Optional. Run this child on one of the operator's routes (provider/model@effort, set in delegation.routes). "
    "Omit to use the default subagent model. Each result reports the model the child ran on."
)

DESCRIPTION_TAIL_WITH_ROUTES = (
    "- Children inherit the parent model unless a task names a `route`, or delegation.provider / delegation.model in "
    "config.yaml pins them."
)


def route_schema_property(catalog: RouteCatalog) -> Optional[dict[str, Any]]:
    if not catalog.routes:
        return None
    return {"type": "string", "enum": list(catalog.ids), "description": ROUTE_DESCRIPTION}

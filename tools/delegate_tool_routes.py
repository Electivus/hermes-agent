"""Operator-owned routes for ``delegate_task``.

A route is one ``(provider, model, effort?)`` triple the operator lists in ``delegation.routes`` as the canonical id
``provider/model[@effort]``. A task may name one with ``route``; that child then runs on that provider, model and
reasoning effort. The model can only pick an id the operator listed: the schema advertises them as an enum, and
dispatch re-reads the live list, so a removed route stops working on the next call.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from hermes_constants import VALID_REASONING_EFFORTS, parse_reasoning_effort

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

ROUTE_EFFORTS = frozenset(VALID_REASONING_EFFORTS) | {"none"}
_PROVIDER_RE = re.compile(r"[A-Za-z0-9_.:-]+")
_LETTERS_RE = re.compile(r"[A-Za-z]+")
# Malformed entries are re-read on every schema rebuild; warn about each one once per process.
_WARNED_ENTRIES: set[str] = set()


class RouteError(ValueError):
    """A route id that does not parse as ``provider/model[@effort]``."""


@dataclasses.dataclass(frozen=True)
class Route:
    """One operator-listed route. It has no field for an endpoint, key, api_mode, command or request overrides: those
    come from the provider's own configuration when the route resolves."""

    provider: str
    model: str
    effort: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}" + (f"@{self.effort}" if self.effort else "")

    def routing_block(self) -> dict[str, str]:
        """The config block this route resolves from and owns its recovery policy: provider and model only, so the
        child gets no ``delegation.base_url``/``api_key``/``request_overrides`` and no fallback chain."""
        return {"provider": self.provider, "model": self.model}

    def reasoning_config(self) -> Optional[dict[str, Any]]:
        return parse_reasoning_effort(self.effort) if self.effort else None


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
        if suffix not in ROUTE_EFFORTS:
            raise RouteError(f"unknown effort {suffix!r} (one of {', '.join(sorted(ROUTE_EFFORTS))})")
        model, effort = head, suffix
    elif at and not suffix:
        raise RouteError("empty effort after '@'")
    if not model:
        raise RouteError("empty model")
    return Route(provider, model, effort)


@dataclasses.dataclass(frozen=True)
class RouteCatalog:
    """The parsed ``delegation.routes``: canonical id -> Route, in operator order, first duplicate wins."""

    routes: Mapping[str, Route] = dataclasses.field(default_factory=dict)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.routes)

    def get(self, route_id: str) -> Optional[Route]:
        return self.routes.get(route_id)


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _WARNED_ENTRIES:
        _WARNED_ENTRIES.add(key)
        logger.warning(message, *args)


def load_route_catalog(delegation_cfg: Optional[Mapping[str, Any]] = None) -> RouteCatalog:
    """Parse ``delegation.routes`` (``delegation_cfg`` None reads the live delegation config).

    Tolerant: a non-list value is an empty catalog and a malformed entry is dropped with a warning, never advertised.
    Pure and cheap, because it runs on every schema rebuild.
    """
    if delegation_cfg is None:
        from tools.delegate_tool_config import _cfg
        delegation_cfg = _cfg()
    raw = delegation_cfg.get("routes")
    if not isinstance(raw, list):
        if raw is not None:
            _warn_once(f"routes:{raw!r}", "delegation.routes must be a list of provider/model[@effort] ids; got %r", raw)
        return RouteCatalog()
    routes: dict[str, Route] = {}
    for entry in raw:
        try:
            route = parse_route_id(entry)
        except RouteError as exc:
            _warn_once(f"entry:{entry!r}", "Ignoring delegation.routes entry %r: %s", entry, exc)
            continue
        routes.setdefault(route.id, route)
    return RouteCatalog(routes)


@dataclasses.dataclass(frozen=True)
class ChildRoute:
    """Everything one child is built on. Tasks of one call that name the same route share one ChildRoute."""

    creds: Mapping[str, Any]  # the _resolve_delegation_credentials bundle
    routing_cfg: Mapping[str, Any]  # the config block that owns this route's fallback policy
    route_id: Optional[str] = None  # None = the call's default route (delegation.* or an internal credentials_cfg)
    reasoning_config: Optional[dict[str, Any]] = None  # None = delegation.reasoning_effort, else the parent's

    @property
    def model(self) -> Optional[str]:
        return self.creds.get("model")

    @property
    def provider(self) -> Optional[str]:
        return self.creds.get("provider")

    def build_kwargs(self) -> dict[str, Any]:
        """The ``_build_child_agent`` kwargs for this route; the one translation from a bundle into a child."""
        creds = self.creds
        return {
            "model": creds["model"], "override_provider": creds["provider"], "override_base_url": creds["base_url"],
            "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
            "override_request_overrides": creds.get("request_overrides"),
            "override_acp_command": creds.get("command"), "override_acp_args": creds.get("args"),
            "routing_cfg": self.routing_cfg, "override_reasoning_config": self.reasoning_config,
        }

    def identity(self, parent_agent: Any) -> dict[str, Optional[str]]:
        """``{model, provider, route}`` as the child will run, for records written before the child exists."""
        model = self.model or getattr(parent_agent, "model", None)
        provider = self.provider or getattr(parent_agent, "provider", None)
        return {
            "model": model if isinstance(model, str) else None,
            "provider": provider if isinstance(provider, str) else None,
            "route": self.route_id,
        }


MIXED_MODEL_LABEL = "mixed"


def unit_model_label(models: Iterable[Optional[str]]) -> Optional[str]:
    """The model a completion unit reports: the children's shared model, ``"mixed"``, or None when none is known."""
    distinct = set(filter(None, models))
    if len(distinct) > 1:
        return MIXED_MODEL_LABEL
    return next(iter(distinct), None)


def default_child_route(routing_cfg: Mapping[str, Any], parent_agent: Any) -> ChildRoute:
    """The route a config block names (``delegation.*``, ``auxiliary.review``, a lifecycle provider pin). Raises
    ValueError with a user-facing message."""
    from tools.delegate_tool import _resolve_delegation_credentials
    return ChildRoute(_resolve_delegation_credentials(routing_cfg, parent_agent), routing_cfg)


def resolve_route(route: Route, parent_agent: Any) -> ChildRoute:
    """Resolve one catalog route from its provider and model alone, on the caller's thread (never inside the
    child-construction lock: resolution reads auth state and may refresh a token over the network)."""
    from tools.delegate_tool import _resolve_delegation_credentials
    block = route.routing_block()
    return ChildRoute(_resolve_delegation_credentials(block, parent_agent), block, route.id, route.reasoning_config())


def _route_problem(index: int, raw: Any, catalog: RouteCatalog) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return f"Task {index} 'route' must be a non-empty string naming one of delegation.routes."
    if not catalog.routes:
        return (f"Task {index} names route {raw!r}, but delegation.routes is empty. Omit 'route' to use the "
                "default subagent model.")
    if catalog.get(raw) is None:
        return f"Task {index} route {raw!r} is not in delegation.routes."
    return None


def plan_child_routes(
    task_list: Sequence[Mapping[str, Any]], *, routing_cfg: Mapping[str, Any], parent_agent: Any,
) -> tuple[list[ChildRoute], Optional[str]]:
    """One ChildRoute per task, or ``([], error)`` naming every failing task.

    Every task's route is checked against the live catalog before any resolution, then each distinct route is
    resolved once in first-appearance order. The default route resolves only when some task omits ``route``, so a
    fully routed call does not depend on ``delegation.provider`` being usable.
    """
    catalog = load_route_catalog()
    routed = {i: task["route"] for i, task in enumerate(task_list) if "route" in task}
    problems = [problem for i, raw in routed.items() if (problem := _route_problem(i, raw, catalog))]
    if problems:
        configured = f" Configured: {', '.join(catalog.ids)}." if catalog.routes else ""
        return [], " ".join(problems) + configured
    keys = [routed.get(i) for i in range(len(task_list))]
    resolved: dict[Optional[str], ChildRoute] = {}
    for key in dict.fromkeys(keys):
        try:
            resolved[key] = (default_child_route(routing_cfg, parent_agent) if key is None
                             else resolve_route(catalog.routes[key], parent_agent))
        except ValueError as exc:
            if key is None:
                return [], str(exc)
            indexes = ", ".join(str(i) for i, k in enumerate(keys) if k == key)
            return [], f"Task(s) {indexes}: route {key!r} (delegation.routes) cannot be used: {exc}"
    return [resolved[key] for key in keys], None


ROUTE_DESCRIPTION = (
    "Optional. Run this child on one of the operator's routes (provider/model@effort, set in delegation.routes). "
    "Omit to use the default subagent model. Each result reports the model the child ran on."
)

DESCRIPTION_TAIL_WITH_ROUTES = (
    "- Children inherit the parent model unless a task names a `route`, or delegation.provider / delegation.model in "
    "config.yaml pins them."
)


def route_schema_property(catalog: RouteCatalog) -> Optional[dict[str, Any]]:
    """The ``tasks.items.properties.route`` fragment, or None when the catalog is empty (no field, zero tokens). Enum
    order is the operator's, so the bytes are a pure function of config.yaml."""
    if not catalog.routes:
        return None
    return {"type": "string", "enum": list(catalog.ids), "description": ROUTE_DESCRIPTION}

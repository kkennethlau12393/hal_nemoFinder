"""Plugin discovery and loading for hal-nemoFinder.

This module is the single entry point for loading customer-provided
extensions (verifiers, knowledge-base clients, aggregators, claim
extractors, ...) into the running framework **without modifying any
framework code**.

Three discovery mechanisms are supported, in this order of increasing
ad-hoc-ness:

1. **Python entry points** (recommended for distributable plugins)

   Ship your plugin as a regular Python package and declare an entry
   point in your ``pyproject.toml``::

       [project.entry-points."hal_nemofinder.verifiers"]
       my_verifier = "mycompany.verifiers:MyVerifier"

   After ``pip install my-company-hal-plugin`` the framework will
   auto-discover and import the module at startup, triggering any
   ``@register_verifier`` decorators.

2. **Module-path config** (recommended for in-repo extensions)

   List the dotted module paths in ``Settings.PLUGIN_MODULES``::

       PLUGIN_MODULES = ["mycompany.verifiers", "mycompany.kb"]

   Each module is imported; any top-level ``@register_verifier`` calls
   run during import.

3. **File-path config** (recommended for quick experiments / notebooks)

   Point ``Settings.PLUGIN_FILES`` at absolute paths to ``.py`` files::

       PLUGIN_FILES = ["/opt/hal/plugins/custom_verifier.py"]

   Files are loaded via :func:`importlib.util.spec_from_file_location`
   and executed top-to-bottom.

All three mechanisms cooperate: entry points are loaded first, then
modules, then files.  Any errors are captured in the
:class:`PluginLoadReport` and logged prominently — they never crash
framework startup.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from src.verifiers.base import BaseVerifier, get_verifier_registry

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class PluginLoadReport:
    """Summary of a plugin-discovery pass.

    Attributes
    ----------
    verifiers_loaded : list[str]
        Names of verifiers newly registered during this load pass.
    kb_clients_loaded : list[str]
        Names of knowledge-base clients newly registered during this
        load pass.
    aggregators_loaded : list[str]
        Names of aggregator plugins loaded.
    extractors_loaded : list[str]
        Names of claim extractor plugins loaded.
    modules_imported : list[str]
        Dotted module paths successfully imported (for diagnostics).
    files_loaded : list[str]
        Absolute file paths successfully loaded (for diagnostics).
    errors : list[tuple[str, str]]
        ``(source, message)`` tuples for every failure encountered.
        *source* is the entry point name, dotted module path, or file
        path that failed.
    """

    verifiers_loaded: list[str] = field(default_factory=list)
    kb_clients_loaded: list[str] = field(default_factory=list)
    aggregators_loaded: list[str] = field(default_factory=list)
    extractors_loaded: list[str] = field(default_factory=list)
    modules_imported: list[str] = field(default_factory=list)
    files_loaded: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"plugins: {len(self.verifiers_loaded)} verifiers, "
            f"{len(self.kb_clients_loaded)} KB clients, "
            f"{len(self.aggregators_loaded)} aggregators, "
            f"{len(self.extractors_loaded)} extractors "
            f"({len(self.errors)} errors)"
        )


# ---------------------------------------------------------------------------
# Entry-point helper
# ---------------------------------------------------------------------------


def discover_entry_points(group: str) -> list[Any]:
    """Return every entry point in *group* across the current environment.

    Uses :mod:`importlib.metadata` and handles the subtle API difference
    between Python 3.10 and 3.11+ (where ``entry_points()`` gained a
    ``group=`` keyword).
    """
    try:
        import importlib.metadata as metadata
    except ImportError:  # pragma: no cover - stdlib always present on 3.11+
        return []

    try:
        eps = metadata.entry_points(group=group)
    except TypeError:
        # Fallback for older API variants that don't accept the kwarg.
        all_eps = metadata.entry_points()
        eps = all_eps.get(group, []) if isinstance(all_eps, dict) else [
            ep for ep in all_eps if getattr(ep, "group", None) == group
        ]
    return list(eps)


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------


class PluginLoader:
    """Discover and load all configured plugins.

    The loader is stateless apart from the accumulating
    :class:`PluginLoadReport` returned from :meth:`load_all`.  Calling
    :meth:`load_all` more than once is safe: the underlying registries
    deduplicate by name.
    """

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self._registry = get_verifier_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> PluginLoadReport:
        """Run the three discovery mechanisms in sequence.

        Returns
        -------
        PluginLoadReport
            Aggregated report of everything that was (or wasn't) loaded.
        """
        report = PluginLoadReport()

        # Snapshot state prior to loading so we can diff what *this*
        # pass contributed versus what was already registered.
        verifiers_before = {v.name for v in self._registry.get_all()}
        kb_before = self._snapshot_kb_clients()

        # 1. Entry points
        self._load_entry_points(report)

        # 2. Dotted modules
        for module_path in self.settings.PLUGIN_MODULES or []:
            self._import_module(module_path, report)

        # 3. File paths
        for file_path in self.settings.PLUGIN_FILES or []:
            self._load_file(file_path, report)

        # Diff registries to figure out what got added.
        verifiers_after = {v.name for v in self._registry.get_all()}
        report.verifiers_loaded = sorted(verifiers_after - verifiers_before)

        kb_after = self._snapshot_kb_clients()
        report.kb_clients_loaded = sorted(kb_after - kb_before)

        if report.errors:
            for source, msg in report.errors:
                logger.error("Plugin load error in %s: %s", source, msg)
        logger.info("Plugin load complete: %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_entry_points(self, report: PluginLoadReport) -> None:
        groups: Iterable[str] = self.settings.PLUGIN_ENTRY_POINT_GROUPS or []
        for group in groups:
            for ep in discover_entry_points(group):
                ep_name = getattr(ep, "name", repr(ep))
                source = f"entry_point[{group}:{ep_name}]"
                try:
                    loaded = ep.load()
                except Exception as exc:  # noqa: BLE001
                    report.errors.append((source, f"{type(exc).__name__}: {exc}"))
                    logger.exception("Failed to load entry point %s", source)
                    continue

                self._classify_loaded_object(group, ep_name, loaded, report)

    def _import_module(self, module_path: str, report: PluginLoadReport) -> None:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(
                (module_path, f"{type(exc).__name__}: {exc}")
            )
            logger.exception("Failed to import plugin module %s", module_path)
            return
        report.modules_imported.append(module_path)

    def _load_file(self, file_path: str, report: PluginLoadReport) -> None:
        module_name = f"hal_nemofinder_plugin_{abs(hash(file_path))}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not build import spec for {file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            report.errors.append((file_path, f"{type(exc).__name__}: {exc}"))
            logger.exception("Failed to load plugin file %s", file_path)
            return
        report.files_loaded.append(file_path)

    def _classify_loaded_object(
        self,
        group: str,
        name: str,
        obj: Any,
        report: PluginLoadReport,
    ) -> None:
        """Route an entry-point payload into the right registry / list.

        Entry points can point at either a class (common) or a module.
        If it's a class and the group is the verifier group, we also
        validate that it subclasses :class:`BaseVerifier` and register
        it.  Modules are assumed to self-register via decorators at
        import time.
        """
        # Modules self-register; nothing more to do.
        if not isinstance(obj, type):
            return

        if group == "hal_nemofinder.verifiers":
            if not issubclass(obj, BaseVerifier):
                msg = (
                    f"entry point {name} -> {obj!r} is not a BaseVerifier "
                    f"subclass; skipping"
                )
                logger.warning(msg)
                report.errors.append((f"entry_point[{group}:{name}]", msg))
                return
            try:
                self._registry.register(obj)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    (f"entry_point[{group}:{name}]", f"{type(exc).__name__}: {exc}")
                )
                logger.exception("Failed to register verifier %s", name)
            return

        if group == "hal_nemofinder.knowledge_clients":
            try:
                from src.knowledge.client_registry import register_kb_client

                register_kb_client(name, obj)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    (f"entry_point[{group}:{name}]", f"{type(exc).__name__}: {exc}")
                )
                logger.exception("Failed to register KB client %s", name)
            return

        if group == "hal_nemofinder.aggregators":
            report.aggregators_loaded.append(name)
            return

        if group == "hal_nemofinder.extractors":
            report.extractors_loaded.append(name)
            return

    @staticmethod
    def _snapshot_kb_clients() -> set[str]:
        """Return the set of currently-registered KB client names."""
        try:
            from src.knowledge.client_registry import list_kb_clients

            return set(list_kb_clients())
        except Exception:  # noqa: BLE001
            return set()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def load_plugins_from_settings(settings: "Settings") -> PluginLoadReport:
    """Convenience wrapper: instantiate a :class:`PluginLoader` and run it.

    Intended to be called once at process startup (FastAPI app, Celery
    worker, CLI invocation, ...).
    """
    loader = PluginLoader(settings)
    return loader.load_all()

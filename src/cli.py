"""Production CLI for hal-nemoFinder.

This module exposes the ``hal-nemofinder`` console script: the single
command customers use to operate every part of the framework — running
verification, calibrating against their own regression sets, starting
the API server or Celery workers, inspecting plugins, and scaffolding
new deployments.

Design goals
------------
* **Never crash on plugin errors.** All plugin loading paths render
  errors into a rich ``Panel`` so operators see exactly which extension
  misbehaved without losing access to the rest of the framework.
* **Human-readable by default, machine-readable on demand.** Every
  command that emits tabular data supports ``--format json`` for
  pipelines, and ``table``/``pretty`` for humans.
* **One source of truth for settings.** Every command funnels through
  :func:`_build_settings` so the precedence rules (env > YAML >
  defaults) declared in :mod:`src.config` are honoured uniformly.

The CLI is registered in ``pyproject.toml`` as::

    [project.scripts]
    hal-nemofinder = "src.cli:app"
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

logger = logging.getLogger("hal_nemofinder.cli")

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------


app = typer.Typer(
    name="hal-nemofinder",
    help="Hallucination detection framework for AI-driven drug discovery.",
    no_args_is_help=True,
    add_completion=False,
)

list_app = typer.Typer(
    help="Introspect registered verifiers, KB clients, and plugins.",
    no_args_is_help=True,
)
config_app = typer.Typer(
    help="Inspect and validate hal-nemoFinder configuration.",
    no_args_is_help=True,
)
audit_app = typer.Typer(
    help="Inspect and export the HMAC-chained audit log.",
    no_args_is_help=True,
)
db_app = typer.Typer(
    help="Run Alembic migrations against the configured database.",
    no_args_is_help=True,
)
auth_app = typer.Typer(
    help="Bootstrap tenants, users, and API keys for multi-tenant deployments.",
    no_args_is_help=True,
)
review_app = typer.Typer(
    help="Active-learning review queue (uncertainty sampling).",
    no_args_is_help=True,
)
drift_app = typer.Typer(
    help="Verifier drift detection.",
    no_args_is_help=True,
)
cost_app = typer.Typer(
    help="Compute and API cost accounting.",
    no_args_is_help=True,
)
shadow_app = typer.Typer(
    help="Shadow-mode champion/challenger comparisons.",
    no_args_is_help=True,
)
app.add_typer(list_app, name="list")
app.add_typer(config_app, name="config")
app.add_typer(audit_app, name="audit")
app.add_typer(db_app, name="db")
app.add_typer(auth_app, name="auth")
app.add_typer(review_app, name="review")
app.add_typer(drift_app, name="drift")
app.add_typer(cost_app, name="cost")
app.add_typer(shadow_app, name="shadow")

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_VERDICT_STYLES: dict[str, str] = {
    "verified": "bold green",
    "refuted": "bold red",
    "partially_supported": "bold yellow",
    "unverifiable": "dim",
}


def _style_verdict(verdict: str) -> Text:
    """Return a colour-coded :class:`Text` for a verdict string."""
    key = str(verdict).lower().replace(" ", "_")
    style = _VERDICT_STYLES.get(key, "white")
    return Text(verdict, style=style)


def _fatal(msg: str, exc: BaseException | None = None) -> "None":
    """Render a red error panel and exit the process with status 1."""
    body = msg
    if exc is not None:
        body = f"{msg}\n\n[dim]{type(exc).__name__}: {exc}[/dim]"
    err_console.print(
        Panel(body, title="hal-nemofinder error", border_style="red", expand=True)
    )
    raise typer.Exit(code=1)


def _build_settings() -> Any:
    """Construct a :class:`Settings` instance, rendering errors nicely."""
    try:
        from src.config import Settings  # local import so --help is fast

        return Settings()
    except Exception as exc:  # noqa: BLE001
        _fatal("Failed to load settings.", exc)


def _load_plugins_safe(settings: Any) -> Any:
    """Invoke the plugin loader, surfacing errors without crashing."""
    try:
        from src.plugins import load_plugins_from_settings

        report = load_plugins_from_settings(settings)
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            Panel(
                f"Plugin discovery failed before running: {type(exc).__name__}: {exc}",
                title="plugin load error",
                border_style="red",
            )
        )
        return None

    if getattr(report, "errors", None):
        table = Table(title="Plugin load errors", header_style="bold red")
        table.add_column("Source", overflow="fold")
        table.add_column("Message", overflow="fold")
        for source, msg in report.errors:
            table.add_row(str(source), str(msg))
        err_console.print(table)
    return report


def _json_default(obj: Any) -> Any:
    """Best-effort JSON serialiser for dataclasses / enums / paths."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _dump_json(data: Any) -> str:
    """Serialise *data* as pretty JSON using :func:`_json_default`."""
    return json.dumps(data, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def _resolve_version() -> str:
    """Best-effort lookup of the installed package version."""
    for dist_name in ("hal-nemofinder", "hal_nemofinder"):
        try:
            return importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "0.0.0+unknown"


@app.command("version")
def version_cmd() -> None:
    """Print the installed version and loaded plugin sources."""
    ver = _resolve_version()
    console.print(Panel(f"hal-nemofinder [bold cyan]{ver}[/bold cyan]", border_style="cyan"))

    settings = _build_settings()
    report = _load_plugins_safe(settings)

    table = Table(title="Loaded plugin sources", header_style="bold magenta")
    table.add_column("Kind")
    table.add_column("Identifier", overflow="fold")
    if report is not None:
        for name in report.verifiers_loaded:
            table.add_row("verifier", name)
        for name in report.kb_clients_loaded:
            table.add_row("kb_client", name)
        for name in report.modules_imported:
            table.add_row("module", name)
        for name in report.files_loaded:
            table.add_row("file", name)
    if table.row_count == 0:
        table.add_row("[dim]none[/dim]", "[dim]built-ins only[/dim]")
    console.print(table)


# ---------------------------------------------------------------------------
# Init scaffold
# ---------------------------------------------------------------------------


_INIT_YAML = """# hal-nemoFinder customer configuration
# See src/config.py for the full list of settings and environment-variable
# equivalents (every field is overridable via HAL_<FIELD>).

database:
  url: postgresql+asyncpg://hal:hal@localhost:5432/hal_nemofinder

redis_url: redis://localhost:6379/0

plugins:
  modules:
    # - mycompany.verifiers
  files:
    # - /absolute/path/to/custom_verifier.py

# Replace built-in KB clients with your internal mirrors.
kb_client_overrides: {}
  # pubchem: mycompany.kb.InternalPubChemMirror

# Override per-verifier reliability from your own calibration runs.
verifier_reliability_overrides: {}
  # chemical:
  #   sensitivity: 0.97
  #   specificity: 0.94

# Regression sets used by `hal-nemofinder calibrate`.
regression_set_paths:
  - ./regression_sets/my_regression_set.jsonl
"""


_INIT_VERIFIER = '''"""Template customer verifier.

Rename the class, fill in the claim types you care about, and wire
``verify`` up to your internal systems.  The framework will auto-load
this file if you list it under ``plugins.files`` in hal_nemofinder.yaml.
"""

from __future__ import annotations

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier


@register_verifier
class MyVerifier(BaseVerifier):
    name = "my_verifier"
    supported_claim_types = [ClaimType.general_biomedical]

    async def verify(self, claim_text, claim_type, context):
        # TODO: plug this into your proprietary system.
        return VerificationOutput(
            verdict=Verdict.unverifiable,
            confidence=0.0,
            reasoning="MyVerifier is a template; replace with real logic.",
            evidence={},
            source_db=self.name,
        )
'''


_INIT_KB_CLIENT = '''"""Template customer knowledge-base client.

Drop-in replacement for a built-in client (e.g. PubChem) backed by your
internal mirror.  Register via KB_CLIENT_OVERRIDES in hal_nemofinder.yaml.
"""

from __future__ import annotations


class MyKBClient:
    """Stand-in for a real knowledge-base client."""

    def __init__(self, cache=None) -> None:
        self._cache = cache

    async def close(self) -> None:
        """Release any open connections."""

    async def search_by_smiles(self, smiles: str) -> dict | None:
        # TODO: call your internal Oracle / Snowflake / REST endpoint.
        return None
'''


_INIT_REGRESSION_SET = (
    '{"claim_text": "Aspirin has a molecular weight of 180.16 g/mol.", '
    '"claim_type": "molecular_property", "is_hallucination": false, '
    '"source": "init_scaffold"}\n'
    '{"claim_text": "Compound XYZ-999 has IC50 of 0 nM against EGFR.", '
    '"claim_type": "target_interaction", "is_hallucination": true, '
    '"source": "init_scaffold"}\n'
    '{"claim_text": "Ibuprofen is a propionic acid NSAID.", '
    '"claim_type": "general_biomedical", "is_hallucination": false, '
    '"source": "init_scaffold"}\n'
)


@app.command("init")
def init_cmd(
    dir: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Target directory (created if missing).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing files instead of skipping them.",
    ),
) -> None:
    """Scaffold a customer plugin directory with example config and stubs."""
    target = dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    files: dict[Path, str] = {
        target / "hal_nemofinder.yaml": _INIT_YAML,
        target / "verifiers" / "my_verifier.py": _INIT_VERIFIER,
        target / "kb_clients" / "my_kb_client.py": _INIT_KB_CLIENT,
        target / "regression_sets" / "my_regression_set.jsonl": _INIT_REGRESSION_SET,
    }

    written: list[Path] = []
    skipped: list[Path] = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.write_text(content, encoding="utf-8")
        written.append(path)

    welcome = Text.assemble(
        ("Welcome to hal-nemoFinder!\n\n", "bold cyan"),
        ("Your customer plugin directory has been scaffolded at:\n  ", "white"),
        (str(target) + "\n\n", "bold"),
        ("Next steps:\n", "bold"),
        ("  1. ", "dim"),
        ("Edit hal_nemofinder.yaml to point at your databases.\n", "white"),
        ("  2. ", "dim"),
        ("Fill in verifiers/my_verifier.py with your proprietary logic.\n", "white"),
        ("  3. ", "dim"),
        ("Curate regression_sets/my_regression_set.jsonl from historical data.\n", "white"),
        ("  4. ", "dim"),
        ("Run `hal-nemofinder calibrate` to see baseline metrics.\n", "white"),
        ("  5. ", "dim"),
        ("Run `hal-nemofinder serve` when you are ready to go live.\n", "white"),
    )
    console.print(Panel(welcome, title="hal-nemofinder init", border_style="cyan"))

    table = Table(title="Scaffolded files", header_style="bold magenta")
    table.add_column("Status")
    table.add_column("Path", overflow="fold")
    for p in written:
        table.add_row("[green]created[/green]", str(p))
    for p in skipped:
        table.add_row("[yellow]skipped[/yellow]", str(p) + "  (exists, use --force)")
    console.print(table)


# ---------------------------------------------------------------------------
# Verify commands
# ---------------------------------------------------------------------------


def _run_verification(
    text: str,
    forced_type: Optional[str],
    settings: Any,
) -> dict[str, Any]:
    """Run the full extract/classify/route/aggregate pipeline on *text*.

    Returns a dict containing the classified claims, per-verifier
    outputs, and the Bayesian aggregate — suitable for either table or
    JSON rendering.
    """
    from src.core.aggregator import BayesianAggregator
    from src.core.claim_classifier import ClaimClassifier
    from src.core.claim_extractor import ClaimExtractor, ExtractedClaim, SourceSpan
    from src.core.router import VerificationRouter
    from src.models.enums import ClaimType

    extractor = ClaimExtractor()
    classifier = ClaimClassifier()
    router = VerificationRouter()

    reliability_overrides = getattr(settings, "VERIFIER_RELIABILITY_OVERRIDES", None) or {}
    if reliability_overrides:
        base = dict(BayesianAggregator().__dict__.get("_reliability", {}))
        base.update(reliability_overrides)
        aggregator = BayesianAggregator(reliability=base)
    else:
        aggregator = BayesianAggregator()

    extracted = extractor.extract(text)
    if not extracted:
        # Fall back to treating the entire input as a single claim so the
        # CLI always produces output for short inputs.
        extracted = [
            ExtractedClaim(
                claim_text=text.strip(),
                source_span=SourceSpan(start=0, end=len(text), context=text),
                confidence=0.3,
            )
        ]

    forced: ClaimType | None = None
    if forced_type:
        try:
            forced = ClaimType(forced_type)
        except ValueError:
            _fatal(
                f"Unknown --type {forced_type!r}. Must be one of: "
                f"{', '.join(t.value for t in ClaimType)}"
            )

    async def _inner() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for claim in extracted:
            ctype = forced or classifier.classify(claim)
            results = await router.verify_claim(claim.claim_text, ctype, {})
            agg = aggregator.aggregate_claim_bayesian(results)
            posterior = aggregator.posterior_probability(results)
            rows.append(
                {
                    "claim_text": claim.claim_text,
                    "claim_type": ctype.value,
                    "extraction_confidence": claim.confidence,
                    "verifier_results": [
                        {
                            "source_db": r.source_db,
                            "verdict": r.verdict.value,
                            "confidence": r.confidence,
                            "reasoning": r.reasoning,
                        }
                        for r in results
                    ],
                    "aggregated": {
                        "verdict": agg.verdict.value,
                        "confidence": agg.confidence,
                        "posterior_hallucination": round(posterior, 4),
                        "reasoning": agg.reasoning,
                    },
                }
            )
        return rows

    rows = asyncio.run(_inner())
    return {"claims": rows}


def _render_verify_table(data: dict[str, Any]) -> None:
    """Pretty-print a verification result as rich tables."""
    for claim in data["claims"]:
        header = Text.assemble(
            ("Claim: ", "bold"),
            (claim["claim_text"], "white"),
        )
        subtitle = (
            f"type={claim['claim_type']}  "
            f"extraction_confidence={claim['extraction_confidence']:.2f}"
        )
        console.print(Panel(header, subtitle=subtitle, border_style="cyan"))

        if claim["verifier_results"]:
            table = Table(header_style="bold magenta", show_lines=False)
            table.add_column("Verifier")
            table.add_column("Verdict")
            table.add_column("Confidence", justify="right")
            table.add_column("Reasoning", overflow="fold")
            for r in claim["verifier_results"]:
                table.add_row(
                    r["source_db"] or "[dim]unnamed[/dim]",
                    _style_verdict(r["verdict"]),
                    f"{r['confidence']:.2f}",
                    r["reasoning"] or "",
                )
            console.print(table)
        else:
            console.print("[dim]No verifiers matched this claim type.[/dim]")

        agg = claim["aggregated"]
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Final verdict:", _style_verdict(agg["verdict"]))
        summary.add_row("Confidence:", f"{agg['confidence']:.2f}")
        summary.add_row(
            "P(hallucination):",
            f"{agg['posterior_hallucination']:.2f}",
        )
        console.print(Panel(summary, title="Bayesian aggregate", border_style="green"))


@app.command("verify")
def verify_cmd(
    text: str = typer.Argument(..., help="Claim text to verify."),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Force a specific ClaimType (else auto-classify)."
    ),
    format: str = typer.Option(
        "pretty",
        "--format",
        "-f",
        help="Output format: json | table | pretty",
        case_sensitive=False,
    ),
) -> None:
    """Verify a single claim from the command line."""
    settings = _build_settings()
    _load_plugins_safe(settings)

    try:
        data = _run_verification(text, type, settings)
    except Exception as exc:  # noqa: BLE001
        _fatal("Verification failed.", exc)

    fmt = format.lower()
    if fmt == "json":
        console.print_json(data=data, default=_json_default)
        return
    if fmt in {"table", "pretty"}:
        _render_verify_table(data)
        return
    _fatal(f"Unknown --format {format!r}. Use json | table | pretty.")


@app.command("verify-file")
def verify_file_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True),
    format: str = typer.Option("pretty", "--format", "-f"),
    type: Optional[str] = typer.Option(None, "--type", "-t"),
) -> None:
    """Verify claim text loaded from a file."""
    text = path.read_text(encoding="utf-8")
    verify_cmd(text=text, type=type, format=format)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------


@app.command("calibrate")
def calibrate_cmd(
    regression_set: Optional[Path] = typer.Option(
        None,
        "--regression-set",
        "-r",
        help="Path to a regression set file. Defaults to REGRESSION_SET_PATHS or the bundled demo set.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write metrics as JSON to this path.",
    ),
) -> None:
    """Run calibration against a regression set and display metrics."""
    from src.core.aggregator import BayesianAggregator
    from src.core.calibration import (
        CalibrationTracker,
        load_regression_set_from_file,
        run_regression_test,
    )
    from src.core.router import VerificationRouter

    settings = _build_settings()
    _load_plugins_safe(settings)

    # Decide which regression set to use.
    claims: list[Any] = []
    if regression_set is not None:
        try:
            claims = load_regression_set_from_file(regression_set)
        except Exception as exc:  # noqa: BLE001
            _fatal(f"Failed to load regression set {regression_set}", exc)
    elif getattr(settings, "REGRESSION_SET_PATHS", None):
        tracker = CalibrationTracker()
        loaded = tracker.load_regression_sets_from_settings(settings)
        if loaded == 0:
            _fatal(
                "REGRESSION_SET_PATHS is set but no claims were loaded; "
                "check file paths and formats."
            )
        claims = tracker.all_loaded_claims()
    # else: use bundled default (run_regression_test handles None)

    router = VerificationRouter()
    overrides = getattr(settings, "VERIFIER_RELIABILITY_OVERRIDES", None) or {}
    aggregator = BayesianAggregator(reliability=overrides or None)

    n_total = len(claims) if claims else 20  # fallback matches bundled set

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Calibrating", total=n_total)

        def on_step(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        try:
            metrics = asyncio.run(
                run_regression_test(
                    verifier_router=router,
                    aggregator=aggregator,
                    regression_set=claims or None,
                    progress_callback=on_step,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _fatal("Calibration run failed.", exc)

    # Headline metrics table
    table = Table(title="Calibration metrics", header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("records", str(metrics.n_records))
    table.add_row("accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("precision", f"{metrics.precision:.4f}")
    table.add_row("recall", f"{metrics.recall:.4f}")
    table.add_row("F1", f"{metrics.f1:.4f}")
    table.add_row("Brier score", f"{metrics.brier_score:.4f}")
    table.add_row("ECE", f"{metrics.ece:.4f}")
    console.print(table)

    # Reliability diagram (per-bin table)
    if metrics.bin_data:
        rel = Table(title="Reliability diagram", header_style="bold cyan")
        rel.add_column("bin", justify="right")
        rel.add_column("pred mean", justify="right")
        rel.add_column("actual mean", justify="right")
        rel.add_column("count", justify="right")
        rel.add_column("bar")
        for lo, hi, mp, ma, count in metrics.bin_data:
            bar_len = int(mp * 30)
            bar = "█" * bar_len + " " * (30 - bar_len)
            rel.add_row(
                f"[{lo:.1f}, {hi:.1f})",
                f"{mp:.3f}",
                f"{ma:.3f}",
                str(count),
                bar,
            )
        console.print(rel)

    if metrics.per_claim_type_accuracy:
        by_type = Table(title="Accuracy by claim type", header_style="bold magenta")
        by_type.add_column("Claim type")
        by_type.add_column("Accuracy", justify="right")
        for k, v in sorted(metrics.per_claim_type_accuracy.items()):
            by_type.add_row(k, f"{v:.4f}")
        console.print(by_type)

    if metrics.per_source_accuracy:
        by_src = Table(title="Accuracy by source", header_style="bold magenta")
        by_src.add_column("Source")
        by_src.add_column("Accuracy", justify="right")
        for k, v in sorted(metrics.per_source_accuracy.items()):
            by_src.add_row(k, f"{v:.4f}")
        console.print(by_src)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_dump_json(metrics.to_dict()), encoding="utf-8")
        console.print(f"[green]Wrote metrics report to[/green] {output}")


# ---------------------------------------------------------------------------
# list subcommands
# ---------------------------------------------------------------------------


@list_app.command("verifiers")
def list_verifiers_cmd(
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """List every registered verifier."""
    settings = _build_settings()
    report = _load_plugins_safe(settings)

    from src.verifiers.base import get_verifier_registry

    registry = get_verifier_registry()
    builtins: set[str] = set()
    try:
        # All verifiers that were not added by this load pass are "built-in".
        if report is not None:
            loaded_names = set(report.verifiers_loaded)
            for v in registry.get_all():
                if v.name not in loaded_names:
                    builtins.add(v.name)
    except Exception:  # noqa: BLE001
        pass

    rows: list[dict[str, Any]] = []
    for v in registry.get_all():
        rows.append(
            {
                "name": v.name,
                "supported_claim_types": [ct.value for ct in v.supported_claim_types],
                "source": "built-in" if v.name in builtins else "plugin",
            }
        )

    if format.lower() == "json":
        console.print_json(data=rows, default=_json_default)
        return

    table = Table(title="Registered verifiers", header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Claim types", overflow="fold")
    table.add_column("Source")
    for row in rows:
        table.add_row(
            row["name"],
            ", ".join(row["supported_claim_types"]) or "[dim]none[/dim]",
            row["source"],
        )
    if not rows:
        table.add_row("[dim]none[/dim]", "", "")
    console.print(table)


@list_app.command("kb-clients")
def list_kb_clients_cmd(
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """List every registered knowledge-base client."""
    settings = _build_settings()
    report = _load_plugins_safe(settings)

    try:
        from src.knowledge.client_registry import get_kb_client, list_kb_clients
    except Exception as exc:  # noqa: BLE001
        _fatal("KB client registry is unavailable.", exc)

    names = list_kb_clients()
    loaded_by_plugin = set(getattr(report, "kb_clients_loaded", []) or [])

    rows: list[dict[str, str]] = []
    for name in names:
        cls = get_kb_client(name)
        rows.append(
            {
                "name": name,
                "class": f"{cls.__module__}.{cls.__name__}" if cls else "?",
                "source": "plugin" if name in loaded_by_plugin else "built-in",
            }
        )

    if format.lower() == "json":
        console.print_json(data=rows, default=_json_default)
        return

    table = Table(title="Registered KB clients", header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Class", overflow="fold")
    table.add_column("Source")
    for row in rows:
        table.add_row(row["name"], row["class"], row["source"])
    if not rows:
        table.add_row("[dim]none[/dim]", "", "")
    console.print(table)


@list_app.command("plugins")
def list_plugins_cmd() -> None:
    """Show everything loaded from entry points, modules, and files."""
    settings = _build_settings()
    report = _load_plugins_safe(settings)
    if report is None:
        return

    tree = Tree("[bold cyan]Plugin load report[/bold cyan]")

    v_node = tree.add(f"Verifiers ({len(report.verifiers_loaded)})")
    for name in report.verifiers_loaded:
        v_node.add(name)

    k_node = tree.add(f"KB clients ({len(report.kb_clients_loaded)})")
    for name in report.kb_clients_loaded:
        k_node.add(name)

    m_node = tree.add(f"Modules imported ({len(report.modules_imported)})")
    for name in report.modules_imported:
        m_node.add(name)

    f_node = tree.add(f"Files loaded ({len(report.files_loaded)})")
    for name in report.files_loaded:
        f_node.add(name)

    a_node = tree.add(f"Aggregators ({len(report.aggregators_loaded)})")
    for name in report.aggregators_loaded:
        a_node.add(name)

    x_node = tree.add(f"Extractors ({len(report.extractors_loaded)})")
    for name in report.extractors_loaded:
        x_node.add(name)

    console.print(tree)

    if report.errors:
        err_table = Table(title="Errors", header_style="bold red")
        err_table.add_column("Source", overflow="fold")
        err_table.add_column("Message", overflow="fold")
        for source, msg in report.errors:
            err_table.add_row(str(source), str(msg))
        console.print(err_table)


# ---------------------------------------------------------------------------
# serve / worker
# ---------------------------------------------------------------------------


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    workers: int = typer.Option(1, "--workers"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)."),
) -> None:
    """Start the FastAPI server via uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        _fatal("uvicorn is not installed.", exc)

    console.print(
        Panel(
            f"Starting hal-nemoFinder on [bold]{host}:{port}[/bold] "
            f"(workers={workers}, reload={reload})",
            border_style="cyan",
        )
    )
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


@app.command("worker")
def worker_cmd(
    queues: str = typer.Option(
        "default", "--queues", "-Q", help="Comma-separated Celery queue names."
    ),
    concurrency: int = typer.Option(4, "--concurrency", "-c"),
    loglevel: str = typer.Option("info", "--loglevel", "-l"),
) -> None:
    """Start a Celery worker."""
    try:
        celery_app = importlib.import_module("src.tasks.celery_app").celery_app
    except Exception as exc:  # noqa: BLE001
        _fatal("Could not import Celery app from src.tasks.celery_app.", exc)

    console.print(
        Panel(
            f"Starting Celery worker (queues={queues}, concurrency={concurrency})",
            border_style="cyan",
        )
    )
    argv = [
        "worker",
        f"--loglevel={loglevel}",
        f"--concurrency={concurrency}",
        f"--queues={queues}",
    ]
    celery_app.worker_main(argv)


# ---------------------------------------------------------------------------
# config subcommands
# ---------------------------------------------------------------------------


def _setting_source(key: str, yaml_keys: set[str]) -> str:
    """Return the effective source of *key* (env, yaml, default)."""
    if f"HAL_{key}" in os.environ or key in os.environ:
        return "env"
    if key in yaml_keys:
        return "yaml"
    return "default"


@config_app.command("show")
def config_show_cmd() -> None:
    """Print the current effective settings as a tree."""
    settings = _build_settings()

    yaml_keys: set[str] = set()
    try:
        from src.config import _load_yaml_file, _resolve_yaml_path

        p = _resolve_yaml_path()
        if p is not None:
            yaml_keys = set(_load_yaml_file(p).keys())
    except Exception:  # noqa: BLE001
        pass

    data = settings.model_dump()
    tree = Tree("[bold cyan]hal-nemofinder settings[/bold cyan]")
    for key in sorted(data.keys()):
        value = data[key]
        source = _setting_source(key, yaml_keys)
        tag = {
            "env": "[green](env)[/green]",
            "yaml": "[yellow](yaml)[/yellow]",
            "default": "[dim](default)[/dim]",
        }[source]
        rendered = json.dumps(value, default=_json_default)
        tree.add(f"[bold]{key}[/bold] = {rendered} {tag}")
    console.print(tree)


@config_app.command("validate")
def config_validate_cmd(
    path: Optional[Path] = typer.Argument(
        None, help="Path to YAML config file. Defaults to auto-discovery."
    ),
) -> None:
    """Validate a YAML config file against the Settings schema."""
    from src.config import Settings

    target = path
    if target is None:
        try:
            from src.config import _resolve_yaml_path

            target = _resolve_yaml_path()
        except Exception:  # noqa: BLE001
            target = None

    if target is None:
        _fatal("No config file specified and no default YAML was found.")

    if not target.exists():
        _fatal(f"Config file does not exist: {target}")

    try:
        Settings.from_yaml(target)
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            Panel(
                f"[red]Invalid config file:[/red] {target}\n\n{traceback.format_exc()}",
                title="config validation failed",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[green]OK[/green] — {target} parses cleanly against Settings.",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# audit subcommands
# ---------------------------------------------------------------------------


def _audit_session_factory() -> Any:
    """Return the async session factory used by audit CLI commands."""
    from src.db.session import AsyncSessionLocal

    return AsyncSessionLocal


@audit_app.command("verify")
def audit_verify_cmd(
    checkpoint: int = typer.Option(
        0,
        "--checkpoint",
        "-c",
        help="Resume verification after this sequence number.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum number of rows to verify."
    ),
) -> None:
    """Run the HMAC chain integrity check and print a summary."""
    from src.audit.recorder import get_audit_recorder

    async def _inner() -> Any:
        recorder = get_audit_recorder()
        factory = _audit_session_factory()
        async with factory() as session:
            return await recorder.verify_integrity(
                session,
                limit=limit,
                start_sequence=checkpoint,
            )

    try:
        report = asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001
        _fatal("Audit verification failed.", exc)

    table = Table(title="Audit chain verification", header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("valid", "[green]yes[/green]" if report.valid else "[red]no[/red]")
    table.add_row("total_checked", str(report.total_checked))
    table.add_row(
        "first_invalid_sequence",
        str(report.first_invalid_sequence) if report.first_invalid_sequence else "-",
    )
    console.print(table)

    if report.errors:
        err_table = Table(title="Errors", header_style="bold red")
        err_table.add_column("#", justify="right")
        err_table.add_column("Message", overflow="fold")
        for i, msg in enumerate(report.errors, 1):
            err_table.add_row(str(i), msg)
        console.print(err_table)
        raise typer.Exit(code=1)


@audit_app.command("export")
def audit_export_cmd(
    since: Optional[str] = typer.Option(
        None, "--since", help="ISO-8601 date/datetime lower bound."
    ),
    tenant: Optional[str] = typer.Option(
        None, "--tenant", help="Tenant slug to filter on."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSONL to this path (default: stdout)."
    ),
) -> None:
    """Dump audit entries as JSONL for compliance reviewers."""
    from datetime import datetime

    from sqlalchemy import select

    from src.audit.recorder import canonical_json
    from src.models.audit import AuditLogEntry

    since_dt: Optional[datetime] = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError as exc:
            _fatal(f"Invalid --since value: {since!r}", exc)

    async def _inner() -> int:
        factory = _audit_session_factory()
        tenant_id = None
        if tenant:
            from src.models.tenant import Tenant

            async with factory() as session:
                row = (
                    await session.execute(
                        select(Tenant).where(Tenant.slug == tenant)
                    )
                ).scalar_one_or_none()
                if row is None:
                    _fatal(f"No tenant with slug {tenant!r}")
                tenant_id = row.id

        stmt = select(AuditLogEntry).order_by(AuditLogEntry.sequence.asc())
        if since_dt is not None:
            stmt = stmt.where(AuditLogEntry.created_at >= since_dt)
        if tenant_id is not None:
            stmt = stmt.where(AuditLogEntry.tenant_id == tenant_id)

        count = 0
        async with factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        out_handle = output.open("w", encoding="utf-8") if output else sys.stdout
        try:
            for row in rows:
                record = {
                    "sequence": row.sequence,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "event_type": row.event_type.value,
                    "tenant_id": str(row.tenant_id) if row.tenant_id else None,
                    "actor_user_id": (
                        str(row.actor_user_id) if row.actor_user_id else None
                    ),
                    "actor_api_key_id": (
                        str(row.actor_api_key_id) if row.actor_api_key_id else None
                    ),
                    "actor_ip": row.actor_ip,
                    "resource_type": row.resource_type,
                    "resource_id": row.resource_id,
                    "action": row.action,
                    "outcome": row.outcome,
                    "payload": row.payload,
                    "prev_hash": row.prev_hash,
                    "integrity_hash": row.integrity_hash,
                }
                out_handle.write(canonical_json(record) + "\n")
                count += 1
        finally:
            if output:
                out_handle.close()
        return count

    try:
        n = asyncio.run(_inner())
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        _fatal("Audit export failed.", exc)

    if output:
        console.print(f"[green]Wrote {n} entries to[/green] {output}")


@audit_app.command("tail")
def audit_tail_cmd(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Poll for new entries every 2 seconds."
    ),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Show the most recent audit events, optionally streaming new ones."""
    import time

    from sqlalchemy import select

    from src.models.audit import AuditLogEntry

    async def _fetch(after_seq: int) -> list[Any]:
        factory = _audit_session_factory()
        async with factory() as session:
            stmt = (
                select(AuditLogEntry)
                .where(AuditLogEntry.sequence > after_seq)
                .order_by(AuditLogEntry.sequence.asc())
                .limit(max(limit, 1))
            )
            return (await session.execute(stmt)).scalars().all()

    def _render(rows: list[Any]) -> None:
        for row in rows:
            console.print(
                f"[dim]{row.sequence:>6}[/dim] "
                f"[cyan]{row.event_type.value:<24}[/cyan] "
                f"{row.action}/{row.outcome} "
                f"[dim]{row.resource_type}[/dim]:{row.resource_id}"
            )

    try:
        # Initial backfill — tail the last N rows.
        async def _initial() -> list[Any]:
            factory = _audit_session_factory()
            async with factory() as session:
                stmt = (
                    select(AuditLogEntry)
                    .order_by(AuditLogEntry.sequence.desc())
                    .limit(limit)
                )
                rows = (await session.execute(stmt)).scalars().all()
                return list(reversed(rows))

        rows = asyncio.run(_initial())
        _render(rows)
        last_seq = rows[-1].sequence if rows else 0

        if not follow:
            return

        while True:
            time.sleep(2.0)
            new_rows = asyncio.run(_fetch(last_seq))
            if new_rows:
                _render(new_rows)
                last_seq = new_rows[-1].sequence
    except KeyboardInterrupt:
        console.print("[dim]stopped[/dim]")
    except Exception as exc:  # noqa: BLE001
        _fatal("Audit tail failed.", exc)


# ---------------------------------------------------------------------------
# db subcommands (alembic wrappers)
# ---------------------------------------------------------------------------


def _alembic_config() -> Any:
    """Build an Alembic :class:`Config` from the repo's alembic.ini."""
    try:
        from alembic.config import Config
    except ImportError as exc:  # pragma: no cover
        _fatal("alembic is not installed.", exc)

    # Use the repo-root alembic.ini — same file that scripts/migrate.sh
    # drives.  Callers can point at a different file via env var.
    ini_path = os.environ.get("HAL_ALEMBIC_INI", "alembic.ini")
    cfg = Config(ini_path)
    return cfg


@db_app.command("migrate")
def db_migrate_cmd(
    revision: str = typer.Option(
        "head", "--revision", "-r", help="Target revision (default: head)."
    ),
) -> None:
    """Run ``alembic upgrade <revision>``."""
    from alembic import command

    cfg = _alembic_config()
    try:
        command.upgrade(cfg, revision)
    except Exception as exc:  # noqa: BLE001
        _fatal("Migration failed.", exc)
    console.print(f"[green]upgraded to[/green] {revision}")


@db_app.command("rollback")
def db_rollback_cmd(
    steps: int = typer.Option(
        1, "--steps", "-s", min=1, help="Number of revisions to downgrade."
    ),
) -> None:
    """Run ``alembic downgrade -<steps>``."""
    from alembic import command

    cfg = _alembic_config()
    try:
        command.downgrade(cfg, f"-{steps}")
    except Exception as exc:  # noqa: BLE001
        _fatal("Rollback failed.", exc)
    console.print(f"[yellow]downgraded {steps} step(s)[/yellow]")


@db_app.command("current")
def db_current_cmd() -> None:
    """Show the currently applied revision."""
    from alembic import command

    cfg = _alembic_config()
    try:
        command.current(cfg)
    except Exception as exc:  # noqa: BLE001
        _fatal("Failed to read current revision.", exc)


@db_app.command("history")
def db_history_cmd() -> None:
    """Show the full migration history."""
    from alembic import command

    cfg = _alembic_config()
    try:
        command.history(cfg)
    except Exception as exc:  # noqa: BLE001
        _fatal("Failed to load history.", exc)


# ---------------------------------------------------------------------------
# MLOps: process-wide singletons used by the CLI subcommands below
# ---------------------------------------------------------------------------


def _get_review_sampler() -> Any:
    """Lazy singleton :class:`UncertaintySampler` for the CLI session.

    A persistent datastore-backed sampler would be wired here in a
    production deployment; the CLI stub keeps state in-process so the
    commands are testable end-to-end without external infrastructure.
    """
    from src.core.active_learning import UncertaintySampler

    sampler = getattr(_get_review_sampler, "_sampler", None)
    if sampler is None:
        sampler = UncertaintySampler(review_queue_size=500)
        _get_review_sampler._sampler = sampler  # type: ignore[attr-defined]
    return sampler


def _get_drift_detector() -> Any:
    """Lazy singleton :class:`DriftDetector` for the CLI session."""
    from src.core.drift import DriftDetector

    detector = getattr(_get_drift_detector, "_detector", None)
    if detector is None:
        detector = DriftDetector()
        _get_drift_detector._detector = detector  # type: ignore[attr-defined]
    return detector


def _get_cost_tracker() -> Any:
    """Lazy singleton :class:`CostTracker` wired to the current settings."""
    from src.mlops.cost_tracker import CostTracker

    tracker = getattr(_get_cost_tracker, "_tracker", None)
    if tracker is None:
        tracker = CostTracker(settings=_build_settings())
        _get_cost_tracker._tracker = tracker  # type: ignore[attr-defined]
    return tracker


# ---------------------------------------------------------------------------
# review subcommands
# ---------------------------------------------------------------------------


@review_app.command("queue")
def review_queue_cmd(
    limit: int = typer.Option(20, "--limit", "-n", help="Max candidates to show."),
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Override uncertainty strategy: least_confident | margin | entropy.",
    ),
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """Show the active-learning review queue, most-uncertain first."""
    sampler = _get_review_sampler()
    candidates = sampler.get_review_queue(limit=limit, strategy=strategy)

    if format.lower() == "json":
        console.print_json(
            data=[c.to_dict() for c in candidates], default=_json_default
        )
        return

    table = Table(
        title=f"Review queue ({len(candidates)} shown)",
        header_style="bold magenta",
    )
    table.add_column("Claim id")
    table.add_column("Verdict")
    table.add_column("Posterior", justify="right")
    table.add_column("Uncertainty", justify="right")
    table.add_column("Claim", overflow="fold")
    for c in candidates:
        table.add_row(
            c.claim_id,
            _style_verdict(c.predicted_verdict.value),
            f"{c.posterior:.3f}",
            f"{c.uncertainty_score:.3f}",
            c.claim_text,
        )
    if not candidates:
        table.add_row("[dim]empty[/dim]", "", "", "", "")
    console.print(table)


@review_app.command("label")
def review_label_cmd(
    claim_id: str = typer.Argument(..., help="Claim id previously shown by `review queue`."),
    label: str = typer.Argument(
        ..., help="Ground-truth label: 'true' if hallucination, 'false' if truthful."
    ),
    reviewer: str = typer.Option(
        "cli", "--reviewer", "-u", help="Reviewer identity recorded on the label."
    ),
) -> None:
    """Attach a ground-truth label to a queued review candidate."""
    lower = label.strip().lower()
    if lower in {"true", "t", "yes", "y", "1", "hallucination"}:
        value = True
    elif lower in {"false", "f", "no", "n", "0", "truthful"}:
        value = False
    else:
        _fatal(f"Unknown label {label!r}. Use 'true' or 'false'.")
        return  # pragma: no cover

    sampler = _get_review_sampler()
    ok = sampler.mark_reviewed(claim_id, value, reviewer)
    if not ok:
        _fatal(f"Claim id {claim_id!r} was not found in the review queue.")
    console.print(
        Panel(
            f"[green]Labelled[/green] {claim_id} as "
            f"[bold]{'hallucination' if value else 'truthful'}[/bold] "
            f"(reviewer={reviewer})",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# drift subcommands
# ---------------------------------------------------------------------------


@drift_app.command("check")
def drift_check_cmd(
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """Run drift detection across all verifiers and surface alerts."""
    detector = _get_drift_detector()
    reports = detector.check_all()

    if format.lower() == "json":
        console.print_json(
            data={k: r.to_dict() for k, r in reports.items()},
            default=_json_default,
        )
        return

    if not reports:
        console.print(
            Panel(
                "[green]No drift detected[/green] across "
                f"{len(detector.verifiers())} verifiers.",
                border_style="green",
            )
        )
        return

    table = Table(title="Drift alerts", header_style="bold red")
    table.add_column("Verifier")
    table.add_column("Baseline", justify="right")
    table.add_column("Window", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Level")
    table.add_column("Recommendation", overflow="fold")
    for name, r in reports.items():
        table.add_row(
            name,
            f"{r.baseline_accuracy:.3f}",
            f"{r.window_accuracy:.3f}",
            f"{r.delta:+.3f}",
            f"{r.p_value:.4f}",
            r.alert_level,
            r.recommendation,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# cost subcommands
# ---------------------------------------------------------------------------


@cost_app.command("report")
def cost_report_cmd(
    since: Optional[str] = typer.Option(
        None, "--since", help="Only include costs recorded since this ISO-8601 date."
    ),
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """Show cost totals by category."""
    tracker = _get_cost_tracker()
    report = tracker.get_total_cost()

    # --since is honoured for display only; the in-process tracker does
    # not persist history across invocations, so we record the filter in
    # the output for traceability.
    if format.lower() == "json":
        payload = report.to_dict()
        if since:
            payload["since_filter"] = since
        console.print_json(data=payload, default=_json_default)
        return

    table = Table(title="Cost report", header_style="bold magenta")
    table.add_column("Category", style="bold")
    table.add_column("USD", justify="right")
    table.add_row("compute", f"${report.compute_cost_usd:.6f}")
    table.add_row("api", f"${report.api_cost_usd:.6f}")
    table.add_row("storage", f"${report.storage_cost_usd:.6f}")
    table.add_row("[bold]total[/bold]", f"[bold]${report.total_cost_usd:.6f}[/bold]")
    console.print(table)

    if report.api_call_counts:
        api_table = Table(title="API calls by client", header_style="bold cyan")
        api_table.add_column("Client")
        api_table.add_column("Calls", justify="right")
        for client, calls in sorted(report.api_call_counts.items()):
            api_table.add_row(client, str(calls))
        console.print(api_table)

    if since:
        console.print(
            f"[dim]--since {since} applied to display only (tracker is process-local).[/dim]"
        )


# ---------------------------------------------------------------------------
# shadow subcommands
# ---------------------------------------------------------------------------


@shadow_app.command("compare")
def shadow_compare_cmd(
    champion: str = typer.Argument(..., help="Champion verifier name."),
    challenger: str = typer.Argument(..., help="Challenger verifier name."),
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """Run a champion/challenger shadow-mode comparison on the regression set."""
    settings = _build_settings()
    _load_plugins_safe(settings)

    from src.core.calibration import CalibrationTracker
    from src.mlops.shadow_mode import ShadowModeRouter
    from src.verifiers.base import get_verifier_registry

    registry = get_verifier_registry()
    try:
        champ_v = registry.get_by_name(champion)
    except KeyError:
        _fatal(f"Champion verifier {champion!r} is not registered.")
        return  # pragma: no cover
    try:
        chall_v = registry.get_by_name(challenger)
    except KeyError:
        _fatal(f"Challenger verifier {challenger!r} is not registered.")
        return  # pragma: no cover

    router = ShadowModeRouter(champion=champ_v, challenger=chall_v)

    # Use the bundled regression set for a quick comparison baseline.
    tracker = CalibrationTracker()
    claims = tracker.get_regression_set()

    async def _run() -> None:
        for c in claims:
            try:
                await router.verify(c.claim_text, c.claim_type, {})
            except Exception as exc:  # noqa: BLE001
                logger.debug("shadow verify failed: %s", exc)

    asyncio.run(_run())
    report = router.get_comparison_report()

    if format.lower() == "json":
        console.print_json(data=report.to_dict(), default=_json_default)
        return

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("champion:", report.champion_name)
    summary.add_row("challenger:", report.challenger_name)
    summary.add_row("total:", str(report.total_comparisons))
    summary.add_row("agreement:", f"{report.agreement_rate:.2%}")
    summary.add_row(
        "mean duration (ms):",
        f"champ={report.mean_champion_duration_ms:.1f} / "
        f"chall={report.mean_challenger_duration_ms:.1f}",
    )
    console.print(Panel(summary, title="Shadow comparison", border_style="cyan"))

    if report.confusion_matrix:
        conf_table = Table(
            title="Confusion (champion -> challenger)", header_style="bold magenta"
        )
        conf_table.add_column("Champion verdict")
        conf_table.add_column("Challenger verdict")
        conf_table.add_column("Count", justify="right")
        for (a, b), c in sorted(report.confusion_matrix.items()):
            conf_table.add_row(a, b, str(c))
        console.print(conf_table)


# ---------------------------------------------------------------------------
# auth subcommands
# ---------------------------------------------------------------------------


def _sync_session_factory() -> Any:
    """Return a blocking SQLAlchemy session for CLI commands.

    CLI commands are one-shot processes, so the overhead of asyncio
    is unnecessary — the sync engine keeps the implementation simple
    and makes error handling straightforward.
    """
    from sqlalchemy.orm import Session

    from src.db.session import get_sync_engine

    return Session(bind=get_sync_engine())


def _render_tenant_table(rows: list[dict[str, Any]]) -> None:
    """Print a rich table of tenant dicts produced by the auth CLI."""
    table = Table(title="Tenants", header_style="bold magenta")
    table.add_column("Slug")
    table.add_column("Name", overflow="fold")
    table.add_column("Active")
    table.add_column("RPM", justify="right")
    table.add_column("Analyze/PM", justify="right")
    table.add_column("ID", overflow="fold")
    for row in rows:
        table.add_row(
            row["slug"],
            row["name"],
            "yes" if row["is_active"] else "no",
            str(row["rate_limit_rpm"]),
            str(row["rate_limit_analyze_pm"]),
            row["id"],
        )
    console.print(table)


@auth_app.command("bootstrap")
def auth_bootstrap_cmd(
    tenant_name: str = typer.Option("Default", "--tenant-name"),
    tenant_slug: str = typer.Option("default", "--tenant-slug"),
    admin_email: str = typer.Option(..., "--admin-email"),
    key_name: str = typer.Option("bootstrap-admin", "--key-name"),
    bootstrap_key: Optional[str] = typer.Option(
        None,
        "--bootstrap-key",
        envvar="HAL_AUTH_BOOTSTRAP_KEY",
        help="Shared secret that must match settings.AUTH_BOOTSTRAP_KEY.",
    ),
) -> None:
    """Create the first tenant, admin user, and admin API key.

    Intended to be run exactly once during installation.  The bootstrap
    key protects this command from being re-run by unauthorised
    operators; remove the ``AUTH_BOOTSTRAP_KEY`` environment variable
    after success.
    """
    settings_obj = _build_settings()
    expected = getattr(settings_obj, "AUTH_BOOTSTRAP_KEY", None)
    if not expected:
        _fatal(
            "AUTH_BOOTSTRAP_KEY is not configured. Set it before running "
            "`hal-nemofinder auth bootstrap`."
        )
    if not bootstrap_key:
        _fatal("--bootstrap-key is required (or set HAL_AUTH_BOOTSTRAP_KEY).")
    import hmac as _hmac

    if not _hmac.compare_digest(expected, bootstrap_key):
        _fatal("Bootstrap key does not match AUTH_BOOTSTRAP_KEY.")

    from src.auth.api_keys import extract_prefix, generate_api_key
    from src.models.tenant import ApiKey, Role, Tenant, User

    session = _sync_session_factory()
    try:
        existing = (
            session.query(Tenant).filter(Tenant.slug == tenant_slug).one_or_none()
        )
        if existing is not None:
            _fatal(
                f"Tenant with slug {tenant_slug!r} already exists. "
                "Use `auth create-tenant` / `auth issue-key` instead."
            )

        tenant = Tenant(
            name=tenant_name,
            slug=tenant_slug,
            is_active=True,
        )
        session.add(tenant)
        session.flush()

        user = User(
            tenant_id=tenant.id,
            email=admin_email,
            role=Role.admin,
            is_active=True,
        )
        session.add(user)
        session.flush()

        plaintext, key_hash = generate_api_key()
        api_key = ApiKey(
            tenant_id=tenant.id,
            user_id=user.id,
            name=key_name,
            key_prefix=extract_prefix(plaintext),
            key_hash=key_hash,
            role=Role.admin,
        )
        session.add(api_key)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _fatal("Bootstrap failed.", exc)
    finally:
        session.close()

    panel = Panel(
        Text.assemble(
            ("Bootstrap complete.\n\n", "bold green"),
            ("Tenant: ", "bold"),
            (f"{tenant_name} ({tenant_slug})\n", "white"),
            ("Admin user: ", "bold"),
            (f"{admin_email}\n", "white"),
            ("API key (store now, shown once):\n\n", "bold yellow"),
            (plaintext + "\n", "bold cyan"),
        ),
        title="hal-nemofinder auth bootstrap",
        border_style="green",
    )
    console.print(panel)


@auth_app.command("create-tenant")
def auth_create_tenant_cmd(
    name: str = typer.Argument(..., help="Human-readable tenant name."),
    slug: Optional[str] = typer.Option(
        None, "--slug", help="Kebab-case slug. Derived from NAME if omitted."
    ),
) -> None:
    """Create a new tenant."""
    from src.models.tenant import Tenant

    slug_value = slug or name.strip().lower().replace(" ", "-")
    session = _sync_session_factory()
    try:
        if (
            session.query(Tenant).filter(Tenant.slug == slug_value).one_or_none()
            is not None
        ):
            _fatal(f"Tenant with slug {slug_value!r} already exists.")
        tenant = Tenant(name=name, slug=slug_value, is_active=True)
        session.add(tenant)
        session.commit()
        console.print(
            Panel(
                f"[green]Created tenant[/green] [bold]{name}[/bold] ({slug_value})",
                border_style="green",
            )
        )
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _fatal("Failed to create tenant.", exc)
    finally:
        session.close()


@auth_app.command("create-user")
def auth_create_user_cmd(
    email: str = typer.Argument(...),
    tenant: str = typer.Option(..., "--tenant", help="Tenant slug."),
    role: str = typer.Option("viewer", "--role"),
) -> None:
    """Create a user inside a tenant."""
    from src.models.tenant import Role, Tenant, User

    try:
        role_enum = Role(role)
    except ValueError:
        _fatal(
            f"Unknown --role {role!r}. Must be one of: admin | analyst | viewer."
        )

    session = _sync_session_factory()
    try:
        tenant_row = (
            session.query(Tenant).filter(Tenant.slug == tenant).one_or_none()
        )
        if tenant_row is None:
            _fatal(f"Tenant with slug {tenant!r} not found.")
        dup = (
            session.query(User)
            .filter(User.tenant_id == tenant_row.id, User.email == email)
            .one_or_none()
        )
        if dup is not None:
            _fatal(f"User {email!r} already exists in tenant {tenant!r}.")
        user = User(
            tenant_id=tenant_row.id,
            email=email,
            role=role_enum,
            is_active=True,
        )
        session.add(user)
        session.commit()
        console.print(
            Panel(
                f"[green]Created user[/green] [bold]{email}[/bold] "
                f"in tenant {tenant} as {role_enum.value}.",
                border_style="green",
            )
        )
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _fatal("Failed to create user.", exc)
    finally:
        session.close()


@auth_app.command("issue-key")
def auth_issue_key_cmd(
    name: str = typer.Argument(...),
    tenant: str = typer.Option(..., "--tenant"),
    user: Optional[str] = typer.Option(
        None, "--user", help="Bind to an existing user's e-mail."
    ),
    role: str = typer.Option("viewer", "--role"),
    expires_in_days: Optional[int] = typer.Option(
        None, "--expires-in-days", min=1
    ),
) -> None:
    """Issue a new API key for a tenant (and optionally a user)."""
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    from src.auth.api_keys import extract_prefix, generate_api_key
    from src.models.tenant import ApiKey, Role, Tenant, User

    try:
        role_enum = Role(role)
    except ValueError:
        _fatal(f"Unknown --role {role!r}.")

    session = _sync_session_factory()
    try:
        tenant_row = (
            session.query(Tenant).filter(Tenant.slug == tenant).one_or_none()
        )
        if tenant_row is None:
            _fatal(f"Tenant with slug {tenant!r} not found.")
        user_row = None
        if user is not None:
            user_row = (
                session.query(User)
                .filter(User.tenant_id == tenant_row.id, User.email == user)
                .one_or_none()
            )
            if user_row is None:
                _fatal(f"User {user!r} not found in tenant {tenant!r}.")

        plaintext, key_hash = generate_api_key()
        expires_at = (
            _dt.utcnow() + _td(days=expires_in_days) if expires_in_days else None
        )
        api_key = ApiKey(
            tenant_id=tenant_row.id,
            user_id=user_row.id if user_row is not None else None,
            name=name,
            key_prefix=extract_prefix(plaintext),
            key_hash=key_hash,
            role=role_enum,
            expires_at=expires_at,
        )
        session.add(api_key)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _fatal("Failed to issue API key.", exc)
    finally:
        session.close()

    console.print(
        Panel(
            Text.assemble(
                ("API key issued (store now, shown once):\n\n", "bold yellow"),
                (plaintext, "bold cyan"),
            ),
            title=f"auth issue-key — {name}",
            border_style="green",
        )
    )


@auth_app.command("list-tenants")
def auth_list_tenants_cmd(
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """Print every tenant in the database."""
    from src.models.tenant import Tenant

    session = _sync_session_factory()
    try:
        tenants = session.query(Tenant).order_by(Tenant.created_at.desc()).all()
    finally:
        session.close()

    rows = [
        {
            "id": str(t.id),
            "slug": t.slug,
            "name": t.name,
            "is_active": t.is_active,
            "rate_limit_rpm": t.rate_limit_rpm,
            "rate_limit_analyze_pm": t.rate_limit_analyze_pm,
        }
        for t in tenants
    ]
    if format.lower() == "json":
        console.print_json(data=rows, default=_json_default)
        return
    if not rows:
        console.print("[dim]No tenants found.[/dim]")
        return
    _render_tenant_table(rows)


@auth_app.command("list-keys")
def auth_list_keys_cmd(
    tenant: str = typer.Option(..., "--tenant"),
    format: str = typer.Option("table", "--format", "-f"),
) -> None:
    """List API keys for a tenant, without the plaintext."""
    from src.models.tenant import ApiKey, Tenant

    session = _sync_session_factory()
    try:
        tenant_row = (
            session.query(Tenant).filter(Tenant.slug == tenant).one_or_none()
        )
        if tenant_row is None:
            _fatal(f"Tenant with slug {tenant!r} not found.")
        keys = (
            session.query(ApiKey)
            .filter(ApiKey.tenant_id == tenant_row.id)
            .order_by(ApiKey.created_at.desc())
            .all()
        )
    finally:
        session.close()

    rows = [
        {
            "prefix": k.key_prefix,
            "name": k.name,
            "role": k.role.value,
            "revoked": k.revoked_at.isoformat() if k.revoked_at else None,
            "expires": k.expires_at.isoformat() if k.expires_at else None,
            "last_used": k.last_used_at.isoformat() if k.last_used_at else None,
        }
        for k in keys
    ]
    if format.lower() == "json":
        console.print_json(data=rows, default=_json_default)
        return
    if not rows:
        console.print("[dim]No API keys for this tenant.[/dim]")
        return
    table = Table(title=f"API keys for {tenant}", header_style="bold magenta")
    table.add_column("Prefix")
    table.add_column("Name", overflow="fold")
    table.add_column("Role")
    table.add_column("Expires")
    table.add_column("Revoked")
    table.add_column("Last used")
    for r in rows:
        table.add_row(
            r["prefix"],
            r["name"],
            r["role"],
            r["expires"] or "[dim]never[/dim]",
            r["revoked"] or "[dim]no[/dim]",
            r["last_used"] or "[dim]never[/dim]",
        )
    console.print(table)


@auth_app.command("revoke-key")
def auth_revoke_key_cmd(
    prefix: str = typer.Argument(
        ..., help="Key prefix returned by `list-keys` (e.g. 'hal_2Kc3u7gV')."
    ),
) -> None:
    """Revoke an API key identified by its prefix."""
    from datetime import datetime as _dt

    from src.models.tenant import ApiKey

    session = _sync_session_factory()
    try:
        matches = session.query(ApiKey).filter(ApiKey.key_prefix == prefix).all()
        if not matches:
            _fatal(f"No API key found with prefix {prefix!r}.")
        if len(matches) > 1:
            _fatal(
                f"Prefix {prefix!r} is ambiguous — {len(matches)} keys match. "
                "Use a longer prefix or revoke via the API."
            )
        api_key = matches[0]
        if api_key.revoked_at is not None:
            console.print(
                Panel(
                    f"[yellow]Key {prefix} was already revoked at "
                    f"{api_key.revoked_at.isoformat()}.[/yellow]",
                    border_style="yellow",
                )
            )
            return
        api_key.revoked_at = _dt.utcnow()
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        _fatal("Failed to revoke key.", exc)
    finally:
        session.close()

    console.print(
        Panel(
            f"[green]Revoked API key[/green] [bold]{prefix}[/bold]",
            border_style="green",
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()

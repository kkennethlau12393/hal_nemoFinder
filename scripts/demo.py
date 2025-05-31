#!/usr/bin/env python3
"""End-to-end demo for hal_nemoFinder hallucination detection.

Loads sample LLM outputs, submits them to the API for analysis,
polls until completion, and pretty-prints colour-coded results.

Usage:
    python scripts/demo.py
    python scripts/demo.py --url http://my-server:9000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# ANSI colour helpers (no external deps)
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

VERDICT_STYLE = {
    "verified": (GREEN, "VERIFIED"),
    "refuted": (RED, "REFUTED"),
    "partially_supported": (YELLOW, "PARTIAL"),
    "unverifiable": (GRAY, "UNVERIFIABLE"),
}

SEVERITY_STYLE = {
    "clean": (GREEN, "CLEAN"),
    "minor": (YELLOW, "MINOR"),
    "major": (RED, "MAJOR"),
    "critical": (f"{BOLD}{BG_RED}{WHITE}", "CRITICAL"),
}


def c(colour: str, text: str) -> str:
    """Wrap *text* in ANSI colour codes."""
    return f"{colour}{text}{RESET}"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def http_post(url: str, payload: dict) -> dict:
    """POST JSON to *url* and return parsed response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def http_get(url: str) -> dict:
    """GET *url* and return parsed JSON response."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------

POLL_INTERVAL = 2.0  # seconds
POLL_TIMEOUT = 120.0  # seconds

TERMINAL_STATUSES = {"completed", "failed"}


def submit_analysis(base_url: str, text: str) -> str:
    """Submit text for analysis and return the job_id."""
    url = f"{base_url}/api/v1/analyze"
    resp = http_post(url, {"text": text})
    return resp["job_id"]


def poll_job(base_url: str, job_id: str) -> dict:
    """Poll GET /api/v1/jobs/{job_id} until terminal status or timeout."""
    url = f"{base_url}/api/v1/jobs/{job_id}"
    start = time.monotonic()
    last_status = ""
    while True:
        elapsed = time.monotonic() - start
        if elapsed > POLL_TIMEOUT:
            raise TimeoutError(
                f"Job {job_id} did not complete within {POLL_TIMEOUT}s"
            )
        job = http_get(url)
        status = job.get("status", "unknown")
        if status != last_status:
            print(f"    {c(DIM, 'status:')} {status} ({elapsed:.1f}s)")
            last_status = status
        if status in TERMINAL_STATUSES:
            return job
        time.sleep(POLL_INTERVAL)


def fetch_report(base_url: str, job_id: str) -> dict:
    """Fetch the full report for a completed job."""
    url = f"{base_url}/api/v1/jobs/{job_id}/report"
    return http_get(url)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_banner() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║    ██╗  ██╗ █████╗ ██╗         ███╗   ██╗███████╗███╗   ███╗  ║
    ║    ██║  ██║██╔══██╗██║         ████╗  ██║██╔════╝████╗ ████║  ║
    ║    ███████║███████║██║         ██╔██╗ ██║█████╗  ██╔████╔██║  ║
    ║    ██╔══██║██╔══██║██║         ██║╚██╗██║██╔══╝  ██║╚██╔╝██║  ║
    ║    ██║  ██║██║  ██║███████╗    ██║ ╚████║███████╗██║ ╚═╝ ██║  ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝  ║
    ║                                                                ║
    ║           nemoFinder  --  Hallucination Detection Demo         ║
    ║             AI Drug Discovery Output Verification              ║
    ╚══════════════════════════════════════════════════════════════════╝
"""
    print(c(CYAN, banner))


def print_separator(char: str = "─", width: int = 72) -> None:
    print(c(DIM, char * width))


def print_verdict_badge(verdict: str) -> str:
    colour, label = VERDICT_STYLE.get(verdict, (GRAY, verdict.upper()))
    return c(colour, f"[{label}]")


def print_severity_badge(severity: str) -> str:
    colour, label = SEVERITY_STYLE.get(severity, (GRAY, severity.upper()))
    return c(colour, f" {label} ")


def print_report(report: dict, sample: dict) -> None:
    """Pretty-print a single analysis report."""
    overall = report.get("overall_score", 0.0)
    severity = report.get("severity", "unknown")
    claims = report.get("claims", [])
    verified = report.get("verified_count", 0)
    refuted = report.get("refuted_count", 0)
    partial = report.get("partial_count", 0)
    unverifiable = report.get("unverifiable_count", 0)
    claim_count = report.get("claim_count", 0)

    # Score colour
    if overall >= 0.8:
        score_colour = RED
    elif overall >= 0.4:
        score_colour = YELLOW
    else:
        score_colour = GREEN

    print()
    print(f"  {c(BOLD, 'Hallucination Score:')}  {c(score_colour, f'{overall:.2f}')}")
    print(f"  {c(BOLD, 'Severity:')}             {print_severity_badge(severity)}")
    print(
        f"  {c(BOLD, 'Claims:')}               "
        f"{claim_count} total  |  "
        f"{c(GREEN, str(verified))} verified  |  "
        f"{c(RED, str(refuted))} refuted  |  "
        f"{c(YELLOW, str(partial))} partial  |  "
        f"{c(GRAY, str(unverifiable))} unverifiable"
    )

    if claims:
        print()
        print(f"  {c(BOLD, 'Claim Details:')}")
        for i, claim in enumerate(claims, 1):
            claim_text = claim.get("claim_text", "")
            claim_type = claim.get("claim_type", "")
            verdicts = claim.get("verdicts", [])

            # Truncate long claim text
            display_text = claim_text
            if len(display_text) > 80:
                display_text = display_text[:77] + "..."

            print()
            print(f"    {c(BOLD, f'Claim {i}:')} {c(DIM, f'[{claim_type}]')}")
            print(f"      {display_text}")

            for v in verdicts:
                verdict = v.get("verdict", "unknown")
                confidence = v.get("confidence", 0.0)
                reasoning = v.get("reasoning", "")
                verifier = v.get("verifier_name", "")

                badge = print_verdict_badge(verdict)
                conf_bar = render_confidence_bar(confidence)

                print(
                    f"      {badge}  {conf_bar}  "
                    f"{c(DIM, verifier)}"
                )
                if reasoning:
                    # Wrap reasoning at ~60 chars
                    for line in _wrap(reasoning, 60):
                        print(f"        {c(DIM, line)}")

    # Known hallucinations comparison
    known = sample.get("known_hallucinations", [])
    if known:
        print()
        print(f"  {c(BOLD, 'Known Embedded Hallucinations')} {c(DIM, '(ground truth):')}")
        for k in known:
            sev = k.get("severity", "")
            sev_colour, sev_label = SEVERITY_STYLE.get(sev, (GRAY, sev.upper()))
            print(
                f"    {c(sev_colour, f'[{sev_label}]')}  "
                f"{c(DIM, k.get('type', ''))}: {k.get('description', '')}"
            )


def render_confidence_bar(confidence: float, width: int = 10) -> str:
    """Render a tiny ASCII confidence bar."""
    filled = int(confidence * width)
    empty = width - filled
    if confidence >= 0.8:
        colour = GREEN
    elif confidence >= 0.5:
        colour = YELLOW
    else:
        colour = RED
    bar = c(colour, "█" * filled) + c(DIM, "░" * empty)
    return f"{bar} {confidence:.0%}"


def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap without textwrap (keeps it dependency-free)."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def print_summary_table(results: list[dict]) -> None:
    """Print a final summary table of all analyses."""
    print()
    print()
    print(c(BOLD + CYAN, "  ╔══════════════════════════════════════════════════════════════╗"))
    print(c(BOLD + CYAN, "  ║               ANALYSIS SUMMARY TABLE                        ║"))
    print(c(BOLD + CYAN, "  ╚══════════════════════════════════════════════════════════════╝"))
    print()

    # Header
    header = (
        f"  {c(BOLD, 'Sample'):40s}"
        f"{c(BOLD, 'Score'):>18s}"
        f"  {c(BOLD, 'Severity'):>18s}"
        f"  {c(BOLD, 'V'):>6s}"
        f"  {c(BOLD, 'R'):>6s}"
        f"  {c(BOLD, 'P'):>6s}"
        f"  {c(BOLD, 'U'):>6s}"
        f"  {c(BOLD, 'Claims'):>8s}"
    )
    print(header)
    print_separator("─", 90)

    for r in results:
        title = r["title"]
        if len(title) > 30:
            title = title[:27] + "..."

        score = r.get("overall_score", 0.0)
        severity = r.get("severity", "?")
        if score >= 0.8:
            sc = c(RED, f"{score:.2f}")
        elif score >= 0.4:
            sc = c(YELLOW, f"{score:.2f}")
        else:
            sc = c(GREEN, f"{score:.2f}")

        sev_colour, sev_label = SEVERITY_STYLE.get(severity, (GRAY, severity.upper()))

        print(
            f"  {title:30s}"
            f"    {sc:>8s}"
            f"      {c(sev_colour, sev_label):>8s}"
            f"    {c(GREEN, str(r.get('verified_count', 0))):>4s}"
            f"    {c(RED, str(r.get('refuted_count', 0))):>4s}"
            f"    {c(YELLOW, str(r.get('partial_count', 0))):>4s}"
            f"    {c(GRAY, str(r.get('unverifiable_count', 0))):>4s}"
            f"    {str(r.get('claim_count', 0)):>4s}"
        )

    print_separator("─", 90)
    print()

    total_claims = sum(r.get("claim_count", 0) for r in results)
    total_refuted = sum(r.get("refuted_count", 0) for r in results)
    avg_score = (
        sum(r.get("overall_score", 0.0) for r in results) / len(results)
        if results
        else 0.0
    )

    print(f"  {c(BOLD, 'Totals:')}")
    print(f"    Samples analysed     : {len(results)}")
    print(f"    Total claims checked : {total_claims}")
    print(f"    Total refuted claims : {c(RED, str(total_refuted))}")
    print(f"    Average hal. score   : {c(YELLOW, f'{avg_score:.2f}')}")
    print()
    legend = (
        f"  {c(DIM, 'Legend:')}  "
        f"V={c(GREEN, 'Verified')}  "
        f"R={c(RED, 'Refuted')}  "
        f"P={c(YELLOW, 'Partial')}  "
        f"U={c(GRAY, 'Unverifiable')}"
    )
    print(legend)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_samples() -> list[dict]:
    """Load sample LLM outputs from seed/sample_llm_outputs.json."""
    # Try several relative paths from likely working directories
    candidates = [
        Path(__file__).resolve().parent.parent / "seed" / "sample_llm_outputs.json",
        Path.cwd() / "seed" / "sample_llm_outputs.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    print(c(RED, f"ERROR: Cannot find seed/sample_llm_outputs.json"))
    print(c(DIM, f"  Searched: {[str(p) for p in candidates]}"))
    sys.exit(1)


def run_demo(base_url: str) -> None:
    print_banner()

    samples = load_samples()
    print(f"  {c(BOLD, 'Loaded')} {len(samples)} sample LLM outputs for analysis")
    print(f"  {c(BOLD, 'Target API:')} {base_url}")
    print()

    # Check health
    try:
        health = http_get(f"{base_url}/api/v1/health")
        db_ok = health.get("database", False)
        redis_ok = health.get("redis", False)
        print(
            f"  {c(BOLD, 'API Health:')} "
            f"status={c(GREEN if health.get('status') == 'ok' else RED, health.get('status', '?'))}  "
            f"db={c(GREEN if db_ok else RED, str(db_ok))}  "
            f"redis={c(GREEN if redis_ok else RED, str(redis_ok))}"
        )
        verifiers = health.get("verifiers", {})
        if verifiers:
            v_str = "  ".join(
                f"{name}={c(GREEN if ok else RED, str(ok))}"
                for name, ok in verifiers.items()
            )
            print(f"  {c(BOLD, 'Verifiers:')}  {v_str}")
        print()
    except Exception as exc:
        print(c(YELLOW, f"  WARNING: Health check failed: {exc}"))
        print(c(YELLOW, "  Continuing anyway -- the API may still accept requests.\n"))

    results: list[dict] = []

    for idx, sample in enumerate(samples, 1):
        title = sample.get("title", f"Sample {sample.get('id', idx)}")
        print_separator("═")
        print(
            f"\n  {c(BOLD + MAGENTA, f'[{idx}/{len(samples)}]')}  "
            f"{c(BOLD, title)}\n"
        )

        # Show a preview of the text
        text = sample["text"]
        preview = text[:120].replace("\n", " ")
        if len(text) > 120:
            preview += "..."
        print(f"  {c(DIM, preview)}\n")

        # Submit
        try:
            print(f"  {c(BLUE, 'Submitting to API...')}")
            job_id = submit_analysis(base_url, text)
            print(f"  {c(BOLD, 'Job ID:')} {job_id}")
        except urllib.error.URLError as exc:
            print(c(RED, f"\n  ERROR: Could not connect to API at {base_url}"))
            print(c(RED, f"  {exc}"))
            print(c(DIM, "\n  Make sure the API server is running:"))
            print(c(DIM, "    docker compose up -d"))
            print(c(DIM, "    # or: uvicorn src.main:app --port 8000\n"))
            sys.exit(1)
        except Exception as exc:
            print(c(RED, f"  ERROR submitting sample: {exc}"))
            continue

        # Poll
        try:
            print(f"  {c(BLUE, 'Polling for completion...')}")
            job = poll_job(base_url, job_id)
            status = job.get("status", "unknown")

            if status == "failed":
                error = job.get("error_message", "Unknown error")
                print(c(RED, f"\n  Job FAILED: {error}\n"))
                continue

            print(f"  {c(GREEN, 'Job completed.')}")
        except TimeoutError as exc:
            print(c(RED, f"  TIMEOUT: {exc}"))
            continue
        except Exception as exc:
            print(c(RED, f"  ERROR polling job: {exc}"))
            continue

        # Fetch report
        try:
            report = fetch_report(base_url, job_id)
            report["title"] = title
            results.append(report)
            print_report(report, sample)
        except Exception as exc:
            print(c(RED, f"  ERROR fetching report: {exc}"))
            continue

        print()

    # Summary
    if results:
        print_separator("═")
        print_summary_table(results)
    else:
        print(c(RED, "\n  No results to summarise.\n"))

    print(c(BOLD + CYAN, "  Demo complete.\n"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hal_nemoFinder end-to-end demo",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the hal_nemoFinder API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    # Strip trailing slash
    base_url = args.url.rstrip("/")
    run_demo(base_url)


if __name__ == "__main__":
    main()

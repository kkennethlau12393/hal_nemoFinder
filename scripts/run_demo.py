"""End-to-end demo of hal_nemoFinder.

Showcases:
* Claim extraction + classification on real LLM-style outputs
* All 8 verifiers (chemical, citation, clinical, pharmacokinetic, target,
  consistency, statistical, pathway)
* Bayesian aggregation across verifiers
* Calibration metrics on a labeled regression set
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.aggregator import BayesianAggregator, ClaimInfo
from src.core.calibration import (
    CalibrationTracker,
    run_regression_test,
    _REGRESSION_SET,
)
from src.core.claim_classifier import ClaimClassifier
from src.core.claim_extractor import ClaimExtractor
from src.core.router import VerificationRouter
from src.models.enums import ClaimType, Verdict
from src.verifiers.base import get_verifier_registry


# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
GRAY = "\033[90m"
CYAN = "\033[96m"
RED_BOLD = "\033[91;1m"
GREEN_BOLD = "\033[92;1m"

VERDICT_COLORS = {
    "verified": GREEN,
    "refuted": RED,
    "partially_supported": YELLOW,
    "unverifiable": GRAY,
}
SEVERITY_COLORS = {
    "clean": GREEN,
    "minor": YELLOW,
    "major": RED,
    "critical": RED_BOLD,
}


def banner(text: str, char: str = "=") -> None:
    line = char * 80
    print(f"\n{BOLD}{line}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{line}{RESET}")


def bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


async def analyze_sample(
    sample,
    extractor,
    classifier,
    router,
    aggregator,
    calibration_tracker,
):
    banner(f"SAMPLE: {sample['title']}")
    text_preview = sample["text"][:250].replace("\n", " ")
    print(f"\n  Input (preview):\n  {GRAY}{text_preview}...{RESET}\n")

    claims = extractor.extract(sample["text"])
    print(f"  {BOLD}Extracted {len(claims)} verifiable claims{RESET}\n")

    all_verdicts = []

    for i, claim in enumerate(claims, 1):
        claim_type = classifier.classify(claim)
        print(f"  {BOLD}Claim {i}{RESET} {CYAN}[{claim_type.value}]{RESET}")
        ct = claim.claim_text
        if len(ct) > 100:
            ct = ct[:100] + "..."
        print(f'    "{ct}"')

        results = await router.verify_claim(claim.claim_text, claim_type, {})

        for r in results:
            color = VERDICT_COLORS.get(r.verdict.value, "")
            reasoning_short = (r.reasoning or "")[:65]
            src = r.source_db or "?"
            print(
                f"    {color}[{r.verdict.value:^22}]{RESET} "
                f"conf={r.confidence:.2f} | {src:14s} | {reasoning_short}"
            )

        # Bayesian aggregation per claim
        if results:
            agg_verdict = aggregator.aggregate_claim_bayesian(results)
            posterior = aggregator.posterior_probability(results)
            color = VERDICT_COLORS.get(agg_verdict.verdict.value, "")
            print(
                f"    {BOLD}→ Bayesian:{RESET} "
                f"{color}{agg_verdict.verdict.value}{RESET} "
                f"(P(hallucination)={posterior:.2f}, conf={agg_verdict.confidence:.2f})"
            )
            all_verdicts.append(agg_verdict)

            # Track for calibration
            calibration_tracker.record_prediction(
                claim_text=claim.claim_text,
                verdict=agg_verdict.verdict,
                confidence=agg_verdict.confidence,
                posterior_prob=posterior,
            )
        print()

    # Sample-level summary
    if all_verdicts:
        v_count = sum(1 for v in all_verdicts if v.verdict == Verdict.verified)
        r_count = sum(1 for v in all_verdicts if v.verdict == Verdict.refuted)
        p_count = sum(
            1 for v in all_verdicts if v.verdict == Verdict.partially_supported
        )
        u_count = sum(1 for v in all_verdicts if v.verdict == Verdict.unverifiable)

        # Hallucination score = fraction refuted, weighted by confidence
        score = (
            sum(v.confidence for v in all_verdicts if v.verdict == Verdict.refuted)
            / len(all_verdicts)
            if all_verdicts
            else 0
        )
        if score > 0.6:
            severity = "critical"
        elif score > 0.3:
            severity = "major"
        elif score > 0.1:
            severity = "minor"
        else:
            severity = "clean"
        sev_color = SEVERITY_COLORS.get(severity, "")

        print(f"  {BOLD}--- SAMPLE REPORT ---{RESET}")
        print(
            f"  Hallucination Score: {BOLD}{score:.2f}{RESET} / 1.00  "
            f"{bar(score)}"
        )
        print(f"  Severity:            {sev_color}{BOLD}{severity.upper()}{RESET}")
        print(
            f"  Claims: {len(all_verdicts)} | "
            f"{GREEN}{v_count} verified{RESET} | "
            f"{RED}{r_count} refuted{RESET} | "
            f"{YELLOW}{p_count} partial{RESET} | "
            f"{GRAY}{u_count} unverifiable{RESET}"
        )

        known = sample.get("known_hallucinations", [])
        if known:
            print(f"\n  {BOLD}Known ground-truth hallucinations:{RESET}")
            for h in known[:3]:  # cap at 3
                desc = h.get("description", h.get("type", str(h)))[:90]
                sev = h.get("severity", "?")
                print(f"    - [{sev:8s}] {desc}")
            if len(known) > 3:
                print(f"    ... and {len(known) - 3} more")

        return (sample["title"], score, severity, v_count, r_count, p_count, u_count)
    return None


async def run_calibration() -> None:
    """Run the framework against the bundled regression set and report metrics."""
    banner("CALIBRATION: regression set evaluation", char="=")
    print(
        f"\n  Running {len(_REGRESSION_SET)} labeled claims through the full pipeline\n"
        f"  (10 known hallucinations + 10 truthful claims)\n"
    )

    router = VerificationRouter(default_timeout=15)
    aggregator = BayesianAggregator()
    tracker = CalibrationTracker()

    metrics = await run_regression_test(router, aggregator, tracker=tracker)

    print(f"  {BOLD}Classification metrics:{RESET}")
    print(f"    Accuracy:  {metrics.accuracy:.2%}")
    print(f"    Precision: {metrics.precision:.2%}  (refuted-as-positive)")
    print(f"    Recall:    {metrics.recall:.2%}")
    print(f"    F1 score:  {metrics.f1:.2%}")
    print(f"\n  {BOLD}Calibration metrics:{RESET}")
    print(f"    Brier score: {metrics.brier_score:.4f}  (lower is better, 0 = perfect)")
    print(f"    ECE:         {metrics.ece:.4f}  (expected calibration error)")
    print(f"    Records:     {metrics.n_records}")

    print(f"\n  {BOLD}Reliability diagram (predicted vs actual hallucination rate):{RESET}")
    print(f"  {DIM}Each bin shows: confidence range → mean predicted vs mean actual ({DIM}n=count){RESET}")
    for lo, hi, pred, actual, count in metrics.bin_data:
        if count == 0:
            continue
        diff = pred - actual
        flag = ""
        if abs(diff) > 0.2:
            flag = f" {RED}(miscalibrated){RESET}"
        elif abs(diff) > 0.1:
            flag = f" {YELLOW}(slightly off){RESET}"
        print(
            f"    {lo:.1f}-{hi:.1f}: predicted {pred:.2f}, actual {actual:.2f}  "
            f"{DIM}(n={count}){RESET}{flag}"
        )


async def main() -> None:
    print(f"\n{BOLD}{GREEN_BOLD}")
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║         hal_nemoFinder — Tier 1-4 Capability Demo               ║")
    print("  ║         Deterministic Hallucination Detection                   ║")
    print("  ║         for AI-Driven Drug Discovery                            ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print(RESET)

    # Show what verifiers are loaded
    registry = get_verifier_registry()
    print(f"\n  {BOLD}{len(registry)} deterministic verifiers loaded:{RESET}")
    for v in registry.get_all():
        types = ", ".join(t.value for t in v.supported_claim_types[:3])
        if len(v.supported_claim_types) > 3:
            types += f", +{len(v.supported_claim_types) - 3} more"
        print(f"    {CYAN}•{RESET} {v.name:20s} → {types}")

    print(
        f"\n  {BOLD}Aggregation:{RESET} BayesianAggregator "
        f"(per-verifier sensitivity/specificity → posterior P(hallucination))"
    )
    print(
        f"  {BOLD}Calibration:{RESET} {len(_REGRESSION_SET)}-claim labeled regression set"
    )

    # Load samples
    seed_path = (
        Path(__file__).resolve().parent.parent / "seed" / "sample_llm_outputs.json"
    )
    with open(seed_path) as f:
        samples = json.load(f)

    extractor = ClaimExtractor()
    classifier = ClaimClassifier()
    router = VerificationRouter(default_timeout=15)
    aggregator = BayesianAggregator()
    tracker = CalibrationTracker()

    reports = []
    for sample in samples:
        result = await analyze_sample(
            sample, extractor, classifier, router, aggregator, tracker
        )
        if result:
            reports.append(result)

    # Summary table
    banner("SAMPLE SUMMARY")
    print(f"  {'Sample':<42s} {'Score':>6s} {'Severity':>10s} {'V':>3s} {'R':>3s} {'P':>3s} {'U':>3s}")
    print(f"  {'-'*42} {'-'*6} {'-'*10} {'-'*3} {'-'*3} {'-'*3} {'-'*3}")
    for title, score, severity, vc, rc, pc, uc in reports:
        sev_color = SEVERITY_COLORS.get(severity, "")
        title_short = title[:42]
        print(
            f"  {title_short:<42s} {score:>6.2f} "
            f"{sev_color}{severity:>10s}{RESET} "
            f"{vc:>3d} {rc:>3d} {pc:>3d} {uc:>3d}"
        )

    total_claims = sum(vc + rc + pc + uc for _, _, _, vc, rc, pc, uc in reports)
    total_refuted = sum(rc for _, _, _, _, rc, _, _ in reports)
    avg_score = sum(s for _, s, _, _, _, _, _ in reports) / len(reports)
    print(
        f"\n  {BOLD}Totals:{RESET} {total_claims} claims analyzed, "
        f"{total_refuted} refuted by Bayesian aggregator"
    )
    print(f"  {BOLD}Average hallucination score:{RESET} {avg_score:.2f}")

    # Calibration on regression set
    await run_calibration()

    print(f"\n{GREEN_BOLD}  ✓ Demo complete{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())

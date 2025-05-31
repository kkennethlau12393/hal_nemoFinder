"""Showcase of the enhanced regex patterns and fabricated-protein detector."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verifiers.statistical import StatisticalVerifier
from src.verifiers.pharmacokinetic import PharmacokineticVerifier
from src.verifiers.target import TargetVerifier
from src.verifiers.pathway import PathwayVerifier
from src.verifiers.chemical import ChemicalVerifier
from src.core.aggregator import BayesianAggregator
from src.core.router import VerificationRouter
from src.models.enums import ClaimType


B = "\033[1m"
R = "\033[0m"
G = "\033[92m"
RD = "\033[91m"
Y = "\033[93m"
GR = "\033[90m"
C = "\033[96m"
M = "\033[95m"

VC = {
    "verified": G,
    "refuted": RD,
    "partially_supported": Y,
    "unverifiable": GR,
}


def section(title: str) -> None:
    print(f"\n{B}{C}{'═' * 78}{R}")
    print(f"{B}{C}  {title}{R}")
    print(f"{C}{'─' * 78}{R}")


def show(label: str, text: str, result) -> None:
    color = VC.get(result.verdict.value, "")
    reasoning = (result.reasoning or "")[:90]
    print(f"\n  {GR}{label}{R}")
    print(f"  {GR}{text!r}{R}")
    print(
        f"  {color}{result.verdict.value:^22}{R} "
        f"conf={result.confidence:.2f}  {reasoning}"
    )


async def main() -> None:
    print(f"\n{B}{M}╔══════════════════════════════════════════════════════════════════════════╗{R}")
    print(f"{B}{M}║          hal_nemoFinder — Enhanced Pattern Showcase                     ║{R}")
    print(f"{B}{M}║          New regex coverage + fabricated-protein detector               ║{R}")
    print(f"{B}{M}╚══════════════════════════════════════════════════════════════════════════╝{R}")

    stat = StatisticalVerifier()
    pk = PharmacokineticVerifier()
    tgt = TargetVerifier()
    path = PathwayVerifier()
    chem = ChemicalVerifier()
    bayes = BayesianAggregator()
    router = VerificationRouter(default_timeout=15)

    # ──────────────────────────────────────────────────────────────────────
    section("ENHANCED STATISTICAL PATTERNS — multiple AE phrasings")
    # ──────────────────────────────────────────────────────────────────────

    cases = [
        ("Inverted AE phrasing",
         "0 of 18000 patients experienced any side effects."),
        ("'None of the' phrasing",
         "None of the 25,000 enrolled patients reported adverse events."),
        ("Patients-first phrasing",
         "12,500 patients were treated; none developed any adverse events."),
        ("Treated/none-experienced phrasing",
         "15,000 subjects were exposed to the drug, none experienced AEs."),
        ("Rephrased zero AEs",
         "Among 20,000 individuals, no patient experienced any side effects."),
    ]
    for label, text in cases:
        r = await stat.verify(text, ClaimType.clinical_outcome, {})
        show(label, text, r)

    # ──────────────────────────────────────────────────────────────────────
    section("ENHANCED CI PATTERNS — flexible connector phrasings")
    # ──────────────────────────────────────────────────────────────────────

    cases = [
        ("CI with 'was' connector",
         "Hazard ratio was 0.5 (95% CI: 0.7 to 0.9, p < 0.001)."),
        ("CI with 'with a' connector",
         "HR was 0.5 with a 95% CI of 0.7 to 0.9."),
        ("CI with 'of' connector",
         "The odds ratio of 2.5 (95% CI: 0.8-1.5) was reported."),
    ]
    for label, text in cases:
        r = await stat.verify(text, ClaimType.clinical_outcome, {})
        show(label, text, r)

    # ──────────────────────────────────────────────────────────────────────
    section("ENHANCED PK PATTERNS — varied phrasings")
    # ──────────────────────────────────────────────────────────────────────

    cases = [
        ("Bioavailability as fraction (F=0.65)",
         "The drug shows F = 0.65 in healthy volunteers."),
        ("Oral bioavailability with adjective",
         "Achieves 99.7% oral bioavailability in humans."),
        ("Half-life with 'terminal' qualifier",
         "Terminal half-life of 380 hours in plasma."),
        ("Plasma half-life",
         "The plasma half-life is approximately 24 hours."),
        ("Protein binding",
         "The drug is 99% protein bound in plasma."),
        ("Multiple PK params",
         "Cmax of 850 ng/mL was reached at Tmax of 2.5 hours, with AUC of 4200 ng·h/mL."),
    ]
    for label, text in cases:
        r = await pk.verify(text, ClaimType.pharmacokinetic, {})
        show(label, text, r)

    # ──────────────────────────────────────────────────────────────────────
    section("FABRICATED PROTEIN DETECTOR — known family bounds")
    # ──────────────────────────────────────────────────────────────────────

    cases = [
        ("Fabricated BRCA9", "The novel inhibitor targets the BRCA9 kinase."),
        ("Fabricated CASP15", "The compound activates CASP15 leading to apoptosis."),
        ("Fabricated CDK99", "The drug inhibits CDK99 with high selectivity."),
        ("Fabricated STAT9", "Phosphorylation of STAT9 was observed downstream."),
        ("Real BRCA1", "The drug targets BRCA1 in DNA damage response."),
        ("Real CASP3", "Caspase activity assay measured CASP3 cleavage."),
        ("Fabricated ERBB7", "The compound binds the ERBB7 receptor with high affinity."),
        ("Real ERBB2 (HER2)", "Trastuzumab binds the ERBB2 receptor."),
    ]
    for label, text in cases:
        r = await tgt.verify(text, ClaimType.target_interaction, {})
        show(label, text, r)

    # ──────────────────────────────────────────────────────────────────────
    section("FULL PIPELINE — fabricated drug claim with multiple violations")
    # ──────────────────────────────────────────────────────────────────────

    print(f"\n  {GR}Claim: 'The novel compound NX-9001 (CC(=O)[GH]) targets BRCA9 kinase{R}")
    print(f"  {GR}with IC50 of 0.05 nM, achieves F = 1.5 oral bioavailability,{R}")
    print(f"  {GR}reduced HbA1c by 5.3%, and 0 of 22000 patients experienced any{R}")
    print(f"  {GR}adverse events.'{R}\n")

    text = (
        "The novel compound NX-9001 (CC(=O)[GH]) targets BRCA9 kinase with "
        "IC50 of 0.05 nM, achieves F = 1.5 oral bioavailability, reduced "
        "HbA1c by 5.3%, and 0 of 22000 patients experienced any adverse events."
    )

    # Run through every applicable verifier manually
    verifiers_for_claim = [
        ("chemical", chem, ClaimType.molecular_property),
        ("target", tgt, ClaimType.target_interaction),
        ("pharmacokinetic", pk, ClaimType.pharmacokinetic),
        ("statistical", stat, ClaimType.clinical_outcome),
    ]
    all_results = []
    for name, v, ct in verifiers_for_claim:
        r = await v.verify(text, ct, {})
        all_results.append(r)
        color = VC.get(r.verdict.value, "")
        reasoning = (r.reasoning or "")[:80]
        print(f"  {color}{r.verdict.value:^22}{R} "
              f"{B}{name:16s}{R} conf={r.confidence:.2f}  {reasoning}")

    # Bayesian aggregation
    agg = bayes.aggregate_claim_bayesian(all_results)
    posterior = bayes.posterior_probability(all_results)
    color = VC[agg.verdict.value]
    print(f"\n  {B}→ Bayesian posterior P(hallucination) = {RD if posterior > 0.5 else G}{B}{posterior:.3f}{R}")
    print(f"  {B}→ Final verdict: {color}{B}{agg.verdict.value.upper()}{R} "
          f"(decisiveness conf={agg.confidence:.2f})")

    print(f"\n{G}{B}━━━ END OF ENHANCED SHOWCASE ━━━{R}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""Capability showcase — exercises every tier of hal_nemoFinder with diverse queries."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.aggregator import BayesianAggregator
from src.core.router import VerificationRouter
from src.models.enums import ClaimType, Verdict
from src.verifiers.chemical import ChemicalVerifier
from src.verifiers.statistical import StatisticalVerifier
from src.verifiers.pathway import PathwayVerifier
from src.verifiers.pharmacokinetic import PharmacokineticVerifier
from src.verifiers.target import TargetVerifier
from src.verifiers.citation import CitationVerifier


# ANSI
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


def section(num: str, title: str) -> None:
    print(f"\n{B}{C}══ TEST {num} — {title}{R}")
    print(f"{C}{'─' * 72}{R}")


def query(label: str, text: str) -> None:
    print(f"\n{B}Query:{R} {GR}{label}{R}")
    print(f"  {GR}{text!r}{R}")


def show_result(verifier_name: str, result) -> None:
    color = VC.get(result.verdict.value, "")
    reasoning = (result.reasoning or "")[:120]
    print(
        f"  {color}{result.verdict.value:^22}{R} "
        f"conf={result.confidence:.2f}  "
        f"{B}{verifier_name:14s}{R}  {reasoning}"
    )


async def main() -> None:
    print(f"\n{B}{M}╔══════════════════════════════════════════════════════════════════════╗{R}")
    print(f"{B}{M}║          hal_nemoFinder — Full Capability Showcase                  ║{R}")
    print(f"{B}{M}║          Diverse queries across all 8 verifiers + Bayesian agg      ║{R}")
    print(f"{B}{M}╚══════════════════════════════════════════════════════════════════════╝{R}")

    chem = ChemicalVerifier()
    stat = StatisticalVerifier()
    path = PathwayVerifier()
    pk = PharmacokineticVerifier()
    target = TargetVerifier()
    cite = CitationVerifier()
    router = VerificationRouter(default_timeout=15)
    bayes = BayesianAggregator()

    # ──────────────────────────────────────────────────────────────────────
    section("1", "TIER 1.1 — Cheminformatics: ground-truth structural checks")
    # ──────────────────────────────────────────────────────────────────────

    query("Valid SMILES, correct MW (aspirin)",
          "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has MW 180.16 g/mol.")
    r = await chem.verify(
        "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has MW 180.16 g/mol.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)

    query("Valid SMILES, wrong MW",
          "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has MW 220.3 g/mol.")
    r = await chem.verify(
        "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has MW 220.3 g/mol.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)

    query("Pseudo-element in SMILES (G is not real)",
          "The compound CC(=O)[GH] is novel.")
    r = await chem.verify(
        "The compound CC(=O)[GH] is novel.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)

    query("Charged species described as 'neutral'",
          "Sodium acetate ([Na+].CC(=O)[O-]) is a neutral compound with no formal charge.")
    r = await chem.verify(
        "Sodium acetate ([Na+].CC(=O)[O-]) is a neutral compound with no formal charge.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)

    query("PAINS substructure (curcumin-like enone)",
          "The compound O=C(/C=C/c1ccccc1)/C=C/c1ccccc1 is a clean drug-like molecule.")
    r = await chem.verify(
        "The compound O=C(/C=C/c1ccccc1)/C=C/c1ccccc1 is a clean drug-like molecule.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)

    query("Drug-likeness check (ibuprofen — passes Lipinski)",
          "Ibuprofen (CC(C)Cc1ccc(C(C)C(O)=O)cc1) is a drug-like NSAID.")
    r = await chem.verify(
        "Ibuprofen (CC(C)Cc1ccc(C(C)C(O)=O)cc1) is a drug-like NSAID.",
        ClaimType.molecular_property, {})
    show_result("chemical", r)
    if "drug_likeness" in (r.evidence or {}):
        dl = r.evidence["drug_likeness"]
        print(f"      {GR}QED={dl.get('qed', 'n/a')}  Lipinski viol={dl.get('lipinski_violations', 'n/a')}{R}")

    # ──────────────────────────────────────────────────────────────────────
    section("2", "TIER 1.2 — Statistical sanity: deterministic stats checks")
    # ──────────────────────────────────────────────────────────────────────

    query("Impossible p-value (negative)",
          "The treatment effect was significant (p = -0.03).")
    r = await stat.verify(
        "The treatment effect was significant (p = -0.03).",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    query("CI doesn't contain point estimate",
          "Hazard ratio was 0.5 (95% CI: 0.7 to 0.9, p < 0.001).")
    r = await stat.verify(
        "Hazard ratio was 0.5 (95% CI: 0.7 to 0.9, p < 0.001).",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    query("Zero AEs in 15,000 patients (binomially impossible)",
          "No treatment-emergent adverse events were reported across 15,000 patients.")
    r = await stat.verify(
        "No treatment-emergent adverse events were reported across 15,000 patients.",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    query("HbA1c reduction of 4.2% in monotherapy (impossible)",
          "Metformin reduced HbA1c by 4.2% in 12 weeks of monotherapy.")
    r = await stat.verify(
        "Metformin reduced HbA1c by 4.2% in 12 weeks of monotherapy.",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    query("HbA1c reduction of 1.2% (clinically plausible)",
          "Metformin reduced HbA1c by 1.2% in 12 weeks of monotherapy.")
    r = await stat.verify(
        "Metformin reduced HbA1c by 1.2% in 12 weeks of monotherapy.",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    query("Multiple p-values without correction",
          "We tested 12 outcomes finding p=0.02, p=0.04, p=0.03, p=0.01, p=0.045, p=0.038, p=0.029.")
    r = await stat.verify(
        "We tested 12 outcomes finding p=0.02, p=0.04, p=0.03, p=0.01, p=0.045, p=0.038, p=0.029.",
        ClaimType.clinical_outcome, {})
    show_result("statistical", r)

    # ──────────────────────────────────────────────────────────────────────
    section("3", "TIER 2.1 — Pathway reasoning")
    # ──────────────────────────────────────────────────────────────────────

    query("True: Erlotinib + EGFR signals through MAPK",
          "Erlotinib inhibits EGFR which signals through the MAPK/ERK pathway.")
    r = await path.verify(
        "Erlotinib inhibits EGFR which signals through the MAPK/ERK pathway.",
        ClaimType.mechanism_of_action, {})
    show_result("pathway", r)

    query("True: Aspirin + COX in prostaglandin pathway",
          "Aspirin inhibits COX-1 and COX-2 in the prostaglandin biosynthesis pathway.")
    r = await path.verify(
        "Aspirin inhibits COX-1 and COX-2 in the prostaglandin biosynthesis pathway.",
        ClaimType.mechanism_of_action, {})
    show_result("pathway", r)

    query("True: Atorvastatin + HMGCR in cholesterol biosynthesis",
          "Atorvastatin inhibits HMGCR in the cholesterol biosynthesis pathway.")
    r = await path.verify(
        "Atorvastatin inhibits HMGCR in the cholesterol biosynthesis pathway.",
        ClaimType.mechanism_of_action, {})
    show_result("pathway", r)

    query("False: BRCA1 in JAK/STAT pathway",
          "BRCA1 is a key signaling component of the JAK/STAT pathway.")
    r = await path.verify(
        "BRCA1 is a key signaling component of the JAK/STAT pathway.",
        ClaimType.mechanism_of_action, {})
    show_result("pathway", r)

    # ──────────────────────────────────────────────────────────────────────
    section("4", "TIER 1.1 — Pharmacokinetic plausibility (already in framework)")
    # ──────────────────────────────────────────────────────────────────────

    query("Bioavailability 150% (physically impossible)",
          "The compound shows oral bioavailability of 150% in humans.")
    r = await pk.verify(
        "The compound shows oral bioavailability of 150% in humans.",
        ClaimType.pharmacokinetic, {})
    show_result("pharmacokinetic", r)

    query("Plausible PK profile",
          "The drug has bioavailability of 65%, half-life of 8 hours, and clearance of 12 L/h.")
    r = await pk.verify(
        "The drug has bioavailability of 65%, half-life of 8 hours, and clearance of 12 L/h.",
        ClaimType.pharmacokinetic, {})
    show_result("pharmacokinetic", r)

    query("Half-life of 9999 hours (way too long)",
          "The compound has a terminal half-life of 9999 hours in plasma.")
    r = await pk.verify(
        "The compound has a terminal half-life of 9999 hours in plasma.",
        ClaimType.pharmacokinetic, {})
    show_result("pharmacokinetic", r)

    # ──────────────────────────────────────────────────────────────────────
    section("5", "TIER 3.1 — Bayesian aggregation across multiple verifiers")
    # ──────────────────────────────────────────────────────────────────────

    print(f"\n{B}Scenario A:{R} Strong consensus that claim is wrong (3 refuted)")
    from src.verifiers.base import VerificationOutput
    results = [
        VerificationOutput(Verdict.refuted, 0.92, "MW computed by RDKit doesn't match", {}, "chemical"),
        VerificationOutput(Verdict.refuted, 0.88, "PK value impossible", {}, "pharmacokinetic"),
        VerificationOutput(Verdict.refuted, 0.85, "Pathway contradiction", {}, "pathway"),
    ]
    v = bayes.aggregate_claim_bayesian(results)
    p = bayes.posterior_probability(results)
    print(f"  Posterior P(hallucination) = {RD}{B}{p:.3f}{R}")
    print(f"  → {VC[v.verdict.value]}{B}{v.verdict.value}{R} (decisiveness conf={v.confidence:.2f})")

    print(f"\n{B}Scenario B:{R} Mild disagreement (1 refuted vs 2 verified)")
    results = [
        VerificationOutput(Verdict.refuted, 0.85, "Stated property mismatch", {}, "chemical"),
        VerificationOutput(Verdict.verified, 0.7, "Target confirmed in DB", {}, "target"),
        VerificationOutput(Verdict.verified, 0.6, "Citation valid", {}, "citation"),
    ]
    v = bayes.aggregate_claim_bayesian(results)
    p = bayes.posterior_probability(results)
    print(f"  Posterior P(hallucination) = {Y}{B}{p:.3f}{R}")
    print(f"  → {VC[v.verdict.value]}{B}{v.verdict.value}{R} (decisiveness conf={v.confidence:.2f})")

    print(f"\n{B}Scenario C:{R} Strong consensus that claim is correct (3 verified)")
    results = [
        VerificationOutput(Verdict.verified, 0.92, "Properties match", {}, "chemical"),
        VerificationOutput(Verdict.verified, 0.88, "PK plausible", {}, "pharmacokinetic"),
        VerificationOutput(Verdict.verified, 0.85, "Target confirmed", {}, "target"),
    ]
    v = bayes.aggregate_claim_bayesian(results)
    p = bayes.posterior_probability(results)
    print(f"  Posterior P(hallucination) = {G}{B}{p:.3f}{R}")
    print(f"  → {VC[v.verdict.value]}{B}{v.verdict.value}{R} (decisiveness conf={v.confidence:.2f})")

    print(f"\n{B}Scenario D:{R} High-confidence statistical refute alone")
    results = [
        VerificationOutput(Verdict.refuted, 0.95, "Binomially impossible", {}, "statistical"),
    ]
    v = bayes.aggregate_claim_bayesian(results)
    p = bayes.posterior_probability(results)
    print(f"  Posterior P(hallucination) = {RD}{B}{p:.3f}{R}")
    print(f"  → {VC[v.verdict.value]}{B}{v.verdict.value}{R} (decisiveness conf={v.confidence:.2f})")

    # ──────────────────────────────────────────────────────────────────────
    section("6", "FULL PIPELINE — claim run through all verifiers + Bayesian")
    # ──────────────────────────────────────────────────────────────────────

    examples = [
        ("Caffeine real-world claim",
         "Caffeine (Cn1c(=O)c2c(ncn2C)n(C)c1=O) has a molecular weight of 194.19 g/mol and acts as an adenosine receptor antagonist.",
         ClaimType.molecular_property),
        ("Hallucinated kinase",
         "The novel inhibitor binds the BRCA9 kinase with IC50 of 2 nM and shows selectivity over 400 kinases.",
         ClaimType.target_interaction),
        ("Impossible clinical claim",
         "In the trial, 0 of 18000 patients experienced any side effects.",
         ClaimType.clinical_outcome),
        ("Plausible PK with valid SMILES",
         "Metformin (CN(C)C(=N)NC(=N)N) has bioavailability of 55%, half-life of 5 hours, and clearance of 510 mL/min.",
         ClaimType.pharmacokinetic),
    ]

    for label, text, ctype in examples:
        print(f"\n{B}{C}► {label}{R}")
        print(f"  {GR}{text}{R}")
        results = await router.verify_claim(text, ctype, {})
        for r in results:
            show_result(r.source_db or "?", r)
        if results:
            v = bayes.aggregate_claim_bayesian(results)
            p = bayes.posterior_probability(results)
            verdict_color = VC[v.verdict.value]
            print(f"  {B}→ Bayesian:{R} {verdict_color}{B}{v.verdict.value}{R}  "
                  f"P(hallucination)={p:.3f}  conf={v.confidence:.2f}")

    print(f"\n{G}{B}━━━ END OF SHOWCASE ━━━{R}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""Statistical verifier — deterministic sanity checks on clinical and quantitative claims.

This verifier performs purely deterministic statistical sanity checks on
claim text, looking for impossible values, internal inconsistencies, and
known red flags of poor statistical reporting.  It uses only the Python
standard library — no scipy, no scikit-learn, no ML, no LLM.

The checks implemented are:

1.  **p-value sanity** — impossible p-values and p-hacking indicators.
2.  **Confidence-interval consistency** — point estimate inside CI, and
    CI/p-value agreement for ratio measures.
3.  **Effect size vs sample size plausibility** — implausibly tight CIs
    from small samples, unrealistically large effect sizes.
4.  **Binomial impossibility checks** — rule-of-three upper bound for
    "zero events in N patients" claims and percentage/count agreement.
5.  **Multiple testing detection** — many p-values reported without any
    mention of correction.
6.  **Clinical metric plausibility** — hard-coded ranges for HbA1c
    reduction, tumour response rates, and survival hazard ratios.

Each check is wrapped in ``try``/``except`` so that pathological inputs
never crash the verifier.  Findings are aggregated into the evidence
payload returned inside :class:`VerificationOutput`.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_IMPOSSIBLE = "impossible"


# ---------------------------------------------------------------------------
# Finding container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Finding:
    """A single check result.

    Attributes
    ----------
    check : str
        Short identifier of the check that produced the finding.
    passed : bool
        ``True`` if the check passed (no issue detected).
    severity : str
        One of ``"info"``, ``"warning"``, ``"impossible"``.
    message : str
        Human-readable description of the finding.
    data : dict
        Optional structured payload associated with the finding.
    """

    check: str
    passed: bool
    severity: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# p-values — captures the operator and the numeric value.
# Matches: "p < 0.05", "p = 0.04", "p-value of 0.001", "P = 0.000001",
# "p<.05", "P-value < 1e-6".
_P_VALUE_RE = re.compile(
    r"""
    \bp
    (?:[\s\-]?value)?              # optional "-value"
    \s*
    (?P<op>[<>=≤≥]{1,2})           # operator
    \s*
    (?P<val>-?\d*\.?\d+(?:[eE][-+]?\d+)?)  # numeric value (allow negative for sanity)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Also match phrasing like "p-value of 0.001" / "p value of .04".
_P_OF_RE = re.compile(
    r"\bp[\s\-]?value\s*of\s*(?P<val>-?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# Confidence intervals.  Captures lower and upper bound.
#   "95% CI: 0.5-1.2"
#   "95% CI [0.5, 1.2]"
#   "(95% CI 0.5 to 1.2)"
_CI_RE = re.compile(
    r"""
    9[05]\s*%\s*CI
    \s*[:\[\(]?\s*
    (?P<lo>-?\d*\.?\d+)
    \s*(?:[-–,]|to)\s*
    (?P<hi>-?\d*\.?\d+)
    \s*[\]\)]?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Point estimates for ratio measures, anchored to their label and captured
# together with a nearby CI if present.  Captures: label, estimate, lo, hi.
# Tolerates short connector phrases between the label and the estimate
# ("HR was 0.5", "the hazard ratio of 0.5"), and between the estimate and
# the CI ("0.5 with a 95% CI of 0.7-0.9").
_POINT_CI_RE = re.compile(
    r"""
    \b(?P<label>HR|OR|RR|hazard\s+ratio|odds\s+ratio|risk\s+ratio|relative\s+risk)
    (?:\s+(?:was|of|is|equals|measured|=|:))?\s+
    (?P<est>-?\d*\.?\d+)
    \s*
    (?:[,;]?\s*(?:with|having|and)\s+(?:a|an)?\s*)?
    [\(\[]?\s*
    (?:9[05]\s*%\s*CI)?
    \s*(?:of|:)?\s*
    [\(\[]?\s*
    (?P<lo>-?\d*\.?\d+)
    \s*(?:[-–,]|to)\s*
    (?P<hi>-?\d*\.?\d+)
    \s*[\]\)]?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sample sizes: "n = 1842", "N=50", "1,000 patients".
_SAMPLE_RE = re.compile(
    r"\b[nN]\s*=\s*(\d[\d,]*)|(\d[\d,]{2,})\s+patients\b",
)

# "X out of Y patients" / "X / Y".
_OUT_OF_RE = re.compile(
    r"(\d[\d,]*)\s*(?:out of|/|of)\s*(\d[\d,]*)\s*(?:patients?|subjects?|participants?)?",
    re.IGNORECASE,
)

# "zero / no adverse events ... in N patients".
# Allows optional adjectives between "no" and "adverse events" (e.g.
# "no treatment-emergent adverse events", "no serious adverse events").
_ZERO_AE_RE = re.compile(
    r"""
    (?:zero|no|0)\s+
    (?:[\w-]+\s+){0,3}?           # up to 3 optional adjectives
    (?:adverse\s+events?|AEs?|side\s+effects?|serious\s+adverse\s+events?|SAEs?|
       treatment[- ]emergent\s+adverse\s+events?)
    .{0,120}?
    (?:in|among|across|from|of)\s+
    (?:a\s+(?:cohort|sample|trial|study|population)\s+of\s+)?
    (\d[\d,]*)\s*
    (?:patients?|subjects?|participants?|individuals?|people)?
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Inverted form: "0 of N patients experienced/reported (any) side effects/AEs".
# Catches "0 of 18000 patients experienced any side effects" and
# "None of the 25,000 enrolled patients reported adverse events".
_ZERO_OF_PATIENTS_RE = re.compile(
    r"""
    (?:zero|none|no|0)\s+
    (?:of\s+(?:the\s+)?)?
    (\d[\d,]*)\s+
    (?:enrolled\s+|treated\s+|exposed\s+|dosed\s+|recruited\s+)?
    (?:patients?|subjects?|participants?|individuals?|people)
    .{0,100}?
    (?:experienced?|reported?|had|developed?|suffered?|presented?|showed?|
       observed|recorded|documented)
    \s+(?:any\s+|an?\s+)?
    (?:adverse\s+events?|AEs?|side\s+effects?|toxicit(?:y|ies)|
       complications?|reactions?|symptoms?|deaths?)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Another inversion: "N patients ... [no/zero] adverse events"
_PATIENTS_ZERO_AE_RE = re.compile(
    r"""
    (\d[\d,]*)\s+
    (?:patients?|subjects?|participants?|individuals?|people)
    .{0,120}?
    (?:had|experienced?|reported?|showed?|developed?|presented?)
    \s+(?:zero|no|0)\s+
    (?:[\w-]+\s+){0,3}?
    (?:adverse\s+events?|AEs?|side\s+effects?|toxicit(?:y|ies)|complications?|deaths?)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# "N treated, none experienced X" / "N enrolled, zero adverse events"
_TREATED_NO_AE_RE = re.compile(
    r"""
    (\d[\d,]*)\s+
    (?:patients?|subjects?|participants?|individuals?|people|enrolled|treated|exposed|dosed)
    .{0,80}?
    [,;]?\s*
    (?:none|zero|0|no\s+(?:patient|subject))
    .{0,40}?
    (?:experienced?|reported?|had|developed?)
    \s+
    (?:any\s+|an?\s+)?
    (?:adverse\s+events?|AEs?|side\s+effects?|toxicit(?:y|ies)|complications?|deaths?)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Percentage mentioned alongside a count: "42 of 100 (42%)".
_COUNT_PCT_RE = re.compile(
    r"(\d[\d,]*)\s*(?:/|of|out of)\s*(\d[\d,]*)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",
    re.IGNORECASE,
)

# Cohen's d: "Cohen's d = 0.8", "d=1.2".
_COHEN_D_RE = re.compile(
    r"(?:Cohen'?s\s+d|effect\s+size\s+d)\s*(?:=|:|of)?\s*(-?\d*\.?\d+)",
    re.IGNORECASE,
)

# HbA1c reduction: "HbA1c reduction of 3.5%", "reduced HbA1c by 2.1%".
_HBA1C_RE = re.compile(
    r"""
    HbA1c
    .{0,60}?
    (?:reduction\s+of|reduced\s+by|decrease\s+of|lowered\s+by|drop\s+of)
    \s*
    (\d+(?:\.\d+)?)
    \s*(?:%|percentage\s+points?|pp)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Verb-first form: "reduced HbA1c by 4.2%", "lowered HbA1c by 1.5%".
_HBA1C_VERB_FIRST_RE = re.compile(
    r"""
    (?:reduce[sd]?|lower(?:ed|s)?|decrease[sd]?|drop(?:ped|s)?|cut)
    \s+HbA1c\s+by\s+
    (\d+(?:\.\d+)?)
    \s*(?:%|percentage\s+points?|pp)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Also accept "X% HbA1c reduction" phrasing.
_HBA1C_ALT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percentage\s+points?)\s+HbA1c\s+(?:reduction|decrease|drop)",
    re.IGNORECASE,
)

# Tumour response rate: "response rate of 98%", "ORR of 97%".
_RESPONSE_RATE_RE = re.compile(
    r"(?:response\s+rate|ORR|objective\s+response\s+rate|complete\s+response\s+rate)"
    r"\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Clinical plausibility lookup
# ---------------------------------------------------------------------------

#: Heuristic ranges from clinical pharmacology used by check 6.
#: ``(threshold, description)`` — values beyond the threshold are flagged.
_CLINICAL_THRESHOLDS: dict[str, dict[str, float | str]] = {
    "hba1c_max_realistic_pct": {
        "value": 2.0,
        "description": (
            "HbA1c reduction > 2% in monotherapy is at the edge of realism; "
            "> 3% is implausible without combination therapy."
        ),
    },
    "hba1c_impossible_pct": {
        "value": 3.0,
        "description": "HbA1c reduction > 3% in monotherapy is implausible.",
    },
    "response_rate_max_pct": {
        "value": 95.0,
        "description": "Tumour response rates > 95% are extremely rare and warrant scrutiny.",
    },
    "survival_hr_floor": {
        "value": 0.3,
        "description": (
            "A survival HR < 0.3 is rare and typically requires long follow-up "
            "and large samples to be credible."
        ),
    },
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _to_int(raw: str) -> int:
    return int(raw.replace(",", ""))


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _extract_p_values(text: str) -> list[tuple[str, float, str]]:
    """Return ``(operator, value, raw_match)`` triples for every p-value."""
    results: list[tuple[str, float, str]] = []
    seen: set[tuple[int, int]] = set()

    for m in _P_VALUE_RE.finditer(text):
        span = m.span()
        if span in seen:
            continue
        seen.add(span)
        try:
            val = float(m.group("val"))
        except (ValueError, TypeError):
            continue
        results.append((m.group("op"), val, m.group(0)))

    for m in _P_OF_RE.finditer(text):
        span = m.span()
        if span in seen:
            continue
        seen.add(span)
        try:
            val = float(m.group("val"))
        except (ValueError, TypeError):
            continue
        results.append(("=", val, m.group(0)))

    return results


def _extract_cis(text: str) -> list[tuple[float, float, str]]:
    """Return ``(lo, hi, raw)`` tuples for every confidence interval."""
    out: list[tuple[float, float, str]] = []
    for m in _CI_RE.finditer(text):
        try:
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
        except (ValueError, TypeError):
            continue
        out.append((lo, hi, m.group(0)))
    return out


def _extract_point_with_ci(
    text: str,
) -> list[tuple[str, float, float, float, str]]:
    """Return ``(label, estimate, lo, hi, raw)`` tuples for ratio measures."""
    out: list[tuple[str, float, float, float, str]] = []
    for m in _POINT_CI_RE.finditer(text):
        try:
            est = float(m.group("est"))
            lo = float(m.group("lo"))
            hi = float(m.group("hi"))
        except (ValueError, TypeError):
            continue
        label = m.group("label").upper().replace(" ", "_")
        out.append((label, est, lo, hi, m.group(0)))
    return out


def _extract_sample_sizes(text: str) -> list[int]:
    out: list[int] = []
    for m in _SAMPLE_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        if not raw:
            continue
        try:
            out.append(_to_int(raw))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@register_verifier
class StatisticalVerifier(BaseVerifier):
    """Deterministic statistical sanity checker for quantitative claims."""

    name = "statistical"
    supported_claim_types = [
        ClaimType.clinical_outcome,
        ClaimType.pharmacokinetic,
        ClaimType.general_biomedical,
    ]

    #: Confidence used when at least one impossible finding is present.
    IMPOSSIBLE_CONFIDENCE = 0.95

    #: Confidence used when only warnings are present.
    WARNING_CONFIDENCE = 0.65

    #: Confidence used when all checks pass.
    PASS_CONFIDENCE = 0.8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        """Run all statistical checks against *claim_text*."""
        try:
            return self._verify_sync(claim_text, claim_type, context)
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception(
                "StatisticalVerifier crashed on claim: %.200s", claim_text
            )
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during statistical verification: {exc}",
                source_db="statistical",
            )

    def _verify_sync(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        findings: list[Finding] = []

        # Run every check.  Each helper is already wrapped in try/except.
        findings.extend(self._check_p_values(claim_text))
        findings.extend(self._check_confidence_intervals(claim_text))
        findings.extend(self._check_effect_vs_sample(claim_text))
        findings.extend(self._check_binomial(claim_text))
        findings.extend(self._check_multiple_testing(claim_text))
        findings.extend(self._check_clinical_metrics(claim_text))

        logger.debug(
            "StatisticalVerifier produced %d findings for claim: %.120s",
            len(findings),
            claim_text,
        )

        return self._aggregate(findings)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, findings: list[Finding]) -> VerificationOutput:
        """Aggregate findings into a single :class:`VerificationOutput`."""
        evidence: dict[str, Any] = {
            "findings": [f.to_dict() for f in findings],
        }

        # Count by severity (only failures matter for aggregation).
        impossibles = [f for f in findings if not f.passed and f.severity == SEVERITY_IMPOSSIBLE]
        warnings = [f for f in findings if not f.passed and f.severity == SEVERITY_WARNING]

        evidence["num_findings"] = len(findings)
        evidence["num_impossible"] = len(impossibles)
        evidence["num_warnings"] = len(warnings)

        if not findings:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=(
                    "No statistical or quantitative content extracted from the claim; "
                    "nothing to verify."
                ),
                evidence=evidence,
                source_db="statistical",
            )

        if impossibles:
            messages = "; ".join(f.message for f in impossibles)
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=self.IMPOSSIBLE_CONFIDENCE,
                reasoning=f"Statistically impossible content detected: {messages}.",
                evidence=evidence,
                source_db="statistical",
            )

        if warnings:
            messages = "; ".join(f.message for f in warnings)
            return VerificationOutput(
                verdict=Verdict.partially_supported,
                confidence=self.WARNING_CONFIDENCE,
                reasoning=(
                    f"Statistical checks passed but raised warnings: {messages}."
                ),
                evidence=evidence,
                source_db="statistical",
            )

        return VerificationOutput(
            verdict=Verdict.verified,
            confidence=self.PASS_CONFIDENCE,
            reasoning=(
                f"All {len(findings)} statistical sanity checks passed."
            ),
            evidence=evidence,
            source_db="statistical",
        )

    # ------------------------------------------------------------------
    # Check 1 — p-value sanity
    # ------------------------------------------------------------------

    def _check_p_values(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        try:
            p_values = _extract_p_values(text)
        except Exception as exc:
            logger.debug("p-value extraction failed: %s", exc)
            return findings

        if not p_values:
            return findings

        for op, val, raw in p_values:
            if val < 0:
                findings.append(
                    Finding(
                        check="p_value_sanity",
                        passed=False,
                        severity=SEVERITY_IMPOSSIBLE,
                        message=f"Negative p-value reported ({raw!r})",
                        data={"operator": op, "value": val},
                    )
                )
            elif val > 1:
                findings.append(
                    Finding(
                        check="p_value_sanity",
                        passed=False,
                        severity=SEVERITY_IMPOSSIBLE,
                        message=f"p-value > 1 reported ({raw!r})",
                        data={"operator": op, "value": val},
                    )
                )
            elif val == 0 and op == "=":
                findings.append(
                    Finding(
                        check="p_value_sanity",
                        passed=False,
                        severity=SEVERITY_IMPOSSIBLE,
                        message=f"p-value reported as exactly 0 ({raw!r})",
                        data={"operator": op, "value": val},
                    )
                )
            else:
                findings.append(
                    Finding(
                        check="p_value_sanity",
                        passed=True,
                        severity=SEVERITY_INFO,
                        message=f"p-value {raw!r} is in-range",
                        data={"operator": op, "value": val},
                    )
                )

        # Suspicious "p < 0.05 but reported as p = 0.0500" pattern.
        suspicious = [
            (op, val, raw)
            for op, val, raw in p_values
            if op == "=" and math.isclose(val, 0.05, abs_tol=1e-9)
        ]
        for op, val, raw in suspicious:
            findings.append(
                Finding(
                    check="p_value_sanity",
                    passed=False,
                    severity=SEVERITY_WARNING,
                    message=(
                        f"p-value exactly at the 0.05 threshold ({raw!r}) — "
                        "suspicious reporting pattern"
                    ),
                    data={"operator": op, "value": val},
                )
            )

        # p-hacking indicator: many p-values clustered just below 0.05.
        clustered = [
            val for _, val, _ in p_values if 0.01 < val < 0.05
        ]
        if len(clustered) >= 3:
            findings.append(
                Finding(
                    check="p_value_sanity",
                    passed=False,
                    severity=SEVERITY_WARNING,
                    message=(
                        f"{len(clustered)} p-values clustered in (0.01, 0.05) — "
                        "possible p-hacking"
                    ),
                    data={"clustered_values": clustered},
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Check 2 — confidence-interval consistency
    # ------------------------------------------------------------------

    def _check_confidence_intervals(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        try:
            point_cis = _extract_point_with_ci(text)
        except Exception as exc:
            logger.debug("Point-CI extraction failed: %s", exc)
            point_cis = []

        for label, est, lo, hi, raw in point_cis:
            if lo > hi:
                lo, hi = hi, lo  # tolerate swapped bounds

            if not (lo <= est <= hi):
                findings.append(
                    Finding(
                        check="ci_consistency",
                        passed=False,
                        severity=SEVERITY_IMPOSSIBLE,
                        message=(
                            f"{label} point estimate {est} falls outside "
                            f"its reported 95% CI [{lo}, {hi}] ({raw!r})"
                        ),
                        data={"label": label, "estimate": est, "lo": lo, "hi": hi},
                    )
                )
            else:
                findings.append(
                    Finding(
                        check="ci_consistency",
                        passed=True,
                        severity=SEVERITY_INFO,
                        message=f"{label} point estimate {est} lies within CI [{lo}, {hi}]",
                        data={"label": label, "estimate": est, "lo": lo, "hi": hi},
                    )
                )

        # CI/p-value agreement for ratio measures: if the caller asserts
        # p < 0.05 but the CI crosses 1.0, they disagree.
        try:
            p_values = _extract_p_values(text)
        except Exception:
            p_values = []

        claimed_significant = any(
            (op.startswith("<") and val <= 0.05) or (op == "=" and val < 0.05)
            for op, val, _ in p_values
        )
        if claimed_significant:
            for label, est, lo, hi, raw in point_cis:
                if label in {"HR", "OR", "RR", "HAZARD_RATIO", "ODDS_RATIO", "RISK_RATIO", "RELATIVE_RISK"}:
                    if lo <= 1.0 <= hi:
                        findings.append(
                            Finding(
                                check="ci_consistency",
                                passed=False,
                                severity=SEVERITY_IMPOSSIBLE,
                                message=(
                                    f"{label} CI [{lo}, {hi}] crosses 1.0 but p < 0.05 "
                                    f"is also claimed ({raw!r})"
                                ),
                                data={
                                    "label": label,
                                    "estimate": est,
                                    "lo": lo,
                                    "hi": hi,
                                },
                            )
                        )

        return findings

    # ------------------------------------------------------------------
    # Check 3 — effect size vs sample size
    # ------------------------------------------------------------------

    def _check_effect_vs_sample(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        try:
            samples = _extract_sample_sizes(text)
            point_cis = _extract_point_with_ci(text)
        except Exception as exc:
            logger.debug("Effect-vs-sample extraction failed: %s", exc)
            return findings

        if not point_cis:
            return findings

        min_n = min(samples) if samples else None

        for label, est, lo, hi, raw in point_cis:
            # Very large effect sizes need large samples.
            if label in {"HR", "OR", "RR", "HAZARD_RATIO", "ODDS_RATIO", "RISK_RATIO", "RELATIVE_RISK"}:
                extreme = est < 0.2 or est > 5.0
                if extreme and min_n is not None and min_n < 200:
                    findings.append(
                        Finding(
                            check="effect_vs_sample",
                            passed=False,
                            severity=SEVERITY_WARNING,
                            message=(
                                f"Extreme effect size {label}={est} claimed on small sample "
                                f"(n={min_n}) — likely overfit"
                            ),
                            data={"label": label, "estimate": est, "n": min_n},
                        )
                    )

                # Implausibly tight CIs from small samples.
                if lo < hi and min_n is not None and min_n > 0:
                    width = hi - lo
                    # Log-scale SE for a ratio ~ (log(hi) - log(lo)) / 3.92.
                    try:
                        if lo > 0 and hi > 0:
                            log_se = (math.log(hi) - math.log(lo)) / 3.92
                            # Rough minimum achievable SE on log scale is
                            # ~ 2 / sqrt(n) for a balanced design.
                            min_log_se = 2.0 / math.sqrt(min_n)
                            if log_se < 0.5 * min_log_se:
                                findings.append(
                                    Finding(
                                        check="effect_vs_sample",
                                        passed=False,
                                        severity=SEVERITY_IMPOSSIBLE,
                                        message=(
                                            f"{label} CI [{lo}, {hi}] is implausibly tight "
                                            f"for n={min_n} (log-SE={log_se:.3f} vs "
                                            f"min~{min_log_se:.3f})"
                                        ),
                                        data={
                                            "label": label,
                                            "lo": lo,
                                            "hi": hi,
                                            "n": min_n,
                                            "log_se": log_se,
                                            "min_log_se": min_log_se,
                                        },
                                    )
                                )
                    except (ValueError, ZeroDivisionError):
                        pass
                    else:
                        _ = width  # width computed for completeness

        # Cohen's d sanity: |d| > 3 is extreme in biomedicine.
        try:
            for m in _COHEN_D_RE.finditer(text):
                d = float(m.group(1))
                if abs(d) > 3.0:
                    findings.append(
                        Finding(
                            check="effect_vs_sample",
                            passed=False,
                            severity=SEVERITY_WARNING,
                            message=f"Cohen's d={d} is extreme; values above 3 are rare",
                            data={"cohen_d": d},
                        )
                    )
        except Exception as exc:
            logger.debug("Cohen's d parsing failed: %s", exc)

        return findings

    # ------------------------------------------------------------------
    # Check 4 — binomial impossibility
    # ------------------------------------------------------------------

    def _check_binomial(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        # Count/percentage consistency.
        try:
            for m in _COUNT_PCT_RE.finditer(text):
                num = _to_int(m.group(1))
                den = _to_int(m.group(2))
                stated_pct = float(m.group(3))
                if den == 0:
                    continue
                actual_pct = 100.0 * num / den
                if not math.isclose(actual_pct, stated_pct, abs_tol=0.6):
                    findings.append(
                        Finding(
                            check="binomial",
                            passed=False,
                            severity=SEVERITY_IMPOSSIBLE,
                            message=(
                                f"Percentage mismatch: {num}/{den} = "
                                f"{actual_pct:.2f}% but stated as {stated_pct}%"
                            ),
                            data={
                                "numerator": num,
                                "denominator": den,
                                "stated_pct": stated_pct,
                                "actual_pct": actual_pct,
                            },
                        )
                    )
        except Exception as exc:
            logger.debug("Count/percentage check failed: %s", exc)

        # "Zero AEs in N patients" — rule of three.  Try every phrasing variant
        # and de-duplicate by sample size so we only emit one finding per N.
        try:
            seen_n: set[int] = set()
            for pattern in (
                _ZERO_AE_RE,
                _ZERO_OF_PATIENTS_RE,
                _PATIENTS_ZERO_AE_RE,
                _TREATED_NO_AE_RE,
            ):
                for m in pattern.finditer(text):
                    n = _to_int(m.group(1))
                    if n <= 0 or n in seen_n:
                        continue
                    seen_n.add(n)
                    upper_95 = 3.0 / n
                    data = {"n": n, "rule_of_three_upper_95": upper_95}
                    if n > 10_000:
                        findings.append(
                            Finding(
                                check="binomial",
                                passed=False,
                                severity=SEVERITY_IMPOSSIBLE,
                                message=(
                                    f"Zero adverse events in n={n} is biologically "
                                    f"implausible for any active drug (rule-of-three "
                                    f"upper 95% bound {upper_95:.2e})"
                                ),
                                data=data,
                            )
                        )
                    else:
                        findings.append(
                            Finding(
                                check="binomial",
                                passed=True,
                                severity=SEVERITY_INFO,
                                message=(
                                    f"Zero AEs in n={n}; rule-of-three upper 95% "
                                    f"bound is {upper_95:.3f}"
                                ),
                                data=data,
                            )
                        )
        except Exception as exc:
            logger.debug("Zero-AE check failed: %s", exc)

        return findings

    # ------------------------------------------------------------------
    # Check 5 — multiple testing detection
    # ------------------------------------------------------------------

    def _check_multiple_testing(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        try:
            p_values = _extract_p_values(text)
        except Exception:
            return findings

        if len(p_values) <= 5:
            return findings

        correction_terms = (
            "bonferroni",
            "fdr",
            "false discovery",
            "benjamini",
            "holm",
            "adjusted",
            "corrected",
            "family-wise",
        )
        lowered = text.lower()
        if any(term in lowered for term in correction_terms):
            findings.append(
                Finding(
                    check="multiple_testing",
                    passed=True,
                    severity=SEVERITY_INFO,
                    message=(
                        f"{len(p_values)} p-values reported with a multiple-testing "
                        "correction mentioned"
                    ),
                    data={"num_p_values": len(p_values)},
                )
            )
        else:
            findings.append(
                Finding(
                    check="multiple_testing",
                    passed=False,
                    severity=SEVERITY_WARNING,
                    message=(
                        f"{len(p_values)} p-values reported without any mention of "
                        "multiple-testing correction (Bonferroni, FDR, etc.)"
                    ),
                    data={"num_p_values": len(p_values)},
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Check 6 — clinical metric plausibility
    # ------------------------------------------------------------------

    def _check_clinical_metrics(self, text: str) -> list[Finding]:
        findings: list[Finding] = []

        # HbA1c
        try:
            hba1c_values: list[float] = []
            for m in _HBA1C_RE.finditer(text):
                hba1c_values.append(float(m.group(1)))
            for m in _HBA1C_ALT_RE.finditer(text):
                hba1c_values.append(float(m.group(1)))
            for m in _HBA1C_VERB_FIRST_RE.finditer(text):
                hba1c_values.append(float(m.group(1)))

            impossible_thr = float(_CLINICAL_THRESHOLDS["hba1c_impossible_pct"]["value"])  # type: ignore[arg-type]
            warn_thr = float(_CLINICAL_THRESHOLDS["hba1c_max_realistic_pct"]["value"])  # type: ignore[arg-type]

            for val in hba1c_values:
                if val > impossible_thr:
                    findings.append(
                        Finding(
                            check="clinical_metrics",
                            passed=False,
                            severity=SEVERITY_IMPOSSIBLE,
                            message=(
                                f"HbA1c reduction of {val}% exceeds the "
                                f"clinically plausible monotherapy ceiling of "
                                f"{impossible_thr}%"
                            ),
                            data={"metric": "hba1c_reduction_pct", "value": val},
                        )
                    )
                elif val > warn_thr:
                    findings.append(
                        Finding(
                            check="clinical_metrics",
                            passed=False,
                            severity=SEVERITY_WARNING,
                            message=(
                                f"HbA1c reduction of {val}% is at the upper edge of "
                                f"realism for monotherapy (>{warn_thr}%)"
                            ),
                            data={"metric": "hba1c_reduction_pct", "value": val},
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            check="clinical_metrics",
                            passed=True,
                            severity=SEVERITY_INFO,
                            message=f"HbA1c reduction of {val}% is clinically plausible",
                            data={"metric": "hba1c_reduction_pct", "value": val},
                        )
                    )
        except Exception as exc:
            logger.debug("HbA1c check failed: %s", exc)

        # Response rate
        try:
            response_thr = float(_CLINICAL_THRESHOLDS["response_rate_max_pct"]["value"])  # type: ignore[arg-type]
            for m in _RESPONSE_RATE_RE.finditer(text):
                rate = float(m.group(1))
                if rate > 100.0:
                    findings.append(
                        Finding(
                            check="clinical_metrics",
                            passed=False,
                            severity=SEVERITY_IMPOSSIBLE,
                            message=f"Response rate {rate}% exceeds 100%",
                            data={"metric": "response_rate_pct", "value": rate},
                        )
                    )
                elif rate > response_thr:
                    findings.append(
                        Finding(
                            check="clinical_metrics",
                            passed=False,
                            severity=SEVERITY_WARNING,
                            message=(
                                f"Tumour response rate {rate}% is extremely rare "
                                f"(threshold {response_thr}%)"
                            ),
                            data={"metric": "response_rate_pct", "value": rate},
                        )
                    )
        except Exception as exc:
            logger.debug("Response-rate check failed: %s", exc)

        # Survival HR
        try:
            survival_floor = float(_CLINICAL_THRESHOLDS["survival_hr_floor"]["value"])  # type: ignore[arg-type]
            if re.search(r"\b(survival|mortality|overall\s+survival|OS)\b", text, re.IGNORECASE):
                for m in re.finditer(
                    r"\b(?:HR|hazard\s+ratio)\s*(?:of|=|:)?\s*(-?\d*\.?\d+)",
                    text,
                    re.IGNORECASE,
                ):
                    try:
                        hr = float(m.group(1))
                    except ValueError:
                        continue
                    if 0 < hr < survival_floor:
                        findings.append(
                            Finding(
                                check="clinical_metrics",
                                passed=False,
                                severity=SEVERITY_WARNING,
                                message=(
                                    f"Survival HR={hr} is below the typical floor "
                                    f"of {survival_floor} and warrants scrutiny"
                                ),
                                data={"metric": "survival_hr", "value": hr},
                            )
                        )
        except Exception as exc:
            logger.debug("Survival HR check failed: %s", exc)

        return findings

    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """The statistical verifier has no external dependencies."""
        return True

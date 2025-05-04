"""Chemical property verifier — validates molecular structures and properties.

This module provides :class:`ChemicalVerifier`, which performs deterministic,
ground-truth cheminformatics checks on SMILES-bearing claims using RDKit. In
addition to basic property comparison (MW, logP, TPSA, HBD, HBA, rotatable
bonds), it runs the following structural/medicinal-chemistry checks:

* Stereochemistry validation (unassigned chiral centers)
* Synthetic accessibility proxy score (1-10 scale)
* PAINS (pan-assay interference) substructure filter
* Brenk (unwanted/reactive group) substructure filter
* Drug-likeness scores: Lipinski Ro5, Veber, QED, Ghose
* Element validity (rejects pseudo-elements like G/J/Q)
* Formal charge check against "neutral" language

All checks are layered on top of the existing property-comparison logic; if
RDKit is not installed, the verifier falls back to UNVERIFIABLE as before.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.models.enums import ClaimType, Verdict
from src.verifiers.base import BaseVerifier, VerificationOutput, register_verifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional RDKit import
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.QED import qed as _qed
    from rdkit.Chem import FilterCatalog as _FilterCatalog

    _RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit is not installed — ChemicalVerifier will return UNVERIFIABLE "
        "for claims requiring structure validation."
    )


# ---------------------------------------------------------------------------
# Filter catalog singletons (lazy)
# ---------------------------------------------------------------------------

_PAINS_CATALOG: Any = None
_BRENK_CATALOG: Any = None


def _get_pains_catalog() -> Any:
    """Return a lazily-initialised PAINS FilterCatalog."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None and _RDKIT_AVAILABLE:
        params = _FilterCatalog.FilterCatalogParams()
        params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = _FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def _get_brenk_catalog() -> Any:
    """Return a lazily-initialised Brenk FilterCatalog."""
    global _BRENK_CATALOG
    if _BRENK_CATALOG is None and _RDKIT_AVAILABLE:
        params = _FilterCatalog.FilterCatalogParams()
        params.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _BRENK_CATALOG = _FilterCatalog.FilterCatalog(params)
    return _BRENK_CATALOG

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_SMILES_RE = re.compile(
    r"(?:SMILES[:\s]*)?([A-Za-z0-9@+\-\[\]\(\)\\\/=#$%&.~]+)"
)

_PROPERTY_PATTERNS: dict[str, re.Pattern[str]] = {
    "mw": re.compile(
        r"(?:molecular\s+weight|MW)\s*(?:of|=|:|\s)\s*([\d.]+)\s*(?:Da|g/mol|g·mol⁻¹)?",
        re.IGNORECASE,
    ),
    "logp": re.compile(
        r"(?:logP|log\s*P|LogP)\s*(?:of|=|:|\s)\s*([+-]?[\d.]+)",
        re.IGNORECASE,
    ),
    "tpsa": re.compile(
        r"(?:TPSA|topological\s+polar\s+surface\s+area)\s*(?:of|=|:|\s)\s*([\d.]+)",
        re.IGNORECASE,
    ),
    "hbd": re.compile(
        r"(?:HBD|hydrogen\s+bond\s+donors?)\s*(?:of|=|:|\s)\s*(\d+)",
        re.IGNORECASE,
    ),
    "hba": re.compile(
        r"(?:HBA|hydrogen\s+bond\s+acceptors?)\s*(?:of|=|:|\s)\s*(\d+)",
        re.IGNORECASE,
    ),
    "rotatable_bonds": re.compile(
        r"(?:rotatable\s+bonds?)\s*(?:of|=|:|\s)\s*(\d+)",
        re.IGNORECASE,
    ),
}

# Tolerances for property comparison
_TOLERANCES: dict[str, tuple[str, float]] = {
    "mw": ("relative", 0.01),        # ±1 %
    "logp": ("absolute", 0.3),       # ±0.3
    "tpsa": ("relative", 0.05),      # ±5 %
    "hbd": ("absolute", 0.0),        # exact integer
    "hba": ("absolute", 0.0),        # exact integer
    "rotatable_bonds": ("absolute", 0.0),
}


def _extract_smiles_candidates(text: str) -> list[str]:
    """Return plausible SMILES strings from *text*."""
    candidates: list[str] = []
    for match in _SMILES_RE.finditer(text):
        token = match.group(1).strip().rstrip(".")
        # Strip balanced outer parentheses that come from prose like "(CC(=O)O)"
        while token.startswith("(") and token.endswith(")"):
            inner = token[1:-1]
            # Only strip if the inner part has balanced parens
            if inner.count("(") == inner.count(")"):
                token = inner
            else:
                break
        # Heuristic filters:
        # 1. Must be at least 3 chars
        # 2. Must contain at least one letter AND one non-alpha character
        # 3. Must not be purely numeric (e.g. "220.3", "180.16")
        if (
            len(token) >= 3
            and re.search(r"[A-Za-z]", token)
            and re.search(r"[^A-Za-z]", token)
            and not re.fullmatch(r"[0-9.]+", token)
        ):
            candidates.append(token)
    return candidates


def _extract_stated_properties(text: str) -> dict[str, float]:
    """Extract numeric property claims from *text*."""
    props: dict[str, float] = {}
    for key, pattern in _PROPERTY_PATTERNS.items():
        match = pattern.search(text)
        if match:
            try:
                props[key] = float(match.group(1))
            except ValueError:
                pass
    return props


def _compute_properties(mol: Any) -> dict[str, float]:
    """Compute molecular properties from an RDKit Mol object."""
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }


def _compare_properties(
    computed: dict[str, float],
    stated: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Compare computed vs stated properties.

    Returns (matches, mismatches) — each a list of human-readable strings.
    """
    matches: list[str] = []
    mismatches: list[str] = []

    for key, stated_val in stated.items():
        if key not in computed:
            continue
        comp_val = computed[key]
        tol_type, tol = _TOLERANCES.get(key, ("absolute", 0.0))

        if tol_type == "relative":
            ok = abs(comp_val - stated_val) <= abs(comp_val * tol) if comp_val != 0 else stated_val == 0
        else:
            ok = abs(comp_val - stated_val) <= tol

        label = key.upper()
        if ok:
            matches.append(f"{label}: stated {stated_val}, computed {comp_val:.4g}")
        else:
            mismatches.append(f"{label}: stated {stated_val}, computed {comp_val:.4g}")

    return matches, mismatches


# ---------------------------------------------------------------------------
# Language cues used to interpret claim intent
# ---------------------------------------------------------------------------

_EASY_SYNTH_RE = re.compile(
    r"\b(easily\s+synthes[iy][sz]ed|trivial(?:ly)?\s+to\s+make|simple\s+synthesis|"
    r"readily\s+synthes[iy][sz]ed|straightforward\s+synthesis)\b",
    re.IGNORECASE,
)

_NEUTRAL_RE = re.compile(
    r"\b(neutral|uncharged|zero\s+charge|no\s+formal\s+charge)\b",
    re.IGNORECASE,
)

_SINGLE_MOLECULE_RE = re.compile(
    r"\b(the\s+(?:compound|molecule|drug|structure|enantiomer|isomer)|"
    r"a\s+single\s+(?:compound|molecule|enantiomer|isomer))\b",
    re.IGNORECASE,
)

# Pseudo-element tokens that occasionally appear in hallucinated SMILES strings.
# These are single-character uppercase letters that do not correspond to any
# real element symbol.
_PSEUDO_ELEMENTS: frozenset[str] = frozenset({"G", "J", "L", "M", "Q", "R", "T", "X", "Z"})


# ---------------------------------------------------------------------------
# Structural / cheminformatics helpers
# ---------------------------------------------------------------------------


def _has_pseudo_element_tokens(smiles: str) -> list[str]:
    """Return a list of pseudo-element tokens appearing in bracketed atoms.

    SMILES uses bracketed atoms ([Fe], [Si], [N+], ...). Any bracketed atom
    whose element symbol does not exist in the RDKit periodic table is a
    clear sign of a hallucinated structure.
    """
    if not _RDKIT_AVAILABLE:
        return []

    pt = Chem.GetPeriodicTable()
    offenders: list[str] = []
    for match in re.finditer(r"\[([^\]]+)\]", smiles):
        token = match.group(1)
        # Extract the element symbol: leading letters after optional isotope digits.
        sym_match = re.match(r"(\d*)([A-Z][a-z]?)", token)
        if not sym_match:
            continue
        symbol = sym_match.group(2)
        try:
            pt.GetAtomicNumber(symbol)
        except Exception:
            offenders.append(symbol)

    return offenders


def _check_stereochemistry(mol: Any) -> dict[str, Any]:
    """Detect assigned and unassigned chiral centers.

    Returns a dict with keys ``total``, ``assigned``, ``unassigned`` and a
    list of ``centers`` formatted as ``(atom_idx, CIP_code)``.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    assigned = [c for c in centers if c[1] not in ("?", "Unassigned")]
    unassigned = [c for c in centers if c[1] in ("?", "Unassigned")]
    return {
        "total": len(centers),
        "assigned": len(assigned),
        "unassigned": len(unassigned),
        "centers": [(idx, code) for idx, code in centers],
    }


def _synthetic_accessibility_proxy(mol: Any) -> dict[str, Any]:
    """Compute a simple Ertl-style synthetic accessibility proxy (1-10).

    This is **not** a reimplementation of the reference SAscore (which
    requires a fragment-frequency database). Instead, we combine four
    RDKit-native complexity descriptors into a bounded score:

    * spiro atoms — rare, generally hard to form
    * bridgehead atoms — fused/bridged systems
    * FractionCSP3 — sp3 richness (high = stereocenters & 3D complexity)
    * ring complexity (number of rings, with a macro-ring penalty)

    Higher values indicate harder-to-synthesize structures. Scores > 7
    are considered "difficult to synthesize".
    """
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    macro_rings = sum(1 for r in ring_info.AtomRings() if len(r) >= 8)
    n_heavy = mol.GetNumHeavyAtoms()
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # Size penalty — very large molecules are harder
    size_penalty = math.log10(max(n_heavy, 1)) * 0.5

    raw = (
        1.0
        + 1.2 * n_spiro
        + 1.0 * n_bridge
        + 2.0 * fsp3
        + 0.3 * n_rings
        + 1.5 * macro_rings
        + 0.25 * n_stereo
        + size_penalty
    )
    score = max(1.0, min(10.0, raw))
    return {
        "score": round(score, 2),
        "num_spiro_atoms": n_spiro,
        "num_bridgehead_atoms": n_bridge,
        "fraction_csp3": round(fsp3, 3),
        "num_rings": n_rings,
        "num_macrocycles": macro_rings,
        "num_stereocenters": n_stereo,
        "difficult": score > 7.0,
    }


def _pains_check(mol: Any) -> dict[str, Any]:
    """Run the PAINS substructure filter against *mol*."""
    catalog = _get_pains_catalog()
    if catalog is None:
        return {"matched": False, "entries": []}
    matches = catalog.GetMatches(mol)
    return {
        "matched": len(matches) > 0,
        "entries": [m.GetDescription() for m in matches],
    }


def _brenk_check(mol: Any) -> dict[str, Any]:
    """Run the Brenk unwanted-functional-group filter against *mol*."""
    catalog = _get_brenk_catalog()
    if catalog is None:
        return {"matched": False, "entries": []}
    matches = catalog.GetMatches(mol)
    return {
        "matched": len(matches) > 0,
        "entries": [m.GetDescription() for m in matches],
    }


def _drug_likeness(mol: Any, props: dict[str, float]) -> dict[str, Any]:
    """Compute Lipinski, Veber, QED and Ghose drug-likeness metrics."""
    mw = props["mw"]
    logp = props["logp"]
    tpsa = props["tpsa"]
    hbd = props["hbd"]
    hba = props["hba"]
    rotb = props["rotatable_bonds"]

    # Lipinski: violations are booleans
    lipinski_violations: list[str] = []
    if mw > 500:
        lipinski_violations.append(f"MW={mw:.1f}>500")
    if logp > 5:
        lipinski_violations.append(f"logP={logp:.2f}>5")
    if hbd > 5:
        lipinski_violations.append(f"HBD={hbd}>5")
    if hba > 10:
        lipinski_violations.append(f"HBA={hba}>10")

    # Veber
    veber_ok = rotb <= 10 and tpsa <= 140
    veber_violations: list[str] = []
    if rotb > 10:
        veber_violations.append(f"RotB={rotb}>10")
    if tpsa > 140:
        veber_violations.append(f"TPSA={tpsa:.1f}>140")

    # Ghose
    mr = Descriptors.MolMR(mol)
    n_atoms = mol.GetNumAtoms()
    ghose_violations: list[str] = []
    if not (160 <= mw <= 480):
        ghose_violations.append(f"MW={mw:.1f} not in [160,480]")
    if not (-0.4 <= logp <= 5.6):
        ghose_violations.append(f"logP={logp:.2f} not in [-0.4,5.6]")
    if not (40 <= mr <= 130):
        ghose_violations.append(f"MR={mr:.2f} not in [40,130]")
    if not (20 <= n_atoms <= 70):
        ghose_violations.append(f"atoms={n_atoms} not in [20,70]")

    # QED
    try:
        qed_score = float(_qed(mol))
    except Exception as exc:  # pragma: no cover
        logger.debug("QED computation failed: %s", exc)
        qed_score = float("nan")

    return {
        "lipinski": {
            "violations": lipinski_violations,
            "n_violations": len(lipinski_violations),
            "passes": len(lipinski_violations) <= 1,
        },
        "veber": {
            "violations": veber_violations,
            "passes": veber_ok,
        },
        "ghose": {
            "violations": ghose_violations,
            "passes": len(ghose_violations) == 0,
            "molar_refractivity": round(mr, 2),
            "num_atoms": n_atoms,
        },
        "qed": round(qed_score, 4) if not math.isnan(qed_score) else None,
    }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@register_verifier
class ChemicalVerifier(BaseVerifier):
    """Verify molecular-property claims using RDKit and public databases.

    Beyond basic property comparison (MW, logP, TPSA, HBD, HBA, rotatable
    bonds), this verifier runs a battery of deterministic structural checks:

    * Stereochemistry sanity (unassigned chiral centers)
    * Synthetic accessibility proxy score
    * PAINS and Brenk substructure filters
    * Lipinski / Veber / Ghose / QED drug-likeness
    * Element validity (pseudo-element detection)
    * Formal-charge vs. "neutral" language consistency

    Each check contributes a structured finding to ``evidence`` and the
    overall verdict is aggregated:

    * ``REFUTED`` — any hard violation (parse failure, mismatched property,
      pseudo-element, "neutral" claim on a charged species)
    * ``PARTIALLY_SUPPORTED`` — soft warnings (PAINS/Brenk hit, unassigned
      stereocenters on a "single molecule" claim, SA mismatch, drug-likeness
      failures that do not contradict a numeric property)
    * ``VERIFIED`` — all checks pass and at least one property matched
    """

    name = "chemical"
    supported_claim_types = [ClaimType.molecular_property]

    def __init__(self) -> None:
        # Knowledge-base clients — instantiated here, will be injected later.
        from src.knowledge.cache import KnowledgeCache  # noqa: F401 — used for typing context

        self._pubchem: Any = None
        self._chembl: Any = None
        self._drugbank: Any = None
        self._triangulation: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Best-effort client initialisation."""
        try:
            from src.knowledge.pubchem import PubChemClient  # type: ignore[import-untyped]
            self._pubchem = PubChemClient()
        except Exception:
            logger.debug("PubChemClient unavailable — external lookups disabled.")

        try:
            from src.knowledge.chembl import ChEMBLClient  # type: ignore[import-untyped]
            self._chembl = ChEMBLClient()
        except Exception:
            logger.debug("ChEMBLClient unavailable — external lookups disabled.")

        try:
            from src.knowledge.drugbank import DrugBankClient  # type: ignore[import-untyped]
            self._drugbank = DrugBankClient()
        except Exception:
            logger.debug("DrugBankClient unavailable — external lookups disabled.")

        try:
            from src.knowledge.triangulation import TriangulationService  # type: ignore[import-untyped]
            self._triangulation = TriangulationService(
                pubchem=self._pubchem,
                chembl=self._chembl,
                drugbank=self._drugbank,
            )
        except Exception:
            logger.debug("TriangulationService unavailable — cross-source checks disabled.")

    def inject_knowledge_base(self, kb: Any) -> None:
        """Inject a fully configured :class:`KnowledgeBase` container.

        Replaces the best-effort self-initialized clients with the shared,
        cache-backed ones from the application-level container.
        """
        self._pubchem = getattr(kb, "pubchem", self._pubchem)
        self._chembl = getattr(kb, "chembl", self._chembl)
        self._drugbank = getattr(kb, "drugbank", self._drugbank)
        self._triangulation = getattr(kb, "triangulation", self._triangulation)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        try:
            return await self._verify_impl(claim_text, claim_type, context)
        except Exception as exc:
            logger.exception("ChemicalVerifier crashed on claim: %.200s", claim_text)
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning=f"Internal error during chemical verification: {exc}",
            )

    async def _verify_impl(
        self,
        claim_text: str,
        claim_type: ClaimType,
        context: dict[str, Any],
    ) -> VerificationOutput:
        if not _RDKIT_AVAILABLE:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="RDKit is not installed; cannot validate molecular properties.",
            )

        smiles_candidates = _extract_smiles_candidates(claim_text)
        stated_props = _extract_stated_properties(claim_text)
        evidence: dict[str, Any] = {"stated_properties": stated_props}

        # Try to also get SMILES from context
        if not smiles_candidates and "smiles" in context:
            smiles_candidates = [context["smiles"]]

        if not smiles_candidates:
            # No structure to validate — check if we can verify by name
            return await self._verify_by_name(claim_text, stated_props, evidence)

        # ------------------------------------------------------------------
        # Element validity — reject pseudo-elements before attempting parse
        # ------------------------------------------------------------------
        pseudo_hits: list[tuple[str, list[str]]] = []
        for smi in smiles_candidates:
            pseudo = _has_pseudo_element_tokens(smi)
            if pseudo:
                pseudo_hits.append((smi, pseudo))

        # Try each SMILES candidate until one parses
        mol = None
        valid_smiles: str | None = None
        for smi in smiles_candidates:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles = smi
                break

        if mol is None:
            if pseudo_hits:
                evidence["pseudo_elements"] = [
                    {"smiles": s, "tokens": toks} for s, toks in pseudo_hits
                ]
                return VerificationOutput(
                    verdict=Verdict.refuted,
                    confidence=0.95,
                    reasoning=(
                        "SMILES contains tokens that are not valid chemical "
                        f"elements: {pseudo_hits[0][1]}. This is a hallucinated "
                        "structure."
                    ),
                    evidence=evidence,
                )
            return VerificationOutput(
                verdict=Verdict.refuted,
                confidence=0.85,
                reasoning=(
                    f"None of the SMILES candidates ({', '.join(smiles_candidates)}) "
                    "could be parsed into a valid molecular structure."
                ),
                evidence=evidence,
            )

        evidence["valid_smiles"] = valid_smiles
        computed = _compute_properties(mol)
        evidence["computed_properties"] = {k: round(v, 4) for k, v in computed.items()}

        # ------------------------------------------------------------------
        # Run the full battery of deterministic structural checks
        # ------------------------------------------------------------------
        hard_violations: list[str] = []   # -> REFUTED
        soft_warnings: list[str] = []     # -> PARTIALLY_SUPPORTED
        positive_notes: list[str] = []    # informational, support VERIFIED

        self._check_element_validity(valid_smiles or "", evidence, hard_violations)
        self._check_charge_balance(mol, claim_text, evidence, hard_violations)
        self._check_stereo(mol, claim_text, evidence, soft_warnings)
        self._check_sa_score(mol, claim_text, evidence, soft_warnings)
        self._check_pains(mol, evidence, soft_warnings)
        self._check_brenk(mol, evidence, soft_warnings)
        self._check_drug_likeness(mol, computed, evidence, soft_warnings, positive_notes)

        # ------------------------------------------------------------------
        # Property comparison
        # ------------------------------------------------------------------
        matches: list[str] = []
        mismatches: list[str] = []
        if stated_props:
            matches, mismatches = _compare_properties(computed, stated_props)
            evidence["matches"] = matches
            evidence["mismatches"] = mismatches
            if mismatches:
                hard_violations.extend(f"Property mismatch: {m}" for m in mismatches)

        db_evidence = await self._cross_reference(valid_smiles, evidence)

        # ------------------------------------------------------------------
        # Cross-database triangulation (two-clocks principle)
        # ------------------------------------------------------------------
        triangulation_boost = 0.0
        triangulation_verdict_override: Verdict | None = None
        triangulation_reasoning: str | None = None

        known_name = self._detect_known_drug_name(claim_text)
        if known_name and self._triangulation is not None:
            try:
                tri = await self._triangulation.triangulate_compound(known_name)
                db_evidence["triangulation"] = self._serialize_triangulation(tri)

                if tri.has_disagreement:
                    # Sources disagree → upstream inconsistency; boost confidence
                    # that something is wrong but don't flip the RDKit-derived verdict.
                    triangulation_boost = 0.05
                    soft_warnings.append(
                        "Cross-database triangulation detected disagreements: "
                        + "; ".join(
                            f"{d.field} ({d.severity}, spread={d.magnitude})"
                            for d in tri.disagreements
                        )
                    )
                elif tri.all_agree and stated_props:
                    # Sources agree — compare claim against consensus.
                    claim_vs_consensus_ok = self._claim_matches_consensus(
                        stated_props, tri
                    )
                    if claim_vs_consensus_ok is True:
                        triangulation_verdict_override = Verdict.verified
                        triangulation_boost = 0.1
                        triangulation_reasoning = (
                            f"Cross-database triangulation of '{known_name}' "
                            f"across {len(tri.sources_with_data)} sources "
                            "confirms the stated properties."
                        )
                    elif claim_vs_consensus_ok is False:
                        triangulation_verdict_override = Verdict.refuted
                        triangulation_boost = 0.15
                        hard_violations.append(
                            "Claim contradicts consensus across "
                            f"{len(tri.sources_with_data)} authoritative "
                            f"sources for '{known_name}'"
                        )
            except Exception as exc:
                logger.debug("Triangulation failed for %r: %s", known_name, exc)

        # ------------------------------------------------------------------
        # Aggregate the overall verdict
        # ------------------------------------------------------------------
        n_checks = 7 + (1 if stated_props else 0)  # rough count for confidence scaling

        if hard_violations:
            verdict = Verdict.refuted
            confidence = min(0.97, 0.75 + 0.05 * len(hard_violations))
            reasoning = (
                f"Hard violations for SMILES '{valid_smiles}': "
                + "; ".join(hard_violations)
                + "."
            )
            if matches:
                reasoning += f" Matched: {'; '.join(matches)}."
        elif matches and not mismatches and soft_warnings:
            # Stated numeric properties verified against computed values —
            # keep the VERIFIED verdict but surface structural warnings in
            # the reasoning and nudge confidence down slightly. A matching
            # numeric property is stronger ground-truth evidence than a
            # substructure advisory (PAINS/Brenk/drug-likeness).
            verdict = Verdict.verified
            confidence = min(
                0.9,
                0.7 + 0.05 * len(matches) - 0.03 * len(soft_warnings),
            )
            reasoning = (
                f"Stated properties match computed values for SMILES "
                f"'{valid_smiles}': {'; '.join(matches)}. "
                f"Structural caveats: {'; '.join(soft_warnings)}."
            )
        elif soft_warnings:
            verdict = Verdict.partially_supported
            confidence = min(0.85, 0.5 + 0.05 * len(soft_warnings) + 0.02 * n_checks)
            reasoning = (
                f"SMILES '{valid_smiles}' is valid but has warnings: "
                + "; ".join(soft_warnings)
                + "."
            )
            if matches:
                reasoning += f" Property checks passed: {'; '.join(matches)}."
        elif matches:
            verdict = Verdict.verified
            confidence = min(0.98, 0.75 + 0.05 * len(matches) + 0.02 * n_checks)
            reasoning = (
                f"All deterministic checks passed for SMILES '{valid_smiles}'. "
                f"Property matches: {'; '.join(matches)}."
            )
            if positive_notes:
                reasoning += " " + " ".join(positive_notes)
        elif stated_props:
            verdict = Verdict.partially_supported
            confidence = 0.45
            reasoning = (
                f"SMILES '{valid_smiles}' is valid and all structural checks "
                "passed, but stated properties could not be mapped to "
                "computable descriptors."
            )
        else:
            # No stated properties; no warnings — acknowledge valid structure.
            verdict = Verdict.partially_supported
            confidence = 0.55
            reasoning = (
                f"SMILES '{valid_smiles}' is a valid molecule "
                f"(MW={computed['mw']:.2f}, logP={computed['logp']:.2f}) and "
                "passes all deterministic structural checks, but no specific "
                "property claims were found to verify."
            )

        # Apply triangulation adjustments after the base aggregation.
        if triangulation_verdict_override is not None:
            verdict = triangulation_verdict_override
        if triangulation_boost:
            confidence = min(0.99, confidence + triangulation_boost)
        if triangulation_reasoning:
            reasoning = f"{reasoning} {triangulation_reasoning}"

        return VerificationOutput(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence=db_evidence,
            source_db="rdkit",
        )

    # ------------------------------------------------------------------
    # Triangulation helpers
    # ------------------------------------------------------------------

    # A very small lexicon of drug names we recognise for triangulation. It
    # mirrors the fallback dataset in the DrugBank client and overlaps with
    # PubChem / ChEMBL fallbacks, so a hit here guarantees at least one
    # source will respond even in fully offline mode.
    _KNOWN_DRUG_NAMES: frozenset[str] = frozenset({
        "aspirin", "ibuprofen", "metformin", "atorvastatin", "omeprazole",
        "losartan", "amlodipine", "simvastatin", "warfarin", "dexamethasone",
        "tamoxifen", "erlotinib", "imatinib", "gefitinib", "sorafenib",
        "sunitinib", "morphine", "codeine", "diazepam", "caffeine",
        "acetaminophen", "penicillin", "amoxicillin", "clopidogrel",
        "lisinopril", "metoprolol", "prednisone",
    })

    def _detect_known_drug_name(self, text: str) -> str | None:
        """Return the first known drug name appearing in *text*, if any."""
        lowered = text.lower()
        for name in self._KNOWN_DRUG_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return name
        return None

    @staticmethod
    def _serialize_triangulation(tri: Any) -> dict[str, Any]:
        """Convert a :class:`TriangulationResult` into an evidence-safe dict."""
        return {
            "query": tri.query,
            "sources_consulted": list(tri.sources_consulted),
            "sources_with_data": list(tri.sources_with_data),
            "consensus_mw": tri.consensus_mw,
            "consensus_logp": tri.consensus_logp,
            "confidence": tri.confidence,
            "disagreements": [
                {
                    "field": d.field,
                    "values": d.values,
                    "magnitude": d.magnitude,
                    "severity": d.severity,
                    "outlier": d.outlier,
                }
                for d in tri.disagreements
            ],
        }

    @staticmethod
    def _claim_matches_consensus(
        stated_props: dict[str, float],
        tri: Any,
    ) -> bool | None:
        """Check whether stated claim values agree with the triangulation consensus.

        Returns
        -------
        bool | None
            ``True`` if at least one stated property matches the consensus
            within tolerance; ``False`` if any stated property contradicts
            the consensus; ``None`` if no comparison could be made.
        """
        checked = False
        any_mismatch = False
        any_match = False

        if "mw" in stated_props and tri.consensus_mw is not None:
            checked = True
            if abs(stated_props["mw"] - tri.consensus_mw) <= max(
                0.5, tri.consensus_mw * 0.01
            ):
                any_match = True
            else:
                any_mismatch = True

        if "logp" in stated_props and tri.consensus_logp is not None:
            checked = True
            if abs(stated_props["logp"] - tri.consensus_logp) <= 0.3:
                any_match = True
            else:
                any_mismatch = True

        if not checked:
            return None
        if any_mismatch:
            return False
        return any_match or None

    # ------------------------------------------------------------------
    # Deterministic structural checks
    # ------------------------------------------------------------------

    def _check_element_validity(
        self,
        smiles: str,
        evidence: dict[str, Any],
        hard_violations: list[str],
    ) -> None:
        """Reject SMILES containing pseudo-element tokens."""
        pseudo = _has_pseudo_element_tokens(smiles)
        if pseudo:
            evidence["pseudo_elements"] = pseudo
            hard_violations.append(
                f"SMILES contains non-element tokens {pseudo}"
            )

    def _check_charge_balance(
        self,
        mol: Any,
        claim_text: str,
        evidence: dict[str, Any],
        hard_violations: list[str],
    ) -> None:
        """Flag contradictions between "neutral" language and formal charge."""
        charge = Chem.GetFormalCharge(mol)
        evidence["formal_charge"] = charge
        if charge != 0 and _NEUTRAL_RE.search(claim_text):
            hard_violations.append(
                f"Claim asserts neutrality but formal charge is {charge:+d}"
            )

    def _check_stereo(
        self,
        mol: Any,
        claim_text: str,
        evidence: dict[str, Any],
        soft_warnings: list[str],
    ) -> None:
        """Detect unassigned stereocenters on a "single molecule" claim."""
        stereo = _check_stereochemistry(mol)
        evidence["stereo_centers"] = stereo
        if stereo["unassigned"] > 0 and _SINGLE_MOLECULE_RE.search(claim_text):
            soft_warnings.append(
                f"{stereo['unassigned']} unassigned stereocenter(s) — "
                "claim treats an undefined stereoisomer as a single compound"
            )

    def _check_sa_score(
        self,
        mol: Any,
        claim_text: str,
        evidence: dict[str, Any],
        soft_warnings: list[str],
    ) -> None:
        """Compare SA-proxy score against "easily synthesized" language."""
        sa = _synthetic_accessibility_proxy(mol)
        evidence["sa_score"] = sa
        if sa["difficult"] and _EASY_SYNTH_RE.search(claim_text):
            soft_warnings.append(
                f"Claim says the compound is easy to synthesize, but SA-proxy "
                f"score is {sa['score']:.1f}/10 (>7 = difficult)"
            )
        if sa["score"] >= 9.0:
            soft_warnings.append(
                f"Molecule has very high synthetic-complexity score ({sa['score']:.1f}/10)"
            )

    def _check_pains(
        self,
        mol: Any,
        evidence: dict[str, Any],
        soft_warnings: list[str],
    ) -> None:
        """Run PAINS filter and register any hits as soft warnings."""
        result = _pains_check(mol)
        evidence["pains_match"] = result
        if result["matched"]:
            entries = ", ".join(result["entries"][:3])
            soft_warnings.append(
                f"PAINS substructure hit ({entries}) — assay interference likely"
            )

    def _check_brenk(
        self,
        mol: Any,
        evidence: dict[str, Any],
        soft_warnings: list[str],
    ) -> None:
        """Run the Brenk filter for reactive/unwanted functional groups."""
        result = _brenk_check(mol)
        evidence["brenk_match"] = result
        if result["matched"]:
            entries = ", ".join(result["entries"][:3])
            soft_warnings.append(
                f"Brenk unwanted-group hit ({entries})"
            )

    def _check_drug_likeness(
        self,
        mol: Any,
        computed: dict[str, float],
        evidence: dict[str, Any],
        soft_warnings: list[str],
        positive_notes: list[str],
    ) -> None:
        """Compute Lipinski / Veber / Ghose / QED and attach to evidence."""
        dl = _drug_likeness(mol, computed)
        evidence["drug_likeness"] = dl

        if not dl["lipinski"]["passes"]:
            soft_warnings.append(
                f"Lipinski Ro5: {dl['lipinski']['n_violations']} violations "
                f"({', '.join(dl['lipinski']['violations'])})"
            )
        if not dl["veber"]["passes"]:
            soft_warnings.append(
                f"Veber rule fails ({', '.join(dl['veber']['violations'])})"
            )
        qed_score = dl["qed"]
        if qed_score is not None:
            if qed_score >= 0.67:
                positive_notes.append(f"QED={qed_score:.2f} (drug-like).")
            elif qed_score < 0.35:
                soft_warnings.append(
                    f"Low QED drug-likeness score ({qed_score:.2f} < 0.35)"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _verify_by_name(
        self,
        claim_text: str,
        stated_props: dict[str, float],
        evidence: dict[str, Any],
    ) -> VerificationOutput:
        """Attempt verification via compound name lookup when no SMILES is present."""
        if self._pubchem is None:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No SMILES found in claim and PubChem client is unavailable.",
                evidence=evidence,
            )

        # Naive compound-name extraction: capitalised words that could be drug names
        name_match = re.search(
            r"\b([A-Z][a-z]{2,}(?:[-\s][a-z]+)*)\b", claim_text
        )
        if not name_match:
            return VerificationOutput(
                verdict=Verdict.unverifiable,
                confidence=0.0,
                reasoning="No SMILES or recognisable compound name found in claim.",
                evidence=evidence,
            )

        compound_name = name_match.group(1)
        try:
            result = await self._pubchem.search_by_name(compound_name)
            if result:
                evidence["pubchem_result"] = result
                return VerificationOutput(
                    verdict=Verdict.partially_supported,
                    confidence=0.4,
                    reasoning=(
                        f"Compound '{compound_name}' found in PubChem, but "
                        "without SMILES the molecular properties cannot be "
                        "independently computed."
                    ),
                    evidence=evidence,
                    source_db="pubchem",
                )
        except Exception as exc:
            logger.warning("PubChem lookup failed for %r: %s", compound_name, exc)

        return VerificationOutput(
            verdict=Verdict.unverifiable,
            confidence=0.0,
            reasoning=f"Could not verify compound '{compound_name}' via PubChem.",
            evidence=evidence,
        )

    async def _cross_reference(
        self,
        smiles: str | None,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Enrich evidence with PubChem / ChEMBL data (best-effort)."""
        if smiles is None:
            return evidence

        if self._pubchem is not None:
            try:
                pc_result = await self._pubchem.search_by_smiles(smiles)
                if pc_result:
                    evidence["pubchem"] = pc_result
            except Exception as exc:
                logger.debug("PubChem cross-ref failed: %s", exc)

        if self._chembl is not None:
            try:
                chembl_result = await self._chembl.search_molecule(smiles)
                if chembl_result:
                    evidence["chembl"] = chembl_result
            except Exception as exc:
                logger.debug("ChEMBL cross-ref failed: %s", exc)

        return evidence

    async def health_check(self) -> bool:
        return _RDKIT_AVAILABLE

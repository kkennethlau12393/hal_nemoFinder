"""Tests for src.verifiers.chemical.ChemicalVerifier."""

from __future__ import annotations

import pytest

from src.models.enums import ClaimType, Verdict

# Guard: skip entire module when RDKit is not installed.
try:
    from rdkit import Chem  # noqa: F401

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit is not installed"),
]


def _make_verifier():
    """Build a ChemicalVerifier with external clients disabled."""
    from src.verifiers.chemical import ChemicalVerifier

    verifier = ChemicalVerifier()
    # Disable external API clients so tests are purely local
    verifier._pubchem = None
    verifier._chembl = None
    return verifier


class TestValidSmilesCorrectMW:
    """Valid SMILES with an accurate molecular-weight claim."""

    async def test_valid_smiles_correct_mw(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has a molecular weight of 180.16 g/mol."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        assert result.verdict == Verdict.verified
        assert result.confidence > 0.5
        assert "match" in result.reasoning.lower() or "verified" in result.reasoning.lower()


class TestValidSmilesWrongMW:
    """Valid SMILES with an incorrect molecular-weight claim."""

    async def test_valid_smiles_wrong_mw(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Aspirin (CC(=O)Oc1ccccc1C(O)=O) has a molecular weight of 300.0 g/mol."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert result.confidence > 0.5
        assert "mismatch" in result.reasoning.lower()


class TestInvalidSmiles:
    """Claim with a completely invalid SMILES string."""

    async def test_invalid_smiles(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "The compound XYZINVALID has a molecular weight of 250 g/mol."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        # XYZINVALID might not be parsed as a SMILES candidate at all (it is
        # purely alphabetical and may fail the heuristic in _extract_smiles_candidates).
        # The verifier should either refute (invalid SMILES) or be unverifiable
        # (no SMILES found).
        assert result.verdict in (Verdict.refuted, Verdict.unverifiable)


class TestNoSmilesInClaim:
    """Claim that contains no SMILES at all."""

    async def test_no_smiles_in_claim(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text="The compound is a white crystalline powder.",
            claim_type=ClaimType.molecular_property,
            context={},
        )
        assert result.verdict == Verdict.unverifiable


class TestValidSmilesNoStatedProperties:
    """Valid SMILES with no numeric property claims to verify."""

    async def test_valid_smiles_no_stated_properties(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "The molecule CC(=O)Oc1ccccc1C(O)=O was synthesised in the lab."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        assert result.verdict == Verdict.partially_supported
        assert result.confidence > 0.0
        assert "valid" in result.reasoning.lower()


class TestDrugLikenessEvidence:
    """Drug-likeness metrics (Lipinski/Veber/Ghose/QED) are attached."""

    async def test_drug_likeness_present_in_evidence(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "Ibuprofen CC(C)Cc1ccc(cc1)C(C)C(=O)O is an NSAID."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        dl = result.evidence.get("drug_likeness")
        assert dl is not None
        assert "lipinski" in dl
        assert "veber" in dl
        assert "ghose" in dl
        assert dl["qed"] is not None
        # Ibuprofen should pass Lipinski
        assert dl["lipinski"]["passes"] is True
        # SA score should also be present and computable
        assert "sa_score" in result.evidence
        assert 1.0 <= result.evidence["sa_score"]["score"] <= 10.0


class TestPainsDetection:
    """A PAINS-matching scaffold triggers a structural warning."""

    async def test_pains_substructure_flagged(self) -> None:
        verifier = _make_verifier()
        # Curcumin-like bis-enone scaffold is a classic PAINS hit.
        result = await verifier.verify(
            claim_text=(
                "The compound "
                "OC1=CC=C(C=C1)/C=C/C(=O)CC(=O)/C=C/C1=CC=C(O)C=C1 "
                "is a promising lead compound."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        # PAINS or Brenk should fire (at minimum the enone is flagged).
        pains = result.evidence.get("pains_match", {})
        brenk = result.evidence.get("brenk_match", {})
        assert pains.get("matched") or brenk.get("matched"), (
            f"Expected PAINS or Brenk hit; got pains={pains} brenk={brenk}"
        )
        assert result.verdict == Verdict.partially_supported


class TestStereochemistryUnassigned:
    """Unassigned stereocenters on a "single compound" claim downgrade verdict."""

    async def test_unassigned_stereocenters_partially_supported(self) -> None:
        verifier = _make_verifier()
        # Alanine CC(N)C(=O)O has one stereocenter but no @ / @@ marker.
        result = await verifier.verify(
            claim_text=(
                "The compound CC(N)C(=O)O is a single compound used as an "
                "amino acid."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        stereo = result.evidence.get("stereo_centers")
        assert stereo is not None
        assert stereo["unassigned"] >= 1
        assert result.verdict == Verdict.partially_supported


class TestChargeBalanceContradiction:
    """"Neutral" language on a formally-charged species is refuted."""

    async def test_neutral_claim_on_charged_species_refuted(self) -> None:
        verifier = _make_verifier()
        result = await verifier.verify(
            claim_text=(
                "The neutral molecule CC(=O)[O-] is a common intermediate."
            ),
            claim_type=ClaimType.molecular_property,
            context={},
        )
        assert result.verdict == Verdict.refuted
        assert result.evidence.get("formal_charge") == -1
        assert "neutral" in result.reasoning.lower() or "charge" in result.reasoning.lower()

"""Seed script that populates a local reference dataset of ~30 well-known drugs.

Uses hardcoded, real-world data instead of calling the ChEMBL API.
Run standalone:  python -m seed.seed_chembl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reference dataset: 30 well-known drugs with real ChEMBL data
# ---------------------------------------------------------------------------
# Sources: ChEMBL 34, PubChem, DrugBank.  Values are consensus/rounded.
# For antibodies and peptides, SMILES is set to None (not small molecules).
# max_clinical_phase: 4 = approved drug.

REFERENCE_DRUGS: list[dict] = [
    {
        "name": "Aspirin",
        "chembl_id": "CHEMBL25",
        "canonical_smiles": "CC(=O)Oc1ccccc1C(O)=O",
        "molecular_weight": 180.16,
        "logP": 1.2,
        "known_targets": ["COX-1 (PTGS1)", "COX-2 (PTGS2)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Ibuprofen",
        "chembl_id": "CHEMBL521",
        "canonical_smiles": "CC(C)Cc1ccc(C(C)C(O)=O)cc1",
        "molecular_weight": 206.28,
        "logP": 3.97,
        "known_targets": ["COX-1 (PTGS1)", "COX-2 (PTGS2)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Metformin",
        "chembl_id": "CHEMBL1431",
        "canonical_smiles": "CN(C)C(=N)NC(=N)N",
        "molecular_weight": 129.16,
        "logP": -1.43,
        "known_targets": ["AMPK", "Mitochondrial complex I"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Caffeine",
        "chembl_id": "CHEMBL113",
        "canonical_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "molecular_weight": 194.19,
        "logP": -0.07,
        "known_targets": ["Adenosine A1 receptor", "Adenosine A2A receptor", "PDE4"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Acetaminophen",
        "chembl_id": "CHEMBL112",
        "canonical_smiles": "CC(=O)Nc1ccc(O)cc1",
        "molecular_weight": 151.16,
        "logP": 0.46,
        "known_targets": ["COX-1 (PTGS1)", "COX-2 (PTGS2)", "COX-3 (CNS variant)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Penicillin G",
        "chembl_id": "CHEMBL29",
        "canonical_smiles": "O=C(CC1=CC=CC=C1)N[C@@H]2C(=O)N3[C@@H]2SC(C)(C)[C@@H]3C(O)=O",
        "molecular_weight": 334.39,
        "logP": 1.83,
        "known_targets": ["Penicillin-binding proteins (PBPs)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Amoxicillin",
        "chembl_id": "CHEMBL1082",
        "canonical_smiles": "N[C@@H](C(=O)N[C@@H]1C(=O)N2[C@@H]1SC(C)(C)[C@@H]2C(O)=O)c1ccc(O)cc1",
        "molecular_weight": 365.40,
        "logP": 0.87,
        "known_targets": ["Penicillin-binding proteins (PBPs)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Atorvastatin",
        "chembl_id": "CHEMBL1487",
        "canonical_smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(O)=O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "molecular_weight": 558.64,
        "logP": 6.36,
        "known_targets": ["HMG-CoA reductase"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Omeprazole",
        "chembl_id": "CHEMBL1503",
        "canonical_smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "molecular_weight": 345.42,
        "logP": 2.23,
        "known_targets": ["H+/K+-ATPase (gastric proton pump)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Losartan",
        "chembl_id": "CHEMBL191",
        "canonical_smiles": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
        "molecular_weight": 422.91,
        "logP": 4.01,
        "known_targets": ["Angiotensin II type 1 receptor (AT1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Metoprolol",
        "chembl_id": "CHEMBL13",
        "canonical_smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "molecular_weight": 267.36,
        "logP": 1.88,
        "known_targets": ["Beta-1 adrenergic receptor (ADRB1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Lisinopril",
        "chembl_id": "CHEMBL1237021",
        "canonical_smiles": "N[C@@H](CCc1ccccc1)C(=O)N1C[C@@H](C(O)=O)C[C@H]1C(=O)O.NCCCC[C@@H](N)C(O)=O",
        "molecular_weight": 405.49,
        "logP": -0.89,
        "known_targets": ["Angiotensin-converting enzyme (ACE)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Amlodipine",
        "chembl_id": "CHEMBL2276",
        "canonical_smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
        "molecular_weight": 408.88,
        "logP": 3.0,
        "known_targets": ["L-type calcium channel (Cav1.2)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Simvastatin",
        "chembl_id": "CHEMBL1064",
        "canonical_smiles": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)CC(=O)O)[C@@H]21",
        "molecular_weight": 418.57,
        "logP": 4.68,
        "known_targets": ["HMG-CoA reductase"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Clopidogrel",
        "chembl_id": "CHEMBL1771",
        "canonical_smiles": "COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1",
        "molecular_weight": 321.82,
        "logP": 3.37,
        "known_targets": ["P2Y12 receptor"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Warfarin",
        "chembl_id": "CHEMBL1464",
        "canonical_smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "molecular_weight": 308.33,
        "logP": 2.7,
        "known_targets": ["Vitamin K epoxide reductase (VKORC1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Dexamethasone",
        "chembl_id": "CHEMBL384467",
        "canonical_smiles": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
        "molecular_weight": 392.46,
        "logP": 1.83,
        "known_targets": ["Glucocorticoid receptor (NR3C1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Prednisone",
        "chembl_id": "CHEMBL635",
        "canonical_smiles": "C[C@]12C=CC(=O)C=C1CC[C@@H]1[C@@H]2[C@@H](O)C[C@]2(C)[C@H]1CC[C@@]2(O)C(=O)CO",
        "molecular_weight": 358.43,
        "logP": 1.46,
        "known_targets": ["Glucocorticoid receptor (NR3C1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Tamoxifen",
        "chembl_id": "CHEMBL83",
        "canonical_smiles": "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "molecular_weight": 371.51,
        "logP": 6.3,
        "known_targets": ["Estrogen receptor alpha (ESR1)", "Estrogen receptor beta (ESR2)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Erlotinib",
        "chembl_id": "CHEMBL553",
        "canonical_smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        "molecular_weight": 393.44,
        "logP": 3.3,
        "known_targets": ["EGFR (ErbB1/HER1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Imatinib",
        "chembl_id": "CHEMBL941",
        "canonical_smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "molecular_weight": 493.60,
        "logP": 3.5,
        "known_targets": ["BCR-ABL", "KIT", "PDGFR-alpha", "PDGFR-beta"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Gefitinib",
        "chembl_id": "CHEMBL939",
        "canonical_smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "molecular_weight": 446.90,
        "logP": 3.2,
        "known_targets": ["EGFR (ErbB1/HER1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Sorafenib",
        "chembl_id": "CHEMBL1336",
        "canonical_smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
        "molecular_weight": 464.82,
        "logP": 3.8,
        "known_targets": ["VEGFR-2", "VEGFR-3", "PDGFR-beta", "RAF-1", "BRAF"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Sunitinib",
        "chembl_id": "CHEMBL535",
        "canonical_smiles": "CCN(CC)CCNC(=O)c1c(C)[nH]c(\\C=C2/C(=O)Nc3ccc(F)cc32)c1C",
        "molecular_weight": 398.47,
        "logP": 2.4,
        "known_targets": ["VEGFR-1", "VEGFR-2", "PDGFR-alpha", "PDGFR-beta", "KIT", "FLT3"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Trastuzumab",
        "chembl_id": "CHEMBL1201585",
        "canonical_smiles": None,
        "molecular_weight": 145531.5,
        "logP": None,
        "known_targets": ["HER2 (ErbB2)"],
        "max_clinical_phase": 4,
        "note": "Monoclonal antibody; no small-molecule SMILES",
    },
    {
        "name": "Rituximab",
        "chembl_id": "CHEMBL1201576",
        "canonical_smiles": None,
        "molecular_weight": 143859.7,
        "logP": None,
        "known_targets": ["CD20"],
        "max_clinical_phase": 4,
        "note": "Monoclonal antibody; no small-molecule SMILES",
    },
    {
        "name": "Insulin",
        "chembl_id": "CHEMBL1566",
        "canonical_smiles": None,
        "molecular_weight": 5807.57,
        "logP": None,
        "known_targets": ["Insulin receptor (INSR)"],
        "max_clinical_phase": 4,
        "note": "Peptide hormone; no standard SMILES representation",
    },
    {
        "name": "Morphine",
        "chembl_id": "CHEMBL70",
        "canonical_smiles": "C[N@@]1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@@H]1C5",
        "molecular_weight": 285.34,
        "logP": 0.89,
        "known_targets": ["Mu opioid receptor (OPRM1)", "Kappa opioid receptor (OPRK1)", "Delta opioid receptor (OPRD1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Codeine",
        "chembl_id": "CHEMBL485",
        "canonical_smiles": "CO[C@H]1C=C[C@H]2[C@@H]3Cc4ccc(O)c5c4[C@]2(CCN3C)[C@@H]1O5",
        "molecular_weight": 299.36,
        "logP": 1.39,
        "known_targets": ["Mu opioid receptor (OPRM1)"],
        "max_clinical_phase": 4,
    },
    {
        "name": "Diazepam",
        "chembl_id": "CHEMBL12",
        "canonical_smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "molecular_weight": 284.74,
        "logP": 2.82,
        "known_targets": ["GABA-A receptor"],
        "max_clinical_phase": 4,
    },
]


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "chembl_reference.json"


def seed() -> None:
    """Write the reference drug dataset to seed/data/chembl_reference.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(REFERENCE_DRUGS, indent=2), encoding="utf-8")

    # Summary
    total = len(REFERENCE_DRUGS)
    small_molecules = sum(1 for d in REFERENCE_DRUGS if d["canonical_smiles"] is not None)
    biologics = total - small_molecules
    targets = {t for d in REFERENCE_DRUGS for t in d["known_targets"]}

    print("=" * 64)
    print("  hal_nemoFinder  --  ChEMBL Reference Seed")
    print("=" * 64)
    print()
    print(f"  Output file : {OUTPUT_FILE}")
    print(f"  Total drugs : {total}")
    print(f"    Small molecules : {small_molecules}")
    print(f"    Biologics       : {biologics}")
    print(f"  Unique targets    : {len(targets)}")
    print()
    print("  Drugs loaded:")
    for drug in REFERENCE_DRUGS:
        tag = ""
        if drug.get("note"):
            tag = f"  ({drug['note'].split(';')[0]})"
        smiles_preview = drug["canonical_smiles"]
        if smiles_preview and len(smiles_preview) > 40:
            smiles_preview = smiles_preview[:37] + "..."
        elif smiles_preview is None:
            smiles_preview = "N/A"
        print(
            f"    {drug['chembl_id']:20s} {drug['name']:20s} "
            f"MW={drug['molecular_weight']:<10.2f} {smiles_preview}{tag}"
        )
    print()
    print(f"  Reference dataset written to {OUTPUT_FILE}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed()

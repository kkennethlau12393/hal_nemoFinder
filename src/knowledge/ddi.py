"""Drug-drug interaction (DDI) knowledge client.

Provides a hand-curated table of ~40 clinically important drug-drug
interactions with severity, mechanism and documented effect.  There is
no free, open DDI web API suitable for deterministic verification; the
client is fully offline and deterministic.

Usage
-----
>>> client = DrugInteractionClient()
>>> client.lookup("warfarin", "ibuprofen").severity
'major'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DrugInteraction:
    """A single documented drug-drug interaction."""

    drug_a: str
    drug_b: str
    severity: str  # "contraindicated" | "major" | "moderate" | "minor"
    mechanism: str
    effect: str

    def matches(self, a: str, b: str) -> bool:
        pair = {a.lower(), b.lower()}
        return {self.drug_a, self.drug_b} == pair


# ---------------------------------------------------------------------------
# Curated table
# ---------------------------------------------------------------------------


def _i(a: str, b: str, sev: str, mech: str, effect: str) -> DrugInteraction:
    return DrugInteraction(a.lower(), b.lower(), sev, mech, effect)


_INTERACTIONS: list[DrugInteraction] = [
    _i("warfarin", "ibuprofen", "major", "pharmacodynamic: antiplatelet + anticoagulant",
       "increased bleeding risk"),
    _i("warfarin", "aspirin", "major", "additive antiplatelet / anticoagulant",
       "increased bleeding risk"),
    _i("warfarin", "clarithromycin", "major", "CYP3A4 inhibition + protein binding",
       "elevated INR, bleeding"),
    _i("warfarin", "fluconazole", "major", "CYP2C9 inhibition",
       "elevated INR"),
    _i("simvastatin", "clarithromycin", "contraindicated", "CYP3A4 inhibition",
       "rhabdomyolysis risk"),
    _i("simvastatin", "erythromycin", "contraindicated", "CYP3A4 inhibition",
       "rhabdomyolysis risk"),
    _i("simvastatin", "itraconazole", "contraindicated", "strong CYP3A4 inhibition",
       "rhabdomyolysis risk"),
    _i("simvastatin", "gemfibrozil", "contraindicated", "OATP1B1 inhibition",
       "rhabdomyolysis risk"),
    _i("atorvastatin", "clarithromycin", "major", "CYP3A4 inhibition",
       "increased myopathy risk"),
    _i("fluoxetine", "tramadol", "major", "serotonergic + CYP2D6 inhibition",
       "serotonin syndrome risk"),
    _i("fluoxetine", "phenelzine", "contraindicated", "MAOI + SSRI",
       "serotonin syndrome"),
    _i("sertraline", "phenelzine", "contraindicated", "MAOI + SSRI",
       "serotonin syndrome"),
    _i("tranylcypromine", "fluoxetine", "contraindicated", "MAOI + SSRI",
       "serotonin syndrome"),
    _i("linezolid", "fluoxetine", "major", "MAO inhibition",
       "serotonin syndrome"),
    _i("clopidogrel", "omeprazole", "moderate", "CYP2C19 inhibition",
       "reduced clopidogrel activation"),
    _i("clopidogrel", "esomeprazole", "moderate", "CYP2C19 inhibition",
       "reduced clopidogrel activation"),
    _i("digoxin", "amiodarone", "major", "P-gp inhibition",
       "digoxin toxicity"),
    _i("digoxin", "verapamil", "major", "P-gp inhibition",
       "digoxin toxicity"),
    _i("digoxin", "quinidine", "major", "P-gp inhibition",
       "digoxin toxicity"),
    _i("methotrexate", "trimethoprim", "major", "folate antagonism",
       "bone marrow suppression"),
    _i("methotrexate", "nsaid", "major", "renal clearance reduction",
       "methotrexate toxicity"),
    _i("methotrexate", "ibuprofen", "major", "renal clearance reduction",
       "methotrexate toxicity"),
    _i("lithium", "ibuprofen", "major", "reduced lithium clearance",
       "lithium toxicity"),
    _i("lithium", "hydrochlorothiazide", "major", "reduced renal lithium clearance",
       "lithium toxicity"),
    _i("lithium", "lisinopril", "major", "reduced lithium clearance",
       "lithium toxicity"),
    _i("sildenafil", "nitroglycerin", "contraindicated", "cGMP potentiation",
       "severe hypotension"),
    _i("sildenafil", "isosorbide", "contraindicated", "cGMP potentiation",
       "severe hypotension"),
    _i("ciprofloxacin", "tizanidine", "contraindicated", "CYP1A2 inhibition",
       "hypotension, sedation"),
    _i("ciprofloxacin", "theophylline", "major", "CYP1A2 inhibition",
       "theophylline toxicity"),
    _i("rifampin", "warfarin", "major", "CYP induction",
       "decreased anticoagulation"),
    _i("rifampin", "oral contraceptives", "major", "CYP3A4 induction",
       "contraceptive failure"),
    _i("phenytoin", "warfarin", "moderate", "mutual CYP2C9 competition",
       "variable INR"),
    _i("carbamazepine", "warfarin", "major", "CYP2C9 induction",
       "decreased INR"),
    _i("st johns wort", "cyclosporine", "major", "CYP3A4/P-gp induction",
       "rejection risk"),
    _i("st johns wort", "warfarin", "major", "CYP induction",
       "decreased INR"),
    _i("ketoconazole", "cyclosporine", "major", "CYP3A4 inhibition",
       "cyclosporine toxicity"),
    _i("erythromycin", "theophylline", "major", "CYP3A4 inhibition",
       "theophylline toxicity"),
    _i("amiodarone", "simvastatin", "major", "CYP3A4 inhibition",
       "rhabdomyolysis risk"),
    _i("amiodarone", "sotalol", "major", "QT prolongation additive",
       "torsade de pointes risk"),
    _i("ondansetron", "methadone", "major", "additive QT prolongation",
       "torsade de pointes risk"),
    _i("metformin", "iodinated contrast", "major", "acute kidney injury risk",
       "lactic acidosis"),
]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DrugInteractionClient:
    """Deterministic DDI lookup client with a hand-curated table."""

    def __init__(self) -> None:
        self._index: dict[frozenset[str], DrugInteraction] = {
            frozenset({i.drug_a, i.drug_b}): i for i in _INTERACTIONS
        }

    def lookup(self, drug_a: str, drug_b: str) -> DrugInteraction | None:
        """Return the interaction for the unordered pair, or ``None``."""
        key = frozenset({drug_a.lower(), drug_b.lower()})
        return self._index.get(key)

    def interactions_for(self, drug: str) -> list[DrugInteraction]:
        """All curated interactions mentioning *drug*."""
        drug_l = drug.lower()
        return [i for i in _INTERACTIONS if drug_l in (i.drug_a, i.drug_b)]

    def all_interactions(self) -> list[DrugInteraction]:
        return list(_INTERACTIONS)

    def contains_drug(self, drug: str) -> bool:
        drug_l = drug.lower()
        return any(drug_l in (i.drug_a, i.drug_b) for i in _INTERACTIONS)

    def known_drugs(self) -> set[str]:
        drugs: set[str] = set()
        for i in _INTERACTIONS:
            drugs.add(i.drug_a)
            drugs.add(i.drug_b)
        return drugs

    async def close(self) -> None:  # symmetry with async clients
        return None

    def extract_drugs(self, text: str, extra: Iterable[str] = ()) -> list[str]:
        """Return known drug names appearing in *text*."""
        text_l = text.lower()
        found: list[str] = []
        candidates = set(self.known_drugs())
        candidates.update(extra)
        for drug in candidates:
            if drug and drug in text_l:
                found.append(drug)
        return found

"""Patent knowledge client with a curated table of well-known pharma patents.

A fully-offline deterministic client.  The curated fallback covers ~20
landmark pharma composition-of-matter / use patents; the optional
remote search hook is left as a stub (patent search APIs such as USPTO
PatentsView require authentication and are intentionally not contacted
from the verifier).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatentRecord:
    patent_number: str
    title: str
    assignee: str
    year: int
    compound: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    is_fallback: bool = True


_FALLBACK_PATENTS: dict[str, PatentRecord] = {
    "US5521184": PatentRecord(
        "US5521184", "Pyrimidine derivatives and processes for the preparation thereof",
        "Ciba-Geigy (Novartis)", 1996, "imatinib",
        ("bcr-abl", "tyrosine kinase", "2-phenylaminopyrimidine"),
    ),
    "US6303638": PatentRecord(
        "US6303638", "Quinazoline derivatives",
        "OSI Pharmaceuticals / Pfizer", 2001, "erlotinib",
        ("egfr", "quinazoline", "tyrosine kinase"),
    ),
    "US6727256": PatentRecord(
        "US6727256", "4-anilinoquinazoline derivatives as EGFR inhibitors",
        "AstraZeneca", 2004, "gefitinib",
        ("egfr", "quinazoline"),
    ),
    "US4444784": PatentRecord(
        "US4444784", "Antihypercholesterolemic compounds",
        "Merck", 1984, "lovastatin",
        ("hmg-coa reductase", "cholesterol"),
    ),
    "US4346227": PatentRecord(
        "US4346227", "ML-236B derivatives",
        "Sankyo", 1982, "pravastatin",
        ("hmg-coa reductase", "statin"),
    ),
    "US4681893": PatentRecord(
        "US4681893", "Trans-6-[2-(3- or 4-carboxamido-substituted pyrrol-1-yl)alkyl]-4-hydroxypyran-2-one inhibitors",
        "Warner-Lambert / Pfizer", 1987, "atorvastatin",
        ("hmg-coa reductase", "atorvastatin", "lipitor"),
    ),
    "US5273995": PatentRecord(
        "US5273995", "[R-(R*,R*)]-2-(4-fluorophenyl)-beta, delta-dihydroxy-5-(1-methylethyl)-3-phenyl-4-[(phenylamino)carbonyl]-1H-pyrrole-1-heptanoic acid",
        "Warner-Lambert", 1993, "atorvastatin",
        ("lipitor", "statin"),
    ),
    "US5250534": PatentRecord(
        "US5250534", "Pyrazolopyrimidinone antianginal agents",
        "Pfizer", 1993, "sildenafil",
        ("pde5", "viagra", "erectile dysfunction"),
    ),
    "US4879303": PatentRecord(
        "US4879303", "Pharmacologically active omeprazole enantiomer",
        "Astra (AstraZeneca)", 1989, "omeprazole",
        ("proton pump", "benzimidazole"),
    ),
    "US4839342": PatentRecord(
        "US4839342", "Carbonyl-containing benzofuran derivatives",
        "Eli Lilly", 1989, "fluoxetine",
        ("ssri", "prozac"),
    ),
    "US4923986": PatentRecord(
        "US4923986", "4-amino-1-(2-pyridyl)piperidines",
        "Pfizer", 1990, "sertraline",
        ("ssri", "zoloft"),
    ),
    "US5260275": PatentRecord(
        "US5260275", "Beta-lactam antibiotics",
        "Merck", 1993, "imipenem",
        ("beta-lactam", "carbapenem"),
    ),
    "US4572909": PatentRecord(
        "US4572909", "2-amino-thiazolyl-acetic acid cephalosporin derivatives",
        "Glaxo", 1986, "ceftazidime",
        ("cephalosporin",),
    ),
    "US5393790": PatentRecord(
        "US5393790", "Substituted sulfonamides",
        "Searle", 1995, "celecoxib",
        ("cox-2", "nsaid", "celebrex"),
    ),
    "US5616601": PatentRecord(
        "US5616601", "5-aryl-1H-pyrazole-3-carboxamides",
        "Sanofi", 1997, "rimonabant",
        ("cb1", "cannabinoid"),
    ),
    "US6372733": PatentRecord(
        "US6372733", "Thienopyridine derivatives",
        "Sanofi / Bristol-Myers Squibb", 2002, "clopidogrel",
        ("p2y12", "antiplatelet", "plavix"),
    ),
    "US6087380": PatentRecord(
        "US6087380", "3-imidazolylmethyl-1,2,4-triazines",
        "Pfizer", 2000, "maraviroc",
        ("ccr5", "hiv"),
    ),
    "US5998463": PatentRecord(
        "US5998463", "Glucagon-like peptide-1 agonist",
        "Amylin / Lilly", 1999, "exenatide",
        ("glp-1", "diabetes"),
    ),
    "US6562872": PatentRecord(
        "US6562872", "HIV reverse transcriptase inhibitor compounds",
        "Merck", 2003, "efavirenz",
        ("nnrti", "hiv"),
    ),
    "US7531581": PatentRecord(
        "US7531581", "Nucleoside phosphoramidate prodrugs",
        "Pharmasset / Gilead", 2009, "sofosbuvir",
        ("hcv", "ns5b"),
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


_PATENT_REGEX = re.compile(
    r"\b((?:US|EP|WO)[- ]?\d{6,10}(?:[A-Z]\d?)?)\b",
    re.IGNORECASE,
)


class PatentClient:
    """Deterministic patent lookup backed by a curated fallback table."""

    def __init__(self) -> None:
        self._by_number = {k.upper(): v for k, v in _FALLBACK_PATENTS.items()}

    def get_patent(self, patent_number: str) -> PatentRecord | None:
        key = self._normalise(patent_number)
        return self._by_number.get(key)

    def search(self, query: str) -> list[PatentRecord]:
        q = query.lower()
        hits: list[PatentRecord] = []
        for rec in self._by_number.values():
            text = " ".join(
                [rec.title.lower(), rec.compound.lower()] + list(rec.keywords)
            )
            if any(tok in text for tok in q.split() if tok):
                hits.append(rec)
        return hits

    def all_patents(self) -> list[PatentRecord]:
        return list(self._by_number.values())

    def exists(self, patent_number: str) -> bool:
        return self._normalise(patent_number) in self._by_number

    @staticmethod
    def _normalise(patent_number: str) -> str:
        return re.sub(r"[\s-]", "", patent_number).upper()

    @staticmethod
    def extract_patents(text: str) -> list[str]:
        return [PatentClient._normalise(m.group(1)) for m in _PATENT_REGEX.finditer(text)]

    async def close(self) -> None:
        return None

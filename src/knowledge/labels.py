"""FDA drug-label client via openFDA /drug/label.json with offline fallback.

Returns a simplified :class:`DrugLabel` record containing approved
indications, contraindications, boxed warnings, and a representative
adult dose range.  Used by the regulatory-label verifier to flag
off-label indication or out-of-range dosage claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .cache import KnowledgeCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoseRange:
    """A simple adult dose range in mg (min / max / unit)."""

    min_mg: float
    max_mg: float
    frequency: str = "daily"


@dataclass(frozen=True, slots=True)
class DrugLabel:
    drug: str
    indications: tuple[str, ...] = field(default_factory=tuple)
    contraindications: tuple[str, ...] = field(default_factory=tuple)
    boxed_warnings: tuple[str, ...] = field(default_factory=tuple)
    adult_dose: DoseRange | None = None
    is_fallback: bool = False

    def has_indication(self, indication: str) -> bool:
        needle = indication.lower()
        return any(needle in ind.lower() for ind in self.indications)


# ---------------------------------------------------------------------------
# Fallback labels (~20 drugs)
# ---------------------------------------------------------------------------

_FALLBACK_LABELS: dict[str, DrugLabel] = {
    "metformin": DrugLabel(
        "metformin",
        indications=("type 2 diabetes mellitus", "glycaemic control"),
        contraindications=("severe renal impairment", "metabolic acidosis"),
        boxed_warnings=("lactic acidosis",),
        adult_dose=DoseRange(500, 2550, "daily"),
        is_fallback=True,
    ),
    "atorvastatin": DrugLabel(
        "atorvastatin",
        indications=("primary hypercholesterolemia", "mixed dyslipidemia",
                     "homozygous familial hypercholesterolemia",
                     "prevention of cardiovascular disease"),
        contraindications=("active liver disease", "pregnancy"),
        boxed_warnings=(),
        adult_dose=DoseRange(10, 80, "daily"),
        is_fallback=True,
    ),
    "warfarin": DrugLabel(
        "warfarin",
        indications=("venous thromboembolism prophylaxis",
                     "atrial fibrillation stroke prevention",
                     "prevention of systemic embolism"),
        contraindications=("active bleeding", "pregnancy"),
        boxed_warnings=("bleeding risk",),
        adult_dose=DoseRange(2, 10, "daily"),
        is_fallback=True,
    ),
    "aspirin": DrugLabel(
        "aspirin",
        indications=("pain", "fever", "inflammation",
                     "secondary prevention of cardiovascular events"),
        contraindications=("active peptic ulcer", "hemophilia"),
        boxed_warnings=(),
        adult_dose=DoseRange(81, 4000, "daily"),
        is_fallback=True,
    ),
    "ibuprofen": DrugLabel(
        "ibuprofen",
        indications=("mild to moderate pain", "fever", "inflammation",
                     "rheumatoid arthritis", "osteoarthritis"),
        contraindications=("peri-operative CABG pain",),
        boxed_warnings=("cardiovascular thrombotic events", "gastrointestinal risk"),
        adult_dose=DoseRange(200, 3200, "daily"),
        is_fallback=True,
    ),
    "simvastatin": DrugLabel(
        "simvastatin",
        indications=("hypercholesterolemia", "mixed dyslipidemia",
                     "prevention of coronary events"),
        contraindications=("active liver disease", "pregnancy",
                           "concomitant strong CYP3A4 inhibitors"),
        boxed_warnings=(),
        adult_dose=DoseRange(5, 40, "daily"),
        is_fallback=True,
    ),
    "lisinopril": DrugLabel(
        "lisinopril",
        indications=("hypertension", "heart failure", "acute myocardial infarction"),
        contraindications=("history of angioedema", "pregnancy", "bilateral renal artery stenosis"),
        boxed_warnings=("fetal toxicity",),
        adult_dose=DoseRange(5, 80, "daily"),
        is_fallback=True,
    ),
    "metoprolol": DrugLabel(
        "metoprolol",
        indications=("hypertension", "angina", "heart failure"),
        contraindications=("severe bradycardia", "second- or third-degree heart block"),
        boxed_warnings=("ischemic heart disease on abrupt discontinuation",),
        adult_dose=DoseRange(25, 450, "daily"),
        is_fallback=True,
    ),
    "amlodipine": DrugLabel(
        "amlodipine",
        indications=("hypertension", "angina", "coronary artery disease"),
        contraindications=("severe aortic stenosis",),
        boxed_warnings=(),
        adult_dose=DoseRange(2.5, 10, "daily"),
        is_fallback=True,
    ),
    "omeprazole": DrugLabel(
        "omeprazole",
        indications=("gerd", "peptic ulcer disease", "zollinger-ellison syndrome",
                     "helicobacter pylori eradication"),
        contraindications=("hypersensitivity",),
        boxed_warnings=(),
        adult_dose=DoseRange(20, 80, "daily"),
        is_fallback=True,
    ),
    "levothyroxine": DrugLabel(
        "levothyroxine",
        indications=("hypothyroidism", "tsh suppression in thyroid cancer"),
        contraindications=("acute myocardial infarction", "uncorrected adrenal insufficiency"),
        boxed_warnings=("not for weight loss",),
        adult_dose=DoseRange(0.025, 0.3, "daily"),
        is_fallback=True,
    ),
    "fluoxetine": DrugLabel(
        "fluoxetine",
        indications=("major depressive disorder", "obsessive compulsive disorder",
                     "bulimia nervosa", "panic disorder"),
        contraindications=("concurrent maoi", "pimozide use"),
        boxed_warnings=("suicidality in children, adolescents and young adults",),
        adult_dose=DoseRange(10, 80, "daily"),
        is_fallback=True,
    ),
    "sertraline": DrugLabel(
        "sertraline",
        indications=("major depressive disorder", "ocd", "ptsd",
                     "social anxiety disorder", "premenstrual dysphoric disorder"),
        contraindications=("maoi use",),
        boxed_warnings=("suicidality",),
        adult_dose=DoseRange(25, 200, "daily"),
        is_fallback=True,
    ),
    "imatinib": DrugLabel(
        "imatinib",
        indications=("chronic myeloid leukemia", "gastrointestinal stromal tumor",
                     "philadelphia chromosome-positive all"),
        contraindications=(),
        boxed_warnings=(),
        adult_dose=DoseRange(400, 800, "daily"),
        is_fallback=True,
    ),
    "erlotinib": DrugLabel(
        "erlotinib",
        indications=("non-small cell lung cancer with egfr mutation", "pancreatic cancer"),
        contraindications=(),
        boxed_warnings=(),
        adult_dose=DoseRange(100, 150, "daily"),
        is_fallback=True,
    ),
    "clopidogrel": DrugLabel(
        "clopidogrel",
        indications=("acute coronary syndrome", "recent myocardial infarction",
                     "recent stroke", "peripheral arterial disease"),
        contraindications=("active bleeding",),
        boxed_warnings=("diminished effectiveness in poor CYP2C19 metabolizers",),
        adult_dose=DoseRange(75, 300, "daily"),
        is_fallback=True,
    ),
    "sildenafil": DrugLabel(
        "sildenafil",
        indications=("erectile dysfunction", "pulmonary arterial hypertension"),
        contraindications=("concomitant nitrates",),
        boxed_warnings=(),
        adult_dose=DoseRange(25, 100, "daily"),
        is_fallback=True,
    ),
    "acetaminophen": DrugLabel(
        "acetaminophen",
        indications=("pain", "fever"),
        contraindications=("severe hepatic impairment",),
        boxed_warnings=("hepatotoxicity",),
        adult_dose=DoseRange(325, 4000, "daily"),
        is_fallback=True,
    ),
    "gabapentin": DrugLabel(
        "gabapentin",
        indications=("postherpetic neuralgia", "partial seizures adjunctive therapy"),
        contraindications=(),
        boxed_warnings=(),
        adult_dose=DoseRange(300, 3600, "daily"),
        is_fallback=True,
    ),
    "pembrolizumab": DrugLabel(
        "pembrolizumab",
        indications=("metastatic melanoma", "non-small cell lung cancer",
                     "classical hodgkin lymphoma", "urothelial carcinoma"),
        contraindications=(),
        boxed_warnings=(),
        adult_dose=DoseRange(200, 400, "every 3 weeks"),
        is_fallback=True,
    ),
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.fda.gov/drug/label.json"
_CACHE_NS = "druglabel"
_CACHE_TTL = 7 * 86400
_DEFAULT_TIMEOUT = 15.0


class DrugLabelClient:
    """Async openFDA drug-label client with caching and offline fallback."""

    def __init__(
        self,
        cache: KnowledgeCache | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._cache = cache
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    async def get_label(self, drug_name: str) -> DrugLabel | None:
        cache_key = f"label:{drug_name.lower()}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return _label_from_dict(cached)

        try:
            resp = await self._http.get(
                _BASE_URL,
                params={
                    "search": f'openfda.generic_name:"{drug_name}"',
                    "limit": "1",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            label = self._parse_label(drug_name, data)
            if label is not None:
                await self._cache_set(cache_key, _label_to_dict(label))
                return label
        except Exception as exc:
            logger.warning("openFDA label fetch failed for %r: %s", drug_name, exc)
        return _FALLBACK_LABELS.get(drug_name.lower())

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_label(drug: str, data: dict[str, Any]) -> DrugLabel | None:
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        indications = tuple(_flatten_strings(r.get("indications_and_usage", [])))
        contra = tuple(_flatten_strings(r.get("contraindications", [])))
        boxed = tuple(_flatten_strings(r.get("boxed_warning", [])))
        return DrugLabel(
            drug=drug,
            indications=indications,
            contraindications=contra,
            boxed_warnings=boxed,
        )

    async def _cache_get(self, key: str) -> Any | None:
        if self._cache is None:
            return None
        try:
            return await self._cache.get(_CACHE_NS, key)
        except Exception as exc:  # pragma: no cover
            logger.debug("DrugLabel cache GET failed: %s", exc)
            return None

    async def _cache_set(self, key: str, value: Any) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.set(_CACHE_NS, key, value, _CACHE_TTL)
        except Exception as exc:  # pragma: no cover
            logger.debug("DrugLabel cache SET failed: %s", exc)

    @staticmethod
    def fallback_label(drug: str) -> DrugLabel | None:
        return _FALLBACK_LABELS.get(drug.lower())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _label_to_dict(label: DrugLabel) -> dict[str, Any]:
    return {
        "drug": label.drug,
        "indications": list(label.indications),
        "contraindications": list(label.contraindications),
        "boxed_warnings": list(label.boxed_warnings),
        "adult_dose": None
        if label.adult_dose is None
        else {
            "min_mg": label.adult_dose.min_mg,
            "max_mg": label.adult_dose.max_mg,
            "frequency": label.adult_dose.frequency,
        },
        "is_fallback": label.is_fallback,
    }


def _label_from_dict(data: dict[str, Any]) -> DrugLabel:
    dose = data.get("adult_dose")
    return DrugLabel(
        drug=str(data.get("drug", "")),
        indications=tuple(data.get("indications", [])),
        contraindications=tuple(data.get("contraindications", [])),
        boxed_warnings=tuple(data.get("boxed_warnings", [])),
        adult_dose=DoseRange(
            min_mg=float(dose["min_mg"]),
            max_mg=float(dose["max_mg"]),
            frequency=str(dose.get("frequency", "daily")),
        )
        if isinstance(dose, dict)
        else None,
        is_fallback=bool(data.get("is_fallback", False)),
    )

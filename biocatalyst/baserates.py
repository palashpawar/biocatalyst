"""Published likelihood-of-approval base rates, used instead of paywalled estimates.

Source: BIO / Informa Biomedtracker / QLS Advisors, "Clinical Development Success
Rates and Contributing Factors 2011-2020" (12,728 phase transitions, 9,704 programs).
Headline figure: 7.9% likelihood of approval from Phase I across all programs.

These are *priors*, not predictions about any specific asset. They exist so the
engine never has to guess and never has to scrape someone's paid model. Edit
freely -- every number here is a dial, not a constant of nature.
"""
from __future__ import annotations

import re

# Likelihood of approval from the start of each phase, all therapeutic areas.
LOA_BY_PHASE: dict[str, float] = {
    "preclinical": 0.04,
    "phase1": 0.079,
    "phase1/2": 0.10,
    "phase2": 0.151,
    "phase2/3": 0.30,
    "phase3": 0.496,
    "nda": 0.853,
    "bla": 0.853,
    "pdufa": 0.90,
    "approved": 1.0,
}

# Single-step phase transition probabilities.
PHASE_TRANSITION: dict[str, float] = {
    "phase1": 0.52,    # P1 -> P2
    "phase2": 0.289,   # P2 -> P3
    "phase3": 0.578,   # P3 -> filing
    "nda": 0.906,      # filing -> approval
    "bla": 0.906,
    "pdufa": 0.906,
}

# Multiplier on LOA_BY_PHASE by therapeutic area. Oncology is the worst performer
# from Phase I in the BIO dataset; rare disease and hematology the best.
TA_MULTIPLIER: dict[str, float] = {
    "oncology": 0.67,
    "hematology": 1.6,
    "rare": 1.9,
    "infectious": 1.2,
    "neurology": 0.75,
    "psychiatry": 0.8,
    "cardiovascular": 0.85,
    "metabolic": 0.9,
    "autoimmune": 0.95,
    "ophthalmology": 1.15,
    "respiratory": 1.0,
    "other": 1.0,
}

# Keyword -> therapeutic area. First match wins, so order matters.
_TA_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("hematology", ("leukemia", "lymphoma", "myeloma", "anemia", "thalassemia",
                    "sickle", "hemophilia", "myelofibrosis", "itp", "mds")),
    ("oncology", ("cancer", "tumor", "tumour", "carcinoma", "sarcoma", "glioma",
                  "melanoma", "oncology", "nsclc", "sclc", "metasta", "neoplas")),
    ("ophthalmology", ("macular", "retina", "retinal", "uveitis", "glaucoma",
                       "ophthalm", "dry eye", "amd", "presbyopia", "myopia",
                       "keratitis", "conjunctivitis", "cataract", "blepharitis",
                       "diabetic macular edema")),
    ("neurology", ("alzheimer", "parkinson", "epilep", "seizure", "als ",
                   "multiple sclerosis", "migraine", "neuropath", "dystrophy",
                   "huntington", "spinal muscular")),
    ("psychiatry", ("depress", "schizophren", "bipolar", "anxiety", "ptsd",
                    "adhd", "autism", "psychiatric")),
    ("cardiovascular", ("cardiac", "heart", "hypertension", "atrial", "thrombo",
                        "cholesterol", "atheroscler", "pulmonary arterial",
                        "cardiomyopathy", "amyloidosis", "amyloid", "hfpef",
                        "heart failure", "angina", "arrhythmi")),
    ("metabolic", ("diabet", "obesity", "nash", "mash", "metabolic", "lipid",
                   "gaucher", "fabry", "phenylketonuria")),
    ("autoimmune", ("lupus", "arthritis", "psoria", "crohn", "colitis",
                    "autoimmune", "sclerosis", "sjogren", "vasculitis",
                    "nephropathy", "nephritis", "igan", "myasthenia")),
    ("infectious", ("hiv", "hepatitis", "influenza", "covid", "vaccine",
                    "bacter", "viral", "virus", "infect", "aureus", "mrsa",
                    "uti", "pyelonephritis", "pneumonia", "sepsis",
                    "tuberculosis", "malaria", "papillomatosis")),
    ("respiratory", ("asthma", "copd", "cystic fibrosis", "respiratory",
                     "sarcoidosis", "pulmonary fibrosis")),
]

# Designations that empirically associate with higher approval odds.
DESIGNATION_BONUS: dict[str, float] = {
    "BTD": 1.35,   # Breakthrough Therapy
    "ODD": 1.25,   # Orphan Drug
    "FTD": 1.15,   # Fast Track
    "RMAT": 1.30,
    "PRIME": 1.20,
}


def as_text(value) -> str:
    """Coerce a possibly-missing cell to a plain string.

    NaN is a float and, crucially, it is *truthy* -- so `if not value` waves it
    straight through and the next `.lower()` raises AttributeError. Whether a
    null arrives as None or NaN depends on pandas dtype inference, which varies
    with how many rows in the column happen to be null, so this failed only in
    CI. Every free-text field from the database goes through here.
    """
    if value is None:
        return ""
    if isinstance(value, float):      # NaN, or a stray numeric cell
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "nat", "none") else text


def _kw_matches(word: str, text: str) -> bool:
    """Keyword match anchored to a word start.

    Plain substring matching quietly misfires: "pyelonephritis" (infectious)
    contains the autoimmune keyword "nephritis", and "autism" contains "uti".
    Anchoring to a word start fixes both, and short keywords additionally
    require a word end so "igan" cannot match inside "ligand". Longer entries
    stay prefix-style so "infect" still catches "infection".
    """
    pattern = r"\b" + re.escape(word)
    if len(word) <= 4:
        pattern += r"\b"
    return re.search(pattern, text) is not None


def classify_ta(indication: str | None) -> str:
    """Map a free-text indication to a therapeutic area bucket."""
    text = as_text(indication).lower()
    if not text:
        return "other"
    # Explicit rare-disease markers, plus the naming conventions that in
    # practice only appear on rare monogenic disease (a named deficiency
    # syndrome, a roman-numeral disease type).
    if ("rare" in text or "orphan" in text
            or re.search(r"\bdeficiency\b", text)
            or re.search(r"\b(?:type|deficiency)[- ][ivx]+\b", text)
            or "dystrophy" in text or "mucopolysacchar" in text):
        return "rare"
    for area, words in _TA_KEYWORDS:
        if any(_kw_matches(w, text) for w in words):
            return area
    return "other"


def loa(stage: str | None, indication: str | None = None,
        designations: str | None = None) -> float:
    """Prior probability of eventual approval, clamped to [0.01, 0.98].

    Therapeutic-area and designation adjustments are damped by how far the
    programme has already come. These multipliers are calibrated on Phase I
    likelihood-of-approval differences, where therapeutic area genuinely
    dominates. At a PDUFA date the filing is accepted and the pivotal data is
    in, so the same adjustment has far less room to move: applying it undamped
    pushed haematology PDUFAs to a 98% prior, which is not a credible number.
    The more certain the base rate, the smaller the adjustment.
    """
    key = as_text(stage).lower().replace(" ", "")
    base = LOA_BY_PHASE.get(key, 0.10)

    multiplier = TA_MULTIPLIER.get(classify_ta(indication), 1.0)
    tags = as_text(designations).upper()
    for tag, bonus in DESIGNATION_BONUS.items():
        if tag in tags:
            multiplier *= bonus

    # Shrink the multiplier toward 1.0 in proportion to the base rate.
    damped = 1.0 + (multiplier - 1.0) * (1.0 - base)
    return max(0.01, min(0.98, base * damped))


def typical_catalyst_move(stage: str | None, market_cap: float | None) -> float:
    """Rough absolute move a binary readout tends to produce, as a fraction.

    Modelled as value-at-risk over market cap rather than a flat percentage.
    A single pipeline asset is worth roughly the same in dollars whoever owns
    it, so the same Phase 3 readout that re-rates a $200M microcap by half is
    noise on a $370B pharma. The earlier flat-percentage version credited
    large caps with double-digit single-asset moves, which manufactured
    spurious "cheap volatility" signals.
    """
    key = as_text(stage).lower().replace(" ", "")

    # Ceiling on the move for a company that is essentially this one asset.
    ceiling = {
        "phase1": 0.22, "phase1/2": 0.25, "phase1,phase2": 0.25,
        "phase2": 0.35, "phase2/3": 0.38, "phase2,phase3": 0.38,
        "phase3": 0.45, "nda": 0.25, "bla": 0.25, "pdufa": 0.30,
    }.get(key, 0.20)

    # Approximate enterprise value at stake in the readout, in dollars.
    at_risk = {
        "phase1": 250e6, "phase1/2": 300e6, "phase1,phase2": 300e6,
        "phase2": 600e6, "phase2/3": 900e6, "phase2,phase3": 900e6,
        "phase3": 1.5e9, "nda": 1.2e9, "bla": 1.2e9, "pdufa": 1.2e9,
    }.get(key, 400e6)

    if not market_cap or market_cap <= 0:
        return ceiling
    return max(0.01, min(ceiling, at_risk / market_cap))

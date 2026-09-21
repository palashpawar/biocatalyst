import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.sources.discovery import _simplify_stage


def test_combined_phases_use_the_earlier_phase():
    # A PHASE1,PHASE2 trial must not inherit Phase 2's higher approval prior.
    assert _simplify_stage("PHASE1,PHASE2") == "phase1/2"
    assert _simplify_stage("PHASE2,PHASE3") == "phase2/3"


def test_single_phases():
    assert _simplify_stage("PHASE3") == "phase3"
    assert _simplify_stage("PHASE1") == "phase1"


def test_missing_phase():
    assert _simplify_stage(None) == "other"
    assert _simplify_stage("") == "other"


def test_drug_id_is_stable_across_runs():
    # Regression: a row counter gave the same catalyst a different id each
    # refresh, so the primary key never matched and rows duplicated.
    import hashlib

    def stable_id(r):
        key = f"{r.get('ticker')}|{r.get('nct_id')}|{r.get('drug_name')}|{r.get('catalyst_date_raw')}"
        return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)

    row = {"ticker": "ABC", "nct_id": "NCT1", "drug_name": "X",
           "catalyst_date_raw": "2026-01-01"}
    other = {**row, "catalyst_date_raw": "2026-02-01"}
    assert stable_id(row) == stable_id(dict(row))
    assert stable_id(row) != stable_id(other)


def test_simplify_stage_handles_nan():
    from biocatalyst.sources.discovery import _simplify_stage
    assert _simplify_stage(float("nan")) == "other"

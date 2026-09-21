import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst import baserates as b


def test_oncology_below_rare():
    assert b.loa("Phase 3", "Metastatic lung cancer") < b.loa("Phase 3", "rare disease")


def test_designations_raise_loa():
    plain = b.loa("Phase 3", "Pulmonary sarcoidosis")
    tagged = b.loa("Phase 3", "Pulmonary sarcoidosis", "BTD ODD")
    assert tagged > plain


def test_loa_is_clamped():
    assert 0.01 <= b.loa("pdufa", "rare", "BTD ODD FTD RMAT PRIME") <= 0.98


def test_later_phase_has_higher_loa():
    assert b.loa("phase1", "x") < b.loa("phase2", "x") < b.loa("phase3", "x")


def test_ta_classification():
    assert b.classify_ta("Metastatic breast cancer") == "oncology"
    assert b.classify_ta("Acute myeloid leukemia") == "hematology"
    assert b.classify_ta("Wet age-related macular degeneration") == "ophthalmology"
    assert b.classify_ta(None) == "other"


def test_microcaps_move_more_than_large_pharma():
    assert (b.typical_catalyst_move("phase3", 80e6)
            > b.typical_catalyst_move("phase3", 4e11))


def test_unknown_stage_falls_back():
    assert b.loa("not-a-phase", "x") > 0


def test_large_pharma_single_asset_move_is_small():
    # Regression: a flat percentage credited mega-caps with double-digit
    # single-asset moves, which fabricated "cheap volatility" signals.
    assert b.typical_catalyst_move("phase3", 371e9) < 0.02


def test_move_declines_monotonically_with_size():
    caps = [80e6, 500e6, 5e9, 50e9, 371e9]
    moves = [b.typical_catalyst_move("phase3", c) for c in caps]
    assert moves == sorted(moves, reverse=True)


def test_missing_market_cap_falls_back_to_ceiling():
    assert b.typical_catalyst_move("phase3", None) == 0.45


def test_comma_separated_phases_from_ctgov_are_recognised():
    # ClinicalTrials.gov returns "PHASE1,PHASE2" rather than "Phase 1/2".
    assert b.typical_catalyst_move("PHASE2,PHASE3", 500e6) == 0.38


def test_ta_adjustment_is_damped_at_late_stages():
    # Regression: undamped multipliers pushed haematology PDUFAs to a 98%
    # prior. The adjustment is calibrated on Phase I differences and has far
    # less room once a filing is accepted.
    assert b.loa("pdufa", "acute myeloid leukemia") < 0.97
    assert b.loa("pdufa", "metastatic lung cancer") > 0.80


def test_early_stage_keeps_full_differentiation():
    onc = b.loa("phase1", "metastatic lung cancer")
    heme = b.loa("phase1", "acute myeloid leukemia")
    assert heme > onc * 1.8


def test_damping_preserves_ordering():
    for stage in ("phase1", "phase2", "phase3", "pdufa"):
        assert (b.loa(stage, "metastatic lung cancer")
                < b.loa(stage, None)
                < b.loa(stage, "acute myeloid leukemia"))


def test_presbyopia_is_ophthalmology():
    assert b.classify_ta("presbyopia") == "ophthalmology"

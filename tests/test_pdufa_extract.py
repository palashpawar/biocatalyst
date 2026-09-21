import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.baserates import classify_ta
from biocatalyst.sources.discovery import (_NOT_A_DRUG, _clean_filing,
                                           _tidy_indication)


def test_exhibit_labels_are_not_drugs():
    # Regression: "EX-99.1" heads every EDGAR exhibit and matched the
    # development-code pattern, so it was captured as the drug name.
    for junk in ["EX-99", "EX-99.1", "EXHIBIT", "FORM", "NDA", "PDUFA", "Q3"]:
        assert _NOT_A_DRUG.match(junk), junk


def test_real_drug_codes_survive():
    for drug in ["INO-3107", "BP-205", "SRP-9001", "BBP-418", "CTx-1301"]:
        assert not _NOT_A_DRUG.match(drug), drug


def test_html_entities_are_decoded():
    out = _clean_filing("<p>WAKIX&#174; for narcolepsy&#8226;</p>")
    assert "®" in out and "WAKIX" in out and "<p>" not in out


def test_indication_trailing_clause_is_cut():
    assert _tidy_indication(
        "IgAN in adults, and assigned a PDUFA target action date of") == "IgAN"
    assert _tidy_indication(
        "high-risk Stage 1 NSCLC announced last quarter") == "high-risk Stage 1 NSCLC"


def test_indication_leading_cohort_is_stripped():
    assert _tidy_indication(
        "adults with Recurrent Respiratory Papillomatosis"
    ) == "Recurrent Respiratory Papillomatosis"


def test_indication_rejects_junk():
    assert _tidy_indication("the") is None
    assert _tidy_indication("a b c") is None
    # Vague or target-only captures carry no therapeutic-area signal.
    assert _tidy_indication("symptomatic") is None
    assert _tidy_indication("epidermal growth factor receptor") is None


def test_long_capture_is_truncated_not_discarded():
    # The leading words carry the disease; the tail is narrative.
    out = _tidy_indication(" ".join(["word"] * 25))
    assert out is not None and len(out.split()) == 10


def test_quote_and_attribution_are_cut():
    assert _tidy_indication(
        'multiple myeloma," said Daniel O\u2019Day, Chairman') == "multiple myeloma"
    assert _tidy_indication(
        "presbyopia and set a PDUFA action date of October 17") == "presbyopia"


def test_keyword_matching_is_word_anchored():
    # Substring matching put pyelonephritis (infectious) in autoimmune via
    # "nephritis", and would have put autism in infectious via "uti".
    assert classify_ta("cUTI, including pyelonephritis") == "infectious"
    assert classify_ta("lupus nephritis") == "autoimmune"
    assert classify_ta("autism spectrum disorder") == "psychiatry"


def test_cardiomyopathy_and_rare_markers():
    assert classify_ta("transthyretin amyloid cardiomyopathy") == "cardiovascular"
    assert classify_ta("Severe Leukocyte Adhesion Deficiency-I") == "rare"

import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.sources import financing as fz
from biocatalyst.sources.insider import OPEN_MARKET, _parse_form4

TODAY = dt.date(2026, 9, 22)


def _filings(pairs):
    return [(f, d) for f, d in pairs]


def test_shelf_expires_after_three_years():
    fresh = fz.financing_at(_filings([("S-3", "2025-01-02")]), TODAY)
    stale = fz.financing_at(_filings([("S-3", "2020-01-02")]), TODAY)
    assert fresh["shelf_live"] and not stale["shelf_live"]


def test_automatic_shelf_counts():
    r = fz.financing_at(_filings([("S-3ASR", "2026-01-05")]), TODAY)
    assert r["shelf_live"] and r["shelf_count"] == 1


def test_future_filings_are_invisible():
    # Point-in-time: a filing dated after `asof` cannot be known.
    r = fz.financing_at(_filings([("424B5", "2026-12-01")]), TODAY)
    assert r["offerings_total"] == 0 and r["last_offering_date"] is None


def test_offering_recency_and_count():
    r = fz.financing_at(_filings([
        ("424B5", "2026-08-01"), ("424B5", "2025-06-01"),
        ("424B5", "2019-01-01")]), TODAY)
    assert r["offerings_24m"] == 2
    assert r["offerings_total"] == 3
    assert r["days_since_offering"] == (TODAY - dt.date(2026, 8, 1)).days


def test_funded_company_is_never_loaded():
    # Regression: a mega-cap with a shelf and many 424B5s scored "likely",
    # but for a large filer those prospectuses are usually debt, not equity.
    # Capability without need must not read as dilution risk.
    rich = fz.financing_at(_filings(
        [("S-3ASR", "2026-01-01")] + [("424B5", "2025-0%d-01" % i) for i in range(1, 7)]),
        TODAY)
    assert fz.dilution_readiness(rich, None)["dilution_label"] == "possible"
    assert fz.dilution_readiness(rich, 60.0)["dilution_label"] == "possible"


def test_broke_serial_issuer_is_loaded():
    broke = fz.financing_at(_filings(
        [("S-3", "2026-01-01"), ("424B5", "2026-07-01"), ("424B5", "2025-11-01")]),
        TODAY)
    out = fz.dilution_readiness(broke, 3.0)
    assert out["dilution_label"] == "loaded"
    assert "offerings in 24mo" in out["dilution_notes"]


def test_readiness_is_ordered():
    cold = fz.dilution_readiness(fz.financing_at([], TODAY), 3.0)
    assert cold["dilution_readiness"] < fz.dilution_readiness(
        fz.financing_at(_filings([("S-3", "2026-01-01")]), TODAY), 3.0
    )["dilution_readiness"]


FORM4 = """<ownershipDocument>
  <nonDerivativeTransaction>
    <transactionCode>F</transactionCode>
    <transactionShares><value>13272</value></transactionShares>
  </nonDerivativeTransaction>
  <nonDerivativeTransaction>
    <transactionCode>P</transactionCode>
    <transactionShares><value>5000</value></transactionShares>
    <transactionPricePerShare><value>3.20</value></transactionPricePerShare>
  </nonDerivativeTransaction>
  <nonDerivativeTransaction>
    <transactionCode>A</transactionCode>
    <transactionShares><value>9999</value></transactionShares>
  </nonDerivativeTransaction>
</ownershipDocument>"""


def test_only_open_market_codes_count():
    # Grants (A), exercises (M) and tax withholding (F) are 74% of Form 4
    # rows. An F is an automatic disposal to cover a tax bill; counting it
    # would make every vesting date look bearish.
    assert OPEN_MARKET == {"P", "S"}
    tx = _parse_form4(FORM4)
    assert len(tx) == 1
    assert tx[0]["code"] == "P" and tx[0]["shares"] == 5000.0
    assert tx[0]["price"] == 3.20


def test_form4_with_no_open_market_activity():
    only_tax = FORM4.replace("<transactionCode>P</transactionCode>",
                             "<transactionCode>F</transactionCode>")
    assert _parse_form4(only_tax) == []


def test_schema_migration_adds_new_columns(tmp_path, monkeypatch):
    # Regression: CREATE TABLE IF NOT EXISTS is a no-op on an existing table,
    # so adding a column to SCHEMA did nothing to an older database and the
    # next insert failed on the missing column.
    import duckdb
    from biocatalyst import db

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE insider (ticker VARCHAR, snapshot_date DATE)")
    db._migrate(con)
    cols = {r[0] for r in con.execute("DESCRIBE insider").fetchall()}
    assert {"senior_net_usd", "cluster_buy", "n_buyers"} <= cols
    con.close()


def test_migration_is_idempotent():
    import duckdb
    from biocatalyst import db
    con = duckdb.connect(":memory:")
    con.execute(db.SCHEMA)
    db._migrate(con)
    before = {r[0] for r in con.execute("DESCRIBE insider").fetchall()}
    db._migrate(con)
    assert {r[0] for r in con.execute("DESCRIBE insider").fetchall()} == before
    con.close()


def test_freshest_tag_wins_over_an_abandoned_one():
    # Regression: _units returned the first tag with any rows. CELZ stopped
    # reporting CashAndCashEquivalentsAtCarryingValue in 2023 and moved to the
    # restricted-cash tag, so the live board showed a three-year-old balance.
    from biocatalyst.sources.edgar import _units
    facts = {"us-gaap": {
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
            {"end": "2023-09-30", "val": 1, "filed": "2023-11-01"}]}},
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": {
            "units": {"USD": [{"end": "2026-06-30", "val": 2,
                               "filed": "2026-08-01"}]}},
    }}
    rows = _units(facts, ["CashAndCashEquivalentsAtCarryingValue",
                          "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
    assert rows and rows[0]["end"] == "2026-06-30"


def test_tags_are_not_merged():
    # Merging would mix definitions and make a tie on period end arbitrary.
    from biocatalyst.sources.edgar import _units
    facts = {"us-gaap": {
        "A": {"units": {"USD": [{"end": "2026-06-30", "val": 10, "filed": "x"}]}},
        "B": {"units": {"USD": [{"end": "2026-06-30", "val": 99, "filed": "x"}]}},
    }}
    assert len(_units(facts, ["A", "B"])) == 1


def test_missing_tags_return_empty():
    from biocatalyst.sources.edgar import _units
    assert _units({"us-gaap": {}}, ["Nope"]) == []

import datetime as dt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from biocatalyst.sources.news import _parse_rss, score_text


def test_clear_positive_and_negative():
    assert score_text("FDA approval granted")[0] > 0
    assert score_text("Shares plunge after clinical hold")[1] > 0


def test_negation_flips_positive_words():
    # "not approved" is the single most important case to get right.
    pos, neg = score_text("Drug not approved by the FDA")
    assert neg > 0 and pos == 0


def test_negated_endpoint_language():
    for phrase in ["Trial did not meet primary endpoint",
                   "Study failed to meet its primary endpoint",
                   "Drug did not achieve statistical significance"]:
        pos, neg = score_text(phrase)
        assert neg > 0, phrase
        assert pos == 0, phrase


def test_met_endpoint_is_positive():
    pos, neg = score_text("Phase 3 trial met primary endpoint")
    assert pos > 0 and neg == 0


def test_complete_response_letter_is_negative():
    assert score_text("FDA issues complete response letter")[1] > 0


def test_financing_language_is_negative():
    # Ties to the one validated setup: offerings are the dilution event.
    for h in ["Announces $100M public offering",
              "Company announces at-the-market facility",
              "Announces reverse stock split"]:
        assert score_text(h)[1] > 0, h


def test_neutral_headline_scores_nothing():
    assert score_text("Company to present at investor conference") == (0, 0)


def test_empty_input():
    assert score_text("") == (0, 0)
    assert score_text(None) == (0, 0)


def test_rss_parsing():
    xml = """<?xml version="1.0"?><rss><channel>
      <item><title>Drug X approved</title>
            <pubDate>Mon, 21 Sep 2026 09:22:40 GMT</pubDate>
            <link>http://e.com/1</link></item>
      <item><title>No date here</title><link>http://e.com/2</link></item>
    </channel></rss>"""
    items = _parse_rss(xml)
    assert len(items) == 2
    assert items[0]["published"] == dt.date(2026, 9, 21)
    assert items[1]["published"] is None


def test_rss_parsing_survives_malformed_xml():
    assert _parse_rss("<rss><channel><item>broken") == []


def test_company_tokens_exclude_generic_words():
    from biocatalyst.sources.news import company_tokens
    toks = company_tokens("MRK", "Merck & Company Inc.")
    assert "merck" in toks and "mrk" in toks
    # "company" would match almost any headline.
    assert "company" not in toks and "inc" not in toks


def test_offtopic_headline_is_rejected():
    from biocatalyst.sources.news import company_tokens, is_relevant
    mrk = company_tokens("MRK", "Merck & Company Inc.")
    # Google News returns mentions, not just subjects.
    assert not is_relevant("Novo Nordisk shares plunge over 7%", mrk)
    assert is_relevant("Merck raises full-year guidance", mrk)


def test_relevance_matches_whole_words_only():
    from biocatalyst.sources.news import company_tokens, is_relevant
    toks = company_tokens("INO", "Inovio Pharmaceuticals")
    assert not is_relevant("Casino stocks rally", toks)
    assert is_relevant("Inovio doses first patient", toks)

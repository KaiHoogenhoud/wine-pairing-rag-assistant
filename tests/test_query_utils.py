"""Tests for the pure query helpers: parsing LLM replies, coverage check, merging results."""

from prompts import COVERAGE_LOOSE, COVERAGE_OK
from query_utils import (
    best_distinct,
    coverage_note,
    empty_results,
    fold,
    format_context,
    merge_results,
    parse_recipe_query,
    parse_wine_query,
    wine_is_covered,
)


def make_results(*rows):
    """Build a search result from (id, title, text, distance) tuples."""
    return {
        "ids": [r[0] for r in rows],
        "metadatas": [{"title": r[1], "source": "Wikipedia"} for r in rows],
        "documents": [r[2] for r in rows],
        "distances": [r[3] for r in rows],
    }


BAROLO = make_results(("wikipedia/barolo::0", "Barolo", "Barolo > Wine styles\nTar and roses.", 0.25))


class TestParseWineQuery:
    def test_wine_question(self):
        reply = "TOPIC: wine\nWINE: Chablis\nSEARCH: Chablis food pairing, body, acidity"
        assert parse_wine_query(reply, "q") == ("wine", "Chablis", "Chablis food pairing, body, acidity")

    def test_dish_question_is_out_of_scope(self):
        reply = "TOPIC: dish\nWINE: none\nSEARCH: none"
        assert parse_wine_query(reply, "What wine goes with curry?")[0] == "dish"

    def test_off_topic_labelled_as_wine_is_caught(self):
        # qwen2.5:7b labels "What's the capital of France?" as a wine question
        question = "What's the capital of France?"
        reply = "TOPIC: wine\nWINE: none\nSEARCH: none"
        assert parse_wine_query(reply, question)[0] == "other"

    def test_generic_wine_without_name_stays_on_topic(self):
        reply = "TOPIC: wine\nWINE: none\nSEARCH: dry white wine food pairing"
        assert parse_wine_query(reply, "q") == ("wine", None, "dry white wine food pairing")

    def test_ignored_format_falls_back_to_question(self):
        question = "I have a Merlot, what should I cook?"
        assert parse_wine_query("Sure! Merlot is great.", question) == ("wine", None, question)


class TestParseRecipeQuery:
    def test_takes_search_line(self):
        reply = "TRAITS: light, crisp, high acidity\nSEARCH: grilled fish, shellfish, salad"
        assert parse_recipe_query(reply) == "grilled fish, shellfish, salad"

    def test_ignored_format_returns_whole_reply(self):
        assert parse_recipe_query('"braised beef, roast lamb"') == "braised beef, roast lamb"


class TestCoverage:
    def test_fold_removes_accents_and_case(self):
        assert fold("Gewürztraminer") == "gewurztraminer"

    def test_named_wine_found(self):
        assert wine_is_covered("Barolo", BAROLO)

    def test_vintage_and_qualifiers_still_match_title(self):
        assert wine_is_covered("Barolo 2019", BAROLO)
        chianti = make_results(("c::0", "Chianti", "Chianti > History\nTuscany.", 0.3))
        assert wine_is_covered("Chianti Classico", chianti)

    def test_unknown_wine_not_covered(self):
        assert not wine_is_covered("Saperavi from Georgia", BAROLO)

    def test_generic_words_never_count(self):
        red = make_results(("r::0", "Red wine", "Red wine > Color\nAnthocyanins.", 0.3))
        assert not wine_is_covered("Georgian red", red)

    def test_note_for_unknown_wine(self):
        note = coverage_note("Saperavi", BAROLO)
        assert "NO information about Saperavi" in note

    def test_note_for_loose_match(self):
        far = make_results(("m::0", "Merlot", "Merlot > History\n...", 0.55))
        assert coverage_note(None, far) == COVERAGE_LOOSE

    def test_note_for_good_match(self):
        assert coverage_note("Barolo", BAROLO) == COVERAGE_OK

    def test_note_for_no_results(self):
        assert coverage_note(None, empty_results()) == COVERAGE_LOOSE


class TestCombiningResults:
    def test_merge_keeps_order(self):
        a = make_results(("a", "A", "a", 0.3))
        b = make_results(("b", "B", "b", 0.1))
        assert merge_results(a, b)["ids"] == ["a", "b"]

    def test_best_distinct_dedupes_and_sorts(self):
        first = make_results(("x", "X", "x", 0.5), ("y", "Y", "y", 0.4))
        second = make_results(("x", "X", "x", 0.2), ("z", "Z", "z", 0.6))
        best = best_distinct([first, second], n_results=2)
        assert best["ids"] == ["x", "y"]
        assert best["distances"] == [0.2, 0.4]

    def test_format_context_numbers_from_start(self):
        text = format_context(BAROLO, start=5)
        assert text.startswith("[5] (Wikipedia: Barolo)")

    def test_format_context_empty(self):
        assert format_context(empty_results()) == "(nothing relevant found)"

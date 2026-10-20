"""Tests for cleaning and naming the downloaded Wikipedia/Wikibooks pages."""

from download_data import clean_text, slugify


def test_slugify_is_snake_case_ascii():
    assert slugify("Cookbook:Crème Brûlée") == "creme_brulee"
    assert slugify("Duck à l'Orange") == "duck_a_lorange"
    assert slugify("Cava (Spanish wine)") == "cava_spanish_wine"


def test_clean_text_removes_breadcrumb_and_reference_sections():
    raw = (
        "Cookbook | Recipes | France\n"
        "A stew from Burgundy.\n"
        "== Ingredients ==\n"
        "1 kg beef\n"
        "=== Optional ===\n"
        "Mushrooms\n"
        "== References ==\n"
        "Some book, 1961.\n"
        "== See also ==\n"
        "Coq au vin\n"
    )
    cleaned = clean_text(raw)
    assert cleaned == "A stew from Burgundy.\n## Ingredients\n1 kg beef\n### Optional\nMushrooms"


def test_clean_text_drops_empty_headings():
    assert clean_text("Intro text.\n== Empty ==\n== Filled ==\nContent") == (
        "Intro text.\n## Filled\nContent"
    )

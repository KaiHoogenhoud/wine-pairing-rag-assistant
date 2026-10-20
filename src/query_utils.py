"""Pure helper functions for the query pipeline.

Nothing here calls an LLM or the vector database, so every function can be
unit-tested directly (see tests/test_query_utils.py).
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from config import GENERIC_WINE_WORDS, MAX_WINE_DISTANCE
from prompts import COVERAGE_LOOSE, COVERAGE_NOT_FOUND, COVERAGE_OK, OUT_OF_SCOPE_ANSWERS

RESULT_KEYS = ("ids", "documents", "metadatas", "distances")


def empty_results() -> Dict[str, List]:
    """A search result with no hits, in the same shape VectorDB.search returns."""
    return {key: [] for key in RESULT_KEYS}


def parse_wine_query(reply: str, question: str) -> Tuple[str, Optional[str], str]:
    """
    Read the TOPIC/WINE/SEARCH lines of the first LLM step.

    Args:
        reply: Raw LLM reply to WINE_QUERY_PROMPT
        question: The user's original question, used as fallback search query

    Returns:
        (topic, wine, search): topic is "wine", "dish" or "other"; wine is the
        named wine or None; search is the query for the wine knowledge.
        If the LLM ignored the format, the question counts as a wine question
        and the raw question is used as search query.
    """
    topic, wine, search = "wine", None, question
    for line in reply.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().upper(), value.strip().strip('"')
        if key == "TOPIC" and value.lower() in OUT_OF_SCOPE_ANSWERS:
            topic = value.lower()
        elif key == "WINE" and value.lower() not in {"", "none", "n/a"}:
            wine = value
        elif key == "SEARCH" and value and value.lower() != "none":
            search = value
    # A real wine question has a wine or at least a search query. Small models
    # sometimes label off-topic questions as "wine" (e.g. anything about France).
    if topic == "wine" and wine is None and search == question and "SEARCH" in reply.upper():
        topic = "other"
    return topic, wine, search


def parse_recipe_query(reply: str) -> str:
    """
    Read the SEARCH line of the recipe query step.

    Args:
        reply: Raw LLM reply to RECIPE_QUERY_PROMPT

    Returns:
        The food search query; the whole reply if the format was ignored.
    """
    for line in reply.splitlines():
        key, _, value = line.partition(":")
        if key.strip().upper() == "SEARCH" and value.strip():
            return value.strip().strip('"')
    return reply.strip().strip('"')


def fold(text: str) -> str:
    """Lowercase and strip accents, so 'Gewurztraminer' matches 'Gewürztraminer'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def wine_is_covered(wine: str, results: Dict[str, Any]) -> bool:
    """
    Check whether the retrieved chunks are about the named wine.

    Matches the full name anywhere in a chunk ("Pinot noir"), or a distinctive
    word of it in a page title ("Barolo 2019" -> Barolo, "Chianti Classico" ->
    Chianti). Generic words like "red" or "wine" never count.

    Args:
        wine: Wine name as extracted from the question
        results: Search results with "documents" and "metadatas"

    Returns:
        True if the knowledge base has information about this wine.
    """
    name = fold(wine)
    titles = [fold(m["title"]) for m in results["metadatas"]]
    texts = [fold(d) for d in results["documents"]]
    if any(name in t for t in titles + texts):
        return True
    words = [w for w in re.findall(r"[a-z]+", name) if len(w) >= 3 and w not in GENERIC_WINE_WORDS]
    return any(re.search(rf"\b{w}\b", t) for w in words for t in titles)


def coverage_note(wine: Optional[str], results: Dict[str, Any]) -> str:
    """
    Tell the LLM, in plain words, whether the knowledge base covers the question.

    Prompt instructions alone are not reliable enough: a small local model
    happily describes a grape that appears nowhere in the context. So the code
    checks coverage itself and states the outcome explicitly.

    Args:
        wine: Named wine, or None if the question names none
        results: Retrieved wine-knowledge chunks

    Returns:
        One of the COVERAGE_* instructions from prompts.py.
    """
    if wine and not wine_is_covered(wine, results):
        return COVERAGE_NOT_FOUND.format(wine=wine)
    if not results["distances"] or min(results["distances"]) > MAX_WINE_DISTANCE:
        return COVERAGE_LOOSE
    return COVERAGE_OK


def merge_results(*results: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate several search results into one, keeping their order."""
    return {key: sum((r[key] for r in results), []) for key in RESULT_KEYS}


def best_distinct(results: List[Dict[str, Any]], n_results: int) -> Dict[str, Any]:
    """
    Combine results of several searches, keeping each chunk once at its best distance.

    Args:
        results: Search results, e.g. one for the question and one for a rewrite
        n_results: How many chunks to keep

    Returns:
        The n_results closest distinct chunks, sorted by distance.
    """
    best = {}
    for res in results:
        for row in zip(*(res[key] for key in RESULT_KEYS)):
            chunk_id, distance = row[0], row[3]
            if chunk_id not in best or distance < best[chunk_id][3]:
                best[chunk_id] = row
    rows = sorted(best.values(), key=lambda r: r[3])[:n_results]
    return {key: [row[i] for row in rows] for i, key in enumerate(RESULT_KEYS)}


def format_context(results: Dict[str, Any], start: int = 1) -> str:
    """
    Number retrieved chunks so the LLM can cite them as [1], [2], ...

    Args:
        results: Search results with "documents" and "metadatas"
        start: Number of the first chunk (recipes continue after wine chunks)

    Returns:
        The chunks as one text block, or a placeholder if there are none.
    """
    blocks = []
    for i, (text, meta) in enumerate(zip(results["documents"], results["metadatas"]), start):
        blocks.append(f"[{i}] ({meta.get('source', 'unknown')}: {meta['title']})\n{text}")
    return "\n\n".join(blocks) if blocks else "(nothing relevant found)"

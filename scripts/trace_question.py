"""Trace one question through the RAG pipeline and print every intermediate step.

Shows the raw LLM replies, each search with its metadata filter and distances,
the coverage check, the prompt sent to the LLM and the final answer. Use it to
understand the pipeline, or to find out where a wrong answer goes wrong:
a bad search query, irrelevant chunks, a wrong coverage verdict or the LLM
ignoring good context.

Usage (uses the LLM configured in .env, and an existing index):
    python scripts/trace_question.py "I was given a bottle of Barolo, what should I cook?"
    python scripts/trace_question.py "..." --full-prompt   # print the complete prompt
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from app import THEORY_FILTER, WINE_FILTER, RAGAssistant, build_index  # noqa: E402
from prompts import OUT_OF_SCOPE_ANSWERS  # noqa: E402

PREVIEW_CHARS = 80


def header(title: str) -> None:
    print(f"\n{'-' * 80}\n{title}\n{'-' * 80}")


def indent(text: str) -> str:
    return "    " + text.strip().replace("\n", "\n    ")


def print_chunks(results: Dict[str, Any]) -> None:
    """One line per chunk: distance, category, 'Title > Section' and a text preview."""
    if not results["documents"]:
        print("    (no results)")
    for meta, distance, doc in zip(results["metadatas"], results["distances"], results["documents"]):
        prefix, _, body = doc.partition("\n")
        preview = body.replace("\n", " ")[:PREVIEW_CHARS]
        print(f"    {distance:.2f}  [{meta['category']}] {prefix}\n          {preview}...")


def trace(assistant: RAGAssistant, question: str, full_prompt: bool) -> None:
    """Run the pipeline for one question and print each step."""
    print(f"\nQUESTION: {question}")
    results = assistant.retrieve(question)

    header("STEP 1  LLM call 1: classify the question and write a wine search query")
    print(indent(results["wine_reply"]))
    print(f"\n    -> topic: {results['topic']}")
    if results["topic"] != "wine":
        print("    -> out of scope: fixed answer, no search and no further LLM calls\n")
        print(indent(OUT_OF_SCOPE_ANSWERS[results["topic"]]))
        return
    print(f"    -> named wine: {results['named_wine']}")
    print(f"    -> wine search query: {results['wine_query']}")

    header(f"STEP 2  Search wine knowledge   filter: {WINE_FILTER}")
    for query in dict.fromkeys([question, results["wine_query"]]):
        print(f"  query: {query!r}")
        print_chunks(assistant.vector_db.search(query, config.N_WINE_RESULTS, WINE_FILTER))
    n_wine = len(results["wine"]["documents"]) - config.N_THEORY_RESULTS
    print(f"  -> kept the {n_wine} best distinct chunks of both searches")

    header(f"STEP 3  Search pairing principles   filter: {THEORY_FILTER}")
    theory = {key: values[n_wine:] for key, values in results["wine"].items()}
    print_chunks(theory)

    header("STEP 4  Coverage check (code, not the LLM)")
    print(f"    best distance: {min(results['wine']['distances']):.2f} "
          f"(loose-match threshold {config.MAX_WINE_DISTANCE})")
    print(f"    -> {results['coverage_note']}")

    header("STEP 5  LLM call 2: wine traits -> food search query, then search recipes")
    print(indent(results["recipe_reply"]))
    print(f"\n    -> recipe search query: {results['recipe_query']}\n")
    print_chunks(results["recipe"])

    header("STEP 6  LLM call 3: write the answer from the numbered chunks")
    inputs = RAGAssistant.answer_inputs(question, results)
    prompt = assistant.prompt_template.format(**inputs)
    if full_prompt:
        print(indent(prompt))
    else:
        for line in prompt.splitlines():
            if line.startswith(("[", "Coverage check", "Wine knowledge", "Recipes", "Question")):
                print(f"    | {line[:100]}")
        print("    (instructions and chunk texts left out; use --full-prompt to see them)")
    print(f"    prompt length: {len(prompt):,} characters (~{len(prompt) // 4:,} tokens)")

    header("ANSWER")
    print(indent(assistant.chain.invoke(inputs)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one question through the RAG pipeline")
    parser.add_argument("question", help="the question to trace")
    parser.add_argument("--full-prompt", action="store_true", help="print the complete final prompt")
    args = parser.parse_args()

    assistant = RAGAssistant()
    build_index(assistant, rebuild=False)
    trace(assistant, args.question, args.full_prompt)


if __name__ == "__main__":
    main()

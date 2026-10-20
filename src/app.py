"""Wine pairing assistant: suggests a dish to cook with a bottle of wine you have.

Entry point. Run from the repository root:

    python src/app.py "I was given a bottle of Barolo, what should I cook?"
    python src/app.py            # interactive mode
    python src/app.py --rebuild  # re-index the data folder first
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import config
from prompts import OUT_OF_SCOPE_ANSWERS, RAG_PROMPT, RECIPE_QUERY_PROMPT, WINE_QUERY_PROMPT
from query_utils import (
    best_distinct,
    coverage_note,
    format_context,
    merge_results,
    parse_recipe_query,
    parse_wine_query,
)
from vectordb import VectorDB

REQUIRED_METADATA = ("title", "category", "source")

# Metadata filters that split the knowledge base into the three search groups
WINE_FILTER = {"category": {"$nin": ["recipe", "pairing_theory"]}}
THEORY_FILTER = {"category": "pairing_theory"}
RECIPE_FILTER = {"category": "recipe"}


def parse_markdown_file(path: Path) -> Dict[str, Any]:
    """
    Split a data file into its metadata header and Markdown body.

    Args:
        path: Markdown file with a "---" delimited "key: value" header

    Returns:
        {"content": str, "metadata": dict}; metadata always has doc_id, title
        and category (falling back to the file name and "other").
    """
    text = path.read_text(encoding="utf-8")
    metadata = {}
    if text.startswith("---\n"):
        header, _, text = text[4:].partition("\n---\n")
        for line in header.splitlines():
            key, _, value = line.partition(": ")
            metadata[key.strip()] = value.strip()
    metadata["doc_id"] = f"{path.parent.name}/{path.stem}"
    metadata.setdefault("title", path.stem)
    metadata.setdefault("category", "other")
    return {"content": text.strip(), "metadata": metadata}


def load_documents(data_dir: Path = config.DATA_DIR) -> List[Dict[str, Any]]:
    """
    Load all Markdown documents from the data directory.

    Files starting with "_" (templates) and the generated SOURCES.md are
    skipped. Unreadable, empty or incomplete files are skipped with a warning
    instead of stopping the whole load.

    Args:
        data_dir: Folder with one sub-folder per source (wikipedia/, wikibooks/, ...)

    Returns:
        List of {"content": str, "metadata": dict} documents
    """
    results = []
    for path in sorted(data_dir.glob("*/*.md")):
        if path.name.startswith("_"):
            continue
        try:
            doc = parse_markdown_file(path)
        except (OSError, UnicodeDecodeError) as e:
            print(f"  warning: skipping unreadable file {path.name}: {e}")
            continue
        missing = [key for key in REQUIRED_METADATA if key not in doc["metadata"]]
        if not doc["content"] or missing:
            reason = "empty file" if not doc["content"] else f"missing {', '.join(missing)}"
            print(f"  warning: skipping {path.name} ({reason})")
            continue
        results.append(doc)
    return results


class RAGAssistant:
    """
    A RAG assistant that suggests dishes for a given wine, using ChromaDB and
    a local (Ollama) or hosted (OpenAI, Groq, Google Gemini) LLM.
    """

    def __init__(self):
        """Initialize the LLM, the vector database and the three LLM chains."""
        config.set_seed()

        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT)

        # Create the chains: two to plan the searches, one to answer
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        self.wine_query_chain = (
            ChatPromptTemplate.from_template(WINE_QUERY_PROMPT) | self.llm | StrOutputParser()
        )
        self.recipe_query_chain = (
            ChatPromptTemplate.from_template(RECIPE_QUERY_PROMPT) | self.llm | StrOutputParser()
        )

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM from the environment variables in .env.

        Uses a local Ollama model if OLLAMA_MODEL is set, otherwise tries
        OpenAI, Groq, and Google Gemini in that order.
        """
        temperature = config.LLM_TEMPERATURE

        # Local model via Ollama (no API key needed)
        if os.getenv("OLLAMA_MODEL"):
            model_name = os.getenv("OLLAMA_MODEL")
            print(f"Using local Ollama model: {model_name}")
            return ChatOllama(
                model=model_name,
                base_url=config.OLLAMA_BASE_URL,
                temperature=temperature,
                num_ctx=config.OLLAMA_NUM_CTX,
                seed=config.SEED,
            )

        # Check for OpenAI API key
        elif os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", config.DEFAULT_MODELS["openai"])
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=temperature,
                seed=config.SEED,
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", config.DEFAULT_MODELS["groq"])
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=temperature
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", config.DEFAULT_MODELS["google"])
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=temperature,
            )

        else:
            raise ValueError(
                "No LLM configured. Set OLLAMA_MODEL, OPENAI_API_KEY, GROQ_API_KEY "
                "or GOOGLE_API_KEY in your .env file (see .env.example)."
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of {"content": str, "metadata": dict} documents
        """
        self.vector_db.add_documents(documents)

    def search_many(
        self, queries: List[str], n_results: int, where: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run several searches and keep the n best distinct chunks overall.

        Args:
            queries: Search queries (duplicates are searched once)
            n_results: Number of chunks to keep
            where: Chroma metadata filter applied to every search

        Returns:
            Search results in the VectorDB.search format.
        """
        results = [
            self.vector_db.search(query, n_results=n_results, where=where)
            for query in dict.fromkeys(queries)
        ]
        return best_distinct(results, n_results)

    def retrieve(self, question: str) -> Dict[str, Any]:
        """
        Classify the question, then retrieve wine knowledge, pairing principles
        and recipes separately.

        A single search tends to return only one kind of document (e.g. five
        chunks about Chianti and no recipe). Searching each group guarantees
        the LLM always sees wine facts, general pairing rules and dishes it
        can actually suggest.

        Args:
            question: The user's question

        Returns:
            Dictionary with the topic and, for wine questions, the results per
            group, both search queries and the coverage note.
        """
        wine_reply = self.wine_query_chain.invoke({"question": question})
        topic, named_wine, wine_query = parse_wine_query(wine_reply, question)
        if topic != "wine":
            return {"topic": topic, "wine_reply": wine_reply}

        # Search with both the original question and the rewritten query: the
        # rewrite adds pairing terms ("food pairing, tannin"), while the original
        # question keeps the search anchored if the LLM writes a poor query.
        queries = [question, wine_query]
        wine = self.search_many(queries, config.N_WINE_RESULTS, where=WINE_FILTER)
        theory = self.search_many(queries, config.N_THEORY_RESULTS, where=THEORY_FILTER)
        wine_results = merge_results(wine, theory)

        recipe_reply = self.recipe_query_chain.invoke(
            {"wine_context": format_context(wine_results), "question": question}
        )
        recipe_query = parse_recipe_query(recipe_reply)
        recipe = self.vector_db.search(
            recipe_query, n_results=config.N_RECIPE_RESULTS, where=RECIPE_FILTER
        )
        return {
            "topic": topic,
            "wine": wine_results,
            "recipe": recipe,
            "named_wine": named_wine,
            "wine_query": wine_query,
            "recipe_query": recipe_query,
            "coverage_note": coverage_note(named_wine, wine_results),
            # raw LLM replies, for debugging (see scripts/trace_question.py)
            "wine_reply": wine_reply,
            "recipe_reply": recipe_reply,
        }

    @staticmethod
    def answer_inputs(question: str, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Build the variables for RAG_PROMPT from the retrieval results.

        Wine chunks are numbered [1]..[n], recipe chunks continue after them,
        so every chunk in the prompt has a unique number the LLM can cite.

        Args:
            question: The user's question
            results: Output of retrieve() for a wine question

        Returns:
            Dictionary with the prompt variables.
        """
        wine, recipe = results["wine"], results["recipe"]
        return {
            "wine_context": format_context(wine),
            "recipe_context": format_context(recipe, start=len(wine["documents"]) + 1),
            "coverage_note": results["coverage_note"],
            "question": question,
        }

    def invoke(self, input: str) -> Dict[str, Any]:
        """
        Query the RAG assistant.

        Args:
            input: User's input

        Returns:
            Dictionary with the answer, both search queries, the coverage note
            and the numbered sources.
        """
        results = self.retrieve(input)
        if results["topic"] != "wine":
            return {
                "answer": OUT_OF_SCOPE_ANSWERS[results["topic"]],
                "wine_query": "-",
                "recipe_query": "-",
                "coverage_note": f"Out of scope ({results['topic']} question), no search done.",
                "sources": [],
            }
        wine, recipe = results["wine"], results["recipe"]

        llm_answer = self.chain.invoke(self.answer_inputs(input, results))

        sources = [
            {"title": meta["title"], "url": meta.get("url", ""), "distance": distance}
            for meta, distance in zip(
                wine["metadatas"] + recipe["metadatas"], wine["distances"] + recipe["distances"]
            )
        ]
        return {
            "answer": llm_answer,
            "wine_query": results["wine_query"],
            "recipe_query": results["recipe_query"],
            "coverage_note": results["coverage_note"],
            "sources": sources,
        }


def print_result(result: Dict[str, Any]) -> None:
    """Print the answer with the search steps and the numbered sources."""
    print(f"\n(wine search: {result['wine_query']})")
    print(f"(recipe search: {result['recipe_query']})")
    print(f"(coverage: {result['coverage_note'].split('.')[0]})")
    print(f"\n{result['answer']}\n\nSources:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['title']} (distance {src['distance']:.2f}) {src['url']}")
    print()


def ask(assistant: RAGAssistant, question: str) -> None:
    """Answer one question; report LLM or connection errors instead of crashing."""
    try:
        print_result(assistant.invoke(question))
    except Exception as e:  # LLM providers raise many different error types
        print(f"\nCould not answer this question: {type(e).__name__}: {e}")
        if os.getenv("OLLAMA_MODEL"):
            print("Is Ollama running? Start it with `ollama serve` and check `ollama list`.")
        else:
            print("Check your API key, model name and internet connection in .env.")


def build_index(assistant: RAGAssistant, rebuild: bool) -> None:
    """Index the data folder on first run or when asked; later runs reuse chroma_db/."""
    if not rebuild and assistant.vector_db.count() > 0:
        print(f"Using existing index ({assistant.vector_db.count()} chunks)")
        return
    if rebuild:
        assistant.vector_db.reset()
    print("\nLoading documents...")
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {config.DATA_DIR}. "
            "Download them first with: python scripts/download_data.py"
        )
    print(f"Loaded {len(docs)} documents")
    assistant.add_documents(docs)


def main() -> None:
    """Main function to run the RAG assistant from the command line."""
    parser = argparse.ArgumentParser(description="Find a dish to cook with your bottle of wine")
    parser.add_argument("question", nargs="?", help="ask one question and exit")
    parser.add_argument(
        "--rebuild", action="store_true", help="re-index the data folder before starting"
    )
    args = parser.parse_args()

    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()
        build_index(assistant, args.rebuild)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    if args.question:
        ask(assistant, args.question)
        return

    while True:
        try:
            question = input("Which wine do you have? ('quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            ask(assistant, question)


if __name__ == "__main__":
    main()

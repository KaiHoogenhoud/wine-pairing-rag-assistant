# Wine Pairing RAG Assistant: What to Cook with Your Bottle

A retrieval-augmented generation (RAG) assistant that suggests a home-cooked dish for a bottle
of wine you already have. Tell it which wine you have (*"I was given a bottle of Chablis, what
should I cook?"*) and it answers with one or two recipes from its cookbook, explains why they
pair well, and cites its sources.

## Overview

People are often given a bottle of wine as a gift and have no idea what to cook with it.
General-purpose chatbots will happily answer, but they make up pairings and recipes and give
no way to check them. This assistant only answers from a small, openly licensed knowledge base:

- **49 Wikipedia pages** on pairing principles (acidity, sweetness, oak, tasting terms), wine styles,
  24 grape varieties and 13 well-known wines and regions
- **51 Wikibooks Cookbook recipes** for home cooking, from boeuf bourguignon to pad thai

It retrieves what the knowledge base says about your wine, works out which foods suit the wine's
traits (body, acidity, tannin, sweetness), finds matching recipes and writes a short, sourced
answer. When a wine is not in the knowledge base, it says so instead of inventing facts.

The assistant deliberately works in **one direction: wine → dish**. The reverse question ("which
wine goes with my curry?") is a different problem and is politely redirected.

The project was built as the capstone of the Ready Tensor
[Agentic AI Essentials](https://app.readytensor.ai/certifications/agentic-ai-essentials-U7HxeL7a)
program, starting from the official
[project template](https://github.com/readytensor/rt-aaidc-project1-template).

## How it works

Every question goes through three LLM calls and four vector searches:

```
question ──► 1. classify + wine search query (LLM)
               │  "wine"  → continue      "dish" / "other" → fixed redirect, no search
               ▼
             search wine knowledge with the question AND the rewritten query
               ├─ 4 chunks about the wine (grapes, regions, styles)
               └─ 2 chunks of general pairing principles
               ▼
             coverage check (code): is the named wine in the retrieved chunks?
               ▼
             2. wine traits → food search query (LLM)
               ▼
             search recipes ─► 4 recipe chunks
               ▼
             3. answer from the 10 numbered chunks + coverage instruction (LLM)
```

Key design decisions, each made after a failed test (see [Evaluation](#evaluation-and-tests)):

| Decision | Why |
|---|---|
| Separate searches for wine knowledge, pairing principles and recipes | One search for "Chianti" returns five Chianti chunks and no recipe. |
| LLM writes the food search query from the wine's traits | Searching the recipes for "Barolo" returned tiramisù: no recipe mentions wine names. Many wine pages (e.g. Chablis) never mention food either, so the step reasons from traits to food. |
| Trait → food examples in that prompt, not wine → food | With a single example, the 7B model copied it for every wine (beef stew for Chablis). |
| Coverage check in code, not only in the prompt | Asked about Saperavi (not in the knowledge base), the model described it from memory despite the prompt rule. Now the code detects this and instructs the model to say so. |
| Scope check (wine / dish / other) | Off-topic questions ("capital of France?") were answered from general knowledge. |
| Search with both the question and the rewritten query | A poor rewrite by a small model cannot take over the results. |
| Markdown-aware chunks of ≤ 800 characters with a "Title > Section" prefix | Fits the 256-token input of the embedding model; a chunk from the "Viticulture" section still says it is about Pinot noir. |

## Project structure

```
wine-pairing-rag-assistant/
├── src/                      # application code
│   ├── app.py                # entry point: CLI, document loading, RAG pipeline
│   ├── vectordb.py           # ChromaDB wrapper: chunking, embedding, search
│   ├── query_utils.py        # pure helpers: parsing LLM replies, coverage check
│   ├── prompts.py            # all prompt templates and fixed answers
│   └── config.py             # paths, model and retrieval settings, random seed
├── scripts/
│   ├── download_data.py      # entry point: downloads the knowledge base
│   ├── sources.py            # list of Wikipedia pages and recipes to download
│   ├── evaluate.py           # entry point: runs the test questions, writes a report
│   └── trace_question.py     # entry point: shows every pipeline step for one question
├── data/
│   ├── wikipedia/            # 49 wine pages (Markdown with metadata header)
│   ├── wikibooks/            # 51 recipes
│   ├── pairing_notes/        # template for optional hand-written pairing notes
│   └── SOURCES.md            # attribution: every page with URL and revision
├── evaluation/
│   ├── test_questions.json   # test questions with expected behaviour
│   └── results.md            # latest evaluation report (sample outputs)
├── tests/                    # pytest unit tests
├── .env.example              # template for API keys and model settings
├── requirements.txt          # runtime dependencies (pinned)
├── requirements-dev.txt      # + test dependencies
└── pyproject.toml            # Python version and pytest settings
```

`chroma_db/` (the vector index) is created on first run and is not part of the repository.

## Prerequisites

- **Python 3.10 or newer** (developed and tested with 3.13 on macOS)
- About **2 GB of disk space** for the dependencies (PyTorch is pulled in by sentence-transformers)
- **No GPU needed**: embedding runs on the CPU in about a minute
- **One LLM**, either
  - a local model through [Ollama](https://ollama.com) (free, no API key; tested with `qwen2.5:7b`), or
  - an API key for [OpenAI](https://platform.openai.com/api-keys),
    [Groq](https://console.groq.com/keys) or [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installation

```bash
git clone https://github.com/KaiHoogenhoud/wine-pairing-rag-assistant.git
cd wine-pairing-rag-assistant

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

Then open `.env` and configure **one** LLM:

```bash
# Option A: local model with Ollama (run `ollama pull qwen2.5:7b` first)
OLLAMA_MODEL=qwen2.5:7b

# Option B: a hosted model, e.g. Google Gemini
GOOGLE_API_KEY=your_key_here
```

The knowledge base is included in `data/`. To download it again, or after editing
`scripts/sources.py`:

```bash
python scripts/download_data.py    # about 8 minutes because of Wikimedia rate limits
```

## Usage

Ask a single question:

```bash
python src/app.py "I was given a bottle of Chablis, what should I cook?"
```

Or start an interactive session:

```bash
python src/app.py
```

The first run builds the vector index in `chroma_db/` (about a minute). Later runs reuse it;
pass `--rebuild` after changing anything in `data/`.

### Example output

```
(wine search: Chablis food pairing, body, acidity)
(recipe search: Grilled fish, shellfish, goat cheese, salad)
(coverage: The wine knowledge covers this question)

Given your Chablis, you might want to consider a Greek Salad [7]. Chablis is known for its high
acidity, which can cut through the richness of the olive oil and the saltiness of the feta cheese
in the salad. The green apple-like acidity in the wine complements the fresh, crisp vegetables
and herbs in the salad, enhancing their flavors. [...]

If you prefer something heartier, you could also try Fish and Chips [10]. The high acidity in
Chablis can help balance the richness of the fried potatoes and the saltiness of the fish,
providing a refreshing contrast. [...]

Sources:
  [1] Chablis (AOC) (distance 0.40) https://en.wikipedia.org/wiki/Chablis_(AOC)
  [2] Chablis (AOC) (distance 0.44) https://en.wikipedia.org/wiki/Chablis_(AOC)
  [3] Chardonnay (distance 0.46) https://en.wikipedia.org/wiki/Chardonnay
  ...
  [7] Greek Salad (distance 0.49) https://en.wikibooks.org/wiki/Cookbook:Greek_Salad
  ...
```

The three lines in brackets show the intermediate steps, which makes wrong answers easy to
debug. A wine outside the knowledge base, and a question outside the scope:

```
> I have a Saperavi from Georgia, what should I cook?
Saperavi is not in my knowledge base, so I'll provide general advice based on the pairing
principles in the context. [...] Roast Leg of Lamb with Gravy [8] [...] Macaroni and Cheese [7]
[...] fats and proteins can soften the perception of tannins [...]

> What wine goes with spicy Thai green curry?
I work the other way around: tell me which bottle of wine you have, and I'll suggest a dish
to cook with it. [...]
```

All answers of the latest evaluation run are in [evaluation/results.md](evaluation/results.md).

## Evaluation and tests

**Unit tests** cover document loading, chunking, cleaning the downloaded pages, parsing the LLM
replies and the coverage check. They need no LLM and run in a few seconds:

```bash
pip install -r requirements-dev.txt
pytest
```

**End-to-end evaluation** runs the eight questions in
[evaluation/test_questions.json](evaluation/test_questions.json) through the full pipeline with
the LLM from `.env`, and checks for each answer:

- **scope**: wine questions are answered, dish → wine and off-topic questions are redirected
- **dish**: the answer suggests at least one dish that suits the wine (keyword list per question)
- **coverage**: for a wine missing from the knowledge base, the answer says so

```bash
python scripts/evaluate.py         # writes evaluation/results.md
```

With `qwen2.5:7b` all 8 questions pass (about 40 seconds per answered question). The keyword
check is deliberately simple: it confirms that a sensible dish is suggested, not that the
explanation is correct, so the answers in `results.md` were also reviewed by hand.

### Debugging a single answer

To see why the assistant gave a particular answer, trace the question through the pipeline:

```bash
python scripts/trace_question.py "I was given a bottle of Barolo, what should I cook?"
```

It prints the raw LLM replies, every search with its metadata filter and distances, the
coverage verdict, the prompt sent to the LLM (`--full-prompt` for the complete text) and the
answer. A wrong answer can then be traced to its cause: a poor search query, irrelevant
chunks, a wrong coverage verdict, or the LLM ignoring good context.

## Configuration

Settings that depend on your machine or contain secrets live in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | (empty) | Local Ollama model; takes priority over the API keys |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Address of the Ollama server |
| `OLLAMA_NUM_CTX` | `8192` | Context window; Ollama's small default would silently truncate the prompt |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI |
| `GROQ_API_KEY` / `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq |
| `GOOGLE_API_KEY` / `GOOGLE_MODEL` | `gemini-2.0-flash` | Google Gemini |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (rebuild the index after changing it) |
| `CHROMA_COLLECTION_NAME` | `wine_pairing` | Name of the vector collection |
| `MAX_WINE_DISTANCE` | `0.5` | Cosine distance above which the wine knowledge counts as a loose match |

Tuning values such as chunk size (800), chunk overlap (100) and the number of chunks per search
(4 wine, 2 pairing principles, 4 recipes) are in [src/config.py](src/config.py). The LLM runs
at temperature 0 and random seeds are fixed (`SEED = 42`), so results are as repeatable as the
chosen provider allows.

## Limitations

- **Small knowledge base.** 24 grapes, 13 regions and 51 recipes. Wines outside it (e.g. Greek
  or Georgian grapes) only get general advice, and the suggestions are limited to these recipes.
- **Reasoning slips with a small model.** With `qwen2.5:7b` the dish choice is usually good, but
  explanations can contain errors, e.g. that roast lamb "features beef". A larger hosted model
  makes fewer such mistakes.
- **Citation numbers are not always precise.** The model sometimes cites a neighbouring chunk
  instead of the one that actually contains the fact.
- **The search queries can contain the model's own knowledge.** The food search query is written
  by the LLM and may include pairing ideas that are not in the knowledge base. The answer itself
  is still restricted to the retrieved context.
- **Coverage check is name-based.** It matches the wine name against the retrieved text and page
  titles, so a synonym that the pages never mention would be reported as missing.
- **Speed.** Three LLM calls per question: about 40 seconds with a local 7B model on a laptop,
  a few seconds with a hosted model.
- **One direction only.** Wine → dish by design; no conversation memory, as each question about
  a bottle stands on its own.

This is a working prototype for learning and demonstration, not a production system.

## Data sources and licenses

- **Code:** this project is derived from the Ready Tensor project template and is therefore
  licensed under the same
  [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  license. See [LICENSE](LICENSE).
- **Knowledge base:** the texts in `data/wikipedia/` and `data/wikibooks/` come from
  [Wikipedia](https://en.wikipedia.org) and the [Wikibooks Cookbook](https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents)
  and are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  They were converted to plain text and trimmed (reference and link sections removed). Every
  page is listed with its URL, revision and download date in [data/SOURCES.md](data/SOURCES.md).
  Commercial wine sites such as Wine Folly and Vivino were deliberately not used: their terms of
  use forbid copying and automated access.
- **Models:** the embedding model
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is licensed
  under Apache 2.0. The LLMs are not distributed with this project; their own terms of use apply.

## Acknowledgements

Built on the [Ready Tensor RAG project template](https://github.com/readytensor/rt-aaidc-project1-template)
for the Agentic AI Essentials certification program.

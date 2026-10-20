"""Download the knowledge base from Wikipedia and the Wikibooks Cookbook.

Pages are fetched as plain text through the MediaWiki API, lightly cleaned,
and saved as Markdown files with a small metadata header:

    data/wikipedia/<slug>.md
    data/wikibooks/<slug>.md
    data/SOURCES.md          (attribution list required by CC BY-SA)

Usage:
    python scripts/download_data.py                 # download everything missing
    python scripts/download_data.py --force         # re-download all pages
    python scripts/download_data.py --only wikibooks
"""

import argparse
import os
import re
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import WIKIBOOKS_RECIPES, WIKIPEDIA_PAGES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SITES = {
    "wikipedia": {
        "api": "https://en.wikipedia.org/w/api.php",
        "pages": WIKIPEDIA_PAGES,
        "source": "Wikipedia",
    },
    "wikibooks": {
        "api": "https://en.wikibooks.org/w/api.php",
        "pages": WIKIBOOKS_RECIPES,
        "source": "Wikibooks Cookbook",
    },
}

LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

# Wikimedia asks API clients to send a descriptive User-Agent.
USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "wine-pairing-rag-assistant/0.1 (educational RAG project; python-requests)",
)

REQUEST_DELAY = 1.0  # seconds between requests, to stay polite
MAX_RETRIES = 5
MIN_CHARS = 400  # pages shorter than this are stubs and not worth embedding

# Trailing sections that add noise rather than knowledge.
DROP_SECTIONS = {
    "references", "see also", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "gallery",
}


def slugify(title: str) -> str:
    """'Cookbook:Crème Brûlée' -> 'creme_brulee' (snake_case, ASCII-only, safe on every OS)."""
    title = title.removeprefix("Cookbook:")
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_-]+", "_", slug).strip("_")


def fetch_page(session: requests.Session, api: str, title: str) -> dict | None:
    """Return {title, text, url, revid} for a page, or None if it doesn't exist."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts|info|revisions",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "inprop": "url",
        "rvprop": "ids",
        "redirects": 1,
        "titles": title,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(api, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = int(resp.headers.get("Retry-After", 5 * attempt))
            print(f"    rate limited (HTTP {resp.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        page = resp.json()["query"]["pages"][0]
        if page.get("missing"):
            return None
        return {
            "title": page["title"],
            "text": page.get("extract", ""),
            "url": page["fullurl"],
            "revid": page["revisions"][0]["revid"],
        }
    raise RuntimeError(f"Giving up on '{title}' after {MAX_RETRIES} retries")


def clean_text(text: str) -> str:
    """Strip navigation/trailing sections and turn wiki headings into Markdown."""
    lines = text.splitlines()

    # Wikibooks extracts start with a breadcrumb line ("Cookbook | Recipes | ...").
    if lines and lines[0].startswith("Cookbook |"):
        lines = lines[1:]

    out, skipping = [], False
    for line in lines:
        heading = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", line)
        if heading:
            name = heading.group(2)
            level = len(heading.group(1))
            if level == 2:
                skipping = name.lower() in DROP_SECTIONS
            if skipping:
                continue
            line = "#" * level + " " + name
        elif skipping:
            continue
        out.append(line)

    cleaned = "\n".join(out)
    # Drop headings left with no content below them, then collapse blank lines.
    cleaned = re.sub(r"(?m)^#{2,6} .*\n\s*(?=^#{2,6} |\Z)", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def to_markdown(page: dict, category: str, source: str) -> str:
    header = "\n".join([
        "---",
        f"title: {page['title'].removeprefix('Cookbook:')}",
        f"category: {category}",
        f"source: {source}",
        f"url: {page['url']}",
        f"revision_id: {page['revid']}",
        f"license: {LICENSE}",
        f"retrieved: {date.today().isoformat()}",
        "---",
    ])
    title = page["title"].removeprefix("Cookbook:")
    return f"{header}\n\n# {title}\n\n{page['text']}\n"


def read_header(path: Path) -> dict:
    """Read the metadata header of an already downloaded file."""
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line == "---":
            break
        key, _, value = line.partition(": ")
        meta[key] = value
    return meta


def download_site(name: str, force: bool) -> tuple[int, list[str]]:
    site = SITES[name]
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    saved, problems = 0, []
    print(f"\n== {site['source']} ({len(site['pages'])} pages) ==")
    for title, category in site["pages"]:
        path = out_dir / f"{slugify(title)}.md"
        if path.exists() and not force:
            print(f"  skip  {title} (already downloaded)")
            saved += 1
            continue

        try:
            page = fetch_page(session, site["api"], title)
        except (requests.RequestException, RuntimeError, KeyError) as e:
            # Network error, repeated rate limiting or an unexpected API reply:
            # report it and continue, a re-run only fetches what is missing.
            print(f"  FAIL  {title} ({type(e).__name__}: {e})")
            problems.append(f"{name}: '{title}' failed to download")
            continue
        finally:
            time.sleep(REQUEST_DELAY)
        if page is None:
            print(f"  MISS  {title} (page not found)")
            problems.append(f"{name}: '{title}' not found")
            continue

        page["text"] = clean_text(page["text"])
        if len(page["text"]) < MIN_CHARS:
            print(f"  STUB  {title} ({len(page['text'])} chars, skipped)")
            problems.append(f"{name}: '{title}' is a stub")
            continue

        path.write_text(to_markdown(page, category, site["source"]), encoding="utf-8")
        print(f"  ok    {title} -> {path.relative_to(ROOT)} ({len(page['text']):,} chars)")
        saved += 1
    return saved, problems


def write_sources_file() -> None:
    """List every downloaded page with its URL and license, for attribution."""
    rows = []
    for name in SITES:
        for path in sorted((DATA_DIR / name).glob("*.md")):
            meta = read_header(path)
            rows.append(
                f"| {meta['title']} | {meta['source']} | {meta['category']} "
                f"| [link]({meta['url']}) | {meta['revision_id']} | {meta['retrieved']} |"
            )
    content = "\n".join([
        "# Data sources",
        "",
        "Texts in `data/wikipedia/` and `data/wikibooks/` are taken from Wikipedia and the",
        f"Wikibooks Cookbook and are licensed under [{LICENSE}]({LICENSE_URL}).",
        "They were converted to plain text and trimmed (reference and link sections removed).",
        "Regenerate them with `python scripts/download_data.py`.",
        "",
        "| Title | Source | Category | URL | Revision | Retrieved |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ])
    (DATA_DIR / "SOURCES.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing pages")
    parser.add_argument("--only", choices=list(SITES), help="download a single source")
    args = parser.parse_args()

    total, problems = 0, []
    for name in [args.only] if args.only else SITES:
        saved, site_problems = download_site(name, args.force)
        total += saved
        problems += site_problems

    write_sources_file()
    print(f"\nDone: {total} pages in {DATA_DIR.relative_to(ROOT)}/, attribution in data/SOURCES.md")
    if problems:
        print("Check these entries in scripts/sources.py:")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()

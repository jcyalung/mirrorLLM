import json
from urllib.parse import quote
from urllib.request import Request, urlopen

from ddgs import DDGS

WIKI_UA = "mirrorLLM/1.0 (local assistant; wikipedia lookups)"
WIKI_API = "https://en.wikipedia.org/w/api.php"


def search_web(query: str, max_results: int = 5) -> str:
    results = DDGS().text(query, max_results=max_results)
    cleaned = [
        {
            "title": item.get("title"),
            "url": item.get("href"),
            "snippet": item.get("body"),
        }
        for item in results
    ]
    if not cleaned:
        return json.dumps({"error": "No search results found."})
    return json.dumps(cleaned, ensure_ascii=False)


def _wiki_get(params: dict) -> dict:
    query = "&".join(f"{key}={quote(str(value))}" for key, value in params.items())
    request = Request(f"{WIKI_API}?{query}", headers={"User-Agent": WIKI_UA})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def get_wikipedia(title: str, resolved: bool = False) -> str:
    data = _wiki_get(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if page.get("missing") or not page.get("extract"):
        if resolved:
            return json.dumps({"error": f"No Wikipedia page found for {title!r}."})
        search = _wiki_get(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srlimit": 1,
                "srsearch": title,
            }
        )
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return json.dumps({"error": f"No Wikipedia page found for {title!r}."})
        return get_wikipedia(hits[0]["title"], resolved=True)

    extract = page.get("extract") or ""
    if len(extract) > 4000:
        extract = extract[:4000] + "..."
    return json.dumps(
        {
            "title": page.get("title"),
            "url": page.get("fullurl"),
            "extract": extract,
        },
        ensure_ascii=False,
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for current information, recipes, news, or articles. "
                "Use this when the user wants recipes or sources from online."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'easy pasta recipes'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia",
            "description": "Fetch a Wikipedia article introduction by title to summarize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Wikipedia page title, e.g. 'Sourdough'.",
                    }
                },
                "required": ["title"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    try:
        if name == "search_web":
            return search_web(arguments["query"])
        if name == "get_wikipedia":
            return get_wikipedia(arguments["title"])
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

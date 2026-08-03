"""Hosted LeanExplore declaration search."""

import os
from typing import Final
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

DEFAULT_API_URL: Final = "https://www.leanexplore.com/api/v2/search"
PACKAGES: Final = ("Mathlib", "Batteries", "Init", "Lean", "Std")


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    module: str | None = None
    source_text: str | None = None
    docstring: str | None = None
    informalization: str | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    results: tuple[SearchResult, ...]


def search(query: str, limit: int = 10) -> SearchResponse:
    """Search the hosted LeanExplore API."""
    params = [("q", query), ("limit", limit), ("packages", ",".join(PACKAGES))]
    api_url = os.environ.get("LEAN_EXPLORE_API_URL", DEFAULT_API_URL)
    request = Request(
        f"{api_url}?{urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": "openprover/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        return SearchResponse.model_validate_json(response.read())

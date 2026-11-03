"""Retrieve cited policy chunks from the persistent PDF index built by ingest.py."""
from ingest import search_policies


def retrieve(question: str, k: int = 2) -> list[tuple[str, str]]:
    """Embed the query and return up to k relevant (source, text) pairs."""
    return search_policies(question, k)

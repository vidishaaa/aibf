"""
entity_linking.py
------------------
Basic ticker mapping utilities using alias dictionaries.

v1 approach:
- Provide a mapping from ticker -> set of aliases (company names, brands)
- Given a headline, return candidate tickers whose aliases appear
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


def build_alias_index(ticker_to_aliases: Dict[str, Iterable[str]]) -> Dict[str, Set[str]]:
    index: Dict[str, Set[str]] = {}
    for ticker, aliases in (ticker_to_aliases or {}).items():
        for alias in aliases or []:
            key = alias.strip().lower()
            if not key:
                continue
            index.setdefault(key, set()).add(ticker.upper())
    return index


def find_tickers_in_text(text: str, alias_index: Dict[str, Set[str]]) -> List[str]:
    if not text:
        return []
    text_l = text.lower()
    found: Set[str] = set()
    # Exact alias phrase match (word-boundary)
    for alias, tickers in alias_index.items():
        # escape regex special chars in alias
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_l):
            found.update(tickers)
    return sorted(found)



"""Normalization helpers shared by deterministic movie-matching rules."""

from __future__ import annotations

import re
import unicodedata


PARENTHETICAL_PATTERN = re.compile(r"\(([^)]*)\)")
DIRECTOR_SEPARATOR_PATTERN = re.compile(r"[,;|]")


def normalize_title(value: str | None) -> str:
    """Return a case-, accent-, whitespace- and punctuation-free key."""
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        character
        for character in decomposed
        if character.isalnum() and not unicodedata.combining(character)
    )


def title_variants(value: str | None) -> set[str]:
    """Return the complete title, base title and parenthetical aliases."""
    if not value:
        return set()
    variants = {normalize_title(value)}
    base_title = PARENTHETICAL_PATTERN.sub(" ", value).strip()
    variants.add(normalize_title(base_title))
    variants.update(
        normalize_title(alias)
        for alias in PARENTHETICAL_PATTERN.findall(value)
    )
    return {variant for variant in variants if variant}


def normalize_directors(value: str | None) -> set[str]:
    """Return normalized director names from a comma-like source list."""
    if not value:
        return set()
    return {
        normalized
        for part in DIRECTOR_SEPARATOR_PATTERN.split(value)
        if (normalized := normalize_title(part))
    }


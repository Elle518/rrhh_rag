"""Catalog of labor agreements with their sectors, provinces and associated archives."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rapidfuzz import fuzz

CATALOGO_CONVENIOS: list[dict[str, Any]] = [
    {
        "id": "contactcenter",
        "sector": "contact center",
        "provincia": None,
        "source_files": ["contactcenter.json"],
        "aliases": [
            "contact center",
            "contactcenter",
            "call center",
            "teleoperadores",
            "telemarketing",
        ],
    },
    {
        "id": "establecimientosanitarios_madrid",
        "sector": "establecimientos sanitarios",
        "provincia": "madrid",
        "source_files": ["establecimientosanitariomadrid.json"],
        "aliases": [
            "establecimientos sanitarios",
            "sanitario",
            "sanidad privada",
            "clinicas privadas",
            "clínicas privadas",
        ],
    },
    {
        "id": "establecimientosanitarios_malaga",
        "sector": "establecimientos sanitarios",
        "provincia": "malaga",
        "source_files": ["establecimientosanitariomalaga.json"],
        "aliases": [
            "establecimientos sanitarios",
            "sanitario",
            "sanidad privada",
            "clinicas privadas",
            "clínicas privadas",
        ],
    },
    {
        "id": "establecimientosanitarios_sevilla",
        "sector": "establecimientos sanitarios",
        "provincia": "sevilla",
        "source_files": [
            "establecimientosanitariosevilla.json",
            "prorrogaconveniosanitariosevilla.json",
        ],
        "aliases": [
            "establecimientos sanitarios",
            "sanitario",
            "sanidad privada",
            "clinicas privadas",
            "clínicas privadas",
        ],
    },
    {
        "id": "hospedaje_madrid",
        "sector": "hospedaje",
        "provincia": "madrid",
        "source_files": ["hospedajemadrid.json"],
        "aliases": [
            "hospedaje",
            "hoteles",
            "alojamiento",
            "hosteleria hospedaje",
            "hostelería hospedaje",
        ],
    },
    {
        "id": "limpieza_valencia",
        "sector": "limpieza",
        "provincia": "valencia",
        "source_files": ["limpiezavalencia.json"],
        "aliases": [
            "limpieza",
            "limpieza de edificios",
            "limpieza de locales",
        ],
    },
    {
        "id": "oficinasydespachos_madrid",
        "sector": "oficinas y despachos",
        "provincia": "madrid",
        "source_files": ["oficinasydespachosmadrid.json"],
        "aliases": [
            "oficinas y despachos",
            "oficinas",
            "despachos",
            "administracion de oficina",
            "administración de oficina",
        ],
    },
    {
        "id": "siderometalurgia_zaragoza",
        "sector": "siderometalurgia",
        "provincia": "zaragoza",
        "source_files": ["siderometalurgiazaragoza.json"],
        "aliases": [
            "siderometalurgia",
            "metal",
            "metalurgia",
            "sidero",
        ],
    },
]

PROVINCIAS = sorted({c["provincia"] for c in CATALOGO_CONVENIOS if c.get("provincia")})


def normalize_text(text: str) -> str:
    """Normalize text for better matching: lowercase,
    remove accents, non-alphanumeric chars, and extra spaces."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_provincia(text: str) -> str | None:
    """Extract provincia from text by looking for known provincia names."""
    norm = normalize_text(text)
    for provincia in PROVINCIAS:
        if provincia in norm:
            return provincia
    return None


def convenio_label(convenio: dict[str, Any]) -> str:
    """Generate a human-readable label for a convenio, e.g. 'Contact Center - Madrid'."""
    provincia = convenio.get("provincia")
    if provincia:
        return f"{convenio['sector'].title()} - {provincia.title()}"
    return convenio["sector"].title()


def get_convenio_by_id(convenio_id: str) -> dict[str, Any] | None:
    """Find a convenio in the catalog by its ID."""
    for c in CATALOGO_CONVENIOS:
        if c["id"] == convenio_id:
            return c
    return None


def find_by_sector_and_provincia(
    sector: str | None,
    provincia: str | None,
) -> dict[str, Any] | None:
    """Find a convenio by matching sector and provincia. If provincia is None, match any convenio with the given sector."""
    if not sector:
        return None

    sector_norm = normalize_text(sector)
    provincia_norm = normalize_text(provincia) if provincia else None

    for c in CATALOGO_CONVENIOS:
        if normalize_text(c["sector"]) != sector_norm:
            continue
        if provincia_norm is None and c["provincia"] is None:
            return c
        if provincia_norm and normalize_text(c["provincia"] or "") == provincia_norm:
            return c
    return None


def get_sector_matches(text: str) -> list[dict[str, Any]]:
    """Find convenios whose sector or aliases match the text."""
    norm = normalize_text(text)
    found: list[dict[str, Any]] = []

    for c in CATALOGO_CONVENIOS:
        candidates = [c["sector"], *c.get("aliases", [])]
        if any(normalize_text(alias) in norm for alias in candidates):
            found.append(c)

    # deduplicar por sector
    unique = {}
    for c in found:
        unique[c["sector"]] = c
    return list(unique.values())


def fuzzy_candidates(text: str, limit: int = 3) -> list[dict[str, Any]]:
    """Find convenios whose sector or aliases have a fuzzy match with the text."""
    norm = normalize_text(text)
    scored: list[tuple[int, dict[str, Any]]] = []

    for c in CATALOGO_CONVENIOS:
        searchable = [
            c["sector"],
            *c.get("aliases", []),
            f"{c['sector']} {c['provincia'] or ''}".strip(),
            convenio_label(c),
        ]
        best_score = max(
            fuzz.token_set_ratio(norm, normalize_text(item)) for item in searchable
        )
        scored.append((best_score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [c for score, c in scored if score >= 55]
    # deduplication by convenio ID, keeping the highest scored ones
    result = []
    seen = set()
    for c in filtered:
        if c["id"] not in seen:
            seen.add(c["id"])
            result.append(c)
        if len(result) >= limit:
            break
    return result


def resolve_convenio_from_text(user_text: str) -> dict[str, Any]:
    """Given a user input text, try to resolve it to a specific convenio in the catalog, following these steps:
    1. Extract provincia from the text.
    2. Look for sector matches in the text.
    3. If exactly one sector match is found, check if it has a provincia. If it does, return it as exact match.
    4. If it has no provincia, but we detected a provincia in the text, try to find a convenio with that sector and provincia.
    5. If multiple sector matches are found, return them as candidates.
    6. If no sector matches are found, perform a fuzzy search and return candidates if any.
    7. If no matches at all, return not found.
    """
    provincia = extract_provincia(user_text)
    sector_matches = get_sector_matches(user_text)

    if len(sector_matches) == 1:
        sector = sector_matches[0]["sector"]
        sector_variants = [c for c in CATALOGO_CONVENIOS if c["sector"] == sector]

        if len(sector_variants) == 1:
            return {
                "status": "exact",
                "convenio": sector_variants[0],
                "provincia_detectada": provincia,
                "candidates": [],
            }

        if provincia:
            exact = find_by_sector_and_provincia(sector, provincia)
            if exact:
                return {
                    "status": "exact",
                    "convenio": exact,
                    "provincia_detectada": provincia,
                    "candidates": [],
                }

        return {
            "status": "needs_provincia",
            "sector": sector,
            "provincia_detectada": provincia,
            "candidates": sector_variants,
        }

    if len(sector_matches) > 1:
        return {
            "status": "candidates",
            "convenio": None,
            "provincia_detectada": provincia,
            "candidates": sector_matches[:3],
        }

    fuzzy = fuzzy_candidates(user_text, limit=3)
    if fuzzy:
        return {
            "status": "candidates",
            "convenio": None,
            "provincia_detectada": provincia,
            "candidates": fuzzy,
        }

    return {
        "status": "not_found",
        "convenio": None,
        "provincia_detectada": provincia,
        "candidates": [],
    }

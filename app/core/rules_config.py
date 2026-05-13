"""
Renovation business rules loaded at import time.

- ``DEFAULT_*`` dicts are the baseline catalog. ``ISSUE_WEIGHTS``, ``COST_MAP``,
  and ``ISSUE_COST_WEIGHT`` are shallow copies so future runtime reloads can replace
  them without mutating the defaults (no DB reload is wired yet; see ``Settings``).
- ``COST_MAP`` values are ``(unit_kind, cost_low, cost_high)`` where ``unit_kind`` is
  ``"sqft"`` (multiply by square feet), ``"unit"`` (multiply by computed unit count),
  or ``"fixed"`` (single lump sum; quantity forced to 1 in line items).
- Keys in ``COST_MAP`` must stay aligned with scope categories and user-intent labels
  in ``renovation_cost_engine`` (e.g. ``window`` not ``windows``).
"""

from __future__ import annotations

DEFAULT_ISSUE_WEIGHTS: dict[str, float] = {
    "foundation cracks": 40,
    "structural damage": 45,
    "major wall cracks": 30,
    "sagging roof": 40,
    "plumbing issues": 25,
    "electrical issues": 25,
    "hvac issues": 20,
    "water damage": 30,
    "mold": 25,
    "outdated cabinets": 15,
    "old bathroom": 18,
    "old tiles": 12,
    "carpet": 10,
    "floor damage": 18,
    "outdated flooring": 12,
    "popcorn ceiling": 15,
    "worn appliances": 12,
    "stains": 8,
    "paint wear": 6,
    "minor wall damage": 10,
    "damaged ceiling": 14,
    "missing grout": 8,
    "dirty grout": 6,
    "broken fixtures": 10,
    "damaged appliances": 12,
    "rotted wood": 22,
    "dirty surfaces": 5,
    "roof damage": 30,
    "peeling exterior paint": 15,
    "cracked driveway": 12,
    "poor landscaping": 10,
    "broken windows": 20,
    "garage damage": 15,
    "smoke damage": 28,
    "fire damage": 40,
}

DEFAULT_COST_MAP: dict[str, tuple[str, float, float]] = {
    "paint": ("sqft", 1.5, 3.5),
    "flooring": ("sqft", 4.0, 10.0),
    "kitchen": ("fixed", 15000.0, 60000.0),
    "bathroom": ("fixed", 8000.0, 25000.0),
    "roof": ("sqft", 5.0, 12.0),
    "foundation": ("fixed", 15000.0, 50000.0),
    "electrical": ("sqft", 3.0, 10.0),
    "plumbing": ("sqft", 4.0, 12.0),
    "hvac": ("fixed", 5000.0, 20000.0),
    "window": ("unit", 300.0, 1200.0),
    "doors": ("unit", 200.0, 1500.0),
    "exterior": ("sqft", 2.0, 6.0),
    "garage": ("fixed", 5000.0, 25000.0),
    "landscaping": ("sqft", 2.0, 8.0),
}

DEFAULT_ISSUE_COST_WEIGHT: dict[str, float] = {
    "foundation": 1.5,
    "structural": 1.5,
    "roof": 1.3,
    "water": 1.3,
    "mold": 1.4,
    "electrical": 1.2,
    "plumbing": 1.2,
    "hvac": 1.1,
    "window": 1.0,
    "floor": 0.9,
    "paint": 0.6,
    "cosmetic": 0.5,
}

ISSUE_WEIGHTS: dict[str, float] = dict(DEFAULT_ISSUE_WEIGHTS)
COST_MAP: dict[str, tuple[str, float, float]] = dict(DEFAULT_COST_MAP)
ISSUE_COST_WEIGHT: dict[str, float] = dict(DEFAULT_ISSUE_COST_WEIGHT)
CANONICAL_ISSUE_TYPES = frozenset(ISSUE_WEIGHTS.keys())

# Static default location multipliers used by renovation estimate requests.
DEFAULT_ADMIN_LABOR_INDEX: float = 1.10
DEFAULT_ADMIN_MATERIAL_INDEX: float = 1.05

# Typical share of gross living area by room (min, max). Midpoint drives localized estimate $/timeline
# when vision/metadata identifies a primary room (single photo — not geometric measurement).
ROOM_AREA_RATIO_RANGES: dict[str, tuple[float, float]] = {
    "kitchen": (0.10, 0.18),
    "living_room": (0.15, 0.30),
    "bedroom": (0.10, 0.20),
    "bathroom": (0.04, 0.08),
    "dining_room": (0.08, 0.15),
    "basement": (0.12, 0.25),
    "hall": (0.05, 0.12),
}

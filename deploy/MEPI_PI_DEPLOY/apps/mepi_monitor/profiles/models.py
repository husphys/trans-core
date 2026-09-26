"""Transformer profiles supported by the research prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def geometric_effective_area_m2(od_mm: float, id_mm: float, height_mm: float) -> float:
    """Return the project's rectangular toroid cross-section approximation.

    ``Ae = height * (OD - ID) / 2``.  With the frozen prototype dimensions this
    reproduces the documented working area exactly: ``1.217268e-4 m2``.
    """

    values = (float(od_mm), float(id_mm), float(height_mm))
    if any(value <= 0.0 for value in values) or od_mm <= id_mm:
        raise ValueError("OD, ID and height must be positive, with OD > ID")
    return (height_mm * ((od_mm - id_mm) / 2.0)) * 1e-6


@dataclass(frozen=True)
class TransformerProfile:
    name: str
    od_mm: float
    id_mm: float
    height_mm: float
    primary_turns: int
    secondary_turns: int
    effective_area_m2: float
    material_label: str | None
    load_resistance_ohm: float
    known_core: bool
    ae_mode: str = "manual"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name is required")
        if self.primary_turns <= 0 or self.secondary_turns <= 0:
            raise ValueError("winding turns must be positive")
        if self.effective_area_m2 <= 0.0 or self.load_resistance_ohm <= 0.0:
            raise ValueError("effective area and load resistance must be positive")
        geometric_effective_area_m2(self.od_mm, self.id_mm, self.height_mm)
        if self.ae_mode not in {"manual", "geometric"}:
            raise ValueError("ae_mode must be 'manual' or 'geometric'")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def prediction_label(self) -> str:
        return "KNOWN CORE" if self.known_core else "UNSEEN CORE — EXPLORATORY PREDICTION"


def builtin_profiles() -> dict[str, TransformerProfile]:
    common = dict(
        od_mm=34.58,
        id_mm=20.81,
        height_mm=17.68,
        primary_turns=10,
        secondary_turns=10,
        effective_area_m2=1.217268e-4,
        load_resistance_ohm=49.6025,
        known_core=True,
        ae_mode="manual",
    )
    profiles = {
        "FE": TransformerProfile(name="FE", material_label="FE", **common),
        "COMMERCIAL": TransformerProfile(
            name="Commercial", material_label="Commercial", **common
        ),
    }
    for profile in profiles.values():
        profile.validate()
    return profiles

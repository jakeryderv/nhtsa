"""Consistent component labels from NHTSA's component descriptions (COMPDESC).

NHTSA's top-level component groups changed over the years (e.g. ENGINE and
ENGINE AND ENGINE COOLING are both in use), so they're merged into one set.
"""

import polars as pl

# Top-level group (the COMPDESC text before the first ":") -> label.
COMPONENT_LABELS = {
    "AIR BAGS": "air_bags",
    "BACK OVER PREVENTION": "driver_assistance",
    "CHILD SEAT": "child_seat",
    "Carry Handle, Shell, Base": "child_seat",
    "Chest Clip, Buckle, Harness": "child_seat",
    "I suspect the car seat is counterfeit": "child_seat",
    "Insert, Padding": "child_seat",
    "Tether, Lower Anchor (on car seat or vehicle)": "child_seat",
    "COMMUNICATION": "communication",
    "COMMUNICATIONS": "communication",
    "ELECTRICAL SYSTEM": "electrical",
    "ELECTRONIC STABILITY CONTROL": "stability_control",
    "ELECTRONIC STABILITY CONTROL (ESC)": "stability_control",
    "TRACTION CONTROL SYSTEM": "stability_control",
    "ENGINE": "engine",
    "ENGINE AND ENGINE COOLING": "engine",
    "EQUIPMENT": "equipment",
    "EQUIPMENT ADAPTIVE/MOBILITY": "equipment",
    "EXTERIOR LIGHTING": "lighting",
    "INTERIOR LIGHTING": "lighting",
    "FIRERELATED": "fire",
    "FORWARD COLLISION AVOIDANCE": "driver_assistance",
    "LANE DEPARTURE": "driver_assistance",
    "FUEL SYSTEM, DIESEL": "fuel_propulsion",
    "FUEL SYSTEM, GASOLINE": "fuel_propulsion",
    "FUEL SYSTEM, OTHER": "fuel_propulsion",
    "FUEL/PROPULSION SYSTEM": "fuel_propulsion",
    "HYBRID PROPULSION SYSTEM": "fuel_propulsion",
    "LATCHES/LOCKS/LINKAGES": "latches_locks",
    "PARKING BRAKE": "brakes",
    "SERVICE BRAKES": "brakes",
    "SERVICE BRAKES, AIR": "brakes",
    "SERVICE BRAKES, ELECTRIC": "brakes",
    "SERVICE BRAKES, HYDRAULIC": "brakes",
    "SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE HOLD": "brakes",
    "POWER TRAIN": "powertrain",
    "SEAT BELTS": "seat_belts",
    "SEATS": "seats",
    "STEERING": "steering",
    "STRUCTURE": "structure",
    "ROLLOVER": "structure",
    "SUSPENSION": "suspension",
    "TIRES": "tires",
    "WHEELS": "wheels",
    "TRAILER HARDWARE": "trailer",
    "TRAILER HITCHES": "trailer",
    "VEHICLE SPEED CONTROL": "speed_control",
    "VISIBILITY": "visibility",
    "VISIBILITY/WIPER": "visibility",
    "UNKNOWN OR OTHER": "unknown",
    "NONE": "unknown",
    "OTHER": "unknown",
    "Other/I am not sure": "unknown",
    "Other/Unknown": "unknown",
}


def component_label(compdesc: pl.Expr) -> pl.Expr:
    """Label for one COMPDESC string; null if its group isn't in COMPONENT_LABELS."""
    group = compdesc.str.split(":").list.first()
    return group.replace_strict(COMPONENT_LABELS, default=None, return_dtype=pl.String)


def component_labels(components: pl.Expr) -> pl.Expr:
    """Sorted distinct labels for a list of COMPDESC strings (load_complaints' COMPONENTS)."""
    return components.list.eval(component_label(pl.element())).list.unique().list.sort()

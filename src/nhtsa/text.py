"""Normalize complaint text (CDESCR) for TF-IDF and embeddings.

Each step is a polars expression, so notebooks can compare them; clean_text()
applies them all. The patterns come from surveying the complaint file, where
they cover hundreds of thousands of complaints each.
"""

import polars as pl

# "TL* ", "TL*- ", "DT*: ": marker on complaints transcribed by NHTSA staff.
_HOTLINE_MARKER = r"(?i)^\W*[a-z]{1,3}\*[\s:;-]*"
# Placeholder text with no content: "NO SUMMARY LISTED FOR ABOVE VEHICLE."
_NO_SUMMARY = r"(?i)^\W*no summary\b[^.]*\.?"
# "The contact owns a 2014 Jeep Grand Cherokee." / "The contact's wife owned a ..."
# Takata recall complaints put "TAKATA RECALL." first; that part is kept.
_VEHICLE_SENTENCE = (
    r"(?i)^(takata recall\.\s*)?"
    r"the contact(?:'s \w+)? (?:owns|owned|own|rented|leased)\b[^.]{0,100}\.\s*"
)
# Analyst initials ("*TR", "*SLC"). Uppercase only, so ordinary words survive;
# run after the hotline marker is gone, since "TL*THE" would otherwise match.
_ANALYST_INITIALS = r"\*[A-Z]{2,3}\b"
# Analyst update notes: "UPDATED 05/08/12", "UPDATED  03-19-08"
_UPDATE_NOTE = r"(?i)\bupdated?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
# The citation is optional because the 2,048-character cap sometimes cuts it off.
_FOIA_NOTE = (
    r"(?i)(?:information redacted|parts of this document have been redacted to"
    r" protect personally identifiable information) pursuant to the freedom of"
    r" information act \(foia\)(?:,?\s*5 u\.?s\.?c\.?\s*552\s*\(b\)\s*\(6\)\.?)?"
)
# 2002 database migration note: "**DIMSII IVOQ ENTRY POSTED AFTER 12/12/02 CUT
# OVER TO ARTEMIS**767939"
_MIGRATION_NOTE = r"(?i)\*+\s*dims\s*i*\b.{0,80}?artemis\*+\s*\d*"
# "[XXX]", "(XXXX)", "XXX"
_REDACTION = r"(?i)[\[(]?\bx{3,}\b[\])]?"
# NHTSA recall campaign IDs: "25V695000", "16V384", "13E-048"
_CAMPAIGN_ID = r"(?i)\b\d{2}[vetc]-?\d{3}(?:\d{3})?\b"
# "however, the part to do the recall repair was not yet available"
_PARTS_UNAVAILABLE = (
    r"(?i)\b(?:however|but),? the parts? (?:\w+ ){0,8}"
    r"(?:was|were) (?:not (?:yet )?available|unavailable)"
)


def strip_boilerplate(text: pl.Expr) -> pl.Expr:
    """Remove the hotline template's opening, analyst and system notes, FOIA notices,
    and "no summary" placeholders (which leaves those texts empty)."""
    return (
        text.str.replace(_HOTLINE_MARKER, "")
        .str.replace(_VEHICLE_SENTENCE, "$1")
        .str.replace(_NO_SUMMARY, "")
        .str.replace_all(_MIGRATION_NOTE, " ")
        .str.replace_all(_ANALYST_INITIALS, " ")
        .str.replace_all(_UPDATE_NOTE, " ")
        .str.replace_all(_FOIA_NOTE, " ")
    )


def mask_identifiers(text: pl.Expr) -> pl.Expr:
    """Replace redactions and recall campaign IDs with placeholder tokens."""
    return text.str.replace_all(_REDACTION, " <redacted> ").str.replace_all(
        _CAMPAIGN_ID, " <campaign> "
    )


def clean_text(text: pl.Expr) -> pl.Expr:
    """strip_boilerplate, mask_identifiers, lowercase, and collapse whitespace."""
    masked = mask_identifiers(strip_boilerplate(text))
    return masked.str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def is_hotline_template(text: pl.Expr) -> pl.Expr:
    """Complaint was transcribed by the hotline ("TL* The contact owns a ...")."""
    return text.str.contains(_HOTLINE_MARKER) | text.str.contains(_VEHICLE_SENTENCE)


def is_recall_parts_unavailable(text: pl.Expr) -> pl.Expr:
    """Complaint is about recall repair parts not being available."""
    return text.str.contains(_PARTS_UNAVAILABLE)

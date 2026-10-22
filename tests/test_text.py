import polars as pl
import pytest

from nhtsa.text import clean_text, is_hotline_template, is_recall_parts_unavailable


def apply(expr_fn, text: str):
    return pl.select(expr_fn(pl.lit(text))).item()


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        (
            "TL* THE CONTACT OWNS A 2014 JEEP GRAND CHEROKEE. THE INSTRUMENT PANEL FLICKERED.",
            "the instrument panel flickered.",
        ),
        (
            "TL*THE CONTACT OWNS A 2005 TOYOTA ECHO. THE VEHICLE STALLED.",
            "the vehicle stalled.",
        ),
        (
            "TL* -  THE CONTACT WAS DRIVING AT 5 MPH WHEN THE PEDAL DETACHED.",
            "the contact was driving at 5 mph when the pedal detached.",
        ),
        (
            "The contact's wife owned a 2012 Ford F-150. The brakes failed.",
            "the brakes failed.",
        ),
        (
            (
                "TL* TAKATA RECALL. THE CONTACT OWNS A 2011 FORD FUSION. THE CONTACT "
                "RECEIVED NOTIFICATION OF NHTSA CAMPAIGN NUMBER: 16V384000 (AIR BAGS)."
            ),
            (
                "takata recall. the contact received notification of nhtsa campaign "
                "number: <campaign> (air bags)."
            ),
        ),
        (
            "THE COST TO REPLACE WAS $352.40. *TR",
            "the cost to replace was $352.40.",
        ),
        (
            "THE MILEAGE WAS 16,000.   UPDATED 05/08/12*LJ  THE DEALER REPLACED IT. *NLM",
            "the mileage was 16,000. the dealer replaced it.",
        ),
        (
            (
                "My car was hit at [XXX] by (XXXX). Information Redacted Pursuant to the "
                "Freedom of Information Act (FOIA), 5 U.S.C. 552(B)(6)."
            ),
            "my car was hit at <redacted> by <redacted> .",
        ),
        (
            "REGARDING RECALL CAMPAIGN 13E-048, THE PART BROKE.",
            "regarding recall campaign <campaign> , the part broke.",
        ),
        (
            "DT*: THE CONTACT STATED THE STEERING WHEEL WAS TIGHT.",
            "the contact stated the steering wheel was tight.",
        ),
        ("NO SUMMARY LISTED FOR ABOV EVEHICLE.  *AK", ""),
        (
            "NO OTHER INFORMATION WAS AVAILABLE.    UPDATED  03-19-08 *BF  UPDATED 03/19/08  *TR",
            "no other information was available.",
        ),
        (
            (
                "IT STALLED. *PH  *NLM      **DIMSII IVOQ ENTRY POSTED AFTER 12/12/02 "
                "CUT OVER TO ARTEMIS**767939"
            ),
            "it stalled.",
        ),
        (
            (
                "IT STALLED. PARTS OF THIS DOCUMENT HAVE BEEN REDACTED TO PROTECT PERSONALLY "
                "IDENTIFIABLE INFORMATION PURSUANT TO THE FREEDOM OF INFORMATION ACT (FOIA), "
                "5 U.S.C. 552(B)(6)."
            ),
            "it stalled.",
        ),
    ],
)
def test_clean_text(raw, cleaned):
    assert apply(clean_text, raw) == cleaned


@pytest.mark.parametrize(
    "text",
    [
        "I own a 2019 Lexus ES 350. The ABS light came on.",  # not the template
        "Later the contact said the seat was loose.",  # "the contact" mid-text
        "*NOTE* THE ENGINE STALLED",  # 4-letter tokens aren't initials
        "the *abs* light stays on",  # lowercase emphasis isn't initials
        "XXXL tire sizes and 1999 models",  # not redactions or campaign IDs
    ],
)
def test_clean_text_only_lowercases_ordinary_text(text):
    assert apply(clean_text, text) == text.lower()


def test_is_hotline_template():
    assert apply(is_hotline_template, "TL* THE CONTACT OWNS A 2014 JEEP.")
    assert apply(is_hotline_template, "The contact owns a 2020 Ford F-350. It rolled.")
    assert not apply(is_hotline_template, "I own a 2020 Ford. The contact was loose.")


def test_is_recall_parts_unavailable():
    assert apply(
        is_recall_parts_unavailable,
        "The contact received notification; however, the part to do the recall "
        "repair was not yet available.",
    )
    assert apply(
        is_recall_parts_unavailable,
        "however, the parts to do the repairs were unavailable",
    )
    assert not apply(is_recall_parts_unavailable, "The part was available and fixed.")

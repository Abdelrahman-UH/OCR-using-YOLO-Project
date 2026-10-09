import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pipeline import LATIN_TO_ARABIC, boxes_to_text


def test_reading_order_letters_and_digits():
    """Synthetic boxes for letters ب ر ص and digits ١ ٢ ٣ ٤ given in shuffled order

    produce plate_text == 'برص١٢٣٤', where letters are sorted right -> left
    (the ب box has the largest x of the letters) and digits left -> right.
    """
    # Letters:
    # ب has largest x (rightmost on plate) = 0.85
    # ر has middle x = 0.70
    # ص has smallest x of letters = 0.55
    #
    # Digits:
    # ١ has smallest x (leftmost on plate) = 0.10
    # ٢ = 0.20
    # ٣ = 0.30
    # ٤ = 0.40

    # Shuffled order of inputs (using Arabic glyphs)
    names_arabic = ["٤", "ر", "١", "ص", "٣", "ب", "٢"]
    xcs = [0.40, 0.70, 0.10, 0.55, 0.30, 0.85, 0.20]

    plate_text, display_text = boxes_to_text(names_arabic, xcs)
    assert plate_text == "برص١٢٣٤"
    assert display_text == "ب ر ص  ١٢٣٤"

    # Also test with Latin class names
    names_latin = ["4", "reh", "1", "sad", "3", "beh", "2"]
    plate_text_lat, display_text_lat = boxes_to_text(names_latin, xcs)
    assert plate_text_lat == "برص١٢٣٤"
    assert display_text_lat == "ب ر ص  ١٢٣٤"


def test_heh_and_yeh_code_points():
    """Verify exact code points for heh (U+06BE) and yeh (U+0649)."""
    heh_char = LATIN_TO_ARABIC["heh"]
    yeh_char = LATIN_TO_ARABIC["yeh"]

    assert heh_char == "ھ"
    assert ord(heh_char) == 0x06BE  # U+06BE Arabic letter heh doachashmee

    assert yeh_char == "ى"
    assert ord(yeh_char) == 0x0649  # U+0649 Arabic letter alef maksura / yeh

    # Test reading order containing heh and yeh
    plate_text, _ = boxes_to_text(["heh", "yeh", "1"], [0.80, 0.60, 0.20])
    assert plate_text == "ھى١"
    assert ord(plate_text[0]) == 0x06BE
    assert ord(plate_text[1]) == 0x0649


def test_plate_with_only_digits():
    """A plate containing only digits should be sorted left -> right."""
    names = ["5", "2", "8"]
    xcs = [0.50, 0.20, 0.80]

    plate_text, display_text = boxes_to_text(names, xcs)
    assert plate_text == "٢٥٨"
    assert display_text == "٢٥٨"


def test_plate_with_only_letters():
    """A plate containing only letters should be sorted right -> left."""
    names = ["sad", "reh"]
    xcs = [0.40, 0.70]  # reh is further right than sad

    plate_text, display_text = boxes_to_text(names, xcs)
    assert plate_text == "رص"
    assert display_text == "ر ص"


def test_empty_reading():
    """Empty inputs should yield empty strings safely."""
    plate_text, display_text = boxes_to_text([], [])
    assert plate_text == ""
    assert display_text == ""

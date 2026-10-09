from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Support both python-bidi import styles
try:
    from bidi.algorithm import get_display  # standard bidi layout
except ImportError:
    from bidi import get_display  # fallback legacy import

import arabic_reshaper

# Bundle font path relative to this script
FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"

# Colors in RGB format
COLOR_VEHICLE = (37, 99, 235)      # royal blue
COLOR_PLATE = (34, 197, 94)        # vibrant green
COLOR_CHAR = (250, 204, 21)        # thin yellow
COLOR_TEXT_WHITE = (255, 255, 255) # white
COLOR_DARK_BG = (15, 23, 42)       # dark slate footer for zoomed plates


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load bundled Arabic font with BASIC layout engine to avoid double-shaping."""
    return ImageFont.truetype(
        str(FONT_PATH),
        size=max(12, size),
        layout_engine=ImageFont.Layout.BASIC,  # prevents Raqm from reversing reshaped bidi
    )


def shape_arabic(text: str) -> str:
    """Reshape Arabic glyphs and apply bidirectional ordering for Pillow drawing."""
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(text)  # connect Arabic letters
    return get_display(reshaped)              # reorder visually for LTR rendering


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    text: str,
    font: ImageFont.FreeTypeFont,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = COLOR_TEXT_WHITE,
    pad: int = 4,
    place_above: bool = True,
) -> None:
    """Draw filled badge with shaped text above (or below) bounding box."""
    if not text:
        return

    shaped_text = shape_arabic(text)
    bbox = font.getbbox(shaped_text)  # (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bx1, by1, bx2, by2 = box

    # Determine vertical position
    if place_above:
        ly2 = by1 - 2
        ly1 = ly2 - text_h - (pad * 2)
        if ly1 < 0:  # clamp if overflowing top edge
            ly1 = by1 + 2
            ly2 = ly1 + text_h + (pad * 2)
    else:
        ly1 = by2 + 2
        ly2 = ly1 + text_h + (pad * 2)

    lx1 = bx1
    lx2 = lx1 + text_w + (pad * 2)

    # Draw filled background badge
    draw.rectangle([lx1, ly1, lx2, ly2], fill=bg_color)

    # Draw text inside badge
    tx = lx1 + pad - bbox[0]
    ty = ly1 + pad - bbox[1]
    draw.text((tx, ty), shaped_text, fill=text_color, font=font)


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    mode: str = "photo",
) -> np.ndarray:
    """Draw vehicle boxes, plate boxes, character boxes, and Arabic readings.

    Returns an RGB numpy array.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB for Pillow
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    w_img, h_img = pil_img.size
    diag = (w_img**2 + h_img**2) ** 0.5
    scale = max(0.5, diag / 1200.0)  # scale factor relative to standard 1080p frame

    line_w = max(2, int(scale * 3))        # vehicle and plate stroke
    char_line_w = max(1, int(scale * 1.5)) # thin yellow character stroke
    font_size_main = max(14, int(scale * 20))
    font_size_sub = max(12, int(scale * 16))

    font_main = get_font(font_size_main)
    font_sub = get_font(font_size_sub)

    for det in detections:
        v_box = det["box"]
        v_cls = det.get("cls", "vehicle")
        v_conf = det.get("conf", 0.0)
        tr_id = det.get("track_id")
        fallback = det.get("fallback", False)
        plate_data = det.get("plate")

        # 1. Vehicle box
        if not fallback:
            draw.rectangle(v_box, outline=COLOR_VEHICLE, width=line_w)

            # Label like "car #12 0.91" (no ID in photo mode)
            if mode == "video" and tr_id is not None:
                v_label = f"{v_cls} #{tr_id} {v_conf:.2f}"
            else:
                v_label = f"{v_cls} {v_conf:.2f}"

            # After lock in video mode: show locked text above vehicle box
            if mode == "video" and plate_data and plate_data.get("locked"):
                v_label = f"{v_label}  |  {plate_data['display_text']}"

            draw_label(draw, v_box, v_label, font_main, bg_color=COLOR_VEHICLE, place_above=True)

        # 2. Plate box and characters
        if plate_data:
            p_box = plate_data["box"]
            p_display = plate_data.get("display_text", "")
            p_ocr_conf = plate_data.get("ocr_conf", 0.0)

            # Green plate outline
            draw.rectangle(p_box, outline=COLOR_PLATE, width=line_w)

            # Draw thin yellow character boxes
            chars = plate_data.get("chars", [])
            for c_info in chars:
                c_box = c_info["box"]
                draw.rectangle(c_box, outline=COLOR_CHAR, width=char_line_w)

            # Filled label above plate with display_text and OCR confidence
            if p_display:
                label_text = f"{p_display}  {p_ocr_conf:.2f}"
                draw_label(draw, p_box, label_text, font_main, bg_color=COLOR_PLATE, place_above=True)

    return np.array(pil_img)  # return RGB numpy array


def plate_zoom(
    plate_crop: np.ndarray,
    char_boxes: list[list[float]] | list[dict[str, Any]],
    display_text: str,
) -> np.ndarray:
    """Upscale plate crop to ~320 px wide with character boxes and Arabic reading underneath.

    Returns an RGB numpy array.
    """
    if plate_crop is None or plate_crop.size == 0:
        blank = Image.new("RGB", (320, 140), color=COLOR_DARK_BG)
        return np.array(blank)

    # Convert to RGB if input is BGR
    if len(plate_crop.shape) == 3 and plate_crop.shape[2] == 3:
        crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    else:
        crop_rgb = plate_crop

    orig_h, orig_w = crop_rgb.shape[:2]
    target_w = 320
    factor = target_w / float(orig_w) if orig_w > 0 else 1.0
    target_h = max(1, int(orig_h * factor))

    # Resize plate crop
    resized_crop = cv2.resize(crop_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    pil_plate = Image.fromarray(resized_crop)
    draw_crop = ImageDraw.Draw(pil_plate)

    # Draw scaled character boxes on crop
    for item in char_boxes:
        box = item["box_local"] if isinstance(item, dict) and "box_local" in item else (item["box"] if isinstance(item, dict) else item)
        scaled_box = [
            box[0] * factor,
            box[1] * factor,
            box[2] * factor,
            box[3] * factor,
        ]
        draw_crop.rectangle(scaled_box, outline=COLOR_CHAR, width=2)

    # Add footer banner for Arabic reading underneath
    footer_h = 50
    total_canvas = Image.new("RGB", (target_w, target_h + footer_h), color=COLOR_DARK_BG)
    total_canvas.paste(pil_plate, (0, 0))

    draw_footer = ImageDraw.Draw(total_canvas)
    if display_text:
        font_reading = get_font(26)
        shaped_text = shape_arabic(display_text)
        bbox = font_reading.getbbox(shaped_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        tx = (target_w - text_w) // 2 - bbox[0]
        ty = target_h + (footer_h - text_h) // 2 - bbox[1]
        draw_footer.text((tx, ty), shaped_text, fill=COLOR_TEXT_WHITE, font=font_reading)

    return np.array(total_canvas)


def render_plate_text_image(
    display_text: str,
    output_path: str | Path = "test_arabic_plate.png",
    font_size: int = 36,
) -> Path:
    """Render a text snippet with NotoNaskhArabic and Pillow to a PNG file."""
    font = get_font(font_size)
    shaped = shape_arabic(display_text)
    bbox = font.getbbox(shaped)
    pad = 20
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = Image.new("RGB", (text_w + pad * 2, text_h + pad * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), shaped, fill=(0, 0, 0), font=font)

    out = Path(output_path)
    img.save(str(out))
    return out


if __name__ == "__main__":
    out_file = render_plate_text_image("ب ر ص  ١٢٣٤", "acceptance_check_3.png")
    print(f"Rendered test reading to {out_file}")

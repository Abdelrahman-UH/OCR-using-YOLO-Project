from __future__ import annotations  # modern type hints

import os
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Mapping from YOLO character class names (Latin) to Arabic characters
LATIN_TO_ARABIC: dict[str, str] = {
    "alef": "\u0623",  # أ (alef with hamza above)
    "beh": "\u0628",   # ب
    "geem": "\u062c",  # ج
    "dal": "\u062f",   # د
    "reh": "\u0631",   # ر
    "seen": "\u0633",  # س
    "sad": "\u0635",   # ص
    "tah": "\u0637",   # ط
    "ain": "\u0639",   # ع
    "feh": "\u0641",   # ف
    "qaf": "\u0642",   # ق
    "lam": "\u0644",   # ل
    "meem": "\u0645",  # م
    "noon": "\u0646",  # ن
    "heh": "\u06be",   # ھ (U+06BE Arabic letter heh doachashmee)
    "waw": "\u0648",   # و
    "yeh": "\u0649",   # ى (U+0649 Arabic letter alef maksura / yeh)
    "1": "\u0661",     # ١
    "2": "\u0662",     # ٢
    "3": "\u0663",     # ٣
    "4": "\u0664",     # ٤
    "5": "\u0665",     # ٥
    "6": "\u0666",     # ٦
    "7": "\u0667",     # ٧
    "8": "\u0668",     # ٨
    "9": "\u0669",     # ٩
}

# Reverse mapping from Arabic characters back to Latin names if needed
ARABIC_TO_LATIN: dict[str, str] = {v: k for k, v in LATIN_TO_ARABIC.items()}

# COCO vehicle class IDs for pretrained YOLO11
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck


class PlateReading(NamedTuple):
    plate_text: str    # letters + digits with no spaces (for voting and CSV)
    display_text: str  # letters with single space, two spaces, then digits


def verify_char_model_classes(model: YOLO) -> None:
    """Check that the character model has exactly the 26 expected Latin names."""
    model_classes = set(model.names.values())
    expected_classes = set(LATIN_TO_ARABIC.keys())
    if model_classes != expected_classes:
        missing = expected_classes - model_classes
        unexpected = model_classes - expected_classes
        raise ValueError(
            f"Character model classes mismatch! Expected 26 classes.\n"
            f"Missing: {missing}\nUnexpected: {unexpected}"
        )


def build_id_to_arabic(model: YOLO) -> dict[int, str]:
    """Build class ID to Arabic character mapping from model.names."""
    verify_char_model_classes(model)
    return {cls_id: LATIN_TO_ARABIC[name] for cls_id, name in model.names.items()}


def boxes_to_text(names: list[str], xcs: list[float]) -> PlateReading:
    """Sort character boxes into Egyptian plate reading order.

    Digits are sorted left-to-right (ascending x-center).
    Letters are sorted right-to-left (descending x-center).
    Returns PlateReading(plate_text, display_text).
    """
    if not names:
        return PlateReading("", "")

    # Convert Latin names to Arabic if Latin strings were passed
    arabic_chars = [LATIN_TO_ARABIC.get(n, n) for n in names]

    # Digits = boxes whose class is a digit, sorted by x-center ascending (left -> right)
    digits = [
        char for _, char in sorted(
            [(x, c) for c, x in zip(arabic_chars, xcs) if c.isdigit()],
            key=lambda item: item[0],
        )
    ]

    # Letters = all other boxes, sorted by x-center descending (right -> left)
    letters = [
        char for _, char in sorted(
            [(x, c) for c, x in zip(arabic_chars, xcs) if not c.isdigit()],
            key=lambda item: item[0],
            reverse=True,
        )
    ]

    plate_text = "".join(letters) + "".join(digits)  # compact text for voting/CSV

    # Format display_text: letters joined by single spaces, then two spaces, then digits
    if letters and digits:
        display_text = " ".join(letters) + "  " + "".join(digits)
    elif letters:
        display_text = " ".join(letters)
    else:
        display_text = "".join(digits)

    return PlateReading(plate_text=plate_text, display_text=display_text)


def load_models(weights_dir: str | Path = "weights") -> dict[str, YOLO]:
    """Load vehicle, plate, and character models onto CUDA or CPU."""
    base = Path(weights_dir)
    plate_path = base / "plate_best.pt"
    char_path = base / "char_best.pt"

    if not plate_path.exists() or not char_path.exists():
        raise FileNotFoundError(
            f"Model weights not found! Expected:\n"
            f"  - {plate_path}\n"
            f"  - {char_path}\n"
            f"Please place 'plate_best.pt' and 'char_best.pt' in the '{weights_dir}' folder."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"  # CUDA if available, else CPU

    # Vehicle detector: pretrained yolo11s.pt from COCO
    vehicle_model = YOLO("yolo11s.pt")
    vehicle_model.to(device)

    # License plate detector
    plate_model = YOLO(str(plate_path))
    plate_model.to(device)

    # Character OCR detector
    char_model = YOLO(str(char_path))
    char_model.to(device)
    verify_char_model_classes(char_model)  # ensure class schema matches

    return {
        "vehicle": vehicle_model,
        "plate": plate_model,
        "char": char_model,
        "id_to_arabic": build_id_to_arabic(char_model),
        "device": device,
    }


def init_video_state(fps: float = 30.0) -> dict[str, Any]:
    """Initialize state dictionary for video tracking and voting."""
    return {
        "tracks": {},       # track_id -> dict(votes, readings, locked, locked_reading, vehicle_cls)
        "locked_rows": [],  # list of locked records for CSV and summary table
        "frame_idx": 0,     # frame index processed
        "fps": fps if fps > 0 else 30.0,
    }


def crop_box(image: np.ndarray, box: list[float] | tuple[float, ...], pad: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
    """Safely crop bounding box from image clamped to dimensions with optional padding."""
    h, w = image.shape[:2]
    x1 = max(0, int(box[0] - pad))
    y1 = max(0, int(box[1] - pad))
    x2 = min(w, int(box[2] + pad))
    y2 = min(h, int(box[3] + pad))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=image.dtype), (x1, y1)

    crop = image[y1:y2, x1:x2].copy()  # detached crop copy
    return crop, (x1, y1)


def process_frame(
    frame_bgr: np.ndarray,
    models: dict[str, Any],
    state: dict[str, Any] | None,
    settings: dict[str, Any],
    mode: str = "photo",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Main processing function for both photo and video modes.

    Returns (annotated_rgb, detections).
    """
    from draw import draw_detections  # avoid circular dependency

    conf_vehicle = settings.get("conf_vehicle", 0.4)
    conf_plate = settings.get("conf_plate", 0.4)
    conf_char = settings.get("conf_char", 0.4)
    char_pad = settings.get("char_pad", 0)
    device = models.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    h_frame, w_frame = frame_bgr.shape[:2]  # frame dimensions
    detections: list[dict[str, Any]] = []

    # 1. Vehicle detection or tracking
    v_model: YOLO = models["vehicle"]
    if mode == "video":
        v_results = v_model.track(
            frame_bgr,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASS_IDS,
            conf=conf_vehicle,
            verbose=False,
            device=device,
        )[0]
    else:
        v_results = v_model.predict(
            frame_bgr,
            classes=VEHICLE_CLASS_IDS,
            conf=conf_vehicle,
            verbose=False,
            device=device,
        )[0]

    vehicle_boxes: list[list[float]] = []
    vehicle_confs: list[float] = []
    vehicle_labels: list[str] = []
    track_ids: list[int | None] = []

    if v_results.boxes is not None and len(v_results.boxes) > 0:
        for b in v_results.boxes:
            box = b.xyxy[0].cpu().numpy().tolist()
            cls_id = int(b.cls[0].item())
            cls_name = v_model.names.get(cls_id, "vehicle")
            conf_val = float(b.conf[0].item())
            tr_id = int(b.id[0].item()) if b.id is not None else None

            vehicle_boxes.append(box)
            vehicle_confs.append(conf_val)
            vehicle_labels.append(cls_name)
            track_ids.append(tr_id)

    # Photo fallback: if no vehicle found, run plate model on the whole image
    fallback_used = False
    if len(vehicle_boxes) == 0 and mode == "photo":
        vehicle_boxes.append([0.0, 0.0, float(w_frame), float(h_frame)])
        vehicle_confs.append(1.0)
        vehicle_labels.append("vehicle")
        track_ids.append(None)
        fallback_used = True

    # 2. Process each vehicle
    p_model: YOLO = models["plate"]
    c_model: YOLO = models["char"]
    id_to_arabic = models.get("id_to_arabic", build_id_to_arabic(c_model))

    if state is None:
        state = init_video_state()

    state["frame_idx"] += 1
    current_time_sec = state["frame_idx"] / state["fps"]
    mins = int(current_time_sec // 60)
    secs = int(current_time_sec % 60)
    time_str = f"{mins:02d}:{secs:02d}"  # mm:ss format for logging

    for idx, (v_box, v_conf, v_label, tr_id) in enumerate(
        zip(vehicle_boxes, vehicle_confs, vehicle_labels, track_ids)
    ):
        v_det: dict[str, Any] = {
            "box": v_box,
            "conf": v_conf,
            "cls": v_label,
            "track_id": tr_id,
            "plate": None,
            "fallback": fallback_used,
        }

        # Check if this track is already locked in video mode
        if mode == "video" and tr_id is not None and tr_id in state["tracks"]:
            tr_info = state["tracks"][tr_id]
            if tr_info.get("locked", False):
                # Stop running plate and char models for this track!
                locked_data = tr_info["locked_reading"]
                v_det["plate"] = {
                    "box": locked_data["box"],
                    "conf": locked_data["conf"],
                    "plate_text": locked_data["plate_text"],
                    "display_text": locked_data["display_text"],
                    "ocr_conf": locked_data["ocr_conf"],
                    "chars": locked_data.get("chars", []),
                    "plate_crop": locked_data.get("plate_crop"),
                    "locked": True,
                    "leading": False,
                }
                detections.append(v_det)
                continue  # skip inference for locked track

        # Crop vehicle from frame
        v_crop, (vx_off, vy_off) = crop_box(frame_bgr, v_box)
        if v_crop.size == 0:
            detections.append(v_det)
            continue

        # Run plate detector on vehicle crop
        p_res = p_model.predict(
            v_crop,
            conf=conf_plate,
            verbose=False,
            device=device,
        )[0]

        best_plate_box: list[float] | None = None
        best_plate_conf = 0.0

        if p_res.boxes is not None and len(p_res.boxes) > 0:
            # Keep highest-confidence plate box
            for pb in p_res.boxes:
                c_val = float(pb.conf[0].item())
                if c_val > best_plate_conf:
                    best_plate_conf = c_val
                    best_plate_box = pb.xyxy[0].cpu().numpy().tolist()

        if best_plate_box is not None:
            # Convert plate box to frame coordinates
            p_frame_box = [
                best_plate_box[0] + vx_off,
                best_plate_box[1] + vy_off,
                best_plate_box[2] + vx_off,
                best_plate_box[3] + vy_off,
            ]

            # 3. Crop plate exactly (with optional padding) and run char model
            plate_crop, (px_off, py_off) = crop_box(frame_bgr, p_frame_box, pad=char_pad)

            char_boxes_local: list[list[float]] = []
            char_boxes_frame: list[list[float]] = []
            char_names: list[str] = []
            char_confs: list[float] = []
            char_xcs_local: list[float] = []

            if plate_crop.size > 0:
                c_res = c_model.predict(
                    plate_crop,
                    imgsz=320,
                    conf=conf_char,
                    agnostic_nms=True,
                    verbose=False,
                    device=device,
                )[0]

                if c_res.boxes is not None and len(c_res.boxes) > 0:
                    for cb in c_res.boxes:
                        cb_xyxy = cb.xyxy[0].cpu().numpy().tolist()
                        cls_idx = int(cb.cls[0].item())
                        c_conf = float(cb.conf[0].item())
                        c_name = id_to_arabic.get(cls_idx, "")

                        # x-center normalized or absolute in crop
                        xc = (cb_xyxy[0] + cb_xyxy[2]) / 2.0

                        char_boxes_local.append(cb_xyxy)
                        char_boxes_frame.append([
                            cb_xyxy[0] + px_off,
                            cb_xyxy[1] + py_off,
                            cb_xyxy[2] + px_off,
                            cb_xyxy[3] + py_off,
                        ])
                        char_names.append(c_name)
                        char_confs.append(c_conf)
                        char_xcs_local.append(xc)

            # 4. Reading order
            reading = boxes_to_text(char_names, char_xcs_local)
            ocr_conf = float(np.mean(char_confs)) if char_confs else 0.0

            plate_info: dict[str, Any] = {
                "box": p_frame_box,
                "conf": best_plate_conf,
                "plate_text": reading.plate_text,
                "display_text": reading.display_text,
                "ocr_conf": ocr_conf,
                "chars": [
                    {"box": b_f, "box_local": b_l, "name": n, "conf": c}
                    for b_f, b_l, n, c in zip(char_boxes_frame, char_boxes_local, char_names, char_confs)
                ],
                "plate_crop": plate_crop,
                "locked": False,
                "leading": False,
            }

            # 5. Video voting per track ID
            if mode == "video" and tr_id is not None:
                if tr_id not in state["tracks"]:
                    state["tracks"][tr_id] = {
                        "votes": Counter(),
                        "readings": {},
                        "locked": False,
                        "locked_reading": None,
                        "vehicle_cls": v_label,
                    }

                tr_data = state["tracks"][tr_id]

                # Keep only readings with len(plate_text) >= 3
                if len(reading.plate_text) >= 3:
                    p_txt = reading.plate_text
                    tr_data["votes"][p_txt] += 1

                    # Keep best reading quality metadata
                    prev = tr_data["readings"].get(p_txt)
                    if prev is None or (ocr_conf > prev["ocr_conf"]):
                        tr_data["readings"][p_txt] = {
                            "box": p_frame_box,
                            "conf": best_plate_conf,
                            "plate_text": p_txt,
                            "display_text": reading.display_text,
                            "ocr_conf": ocr_conf,
                            "chars": plate_info["chars"],
                            "plate_crop": plate_crop,
                        }

                    # Check locking condition: top reading has >= 3 votes and strictly more than 2nd
                    top_two = tr_data["votes"].most_common(2)
                    top_text, top_count = top_two[0]
                    second_count = top_two[1][1] if len(top_two) > 1 else 0

                    if top_count >= 3 and top_count > second_count:
                        tr_data["locked"] = True
                        tr_data["locked_reading"] = tr_data["readings"][top_text]
                        plate_info["locked"] = True
                        plate_info["plate_text"] = top_text
                        plate_info["display_text"] = tr_data["locked_reading"]["display_text"]

                        # Log one row for this locked track
                        log_row = {
                            "time": time_str,
                            "track_id": tr_id,
                            "vehicle": v_label,
                            "plate": top_text,
                            "det_conf": round(best_plate_conf, 2),
                            "ocr_conf": round(tr_data["locked_reading"]["ocr_conf"], 2),
                        }
                        state["locked_rows"].append(log_row)
                    else:
                        # Before lock: display leading reading with '?'
                        plate_info["leading"] = True
                        leading_reading = tr_data["readings"][top_text]
                        plate_info["display_text"] = f"{leading_reading['display_text']} ?"

            v_det["plate"] = plate_info

        detections.append(v_det)

    # Render frame using Pillow Arabic drawing
    annotated_rgb = draw_detections(frame_bgr, detections, mode=mode)
    return annotated_rgb, detections

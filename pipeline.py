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
        "tracks": {},                      # track_id -> votes, best frame per reading, vehicle class votes
        "fps": fps if fps > 0 else 30.0,
        "time_sec": 0.0,                   # time of the current frame, set by app.py before each frame
    }


def fmt_time(sec: float) -> str:
    """Seconds -> mm:ss."""
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


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


def add_reading(
    state: dict[str, Any],
    tr_id: int,
    v_label: str,
    plate_info: dict[str, Any],
    v_crop: np.ndarray,
    plate_box_local: list[float],
) -> None:
    """Add one frame's reading to the votes of a track, and keep the best frame per reading."""
    tracks = state["tracks"]
    if tr_id not in tracks:
        tracks[tr_id] = {
            "votes": Counter(),            # plate_text -> number of frames that read it
            "vehicle": Counter(),          # car / truck can flicker between frames, so vote it too
            "best": {},                    # plate_text -> best frame for that reading
            "first_sec": state["time_sec"],
        }
    tr = tracks[tr_id]
    tr["vehicle"][v_label] += 1

    text = plate_info["plate_text"]
    if len(text) < 3:
        return                                             # too short to be a real plate

    tr["votes"][text] += 1

    score = plate_info["ocr_conf"] * plate_info["conf"]    # how sure the models were in this frame
    old = tr["best"].get(text)
    if old is not None and old["score"] >= score:
        return                                             # an earlier frame was better

    # small copy of the car with its plate box, to show the best frame at the end
    factor = min(1.0, 400 / max(1, v_crop.shape[1]))       # max 400 px wide to save memory
    car = cv2.resize(v_crop, None, fx=factor, fy=factor)
    x1, y1, x2, y2 = [int(v * factor) for v in plate_box_local]
    cv2.rectangle(car, (x1, y1), (x2, y2), (94, 197, 34), 2)   # green (BGR)

    tr["best"][text] = {
        "score": score,
        "display_text": plate_info["display_text"],
        "ocr_conf": plate_info["ocr_conf"],
        "conf": plate_info["conf"],
        "chars": plate_info["chars"],
        "plate_crop": plate_info["plate_crop"],
        "car_view": cv2.cvtColor(car, cv2.COLOR_BGR2RGB),
        "time_sec": state["time_sec"],
    }


def leading_display(state: dict[str, Any], tr_id: int, min_votes: int) -> str | None:
    """Text to show on the live frame: the reading with most votes so far, '?' until confirmed."""
    tr = state["tracks"].get(tr_id)
    if tr is None or not tr["votes"]:
        return None
    text, votes = tr["votes"].most_common(1)[0]
    shown = tr["best"][text]["display_text"]
    return shown if votes >= min_votes else f"{shown} ?"


def summarize_tracks(state: dict[str, Any], min_votes: int = 3) -> list[dict[str, Any]]:
    """One result per car: winning reading (most votes) + the best frame of that reading.

    The same plate under two track IDs (ByteTrack lost the car for a moment) is merged.
    """
    by_plate: dict[str, dict[str, Any]] = {}

    for tr_id, tr in state["tracks"].items():
        if not tr["votes"]:
            continue
        text, votes = tr["votes"].most_common(1)[0]       # winning reading of this track
        row = dict(tr["best"][text])                       # copy of the best frame info
        row.update({
            "plate_text": text,
            "track_ids": [tr_id],
            "votes": votes,
            "frames_read": sum(tr["votes"].values()),      # frames where a plate was read
            "vehicle": tr["vehicle"].most_common(1)[0][0],
            "first_sec": tr["first_sec"],
        })

        old = by_plate.get(text)
        if old is None:
            by_plate[text] = row
            continue

        # same plate seen under another track ID -> merge into one car
        old["track_ids"].append(tr_id)
        old["votes"] += row["votes"]
        old["frames_read"] += row["frames_read"]
        old["first_sec"] = min(old["first_sec"], row["first_sec"])
        if row["score"] > old["score"]:
            for k in ["score", "display_text", "ocr_conf", "conf", "chars", "plate_crop", "car_view", "time_sec"]:
                old[k] = row[k]

    cars = [r for r in by_plate.values() if r["votes"] >= min_votes]   # enough frames agree
    return sorted(cars, key=lambda r: r["first_sec"])                    # in order of appearance


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
    id_to_arabic = models["id_to_arabic"]  # built once in load_models

    if state is None:
        state = init_video_state()

    min_votes = settings.get("min_votes", 3)

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
            }

            # 5. Video voting per track ID: every frame votes, best frame per reading is kept
            if mode == "video" and tr_id is not None:
                add_reading(state, tr_id, v_label, plate_info, v_crop, best_plate_box)
                shown = leading_display(state, tr_id, min_votes)
                if shown is not None:
                    plate_info["display_text"] = shown     # stable text instead of this frame's guess

            v_det["plate"] = plate_info

        detections.append(v_det)

    # Render frame using Pillow Arabic drawing
    annotated_rgb = draw_detections(frame_bgr, detections, mode=mode)
    return annotated_rgb, detections

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio

from pipeline import init_video_state, load_models, process_frame

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Egyptian License Plate Recognition using YOLO11"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input photo or video file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Directory to save annotated output file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights",
        help="Path to folder containing plate_best.pt and char_best.pt",
    )
    parser.add_argument(
        "--conf-vehicle",
        type=float,
        default=0.4,
        help="Confidence threshold for vehicle detector (default 0.4)",
    )
    parser.add_argument(
        "--conf-plate",
        type=float,
        default=0.4,
        help="Confidence threshold for plate detector (default 0.4)",
    )
    parser.add_argument(
        "--conf-char",
        type=float,
        default=0.4,
        help="Confidence threshold for character detector (default 0.4)",
    )
    return parser.parse_args()


def predict_image(
    source_path: Path,
    out_dir: Path,
    models: dict,
    settings: dict,
) -> None:
    frame_bgr = cv2.imread(str(source_path))  # load input image
    if frame_bgr is None:
        print(f"Error: Could not read image at '{source_path}'", file=sys.stderr)
        sys.exit(1)

    annotated_rgb, detections = process_frame(
        frame_bgr,
        models=models,
        state=None,
        settings=settings,
        mode="photo",
    )

    plates_found = 0
    print(f"\n--- Detections for {source_path.name} ---")
    for det in detections:
        v_cls = det.get("cls", "vehicle")
        v_conf = det.get("conf", 0.0)
        plate_data = det.get("plate")

        if plate_data:
            plates_found += 1
            p_text = plate_data["plate_text"]
            p_display = plate_data["display_text"]
            det_conf = plate_data["conf"]
            ocr_conf = plate_data["ocr_conf"]
            print(
                f"  Vehicle: {v_cls} ({v_conf:.2f}) | "
                f"Plate: '{p_text}' (Display: '{p_display}') | "
                f"Plate Conf: {det_conf:.2f} | OCR Conf: {ocr_conf:.2f}"
            )
        else:
            print(f"  Vehicle: {v_cls} ({v_conf:.2f}) | No plate detected")

    if plates_found == 0:
        print("  No license plates recognized.")

    # Save output image
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_{source_path.stem}.png"
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), annotated_bgr)
    print(f"\nSaved annotated image to: {out_path}\n")


def predict_video(
    source_path: Path,
    out_dir: Path,
    models: dict,
    settings: dict,
) -> None:
    cap = cv2.VideoCapture(str(source_path))  # open video source
    if not cap.isOpened():
        print(f"Error: Could not open video file '{source_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = init_video_state(fps=fps)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_{source_path.stem}.mp4"

    # Use imageio with libx264 for browser and universal player compatibility
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    frame_count = 0
    print(f"\nProcessing video '{source_path.name}' ({total_frames} frames at {fps:.1f} FPS)...")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1

            annotated_rgb, _ = process_frame(
                frame_bgr,
                models=models,
                state=state,
                settings=settings,
                mode="video",
            )
            writer.append_data(annotated_rgb)

            if frame_count % 30 == 0 or frame_count == total_frames:
                print(
                    f"  Processed {frame_count}/{total_frames} frames "
                    f"({(frame_count / total_frames * 100):.1f}%) | "
                    f"Locked plates: {len(state['locked_rows'])}"
                )
    finally:
        cap.release()
        writer.close()

    print(f"\nVideo processing complete! Saved to: {out_path}")
    print("\n--- Locked License Plates ---")
    if state["locked_rows"]:
        for row in state["locked_rows"]:
            print(
                f"  [{row['time']}] Track #{row['track_id']} ({row['vehicle']}): "
                f"Plate = '{row['plate']}' | Det Conf = {row['det_conf']} | OCR Conf = {row['ocr_conf']}"
            )
    else:
        print("  No tracks locked during this video.")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"Error: Source file '{source_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        models = load_models(weights_dir=args.weights)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    settings = {
        "conf_vehicle": args.conf_vehicle,
        "conf_plate": args.conf_plate,
        "conf_char": args.conf_char,
        "char_pad": 0,
    }

    out_dir = Path(args.out)
    ext = source_path.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        predict_image(source_path, out_dir, models, settings)
    elif ext in VIDEO_EXTENSIONS:
        predict_video(source_path, out_dir, models, settings)
    else:
        print(
            f"Unsupported file format '{ext}'. Supported formats:\n"
            f"  Images: {', '.join(IMAGE_EXTENSIONS)}\n"
            f"  Videos: {', '.join(VIDEO_EXTENSIONS)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

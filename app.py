from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from ultralytics import YOLO

from draw import plate_zoom
from pipeline import (
    fmt_time,
    init_video_state,
    load_models,
    process_frame,
    summarize_tracks,
)

# Page configuration
st.set_page_config(
    page_title="OCR using YOLO 11 | Egyptian License Plate Recognition ",
    page_icon="🚗",
    layout="wide",

    initial_sidebar_state="expanded",
)

# Custom styling for a polished graduate diploma presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_models(weights_dir: str = "weights") -> dict[str, Any] | None:
    """Load and cache plate and char models once across sessions."""
    p_path = Path(weights_dir) / "plate_best.pt"
    c_path = Path(weights_dir) / "char_best.pt"

    if not p_path.exists() or not c_path.exists():
        return None  # signal missing weights gracefully

    return load_models(weights_dir=weights_dir)


def get_fresh_models(cached: dict[str, Any]) -> dict[str, Any]:
    """Return model set with a fresh vehicle tracker instance for clean video state."""
    device = cached["device"]
    fresh_vehicle = YOLO("yolo11s.pt")  # fresh vehicle model resets ByteTrack IDs
    fresh_vehicle.to(device)

    return {
        "vehicle": fresh_vehicle,
        "plate": cached["plate"],
        "char": cached["char"],
        "id_to_arabic": cached["id_to_arabic"],
        "device": device,
    }


def main() -> None:
    # Header and diploma project summary
    st.markdown('<div class="main-header">🚗 Egyptian License Plate Recognition with YOLO</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'AMIT Data Science & AI Diploma — Graduation Project (OCR using YOLO). '
        
        
        
        
        '</div>',
        unsafe_allow_html=True,
    )

    # Check for weights availability early
    cached_models = get_cached_models(weights_dir="weights")
    if cached_models is None:
        st.error(
            "⚠️ **Model Weights Missing!**\n\n"
            "The required fine-tuned YOLO11s weights could not be found. Please ensure:\n"
            "- `weights/plate_best.pt` (License Plate Detector)\n"
            "- `weights/char_best.pt` (Character OCR Detector)\n\n"
            "are placed inside the `weights/` directory relative to the project root.",
            icon="🚫",
        )
        st.info("Run `python predict.py --help` or see `weights/README.md` for more information.")
        return

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio("Detection Mode", ["Photo", "Video"], index=0)

    st.sidebar.subheader("Confidence Thresholds")
    conf_vehicle = st.sidebar.slider("Vehicle Confidence", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    conf_plate = st.sidebar.slider("Plate Confidence", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    conf_char = st.sidebar.slider("Character Confidence", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    char_pad = st.sidebar.slider("Plate Crop Padding (px)", min_value=0, max_value=20, value=0, step=1)

    settings = {
        "conf_vehicle": conf_vehicle,
        "conf_plate": conf_plate,
        "conf_char": conf_char,
        "char_pad": char_pad,
    }

    if mode == "Photo":
        run_photo_mode(cached_models, settings)
    else:
        run_video_mode(cached_models, settings)


def run_photo_mode(models: dict[str, Any], settings: dict[str, Any]) -> None:
    st.subheader("📸 Photo Recognition")
    uploaded_file = st.file_uploader(
        "Upload vehicle photo (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a street vehicle photo or plate close-up",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image file above to run license plate recognition.")
        return

    # Load and correct EXIF orientation for phone cameras
    raw_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(raw_bytes))
    pil_image = ImageOps.exif_transpose(pil_image)

    # Convert to BGR numpy array for OpenCV/YOLO
    rgb_arr = np.array(pil_image.convert("RGB"))
    frame_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    with st.spinner("Processing image through YOLO11 pipeline..."):
        annotated_rgb, detections = process_frame(
            frame_bgr,
            models=models,
            state=None,
            settings=settings,
            mode="photo",
        )

    # Layout: side-by-side original and annotated images
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.markdown("**Original Image**")
        st.image(pil_image, use_container_width=True)

    with col_annot:
        st.markdown("**Annotated Detection**")
        st.image(annotated_rgb, use_container_width=True)

    # Collect plate detections
    plate_rows = []
    plate_crops_data = []

    for det in detections:
        v_cls = det.get("cls", "vehicle")
        v_conf = det.get("conf", 0.0)
        plate_data = det.get("plate")

        if plate_data and plate_data.get("plate_text"):
            p_text = plate_data["plate_text"]
            p_disp = plate_data["display_text"]
            det_conf = plate_data["conf"]
            ocr_conf = plate_data["ocr_conf"]

            plate_rows.append({
                "Vehicle": v_cls,
                "Plate (Arabic)": p_text,
                "Display Reading": p_disp,
                "Plate Conf": round(det_conf, 2),
                "OCR Conf": round(ocr_conf, 2),
            })

            plate_crops_data.append({
                "crop": plate_data.get("plate_crop"),
                "chars": plate_data.get("chars", []),
                "display": p_disp,
            })

    # Results table & gallery
    st.markdown("---")
    if plate_rows:
        st.subheader("📋 Recognized License Plates")
        df_plates = pd.DataFrame(plate_rows)
        st.dataframe(df_plates[["Vehicle", "Plate (Arabic)", "Plate Conf", "OCR Conf"]], use_container_width=True)

        # Plate zoom gallery
        st.subheader("🔍 Plate Detail Gallery (Zoom & Characters)")
        gallery_cols = st.columns(min(len(plate_crops_data), 3))
        for idx, item in enumerate(plate_crops_data):
            col = gallery_cols[idx % 3]
            with col:
                zoom_img = plate_zoom(item["crop"], item["chars"], item["display"])
                st.image(zoom_img, caption=f"Plate #{idx + 1}: {item['display']}", use_container_width=True)
    else:
        # Check if vehicles were found at all
        has_vehicles = any(not d.get("fallback") for d in detections)
        if not has_vehicles and not detections:
            st.warning("No vehicles were found in the image. Try lowering the vehicle confidence threshold.")
        else:
            st.warning("Vehicle detected, but no license plate characters could be recognized at the current thresholds.")

    # Download annotated image button
    out_buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(out_buf, format="PNG")
    st.download_button(
        label="💾 Download Annotated Image (PNG)",
        data=out_buf.getvalue(),
        file_name=f"annotated_{Path(uploaded_file.name).stem}.png",
        mime="image/png",
    )


def run_video_mode(cached_models: dict[str, Any], settings: dict[str, Any]) -> None:
    st.subheader("🎥 Video Recognition & Vehicle Tracking")

    # Video specific sidebar parameters
    st.sidebar.subheader("Video Parameters")
    frame_step = st.sidebar.slider("Process Every Nth Frame", min_value=1, max_value=10, value=1, step=1)
    max_frame_width = st.sidebar.slider("Max Frame Width (px)", min_value=640, max_value=1920, value=1280, step=80)
    min_votes = st.sidebar.slider(
        "Votes needed to accept a plate", min_value=1, max_value=10, value=3, step=1,
        help="A car is listed only if at least this many frames read the same plate.",
    )
    settings = {**settings, "min_votes": min_votes}

    uploaded_video = st.file_uploader(
        "Upload vehicle video (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload street traffic video for tracking and plate voting",
    )

    if uploaded_video is None:
        st.info("👆 Please upload a video file above to start video recognition.")
        return

    video_key = f"{uploaded_video.name}-{uploaded_video.size}"   # a new upload = new results

    # Processing runs only when the button is pressed. The results are kept in session_state,
    # so clicking a download button or moving a slider does NOT process the whole video again.
    if st.button("▶️ Read plates in this video", type="primary"):
        result = process_video(uploaded_video, cached_models, settings, frame_step, max_frame_width)
        if result is not None:
            result["key"] = video_key
            st.session_state["video_result"] = result

    result = st.session_state.get("video_result")
    if result is None or result["key"] != video_key:
        st.info("Press **Read plates in this video** to start. To cancel while it runs, press Stop (top right).")
        return

    show_video_result(result, min_votes)


def process_video(
    uploaded_video: Any,
    cached_models: dict[str, Any],
    settings: dict[str, Any],
    frame_step: int,
    max_frame_width: int,
) -> dict[str, Any] | None:
    """Read the video frame by frame, vote per car, and write the annotated MP4."""
    models = get_fresh_models(cached_models)   # fresh tracker -> track IDs start from 1 again

    # Save uploaded video to temporary file (OpenCV needs a path)
    ext = Path(uploaded_video.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
        tmp_in.write(uploaded_video.getvalue())
        temp_input_path = tmp_in.name

    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("Could not open this video. Try converting it to MP4 (H.264).")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state = init_video_state(fps=video_fps)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output_path = tmp_out.name
    tmp_out.close()
    writer = imageio.get_writer(
        temp_output_path,
        fps=video_fps / frame_step,      # keeps the output at real speed when frames are skipped
        codec="libx264",                 # browser-playable MP4
        pixelformat="yuv420p",
    )

    st.markdown("### ⚡ Live Tracking & Inference")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    col_fps, col_veh, col_ok = st.columns(3)
    fps_metric = col_fps.empty()
    veh_metric = col_veh.empty()
    ok_metric = col_ok.empty()

    frame_idx = -1                       # index of the frame in the original video
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue

            # Resize larger frames for speed
            if frame_bgr.shape[1] > max_frame_width:
                factor = max_frame_width / float(frame_bgr.shape[1])
                frame_bgr = cv2.resize(frame_bgr, (max_frame_width, int(frame_bgr.shape[0] * factor)))

            state["time_sec"] = frame_idx / video_fps     # real video time, also when frames are skipped
            annotated_rgb, _ = process_frame(frame_bgr, models=models, state=state, settings=settings, mode="video")
            writer.append_data(annotated_rgb)
            processed_count += 1

            frame_placeholder.image(annotated_rgb, use_container_width=True)
            progress_bar.progress(min(1.0, (frame_idx + 1) / max(1, total_frames)))
            fps_metric.metric("Processing Speed", f"{processed_count / max(0.001, time.time() - start_time):.1f} FPS")
            veh_metric.metric("Vehicles Tracked", f"{len(state['tracks'])}")
            ok_metric.metric("Plates Accepted", f"{len(summarize_tracks(state, settings['min_votes']))}")
    finally:
        cap.release()
        writer.close()
        try:
            os.remove(temp_input_path)
        except OSError:
            pass

    with open(temp_output_path, "rb") as vf:
        video_bytes = vf.read()
    try:
        os.remove(temp_output_path)
    except OSError:
        pass

    frame_placeholder.empty()            # the finished video is shown below instead
    progress_bar.empty()

    return {
        "file_name": uploaded_video.name,
        "video_bytes": video_bytes,
        "state": state,                  # all votes are kept, so the min-votes slider works without re-running
        "frames": processed_count,
    }


def show_video_result(result: dict[str, Any], min_votes: int) -> None:
    """Show the results like photo mode: one entry per car with its best frame and characters."""
    stem = Path(result["file_name"]).stem
    st.success(f"Done: {result['frames']} frames read from {result['file_name']}.")

    st.subheader("🎬 Annotated Video Result")
    st.video(result["video_bytes"])
    st.download_button(
        label="💾 Download Annotated Video (MP4)",
        data=result["video_bytes"],
        file_name=f"annotated_{stem}.mp4",
        mime="video/mp4",
    )

    st.markdown("---")
    cars = summarize_tracks(result["state"], min_votes)
    if not cars:
        st.warning(
            f"No car had the same plate read in at least {min_votes} frames. "
            "Lower 'Votes needed to accept a plate' in the sidebar, or lower the confidence thresholds and run again."
        )
        return

    # Table: one row per car
    rows = []
    for i, car in enumerate(cars):
        rows.append({
            "Car": i + 1,
            "Vehicle": car["vehicle"],
            "Plate (Arabic)": car["plate_text"],
            "Votes": f"{car['votes']} / {car['frames_read']}",     # frames that agree / frames read
            "First seen": fmt_time(car["first_sec"]),
            "Best frame at": fmt_time(car["time_sec"]),
            "Plate Conf": round(car["conf"], 2),
            "OCR Conf": round(car["ocr_conf"], 2),
            "Track IDs": ", ".join(str(t) for t in car["track_ids"]),
        })
    df_cars = pd.DataFrame(rows)

    st.subheader(f"📋 Recognized License Plates ({len(cars)} cars)")
    st.dataframe(df_cars, use_container_width=True, hide_index=True)
    st.download_button(
        label="📥 Download Plates Log (CSV)",
        data=df_cars.to_csv(index=False).encode("utf-8-sig"),     # utf-8-sig so Excel shows Arabic
        file_name=f"plates_log_{stem}.csv",
        mime="text/csv",
    )

    # Gallery: best frame of each car + zoomed plate with character boxes
    st.subheader("🔍 Best Frame per Car")
    for i, car in enumerate(cars):
        col_car, col_plate = st.columns(2)
        with col_car:
            st.image(
                car["car_view"],
                caption=f"Car {i + 1} ({car['vehicle']}) at {fmt_time(car['time_sec'])}",
                use_container_width=True,
            )
        with col_plate:
            zoom_img = plate_zoom(car["plate_crop"], car["chars"], car["display_text"])
            st.image(zoom_img, use_container_width=True)
            st.markdown(f"**Plate:** {car['display_text']}")
            st.caption(
                f"{car['votes']} of {car['frames_read']} frames read this plate, "
                f"OCR conf {car['ocr_conf']:.2f}, plate conf {car['conf']:.2f}"
            )
        st.markdown("---")


if __name__ == "__main__":
    main()

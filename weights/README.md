# Model Weights

This directory holds the fine-tuned YOLO11s model weights used by the pipeline:

1. **`plate_best.pt`**
   - **Task**: License Plate Detection.
   - **Architecture**: YOLO11s (Ultralytics).
   - **Classes**: 1 (`plate`).
   - **Input Resolution**: `imgsz = 640`.
   - **Trained On**: Full vehicle photos from the EALPR dataset.
   - **Role**: Detects license plate bounding boxes on vehicle crops (or the full image in close-up fallback mode).

2. **`char_best.pt`**
   - **Task**: Plate Character Recognition ("OCR using YOLO").
   - **Architecture**: YOLO11s (Ultralytics).
   - **Classes**: 26 classes (17 Arabic letters + 9 Arabic-Indic digits `١` through `٩`; class `٠` was omitted due to dataset sparsity).
   - **Class Labels**: Trained with Latin names mapped to Arabic:
     - Letters: `alef` (أ), `beh` (ب), `geem` (ج), `dal` (د), `reh` (ر), `seen` (س), `sad` (ص), `tah` (ط), `ain` (ع), `feh` (ف), `qaf` (ق), `lam` (ل), `meem` (م), `noon` (ن), `heh` (ھ, U+06BE), `waw` (و), `yeh` (ى, U+0649).
     - Digits: `'1'` (١), `'2'` (٢), `'3'` (٣), `'4'` (٤), `'5'` (٥), `'6'` (٦), `'7'` (٧), `'8'` (٨), `'9'` (٩).
   - **Input Resolution**: `imgsz = 320`.
   - **Trained On**: Tightly cropped license plates.
   - **Role**: Detects each individual character and digit on the license plate crop.

3. **Pretrained Vehicle Detector**
   - Standard `yolo11s.pt` (COCO) is automatically downloaded by Ultralytics on first run.
   - It filters for vehicle classes: `2` (car), `3` (motorcycle), `5` (bus), `7` (truck).

### Where to Place Weights
Ensure that `plate_best.pt` and `char_best.pt` are placed directly in this directory (`weights/` relative to the repository root):
```
weights/
├── plate_best.pt
├── char_best.pt
└── README.md
```
If weights are missing, the pipeline and Streamlit application will display an informative error instructing the user to supply these files.

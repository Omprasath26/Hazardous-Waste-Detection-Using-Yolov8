from pathlib import Path
from ultralytics import YOLO
import uuid
import shutil
import cv2

# ============================================================
# BASE PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# TEMPORARY: use the original working trained model directly
MODEL_PATH = BASE_DIR / "runs" / "hazwaste_detection" / "yolov8n_baseline" / "weights" / "best.pt"

UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "app" / "static" / "outputs"

for path_obj in [UPLOAD_DIR, OUTPUT_DIR]:
    if path_obj.exists() and not path_obj.is_dir():
        raise RuntimeError(f"{path_obj} exists but is not a directory. Delete or rename it.")
    path_obj.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ============================================================
# LOAD MODEL ONCE
# ============================================================
model = YOLO(str(MODEL_PATH))

CLASS_NAMES = {
    0: "Cylinder",
    1: "Shock_Absorber"
}

def save_upload_file(upload_file) -> Path:
    """
    Save uploaded file to static/uploads and return saved file path.
    """
    ext = Path(upload_file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return save_path

def run_inference(image_path: Path, conf: float = 0.25):
    """
    Run YOLO inference on one image.
    Saves annotated output image and returns prediction details.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=640,
        save=False,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    # Save annotated output image
    plotted_img = result.plot()
    output_name = f"pred_{image_path.name}"
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), plotted_img)

    detections = []
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": round(conf_score, 4),
                "bbox": [round(v, 2) for v in xyxy]
            })

    return {
        "input_image_name": image_path.name,
        "output_image_name": output_name,
        "detections": detections
    }
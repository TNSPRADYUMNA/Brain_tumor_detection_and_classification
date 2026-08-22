import os
import uuid
from pathlib import Path
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from gradcam_helper import make_gradcam_and_overlay

# -------------- USER SETTINGS --------------
MODEL_PATH = r"E:\brain tumor\saved models\densenet201_final.h5" # update if needed
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# -------------- HELPERS --------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def preprocess_image_for_model(filepath, model):
    """
    Returns (x_np, orig_pil)
    x_np : preprocessed numpy array shaped (1,H,W,3) ready for model.predict
    orig_pil : PIL.Image resized to target dims (RGB)
    """
    in_shape = None
    try:
        in_shape = model.input_shape
    except Exception:
        in_shape = None

    try:
        if isinstance(in_shape, (list, tuple)) and isinstance(in_shape[0], (list, tuple)):
            _, h, w, c = in_shape[0]
        elif isinstance(in_shape, (list, tuple)) and len(in_shape) == 4:
            _, h, w, c = in_shape
        else:
            h, w, c = (224, 224, 3)
    except Exception:
        h, w, c = (224, 224, 3)

    target = (int(w), int(h))
    img = Image.open(filepath).convert("RGB")
    img_rs = img.resize(target)
    arr = np.asarray(img_rs).astype("float32")
    arr = densenet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr, img_rs

# ---------------- Load model once ----------------
print("Loading model from:", MODEL_PATH)
MODEL = load_model(MODEL_PATH, compile=False)
print("Model loaded. Top-level layer count:", len(MODEL.layers))
print("Top-level layers (first 40):", [L.name for L in MODEL.layers[:40]])

# ---------------- Flask routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(filepath)

    # 1) Detection / preprocessing
    try:
        x, orig_pil = preprocess_image_for_model(filepath, MODEL)
    except Exception as e:
        detection = {"status": "Error", "explanation": f"Preprocess error: {e}"}
        return render_template("result.html", filename=fname, detection=detection,
                               segmentation={"overlay_file": None, "heatmap_file": None, "explanation": "Preprocess error"},
                               pred_label="N/A", top_prob=0.0, prob_map={})

    try:
        preds = MODEL.predict(x, verbose=0)
        probs = preds[0]
        pred_idx = int(np.argmax(probs))
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        pred_label = class_names[pred_idx]
        top_prob = float(probs[pred_idx])
        prob_map = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    except Exception as e:
        detection = {"status": "Error", "explanation": f"Predict error: {e}"}
        return render_template("result.html", filename=fname, detection=detection,
                               segmentation={"overlay_file": None, "heatmap_file": None, "explanation": "Predict error"},
                               pred_label="N/A", top_prob=0.0, prob_map={})

    if pred_label == "notumor":
        detection = {"status": "No tumor", "explanation": "Model predicted no tumor."}
    else:
        detection = {"status": "Tumor found", "explanation": f"Predicted class: {pred_label} (p={top_prob:.4f})"}

    # Generate both the Grad-CAM overlay shown on the page and its raw heatmap.
    stem = Path(fname).stem
    overlay_name = f"{stem}_gradcam.png"
    heatmap_name = f"{stem}_heatmap.png"
    overlay_path = UPLOAD_FOLDER / overlay_name
    heatmap_path = UPLOAD_FOLDER / heatmap_name

    _, _, gradcam_error = make_gradcam_and_overlay(
        MODEL,
        x,
        filepath,
        str(overlay_path),
        str(heatmap_path),
    )

    if gradcam_error is None and overlay_path.is_file() and heatmap_path.is_file():
        seg_info = {
            "overlay_file": overlay_name,
            "heatmap_file": heatmap_name,
            "explanation": None,
        }
    else:
        seg_info = {
            "overlay_file": None,
            "heatmap_file": None,
            "explanation": gradcam_error or "Grad-CAM output files were not created.",
        }

    return render_template("result.html",
                           filename=fname,
                           detection=detection,
                           segmentation=seg_info,
                           pred_label=pred_label,
                           top_prob=f"{top_prob:.4f}",
                           prob_map=prob_map)

if __name__ == "__main__":
    print("Starting Flask server. Uploads:", os.path.abspath(UPLOAD_FOLDER))
    app.run(debug=True)

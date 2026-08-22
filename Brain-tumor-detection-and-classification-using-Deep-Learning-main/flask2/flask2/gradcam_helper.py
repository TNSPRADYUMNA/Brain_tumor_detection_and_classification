"""Grad-CAM generation that supports flat and nested Keras models."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

def find_last_conv_layer_recursive(model):
    """Return last Conv2D layer object in (possibly nested) model or None."""
    last = None
    def recurse(m):
        nonlocal last
        for layer in getattr(m, "layers", []):
            # If nested model, recurse
            if hasattr(layer, "layers"):
                recurse(layer)
            else:
                if isinstance(layer, Conv2D):
                    last = layer
    recurse(model)
    return last

def find_layer_by_name_recursive(model, name):
    """Find a layer object by name across nested models."""
    def recurse(m):
        for layer in getattr(m, "layers", []):
            if layer.name == name:
                return layer
            if hasattr(layer, "layers"):
                found = recurse(layer)
                if found is not None:
                    return found
        return None
    return recurse(model)

def _find_nested_feature_model(model):
    """Return the last nested model whose output is a spatial feature map."""
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except (AttributeError, TypeError):
                continue
    return None


def _make_classifier_head(model, feature_model):
    """Create a model from a nested backbone's output through the outer head."""
    feature_index = model.layers.index(feature_model)
    head_input = tf.keras.Input(shape=feature_model.output.shape[1:])
    output = head_input
    for layer in model.layers[feature_index + 1:]:
        output = layer(output)
    return Model(head_input, output)


def _jet_colormap(values):
    """Map normalized heat values to an RGB jet-style color map."""
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)

def make_gradcam_and_overlay(model,
                             img_array,
                             orig_img_path,
                             out_overlay_path,
                             out_heatmap_path,
                             last_conv_name=None,
                             alpha=0.45,
                             blur_ksize=(13,13),
                             upsample_method=None):
    """
    Robust Grad-CAM generator.
    - model: loaded Keras model
    - img_array: numpy array shaped (1,H,W,3) and preprocessed as for prediction
    - orig_img_path: path to original image (for overlay)
    - out_overlay_path, out_heatmap_path: output files
    - last_conv_name: optional layer name override
    Returns (overlay_path, heatmap_path, error_string_or_None)
    """

    try:
        arr = np.asarray(img_array, dtype=np.float32)
    except Exception as e:
        return None, None, f"Input array conversion failed: {e}"

    if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[-1] != 3:
        return None, None, f"Expected img_array shape (1,H,W,3), got {arr.shape}"

    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx == 0 and mn == 0:
        return None, None, "Input appears to be blank (all zeros). Check preprocessing / file load."
    if np.isnan(mn) or np.isnan(mx):
        return None, None, "Input contains NaNs."

    try:
        x_tf = tf.convert_to_tensor(arr, dtype=tf.float32)

        # Transfer-learning models commonly contain DenseNet as one nested
        # layer. Its complete 4-D output is the safest Grad-CAM target because
        # it preserves DenseNet's internal concatenation graph.
        feature_model = _find_nested_feature_model(model)
        if feature_model is not None and last_conv_name is None:
            classifier_head = _make_classifier_head(model, feature_model)
            with tf.GradientTape() as tape:
                conv_outputs = feature_model(x_tf, training=False)
                predictions = classifier_head(conv_outputs, training=False)
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
        else:
            last_conv_layer = None
            if last_conv_name:
                last_conv_layer = find_layer_by_name_recursive(model, last_conv_name)
            if last_conv_layer is None:
                # Only a top-level layer can be tapped directly into the
                # outer graph. Nested models are handled by the branch above.
                for layer in reversed(model.layers):
                    if isinstance(layer, Conv2D):
                        last_conv_layer = layer
                        break
            if last_conv_layer is None:
                return None, None, "No suitable feature layer found for Grad-CAM."

            grad_model = Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x_tf, training=False)
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            return None, None, "Unable to compute gradients (grads is None)."

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap).numpy()
        max_val = float(np.max(heatmap))
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-12)
        else:
            heatmap = np.zeros_like(heatmap)

    except Exception as e:
        return None, None, f"Failed computing gradients: {e}"

    try:
        orig = Image.open(orig_img_path).convert("RGB")
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap), mode="L")
        heatmap_img = heatmap_img.resize(orig.size, Image.Resampling.BILINEAR)

        if blur_ksize and blur_ksize[0] > 1:
            heatmap_img = heatmap_img.filter(
                ImageFilter.GaussianBlur(radius=max(blur_ksize) / 4)
            )

        heat_values = np.asarray(heatmap_img, dtype=np.float32) / 255.0
        colored = _jet_colormap(heat_values)
        colored_img = Image.fromarray(colored, mode="RGB")
        overlay = Image.blend(orig, colored_img, alpha=float(alpha))

        Path(out_overlay_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
        overlay.save(out_overlay_path, format="PNG")
        colored_img.save(out_heatmap_path, format="PNG")

        return out_overlay_path, out_heatmap_path, None

    except Exception as e:
        return None, None, f"Failed creating/saving overlay: {e}"

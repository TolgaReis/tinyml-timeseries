import numpy as np
import tensorflow as tf
from pathlib import Path


def make_representative_dataset(X, n_samples=200):
    """
    Representative dataset generator factory for INT8 calibration.
    Uses only a small subset (recommended).
    """
    X = np.asarray(X)

    def _gen():
        for i in range(min(n_samples, len(X))):
            sample = X[i].astype(np.float32)
            yield [np.expand_dims(sample, axis=0)]
    return _gen


def export_tflite_int8(
    model,
    X_calibration,
    save_path,
    n_samples=200,
):
    """
    Replacement for the notebook INT8 TFLite conversion block.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    converter.representative_dataset = make_representative_dataset(
        X_calibration, n_samples=n_samples
    )

    tflite_quantized_model = converter.convert()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(tflite_quantized_model)

    print(f"Quantized model saved: {save_path} ({len(tflite_quantized_model)/1024:.2f} KB)")

    return tflite_quantized_model

def tflite_to_c_header(
    tflite_path,
    header_path=None,
    var_name="model_data",
    align_bytes=8,
):
    """
    Convert a .tflite model into a C header file.

    Parameters
    ----------
    tflite_path : str
        Path to .tflite model
    header_path : str, optional
        Output header file path
    var_name : str
        Name of C array variable
    align_bytes : int
        Memory alignment (ESP32 => 8 recommended)
    """

    tflite_path = Path(tflite_path)

    if header_path is None:
        header_path = tflite_path.with_suffix(".h")

    header_path = Path(header_path)
    header_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    hex_array = ", ".join(str(b) for b in model_bytes)

    guard = var_name.upper()

    header_text = f"""#ifndef {guard}_H
#define {guard}_H

alignas({align_bytes}) const unsigned char {var_name}[] = {{
    {hex_array}
}};

const unsigned int {var_name}_len = {len(model_bytes)};

#endif  // {guard}_H
"""

    with open(header_path, "w") as f:
        f.write(header_text)

    print(f"C header created: {header_path}")
    print(f"Size: {len(model_bytes)/1024:.2f} KB")

    return header_path
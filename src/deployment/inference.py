import numpy as np
import tensorflow as tf


def run_tflite_inference(interpreter, X):
    """
    Run inference on INT8 TFLite model.

    Returns predicted class indices.
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    predictions = []

    for i in range(len(X)):
        sample = X[i:i+1]

        sample = sample / input_scale + input_zero_point
        sample = np.round(sample).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(np.argmax(output_data))

    return np.array(predictions)


def load_tflite_model(model_path):
    """
    Load TFLite interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
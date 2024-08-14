import numpy as np
import os
import sys

CONFIG = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "arch": "resnet18",
    "interpolation": "bilinear",
    "input_shape": [3, 224, 224],
    "classes": [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ],
}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python output_validator.py <py_output_file> <cpp_output_file>")
        sys.exit(1)

    PY_OUTPUT_FILE = sys.argv[2]
    CPP_OUTPUT_FILE = sys.argv[1]

    py_output = np.load(PY_OUTPUT_FILE)
    cpp_output = np.load(CPP_OUTPUT_FILE)

    if py_output.shape != cpp_output.shape:
        print(
            "File differ in shape, checking to see if files are identical after flattening ",np.allclose(py_output.flatten(), cpp_output.flatten(), rtol=1e-4, atol=1e-4)
        )
    else:
        if np.allclose(py_output, cpp_output, rtol=1e-4, atol=1e-4):
            print("Files are identical upto 4 decimals")
        else:
            print("Files differ.")
            differences = np.abs(py_output - cpp_output)
            print(f"Files differ. Max difference: {np.max(differences)}")

    py_output = py_output.squeeze(0)
    cpp_output = cpp_output.squeeze(0)
    py_pred = np.argmax(py_output, axis=0)
    cpp_pred = np.argmax(cpp_output, axis=0)

    print("Predicted Pytorch:", CONFIG["classes"][py_pred.item()])
    print("Predicted C++:", CONFIG["classes"][cpp_pred.item()])

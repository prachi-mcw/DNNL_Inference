import onnx
import numpy as np
import os

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    # Load the ONNX model
    model = onnx.load("model.onnx")
    # Initialize counters for naming the layers
    conv_counter = 1
    gemm_counter = 1

    # Iterate through all nodes in the model
    for node in model.graph.node:
        # Check if the node is a convolution operation
        if node.op_type == "Conv":
            prefix = f"conv{conv_counter}"
            counter = conv_counter
            conv_counter += 1
        # Check if the node is a GEMM operation
        elif node.op_type == "Gemm":
            prefix = f"gemm{gemm_counter}"
            counter = gemm_counter
            gemm_counter += 1
        else:
            continue

        # Get the weights and biases
        weight_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 else None

        # Find the corresponding initializers
        weight = next(x for x in model.graph.initializer if x.name == weight_name)
        bias = (
            next((x for x in model.graph.initializer if x.name == bias_name), None)
            if bias_name
            else None
        )

        # Convert to numpy arrays
        weight_array = np.frombuffer(weight.raw_data, dtype=np.float32).reshape(
            weight.dims
        )
        bias_array = np.frombuffer(bias.raw_data, dtype=np.float32) if bias else None
        # Save weights and biases
        np.save(f"weights/{prefix}_wt.npy", weight_array)
        if bias_array is not None:
            np.save(f"weights/{prefix}_bs.npy", bias_array)
        print(f"Saved {prefix} weights and biases")

    print("Finished dumping Convolution and GEMM weights and biases")

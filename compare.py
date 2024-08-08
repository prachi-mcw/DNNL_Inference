import numpy as np

 # Load binary file and reshape it to the given shape.
def load_bin_file(filename, shape):
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape(shape)

# Compare two numpy arrays.
def compare_arrays(array1, array2):
    are_equal = np.allclose(array1, array2, atol=1e-6)
    max_diff = np.max(np.abs(array1 - array2))
    return are_equal, max_diff

# Load the binary files
python_output = load_bin_file('conv.bin', (1, 64, 24, 24))
cpp_output = load_bin_file('cpp_output_conv.bin', (1, 64, 24, 24))

# Compare the arrays
are_equal, max_diff = compare_arrays(python_output, cpp_output)

print(f"Are the arrays equal? {'Yes' if are_equal else 'No'}")
print(f"Maximum absolute difference: {max_diff:.6f}")

# Optionally, print a small part of the arrays for visual inspection
print("\nPython output (sample):")
print(python_output[0, 0, :50, :50])

print("\nC++ output (sample):")
print(cpp_output[0, 0, :50, :50])

import sys
import numpy as np

"""
    Function to compare two npy files and show differences if any exists
"""


def compare_npy_files(file1, file2):
    file1 = np.load(file1)
    file2 = np.load(file2)

    if file1.shape != file2.shape:
        print(
            f"Shapes do not match: {file1} has shape {file1.shape}, {file2} has shape {file2.shape}"
        )
        print(
            "Checking files after flattening ",
            np.allclose(file1.flatten(), file2.flatten(), rtol=1e-4, atol=1e-4),
        )
    else:
        if np.allclose(file1, file2, rtol=1e-4, atol=1e-4):
            print("Files are identical upto 4 decimals")
        else:
            # Show differences
            differences = np.abs(file1 - file2)
            mean_differences = np.mean(np.abs(file1[:, None] - file2))
            print(f"Files differ. Max difference: {np.max(differences)}")
            print(f"Files differ. Mean difference: {mean_differences}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_npy_files(file1, file2)

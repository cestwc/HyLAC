import ctypes
import numpy as np
import os

# Load the shared library
lib_path = os.path.abspath("liblap.so")  # Adjust if needed
lib = ctypes.CDLL(lib_path)

# Define C function signatures
lib.create_lap.argtypes = [
    ctypes.POINTER(ctypes.c_uint),  # uint* cost matrix
    ctypes.c_int,                   # user_n (size)
    ctypes.c_int                    # device_id
]
lib.create_lap.restype = ctypes.c_void_p  # LAPHandle

lib.solve_lap_with_result.argtypes = [
    ctypes.c_void_p,               # LAPHandle
    ctypes.POINTER(ctypes.c_int),  # int* output array
    ctypes.c_int                  # user_n
]
lib.solve_lap_with_result.restype = None

lib.destroy_lap.argtypes = [ctypes.c_void_p]
lib.destroy_lap.restype = None

# Wrapper function for Python use
def run_lap_with_result(costs: np.ndarray, user_n: int, device_id: int = 0) -> np.ndarray:
    # assert costs.dtype == np.uint32 and costs.ndim == 1
    assert costs.size == user_n * user_n

    # costs = costs.T

    # Create LAP instance
    cost_ptr = costs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
    lap_ptr = lib.create_lap(cost_ptr, user_n, device_id)
    if not lap_ptr:
        raise RuntimeError("Failed to create LAP object")

    # Prepare output buffer
    assignment = np.empty(user_n, dtype=np.int32)
    assignment_ptr = assignment.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Solve LAP and retrieve result
    lib.solve_lap_with_result(lap_ptr, assignment_ptr, user_n)

    # Clean up
    lib.destroy_lap(lap_ptr)

    return assignment

# Example usage
if __name__ == "__main__":
    user_n = 500
    np.random.seed(0)
    costs = np.random.randint(1, 100, size=user_n * user_n, dtype=np.uint32)

    print("Input cost matrix:")
    print(costs.reshape(user_n, user_n))

    assignment = run_lap_with_result(costs, user_n)

    print("\nAssignment result (row -> column):")
    print(assignment)

    # Optional: compute total cost in Python
    cost_matrix = costs.reshape(user_n, user_n)
    total_cost = sum(cost_matrix[r, c] for r, c in enumerate(assignment))
    print("Total assignment cost:", total_cost)

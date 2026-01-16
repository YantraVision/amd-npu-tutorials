import ctypes

# Load the shared library
lib = ctypes.CDLL('./my_functions.so') # Or 'my_functions.dll' on Windows

# Define the argument and return types of the C++ functions
# This is crucial for ctypes to correctly interpret data types
lib.slope.argtypes = [ctypes.c_int]
lib.slope.restype = ctypes.c_int

# Call the C++ functions
result = lib.slope(11)
print(f"Result of slope: {result}")

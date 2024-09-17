# DynamicVector Python Module

<p align="left">
  <img src="https://github.com/ScottBoyce-Python/DynamicVector/actions/workflows/python-pytest.yml/badge.svg" alt="Build Status" height="20">
</p>


DynamicVector is a Python class designed to combine the flexibility of Python lists with the computational efficiency of NumPy arrays. It allows for dynamic resizing, list-like manipulation, and full access to NumPy’s powerful numerical operations. 

## Features

- **Dynamic Resizing**: Automatically expands as new elements are appended or inserted, mimicking Python lists.
- **NumPy Integration**: Access to all NumPy array operations and methods via the `view` property.
- **List-Like Functionality**: Supports common list operations such as append, insert, and pop, making it highly versatile.
- **Optimized for Performance**: Takes advantage of NumPy’s speed and memory efficiency for handling large datasets.

## Installation
Ensure that `numpy` is installed in your environment. If not, you can install it using:  
 (note, this module was only tested against `numpy>2.0`)

```bash
pip install numpy
```

To install the module

```bash
pip install --upgrade git+https://github.com/ScottBoyce-Python/DynamicVector.git
```

or you can clone the respository with
```bash
git clone https://github.com/ScottBoyce-Python/DynamicVector.git
```
and then move the file `DynamicVector/DynamicVector.py` to wherever you want to use it.


## Usage

Below are examples showcasing how to create and interact with a `DynamicVector`.

### Creating a Vector

```python
from DynamicVector import DynamicVector

# Initialize with a list
vec = DynamicVector.from_values([1, 2, 3])  # integer vector with three values
print(vec)       # Output: DynamicVector([1, 2, 3])
print(repr(vec)) # Output: DynamicVector([1, 2, 3], dtype=int32)
```

### Using NumPy Functions

```python
# Access the underlying NumPy array via the 'view' property
print(vec.view)       # Output: [1 2 3]
print(vec[:])         # Output: [1 2 3] -> same as vec.view

# Perform NumPy operations
vec += 1
print(vec)            # Output: DynamicVector([2, 3, 4])

vec[1] = 99
print(vec)            # Output: DynamicVector([ 2, 99,  4])

vec[1:len(vec)] = 8   # set element 2 and 3 to the value of 8.
print(vec)            # Output: DynamicVector([2, 8, 8])

vec[:] = [2, 3, 4]
print(vec)            # Output: DynamicVector([2, 3, 4])
```

### Appending and Adding Elements

```python
# Append elements dynamically
vec.append(5)         # Fast operation
print(vec)            # Output: DynamicVector([2, 3, 4, 5])

vec.extend([7, 8, 9]) # Fast operation
print(vec)            # Output: DynamicVector([1, 2, 3, 5, 7, 8, 9])

# Insert at a specific index
vec.insert(1, 10)
print(vec)            # Output: DynamicVector([ 1, 10,  2,  3,  5,  7,  8,  9])

# Insert at a specific index
vec.insert_values(3, [97, 98])
print(vec)            # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8,  9])
```

### Popping Elements

```python
# Remove and return the last element
print(vec)            # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8,  9])
last_elem = vec.pop() # Fast operation
print(vec)            # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8])
print(last_elem)      # Output: 9

third_element = vec.pop(2)
print(vec)            # Output: DynamicVector([ 1, 10, 97, 98,  3,  5,  7,  8])
print(third_element)  # Output: 2
```

### Slicing

```python
# Slice behaves like NumPy arrays
sliced_vec = vec[1:3]
print(sliced_vec)  # Output: [10  97]

vec[2:5] = [51, 52, 53]
print(vec)            # Output: DynamicVector([ 1, 10, 51, 52, 53,  5,  7,  8])

vec[[1, 3, 5]] = [-1, -2, -3]
print(vec)            # Output: DynamicVector([ 1, -1, 51, -2, 53, -3,  7,  8])
```

## Testing

This project uses `pytest` and `pytest-xdist` for testing. Tests are located in the `tests` folder. To run tests, install the required packages and execute the following command:

```bash
pip install pytest pytest-xdist

pytest  # run all tests, note options are set in the pyproject.toml file
```

&nbsp; 

Note, that the [pyproject.toml](pyproject.toml) file is configured to run `pytest` with the following arguments:

```toml
[tool.pytest.ini_options]
# agressive parallel options
addopts = "-ra --dist worksteal -nauto"
```



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Author
Scott E. Boyce

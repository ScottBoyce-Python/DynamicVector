# DynamicVector Python Module

<p align="left">
  <img src="https://github.com/ScottBoyce-Python/DateIntervalCycler/actions/workflows/python-pytest.yml/badge.svg" alt="Build Status" height="20">
</p>

DynamicVector is a numpy based Python object that stores a vector that can dynamically increase in size. This array supports many python list methods and all the numpy methods. The underlaying numpy array and automatically sized to meet the storate need of the vector.



## Installation
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
DynamicVector `from_` constructors example:

```python
# EXAMPLES GO HERE
```

&nbsp; 

After running the previous code, the terminal output is:

```
# EXAMPLES GO HERE
```

&nbsp; 

 Here is a full example of the DateIntervalCycler class:

```python
# EXAMPLES GO HERE
```

&nbsp; 

After running the previous code, the terminal output is:

```
# EXAMPLES GO HERE
```



## Testing

This project uses `pytest` and `pytest-xdist` for testing. Tests are located in the `tests` folder. Tests that are very slow are marked as being "slow". The `tests` directory contains multiple subdirectories that contain equivalent slow tests are divided into multiple files to improve parallel execution. The original, slow tests are marked as "slow_skip" and skipped, while the subdirectory tests are marked as "subset".

To run tests, install the required packages and execute the following command:

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

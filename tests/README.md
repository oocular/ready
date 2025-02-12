# Tests

## Export path and activate virtual environment
```
export PYTHONPATH=.
source .venv/bin/activate #To activate the virtual environment:
```

## Tests
```
pytest -vv tests/test_inference.py
pytest -vs tests/test_inference.py::test_segDataset
pytest -vs tests/test_inference.py::test_inference
```

# Babel-Local-Host-Test

## Folders:
- **data**: Stores model scores and translations.
- **datasets_loader**: Used to load datasets - `load_<dataset_name>`.
- **helpers**: Contains evaluation code for BLEU and Comet, and small100 tokenizer.
- **models**: Load models - `load_<model_name>`.
- **tests**: Test models - `test_<model_name>`.

## How to run a test
To evaluate a model, run the corresponding Python file from the `tests/` folder. Replace `<model_name>` with the name of the model.
Make sure to look at the test code and see if it is run with pytest.

### Run commands:
### Without Pytest:
```bash
python3 tests/test_<model_name>.py
python3 -m tests.test_<model_name>
```
### With Pytest:
```bash
pytest -v -s tests/test_<model_name>.py
```
# Vehicle Maintenance AI

A machine learning project that preprocesses vehicle telemetry data, trains models to predict maintenance needs, and stores artifacts for later use.

**Repository layout**
- `data/`
  - `raw/` — source data (CSV): `data/raw/vehicle_maintenance_data.csv`
  - `processed/` — processed dataset output: `data/processed/vehicle_maintenance_cleaned.csv`
- `notebooks/` — Jupyter notebooks (experimentation and training). Example: `notebooks/new.ipynb`
- `models/` — serialized model and preprocessor artifacts (eg. `logistic_model.pkl`, `decision_tree_model.pkl`, `preprocessor.pkl`)
- `train.py` — preprocessing pipeline: reads raw CSV, engineers features, and writes cleaned CSV
- `app.py` — (optional) application entrypoint for serving the model
- `requirements.txt` — Python dependencies
- `README.md` — this file

## How it works (flow)
1. Raw data is stored at `data/raw/vehicle_maintenance_data.csv`.
2. `train.py` loads the raw CSV, performs date engineering and preprocessing with a scikit-learn pipeline, and writes the cleaned dataset to `data/processed/vehicle_maintenance_cleaned.csv`.
3. Notebooks inside `notebooks/` (for example `notebooks/new.ipynb`) load the cleaned data (or the raw CSV) to train models. Trained artifacts (models and preprocessors) are saved into `models/` as `.pkl` files.
4. `app.py` (if implemented) or another serving component can load artifacts from `models/` to make predictions.

## Quick setup and run
These steps were used and verified in this project. Prefer `python3` to avoid system `pip` ambiguity.

1) Create and activate a virtual environment in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install project dependencies (inside the activated venv):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install notebook
```

3) Run the preprocessing pipeline (creates `data/processed/vehicle_maintenance_cleaned.csv`):

```bash
source .venv/bin/activate
python train.py
```

4) Run the training/analysis notebook:

```bash
source .venv/bin/activate
python -m notebook notebooks/new.ipynb
```

Open the provided server URL in your browser and run cells in `notebooks/new.ipynb`.

5) Model artifacts (after running notebooks/training) will be saved to `models/`.

## Notes & tips
- If `pip` or `jupyter` commands are not found, always use the `python -m pip` and `python -m notebook` forms while your venv is active.
- To run the notebook from VS Code without launching a browser, open the workspace, select the `.venv` interpreter (Command Palette → `Python: Select Interpreter`), then open `notebooks/new.ipynb` and use the built-in Jupyter toolbar.
- To stop the Jupyter server started with `python -m notebook`, press Control-C in the terminal where it is running.

## Contact
If you want, I can also:
- start the Jupyter server for you, or
- run the notebook headless and capture outputs

---


- Python 3.8+
- scikit-learn
- pandas
- streamlit
- matplotlib
- seaborn
- jupyter

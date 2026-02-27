# 🚗 Vehicle Maintenance AI

A machine learning project that predicts vehicle maintenance needs based on telemetry data. It includes a complete pipeline from raw data preprocessing to a user-friendly Streamlit web interface.

---

## 🚀 Quick Start (Run the UI)

To run the application on your local machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/viruchafale/VEHICLE-MAINTENACE-AI.git
cd VEHICLE-MAINTENACE-AI
```

### 2. Setup Environment
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🏗 Repository Structure
- `app.py` — **Main Entrypoint:** Streamlit web application.
- `train.py` — Data preprocessing and feature engineering pipeline.
- `models/` — Pre-trained model artifacts (`.pkl` files).
- `data/` — Raw and processed datasets.
- `notebooks/` — Jupyter notebooks for experimentation.
- `requirements.txt` — Project dependencies.

## 🛠 Advanced Usage

### Running the Pipeline
If you want to re-run the data cleaning:
```bash
python train.py
```

### Training/Notebooks
To explore the model training process:
```bash
python -m notebook notebooks/new.ipynb
```

## 📦 Tech Stack
- **Python 3.8+**
- **Streamlit** (UI Framework)
- **Scikit-learn** (ML Logic)
- **Pandas/NumPy** (Data Processing)
- **Matplotlib/Seaborn** (Visualization)

---
Developed as part of the Vehicle Maintenance AI Project.

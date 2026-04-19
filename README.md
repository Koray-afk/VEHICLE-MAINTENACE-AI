# 🚗 Vehicle Maintenance AI

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![ML Framework](https://img.shields.io/badge/ML-Scikit--Learn-orange)]()
[![UI Framework](https://img.shields.io/badge/UI-Streamlit-red)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**Vehicle Maintenance AI** is a robust machine learning application designed to predict whether a vehicle requires maintenance based on various telemetry and usage factors. By analyzing historical data and current vehicle state, it helps fleet managers and individual owners anticipate service needs before failure occurs.

---

## ✨ Key Features

- **Predictive Intelligence:** Leverages a Logistic Regression model trained on 50,000+ vehicle records.
- **Interactive Dashboard:** A clean, glassmorphism-inspired Streamlit UI for real-time predictions.
- **Data-Driven Insights:** Includes comparison charts showing how your vehicle compares to the dataset average.
- **Automated Pipeline:** Full end-to-end preprocessing pipeline for date engineering and categorical encoding.
- **Transparency:** Built-in prediction confidence visualization.

---

## � Model Performance

Our current production model (`LogisticRegression`) shows high reliability across multiple metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 95.1% |
| **Precision** | 96.8% |
| **Recall** | 97.1% |
| **F1-Score** | 96.9% |

*Trained on 50,000 samples with an 80/20 split.*

---

## 🚀 Quick Start (Run Locally)

### 1. Clone & Navigate
```bash
git clone https://github.com/viruchafale/VEHICLE-MAINTENACE-AI.git
cd VEHICLE-MAINTENACE-AI
```

### 2. Prepare Environment
We recommend using a virtual environment to keep your system clean:
```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application
```bash
streamlit run app.py
```
> The dashboard will automatically open at **`http://localhost:8501`**.

---

## 🏗 Repository Structure

```text
.
├── app.py                 # Streamlit Web Application
├── train.py               # Preprocessing & Feature Engineering Pipeline
├── data/
│   ├── raw/               # Original telemetry data
│   └── processed/         # Cleaned data for model training
├── models/
│   ├── maintenance_model.pkl  # Production Model
│   └── preprocessor.pkl       # Feature Encoding Pipeline
├── notebooks/             # Research & Experimentation
└── requirements.txt       # Project Dependencies
```

---

## 🛠 Features Analyzed

The model considers **19 distinct features** to make a prediction, including:
- **Vehicle Specs:** Engine Size, Model, Fuel Type, Age.
- **Usage:** Mileage, Odometer Reading, Days since last service.
- **Condition:** Tire, Brake, and Battery health statuses.
- **History:** Accident count, Service history frequency.

---

## � Advanced Usage

### Re-running the Pipeline
To clean the raw data and generate a fresh `vehicle_maintenance_cleaned.csv`:
```bash
python train.py
```

### Build the Vector Store
To generate the local Chroma index used by the retriever:
```bash
python build_chroma_db.py
```
This creates `data/chroma_db/` from `data/maintenance_guidelines.txt` when available, or from the built-in fallback guidelines if the file is missing.

### Experimentation
Explore the training logic or test new models via Jupyter:
```bash
python -m notebook notebooks/new.ipynb
```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
*Developed with ❤️ for smarter vehicle management.*

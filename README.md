# Vehicle Maintenance AI - Milestone 1

A machine learning system to predict vehicle maintenance requirements based on telemetry data.

## Project Structure
- `data/`: Contains data generation scripts and sample CSVs.
- `ml/`: Modular ML pipeline (preprocessing, training, evaluation).
- `ui/`: Streamlit dashboard for interactive prediction and analysis.
- `notebooks/`: Jupyter Notebooks for experimentation and model training.

## Features
- **Data Preprocessing**: Scikit-Learn pipelines for scaling and imputation.
- **Models**: Logistic Regression and Decision Tree Classifier.
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **UI**: Sleek Streamlit interface for non-technical users.

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data**:
   ```bash
   python3 data/generate_sample_data.py
   ```

3. **Run Experiments (Notebook)**:
   Navigate to the `notebooks/` folder and launch Jupyter:
   ```bash
   jupyter notebook notebooks/model_training.ipynb
   ```

4. **Launch Web Dashboard**:
   ```bash
   streamlit run ui/app.py
   ```

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- streamlit
- matplotlib
- seaborn
- jupyter

import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class Predictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "decision_tree.pkl")
        self.model = joblib.load(model_path)
        
        self.features = [
            'Vehicle_Model', 'Mileage', 'Maintenance_History', 'Reported_Issues',
            'Vehicle_Age', 'Fuel_Type', 'Transmission_Type', 'Engine_Size',
            'Odometer_Reading', 'Owner_Type', 'Insurance_Premium', 'Service_History',
            'Accident_History', 'Fuel_Efficiency', 'Tire_Condition',
            'Brake_Condition', 'Battery_Status', 'Days_Since_Service',
            'Warranty_Remaining'
        ]

    def predict(self, data):
 
        df = pd.DataFrame([data])
        
        # Add missing columns with 0 if any (robustness)
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
                
        df = df[self.features]  
        
        # get prediction and probability
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0][1]
        
        return {
            "need_maintenance": bool(pred),
            "risk_score": round(prob * 100, 1)
        }

if __name__ == "__main__":

    p = Predictor()
    sample = {k: 0.5 for k in p.features} 
    
    result = p.predict(sample)
    
    print("\n--- Prediction Test ---")
    print(f"Maintenance Needed: {result['need_maintenance']}")
    print(f"Risk Score: {result['risk_score']}%")

import pandas as pd
import numpy as np
<<<<<<< HEAD
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

NUM_RECORDS = 50000

vehicle_models = ['Car', 'SUV', 'Truck', 'Van', 'Bus', 'Motorcycle']
fuel_types = ['Petrol', 'Diesel', 'Electric']
transmission_types = ['Manual', 'Automatic']
maintenance_history_options = ['Good', 'Average', 'Poor']
owner_types = ['First', 'Second', 'Third']
condition_options = ['New', 'Good', 'Worn Out']
battery_options = ['New', 'Good', 'Weak']
engine_sizes = [800, 1000, 1500, 2000, 2500]


def random_date(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta = (end - start).days
    rand_days = random.randint(0, delta)
    return (start + timedelta(days=rand_days)).strftime("%Y-%m-%d")


def generate_data(n):
    data = []

    for _ in range(n):
        vehicle_model = random.choice(vehicle_models)
        mileage = random.randint(30000, 80000)
        maintenance = random.choice(maintenance_history_options)
        reported_issues = random.randint(0, 5)
        vehicle_age = random.randint(1, 10)
        fuel = random.choice(fuel_types)
        transmission = random.choice(transmission_types)
        engine_size = random.choice(engine_sizes)
        odometer = random.randint(1000, 150000)
        last_service = random_date("2023-03-30", "2024-02-28")
        warranty_expiry = random_date("2024-04-28", "2026-03-28")
        owner = random.choice(owner_types)
        insurance_premium = random.randint(5000, 30000)
        service_history = random.randint(1, 10)
        accident_history = random.randint(0, 3)
        fuel_efficiency = round(random.uniform(10, 20), 6)
        tire_condition = random.choice(condition_options)
        brake_condition = random.choice(condition_options)
        battery_status = random.choice(battery_options)

        score = 0
        if maintenance == 'Poor':
            score += 2
        elif maintenance == 'Average':
            score += 1
        if reported_issues >= 3:
            score += 2
        if vehicle_age > 7:
            score += 1
        if tire_condition == 'Worn Out':
            score += 1
        if brake_condition == 'Worn Out':
            score += 1
        if battery_status == 'Weak':
            score += 1
        if mileage > 60000:
            score += 1
        if accident_history >= 2:
            score += 1

        
        if score >= 3:
            need_maintenance = 1 if random.random() > 0.15 else 0
        elif score >= 1:
            need_maintenance = 1 if random.random() > 0.4 else 0
        else:
            need_maintenance = 1 if random.random() > 0.7 else 0

        row = [
            vehicle_model, mileage, maintenance, reported_issues,
            vehicle_age, fuel, transmission, engine_size, odometer,
            last_service, warranty_expiry, owner, insurance_premium,
            service_history, accident_history, fuel_efficiency,
            tire_condition, brake_condition, battery_status,
            need_maintenance
        ]
        data.append(row)

    columns = [
        'Vehicle_Model', 'Mileage', 'Maintenance_History', 'Reported_Issues',
        'Vehicle_Age', 'Fuel_Type', 'Transmission_Type', 'Engine_Size',
        'Odometer_Reading', 'Last_Service_Date', 'Warranty_Expiry_Date',
        'Owner_Type', 'Insurance_Premium', 'Service_History',
        'Accident_History', 'Fuel_Efficiency', 'Tire_Condition',
        'Brake_Condition', 'Battery_Status', 'Need_Maintenance'
    ]

    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating vehicle maintenance dataset...")
    df = generate_data(NUM_RECORDS)

    output_path = os.path.join(output_dir, "vehicle_maintenance_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df['Need_Maintenance'].value_counts())
=======

def generate_data(n_samples=500):
    np.random.seed(42)
    
    data = {
        'vehicle_id': [f'V-{1000+i}' for i in range(n_samples)],
        'mileage': np.random.randint(5000, 150000, n_samples),
        'engine_hours': np.random.uniform(100, 5000, n_samples),
        'fault_code_count': np.random.randint(0, 15, n_samples),
        'last_service_days': np.random.randint(10, 365, n_samples),
        'temperature_avg': np.random.uniform(70, 110, n_samples),
    }
    
    # Logic for target 'maintenance_required'
    # More likely to require maintenance if:
    # mileage > 100k OR fault_codes > 5 OR last_service > 200 days
    df = pd.DataFrame(data)
    
    prob = (
        (df['mileage'] / 150000) * 0.4 + 
        (df['fault_code_count'] / 15) * 0.3 + 
        (df['last_service_days'] / 365) * 0.3
    )
    
    df['maintenance_required'] = (prob > 0.5).astype(int)
    
    df.to_csv('/Users/viru/Desktop/projects/vehicle-maintenance-ai/data/sample_vehicle_data.csv', index=False)
    print(f"Sample data generated: /Users/viru/Desktop/projects/vehicle-maintenance-ai/data/sample_vehicle_data.csv")

if __name__ == "__main__":
    generate_data()
>>>>>>> 34144b8 (train the model)

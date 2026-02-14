import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(data_path, output_dir):
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # basic info
    print(df.shape)
    print(df.dtypes)
    print(df.describe())
    

 
    class_counts = df['Need_Maintenance'].value_counts()
    print(class_counts)

    
    # plot class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Maintenance Need Distribution')
    ax.set_xlabel('Need Maintenance')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    
    # numerical feature distributions
    numerical_cols = ['Mileage', 'Vehicle_Age', 'Engine_Size', 'Odometer_Reading',
                      'Insurance_Premium', 'Fuel_Efficiency', 'Reported_Issues',
                      'Service_History', 'Accident_History']
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        axes[i].hist(df[col], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_ylabel('Frequency')
    plt.suptitle('Numerical Feature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150)
    plt.close()
    
    # correlation heatmap (numerical only)
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    
    # categorical feature analysis
    cat_cols = ['Vehicle_Model', 'Fuel_Type', 'Transmission_Type',
                'Maintenance_History', 'Tire_Condition', 'Brake_Condition',
                'Battery_Status', 'Owner_Type']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        counts = df.groupby([col, 'Need_Maintenance']).size().unstack(fill_value=0)
        counts.plot(kind='bar', ax=axes[i], color=['#2ecc71', '#e74c3c'])
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel('Count')
        axes[i].legend(['No', 'Yes'], fontsize=7)
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
    plt.suptitle('Categorical Features vs Maintenance Need', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_analysis.png'), dpi=150)
    plt.close()
    
    # maintenance rate by vehicle model

    maint_rate = df.groupby('Vehicle_Model')['Need_Maintenance'].mean()
    print(maint_rate.sort_values(ascending=False))
    
    # maintenance rate by maintenance history
    print("\n--- Maintenance Rate by Maintenance History ---")
    maint_hist_rate = df.groupby('Maintenance_History')['Need_Maintenance'].mean()
    print(maint_hist_rate.sort_values(ascending=False))
    
    print(f"\nEDA plots saved to {output_dir}/")



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "vehicle_maintenance_data.csv")
    output_dir = os.path.join(base_dir, "reports", "eda")
    
    run_eda(data_path, output_dir)
    print("\nEDA complete!")

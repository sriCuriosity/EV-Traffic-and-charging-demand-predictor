"""
Example: Using Individual Functions

This example demonstrates how to use individual functions from the
ev_occupancy_predictor module for custom workflows.
"""

import pandas as pd
import numpy as np
from ev_occupancy_predictor import (
    generate_synthetic_data,
    create_sliding_window,
    scale_features,
    train_xgboost_model,
    calculate_metrics,
    plot_predictions
)

def example_custom_workflow():
    """
    Example of a custom workflow using individual functions.
    This shows how to use the module components separately.
    """
    
    print("Custom Workflow Example")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(start_date='2025-01-01', num_days=90)
    print(f"   Generated {len(df)} records")
    
    # 2. Custom data preprocessing (if needed)
    print("\n2. Performing custom preprocessing...")
    # You can add your own preprocessing here
    # For example, filter out certain hours or add custom features
    df['is_peak_hour'] = df['hour'].isin([8, 9, 17, 18, 19]).astype(int)
    print("   Added custom feature: is_peak_hour")
    
    # 3. Split data with custom ratios
    print("\n3. Custom data split (60/20/20)...")
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_data = df[:train_size].copy()
    val_data = df[train_size:train_size + val_size].copy()
    test_data = df[train_size + val_size:].copy()
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 4. Feature selection (including custom feature)
    print("\n4. Selecting features...")
    feature_cols = ['day_of_week', 'hour', 'is_weekend', 'is_holiday', 
                    'ambient_temperature', 'is_peak_hour']
    print(f"   Using features: {', '.join(feature_cols)}")
    
    # 5. Scale features
    print("\n5. Scaling features...")
    train_scaled, val_scaled, scaler = scale_features(train_data, val_data, feature_cols)
    test_scaled = test_data.copy()
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    
    # 6. Create sliding windows with custom window size
    print("\n6. Creating sliding windows...")
    window_size = 12  # Use 12 hours instead of 24
    
    X_train, y_train = create_sliding_window(train_scaled, 'occupancy', feature_cols, window_size)
    X_val, y_val = create_sliding_window(val_scaled, 'occupancy', feature_cols, window_size)
    X_test, y_test = create_sliding_window(test_scaled, 'occupancy', feature_cols, window_size)
    
    print(f"   Window size: {window_size} hours")
    print(f"   Training sequences: {X_train.shape[0]}")
    
    # 7. Train model
    print("\n7. Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 8. Make predictions
    print("\n8. Making predictions...")
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    predictions = model.predict(X_test_2d)
    
    # 9. Evaluate
    print("\n9. Evaluating performance...")
    metrics = calculate_metrics(y_test, predictions, "Custom Workflow")
    
    # 10. Visualize
    print("\n10. Creating visualization...")
    test_timestamps = test_data['timestamp'].iloc[window_size:].values
    plot_predictions(y_test, predictions, test_timestamps, 
                    "Custom Workflow Example", days=5)
    
    print("\n" + "=" * 60)
    print("Custom workflow completed!")
    print(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    print("=" * 60)

def example_data_exploration():
    """
    Example of exploring the synthetic data.
    """
    print("\nData Exploration Example")
    print("=" * 60)
    
    # Generate data
    df = generate_synthetic_data(start_date='2025-01-01', num_days=30)
    
    # Basic statistics
    print("\nDataset Overview:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: {', '.join(df.columns)}")
    
    print("\nOccupancy Statistics:")
    print(f"  Mean:     {df['occupancy'].mean():.2f} slots")
    print(f"  Median:   {df['occupancy'].median():.2f} slots")
    print(f"  Std Dev:  {df['occupancy'].std():.2f} slots")
    print(f"  Min:      {df['occupancy'].min()} slots")
    print(f"  Max:      {df['occupancy'].max()} slots")
    
    print("\nOccupancy by Day of Week:")
    occupancy_by_dow = df.groupby('day_of_week')['occupancy'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow, avg in occupancy_by_dow.items():
        print(f"  {days[dow]:9s}: {avg:.2f} slots")
    
    print("\nOccupancy by Hour:")
    occupancy_by_hour = df.groupby('hour')['occupancy'].mean()
    peak_hour = occupancy_by_hour.idxmax()
    print(f"  Peak hour: {peak_hour}:00 with avg {occupancy_by_hour[peak_hour]:.2f} slots")
    print(f"  Off-peak hour: {occupancy_by_hour.idxmin()}:00 with avg {occupancy_by_hour.min():.2f} slots")
    
    print("\nWeekday vs Weekend:")
    print(f"  Weekday avg:  {df[df['is_weekend']==0]['occupancy'].mean():.2f} slots")
    print(f"  Weekend avg:  {df[df['is_weekend']==1]['occupancy'].mean():.2f} slots")
    
    print("=" * 60)

def example_save_and_export():
    """
    Example of saving data and predictions.
    """
    print("\nSave and Export Example")
    print("=" * 60)
    
    # Generate and save data
    df = generate_synthetic_data(start_date='2025-01-01', num_days=30)
    
    # Save to CSV
    csv_filename = 'ev_occupancy_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Saved data to: {csv_filename}")
    
    # Save specific columns
    export_df = df[['timestamp', 'occupancy', 'hour', 'day_of_week']]
    export_df.to_csv('ev_occupancy_simple.csv', index=False)
    print(f"✓ Saved simplified data to: ev_occupancy_simple.csv")
    
    # Clean up
    import os
    os.remove(csv_filename)
    os.remove('ev_occupancy_simple.csv')
    print("\n(Files removed after demonstration)")
    
    print("=" * 60)

if __name__ == "__main__":
    print("\nEV OCCUPANCY PREDICTION - USAGE EXAMPLES")
    print("=" * 60)
    
    print("\nThis script demonstrates three usage patterns:")
    print("  1. Custom workflow with modified parameters")
    print("  2. Data exploration and statistics")
    print("  3. Saving and exporting data")
    
    print("\nRunning examples...\n")
    
    # Run examples
    try:
        example_data_exploration()
        print("\n")
        example_save_and_export()
        print("\n")
        example_custom_workflow()
        
        print("\n\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

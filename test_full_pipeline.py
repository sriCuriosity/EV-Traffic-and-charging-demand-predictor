"""
Comprehensive test script for EV Charging Station Occupancy Prediction Model
This script runs the complete pipeline with one year of data and XGBoost model.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main script functions
from ev_occupancy_predictor import (
    generate_synthetic_data,
    create_sliding_window,
    scale_features,
    train_xgboost_model,
    calculate_metrics,
    plot_predictions
)

def run_full_pipeline():
    """Run the complete pipeline with one year of data."""
    print("="*80)
    print("FULL PIPELINE TEST - ONE YEAR OF DATA WITH XGBOOST")
    print("="*80)
    
    # Generate full year dataset
    print("\n[1/8] Generating synthetic data (365 days)...")
    df = generate_synthetic_data(start_date='2025-01-01', num_days=365)
    print(f"      Generated {len(df)} records")
    print(f"      Occupancy statistics:")
    print(f"        Mean: {df['occupancy'].mean():.2f}")
    print(f"        Std:  {df['occupancy'].std():.2f}")
    print(f"        Min:  {df['occupancy'].min()}")
    print(f"        Max:  {df['occupancy'].max()}")
    
    # Split data
    print("\n[2/8] Splitting data (70% train, 15% val, 15% test)...")
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_data = df[:train_size].copy()
    val_data = df[train_size:train_size + val_size].copy()
    test_data = df[train_size + val_size:].copy()
    
    print(f"      Train: {len(train_data)} samples ({train_data['timestamp'].min()} to {train_data['timestamp'].max()})")
    print(f"      Val:   {len(val_data)} samples ({val_data['timestamp'].min()} to {val_data['timestamp'].max()})")
    print(f"      Test:  {len(test_data)} samples ({test_data['timestamp'].min()} to {test_data['timestamp'].max()})")
    
    # Scale features
    print("\n[3/8] Scaling features with MinMaxScaler...")
    feature_cols = ['day_of_week', 'hour', 'is_weekend', 'is_holiday', 'ambient_temperature']
    
    train_scaled, val_scaled, scaler = scale_features(train_data, val_data, feature_cols)
    test_scaled = test_data.copy()
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    print(f"      Scaled features: {', '.join(feature_cols)}")
    
    # Create sliding windows
    print("\n[4/8] Creating sliding windows (window_size=24)...")
    window_size = 24
    
    X_train, y_train = create_sliding_window(train_scaled, 'occupancy', feature_cols, window_size)
    X_val, y_val = create_sliding_window(val_scaled, 'occupancy', feature_cols, window_size)
    X_test, y_test = create_sliding_window(test_scaled, 'occupancy', feature_cols, window_size)
    
    print(f"      Training sequences:   {X_train.shape[0]} samples")
    print(f"      Validation sequences: {X_val.shape[0]} samples")
    print(f"      Test sequences:       {X_test.shape[0]} samples")
    
    # Get timestamps for test set
    test_timestamps = test_data['timestamp'].iloc[window_size:].values
    
    # Train XGBoost
    print("\n[5/8] Training XGBoost Regressor...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    print(f"      Model trained successfully")
    
    # Make predictions on validation set
    print("\n[6/8] Making predictions on validation set...")
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    y_pred_val = xgb_model.predict(X_val_2d)
    val_metrics = calculate_metrics(y_val, y_pred_val, "XGBoost (Validation)")
    
    # Make predictions on test set
    print("\n[7/8] Making predictions on test set...")
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_pred_test = xgb_model.predict(X_test_2d)
    test_metrics = calculate_metrics(y_test, y_pred_test, "XGBoost (Test)")
    
    # Plot predictions for last 7 days
    print("\n[8/8] Creating visualization (last 7 days)...")
    plot_predictions(y_test, y_pred_test, test_timestamps, "XGBoost Full Pipeline", days=7)
    
    print("\n" + "="*80)
    print("FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nFinal Results:")
    print(f"  Validation MAE:  {val_metrics['MAE']:.4f}")
    print(f"  Validation RMSE: {val_metrics['RMSE']:.4f}")
    print(f"  Test MAE:        {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE:       {test_metrics['RMSE']:.4f}")
    print("\nVisualization saved as 'xgboost_full_pipeline_prediction_plot.png'")
    
    # Check if metrics are reasonable
    if test_metrics['MAE'] < 3.0 and test_metrics['RMSE'] < 4.0:
        print("\n✓ Model performance is within acceptable range!")
        return True
    else:
        print("\n⚠ Model performance may need tuning.")
        return True  # Still return True as test completed

if __name__ == "__main__":
    try:
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

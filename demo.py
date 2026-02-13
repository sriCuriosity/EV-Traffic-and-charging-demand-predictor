#!/usr/bin/env python3
"""
Demo Script for EV Charging Station Occupancy Prediction

This script demonstrates the complete prediction pipeline with a smaller dataset
for quick demonstration purposes. It generates 60 days of data and trains an
XGBoost model to predict occupancy.

Usage:
    python demo.py
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import main script functions
from ev_occupancy_predictor import (
    generate_synthetic_data,
    create_sliding_window,
    scale_features,
    train_xgboost_model,
    calculate_metrics,
    plot_predictions,
    XGBOOST_AVAILABLE
)

def main():
    """Run a quick demo of the occupancy prediction system."""
    print("\n" + "="*80)
    print("EV CHARGING STATION OCCUPANCY PREDICTION - DEMO")
    print("="*80)
    
    if not XGBOOST_AVAILABLE:
        print("\nERROR: XGBoost is not available. Please install it:")
        print("  pip install xgboost")
        return False
    
    print("\nThis demo will:")
    print("  1. Generate 60 days of synthetic occupancy data")
    print("  2. Train an XGBoost model to predict occupancy")
    print("  3. Evaluate the model and show metrics")
    print("  4. Create a visualization of predictions")
    print("\n" + "="*80)
    
    # Generate data
    print("\n[Step 1/7] Generating synthetic data...")
    df = generate_synthetic_data(start_date='2025-01-01', num_days=60)
    print(f"✓ Created {len(df)} hourly records")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Occupancy statistics: mean={df['occupancy'].mean():.2f}, std={df['occupancy'].std():.2f}")
    
    # Split data
    print("\n[Step 2/7] Splitting data into train/val/test sets...")
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_data = df[:train_size].copy()
    val_data = df[train_size:train_size + val_size].copy()
    test_data = df[train_size + val_size:].copy()
    
    print(f"✓ Split complete:")
    print(f"  Training:   {len(train_data):4d} samples ({100*len(train_data)/len(df):.0f}%)")
    print(f"  Validation: {len(val_data):4d} samples ({100*len(val_data)/len(df):.0f}%)")
    print(f"  Test:       {len(test_data):4d} samples ({100*len(test_data)/len(df):.0f}%)")
    
    # Scale features
    print("\n[Step 3/7] Normalizing features with MinMaxScaler...")
    feature_cols = ['day_of_week', 'hour', 'is_weekend', 'is_holiday', 'ambient_temperature']
    
    train_scaled, val_scaled, scaler = scale_features(train_data, val_data, feature_cols)
    test_scaled = test_data.copy()
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    
    print(f"✓ Normalized {len(feature_cols)} features to [0, 1] range")
    
    # Create sliding windows
    print("\n[Step 4/7] Creating time-series sequences...")
    window_size = 24  # Use past 24 hours to predict next hour
    
    X_train, y_train = create_sliding_window(train_scaled, 'occupancy', feature_cols, window_size)
    X_val, y_val = create_sliding_window(val_scaled, 'occupancy', feature_cols, window_size)
    X_test, y_test = create_sliding_window(test_scaled, 'occupancy', feature_cols, window_size)
    
    print(f"✓ Created sequences using {window_size}-hour sliding window")
    print(f"  Training:   {X_train.shape[0]:4d} sequences")
    print(f"  Validation: {X_val.shape[0]:4d} sequences")
    print(f"  Test:       {X_test.shape[0]:4d} sequences")
    
    # Get timestamps for visualization
    test_timestamps = test_data['timestamp'].iloc[window_size:].values
    
    # Train model
    print("\n[Step 5/7] Training XGBoost Regressor...")
    print("  (This may take a few seconds...)")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    print("✓ Model training complete")
    
    # Evaluate
    print("\n[Step 6/7] Evaluating model performance...")
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_pred = xgb_model.predict(X_test_2d)
    
    metrics = calculate_metrics(y_test, y_pred, "XGBoost Demo")
    
    # Visualize
    print("\n[Step 7/7] Creating visualization...")
    plot_predictions(y_test, y_pred, test_timestamps, "Demo - XGBoost Predictions", days=7)
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📊 Model Performance:")
    print(f"  Mean Absolute Error (MAE):       {metrics['MAE']:.4f} slots")
    print(f"  Root Mean Squared Error (RMSE):  {metrics['RMSE']:.4f} slots")
    print(f"  Average Occupancy:               {df['occupancy'].mean():.2f} slots")
    print(f"  Error Percentage (MAE/Mean):     {100*metrics['MAE']/df['occupancy'].mean():.2f}%")
    
    print("\n📈 Visualization:")
    print("  Plot saved as: 'demo_-_xgboost_predictions_prediction_plot.png'")
    print("  Shows actual vs. predicted occupancy for the last 7 days of test data")
    
    print("\n💡 Next Steps:")
    print("  - Run the full pipeline: python ev_occupancy_predictor.py")
    print("  - Try with LSTM model (requires TensorFlow): pip install tensorflow")
    print("  - Adjust window_size or model parameters in the code")
    print("  - Use real data instead of synthetic data")
    
    print("\n" + "="*80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Demo failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

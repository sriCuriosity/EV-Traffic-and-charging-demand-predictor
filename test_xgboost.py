"""
Test script for EV Charging Station Occupancy Prediction Model (XGBoost only)
This script tests the pipeline without LSTM to ensure core functionality works.
"""

import sys
import os

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

def test_pipeline():
    """Test the complete pipeline with XGBoost only."""
    print("="*80)
    print("TESTING EV CHARGING OCCUPANCY PREDICTION - XGBoost Only")
    print("="*80)
    
    # Generate smaller dataset for testing (30 days)
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(start_date='2025-01-01', num_days=30)
    print(f"   Generated {len(df)} records")
    print(f"   Sample data:\n{df.head()}")
    
    # Split data
    print("\n2. Splitting data...")
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_data = df[:train_size].copy()
    val_data = df[train_size:train_size + val_size].copy()
    test_data = df[train_size + val_size:].copy()
    
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Scale features
    print("\n3. Scaling features...")
    feature_cols = ['day_of_week', 'hour', 'is_weekend', 'is_holiday', 'ambient_temperature']
    
    train_scaled, val_scaled, scaler = scale_features(train_data, val_data, feature_cols)
    test_scaled = test_data.copy()
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    
    # Create sliding windows
    print("\n4. Creating sliding windows...")
    window_size = 24
    
    X_train, y_train = create_sliding_window(train_scaled, 'occupancy', feature_cols, window_size)
    X_val, y_val = create_sliding_window(val_scaled, 'occupancy', feature_cols, window_size)
    X_test, y_test = create_sliding_window(test_scaled, 'occupancy', feature_cols, window_size)
    
    print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Get timestamps for test set
    test_timestamps = test_data['timestamp'].iloc[window_size:].values
    
    # Train XGBoost
    print("\n5. Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("\n6. Making predictions...")
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    y_pred = xgb_model.predict(X_test_2d)
    
    # Calculate metrics
    print("\n7. Evaluating model...")
    metrics = calculate_metrics(y_test, y_pred, "XGBoost")
    
    # Plot predictions (last 7 days, but we only have 30 days total)
    print("\n8. Creating visualization...")
    plot_predictions(y_test, y_pred, test_timestamps, "XGBoost Test", days=7)
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nFinal Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

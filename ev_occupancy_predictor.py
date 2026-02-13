"""
EV Charging Station Occupancy Prediction Model

This module implements a time-series prediction model for EV charging station occupancy.
It includes:
- Synthetic data generation for one year of hourly occupancy data
- Data preprocessing with MinMaxScaler and sliding window function
- LSTM and XGBoost Regressor implementations
- Model evaluation with MAE and RMSE metrics
- Visualization of actual vs. predicted occupancy

Author: EV Traffic and Charging Demand Predictor
Date: 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM model will not be available.")

# XGBoost Library
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost Regressor will not be available.")


def generate_synthetic_data(start_date='2025-01-01', num_days=365):
    """
    Generate synthetic EV charging station occupancy data for time-series analysis.
    
    Parameters:
    -----------
    start_date : str
        Starting date for data generation (format: 'YYYY-MM-DD')
    num_days : int
        Number of days to generate data for (default: 365 for one year)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing hourly occupancy data with features:
        - timestamp: datetime of the record
        - day_of_week: day of week (0=Monday, 6=Sunday)
        - hour: hour of the day (0-23)
        - is_weekend: binary flag for weekend (1) or weekday (0)
        - is_holiday: binary flag for holidays
        - ambient_temperature: temperature in Celsius
        - occupancy: number of occupied charging slots (0-10)
    """
    print("Generating synthetic data...")
    
    # Create hourly timestamps for the specified period
    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=num_days*24, freq='h')
    
    # Initialize DataFrame
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Extract time-based features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Define some holidays (simplified: first day of each month for demonstration)
    holidays = pd.date_range(start=start, periods=12, freq='MS')
    df['is_holiday'] = df['timestamp'].dt.date.isin(holidays.date).astype(int)
    
    # Generate ambient temperature with seasonal patterns
    day_of_year = df['timestamp'].dt.dayofyear
    # Base temperature with seasonal variation (sine wave)
    base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
    # Add daily variation
    daily_variation = 5 * np.sin(2 * np.pi * df['hour'] / 24 - np.pi/2)
    # Add random noise
    noise = np.random.normal(0, 2, len(df))
    df['ambient_temperature'] = base_temp + daily_variation + noise
    
    # Generate occupancy with realistic patterns
    # Base occupancy depends on hour of day, day of week, and holidays
    base_occupancy = np.zeros(len(df))
    
    # Weekday patterns (higher during commute hours)
    weekday_pattern = np.where(
        (df['hour'] >= 7) & (df['hour'] <= 9), 8,  # Morning rush
        np.where((df['hour'] >= 17) & (df['hour'] <= 19), 9,  # Evening rush
                 np.where((df['hour'] >= 10) & (df['hour'] <= 16), 6,  # Mid-day
                          np.where((df['hour'] >= 20) & (df['hour'] <= 22), 4, 2)))  # Evening/Night
    )
    
    # Weekend patterns (more spread out, moderate throughout day)
    weekend_pattern = np.where(
        (df['hour'] >= 9) & (df['hour'] <= 20), 6,  # Day time
        np.where((df['hour'] >= 21) & (df['hour'] <= 23), 4, 2)  # Evening/Night
    )
    
    # Choose pattern based on weekend flag
    base_occupancy = np.where(df['is_weekend'] == 1, weekend_pattern, weekday_pattern)
    
    # Reduce occupancy on holidays
    base_occupancy = np.where(df['is_holiday'] == 1, base_occupancy * 0.6, base_occupancy)
    
    # Add random variation
    occupancy_noise = np.random.normal(0, 1.5, len(df))
    df['occupancy'] = np.clip(base_occupancy + occupancy_noise, 0, 10).round().astype(int)
    
    print(f"Generated {len(df)} hourly records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Occupancy range: {df['occupancy'].min()} to {df['occupancy'].max()}")
    
    return df


def create_sliding_window(data, target_col, feature_cols, window_size=24):
    """
    Create sliding window sequences for time-series prediction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with features and target
    target_col : str
        Name of the target column (e.g., 'occupancy')
    feature_cols : list
        List of feature column names to include in sequences
    window_size : int
        Number of time steps to look back (default: 24 hours)
    
    Returns:
    --------
    tuple
        (X, y) where:
        - X: numpy array of shape (samples, window_size, num_features)
        - y: numpy array of shape (samples,) containing target values
    """
    print(f"Creating sliding windows with window_size={window_size}...")
    
    X, y = [], []
    
    # Extract feature and target arrays
    features = data[feature_cols].values
    targets = data[target_col].values
    
    # Create sequences
    for i in range(window_size, len(data)):
        X.append(features[i-window_size:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} samples with shape X: {X.shape}, y: {y.shape}")
    
    return X, y


def scale_features(train_data, test_data, feature_cols):
    """
    Scale features using MinMaxScaler.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training dataset
    test_data : pandas.DataFrame
        Testing dataset
    feature_cols : list
        List of feature columns to scale
    
    Returns:
    --------
    tuple
        (train_scaled, test_scaled, scaler) where:
        - train_scaled: scaled training data (DataFrame)
        - test_scaled: scaled testing data (DataFrame)
        - scaler: fitted MinMaxScaler object
    """
    print("Scaling features using MinMaxScaler...")
    
    scaler = MinMaxScaler()
    
    # Fit on training data and transform both
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    
    print(f"Scaled {len(feature_cols)} features")
    
    return train_scaled, test_scaled, scaler


def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    """
    Build and compile LSTM model for time-series prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (window_size, num_features)
    lstm_units : int
        Number of LSTM units in each layer (default: 50)
    dropout_rate : float
        Dropout rate for regularization (default: 0.2)
    
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not available. Please install it to use LSTM.")
    
    print("Building LSTM model...")
    
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print("LSTM model architecture:")
    model.summary()
    
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train LSTM model.
    
    Parameters:
    -----------
    model : keras.Model
        Compiled LSTM model
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation targets
    epochs : int
        Number of training epochs (default: 50)
    batch_size : int
        Batch size for training (default: 32)
    
    Returns:
    --------
    keras.callbacks.History
        Training history object
    """
    print("Training LSTM model...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print("LSTM training completed.")
    
    return history


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost Regressor model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features (needs to be 2D)
    y_train : numpy.ndarray
        Training targets
    X_val : numpy.ndarray
        Validation features (needs to be 2D)
    y_val : numpy.ndarray
        Validation targets
    
    Returns:
    --------
    XGBRegressor
        Trained XGBoost model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Please install it to use XGBoost Regressor.")
    
    print("Training XGBoost model...")
    
    # Reshape 3D data to 2D for XGBoost
    if len(X_train.shape) == 3:
        n_samples, n_steps, n_features = X_train.shape
        X_train_2d = X_train.reshape(n_samples, n_steps * n_features)
        X_val_2d = X_val.reshape(X_val.shape[0], n_steps * n_features)
    else:
        X_train_2d = X_train
        X_val_2d = X_val
    
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(
        X_train_2d, y_train,
        eval_set=[(X_val_2d, y_val)],
        verbose=False
    )
    
    print("XGBoost training completed.")
    
    return model


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate and display evaluation metrics (MAE and RMSE).
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    model_name : str
        Name of the model for display purposes
    
    Returns:
    --------
    dict
        Dictionary containing MAE and RMSE values
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse}


def plot_predictions(y_true, y_pred, timestamps, model_name="Model", days=7):
    """
    Plot actual vs. predicted occupancy for the last N days.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    timestamps : pandas.DatetimeIndex or array-like
        Timestamps corresponding to predictions
    model_name : str
        Name of the model for plot title
    days : int
        Number of days to plot (default: 7)
    """
    print(f"Creating prediction plot for last {days} days...")
    
    # Get last N days of data
    hours = days * 24
    last_hours = min(hours, len(y_true))
    
    y_true_plot = y_true[-last_hours:]
    y_pred_plot = y_pred[-last_hours:]
    timestamps_plot = timestamps[-last_hours:]
    
    # Create figure
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps_plot, y_true_plot, label='Actual Occupancy', 
             color='blue', linewidth=2, alpha=0.7)
    plt.plot(timestamps_plot, y_pred_plot, label='Predicted Occupancy', 
             color='red', linewidth=2, alpha=0.7, linestyle='--')
    
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Occupancy (Number of Slots)', fontsize=12)
    plt.title(f'{model_name}: Actual vs. Predicted Occupancy (Last {days} Days)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'{model_name.lower().replace(" ", "_")}_prediction_plot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{filename}'")
    
    plt.show()


def main():
    """
    Main function to execute the complete EV charging occupancy prediction pipeline.
    """
    print("="*80)
    print("EV CHARGING STATION OCCUPANCY PREDICTION")
    print("="*80)
    
    # Step 1: Generate synthetic data
    print("\n" + "="*80)
    print("STEP 1: DATA GENERATION")
    print("="*80)
    df = generate_synthetic_data(start_date='2025-01-01', num_days=365)
    
    # Display sample data
    print("\nSample of generated data:")
    print(df.head(10))
    print("\nData statistics:")
    print(df.describe())
    
    # Step 2: Split data into train, validation, and test sets
    print("\n" + "="*80)
    print("STEP 2: DATA SPLITTING")
    print("="*80)
    
    # Use 70% for training, 15% for validation, 15% for testing
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_data = df[:train_size].copy()
    val_data = df[train_size:train_size + val_size].copy()
    test_data = df[train_size + val_size:].copy()
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Step 3: Feature scaling
    print("\n" + "="*80)
    print("STEP 3: FEATURE SCALING")
    print("="*80)
    
    feature_cols = ['day_of_week', 'hour', 'is_weekend', 'is_holiday', 'ambient_temperature']
    
    train_scaled, val_scaled, scaler = scale_features(train_data, val_data, feature_cols)
    test_scaled, _, _ = scale_features(test_data, test_data, feature_cols)
    test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])
    
    # Step 4: Create sliding windows
    print("\n" + "="*80)
    print("STEP 4: SLIDING WINDOW CREATION")
    print("="*80)
    
    window_size = 24  # Use 24 hours of history to predict next hour
    
    X_train, y_train = create_sliding_window(train_scaled, 'occupancy', feature_cols, window_size)
    X_val, y_val = create_sliding_window(val_scaled, 'occupancy', feature_cols, window_size)
    X_test, y_test = create_sliding_window(test_scaled, 'occupancy', feature_cols, window_size)
    
    # Get timestamps for test set (for plotting)
    test_timestamps = test_data['timestamp'].iloc[window_size:].values
    
    # Step 5: Train and evaluate models
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    # Try LSTM if available
    if TENSORFLOW_AVAILABLE:
        print("\n--- LSTM Model ---")
        lstm_model = build_lstm_model(input_shape=(window_size, len(feature_cols)))
        lstm_history = train_lstm_model(lstm_model, X_train, y_train, X_val, y_val, 
                                       epochs=50, batch_size=32)
        
        # Make predictions
        y_pred_lstm = lstm_model.predict(X_test).flatten()
        
        # Calculate metrics
        lstm_metrics = calculate_metrics(y_test, y_pred_lstm, "LSTM")
        
        # Plot predictions
        plot_predictions(y_test, y_pred_lstm, test_timestamps, "LSTM Model", days=7)
    
    # Try XGBoost if available
    if XGBOOST_AVAILABLE:
        print("\n--- XGBoost Model ---")
        xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
        
        # Make predictions
        if len(X_test.shape) == 3:
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
        else:
            X_test_2d = X_test
        
        y_pred_xgb = xgb_model.predict(X_test_2d)
        
        # Calculate metrics
        xgb_metrics = calculate_metrics(y_test, y_pred_xgb, "XGBoost")
        
        # Plot predictions
        plot_predictions(y_test, y_pred_xgb, test_timestamps, "XGBoost Model", days=7)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()

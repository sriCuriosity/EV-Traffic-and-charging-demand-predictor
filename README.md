# EV Traffic and Charging Demand Predictor

A machine learning project for predicting Electric Vehicle (EV) charging station occupancy using time-series analysis with LSTM and XGBoost models.

## Features

- **Synthetic Data Generation**: Creates realistic hourly occupancy data for one year with multiple features
- **Data Preprocessing**: Includes MinMaxScaler for feature normalization and sliding window function for time-series preparation
- **Multiple Models**: Implements both LSTM (deep learning) and XGBoost (gradient boosting) models
- **Evaluation Metrics**: Calculates MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)
- **Visualization**: Generates plots comparing actual vs. predicted occupancy for the last 7 days

## Dataset Features

The synthetic dataset includes the following features:
- `timestamp`: Date and time of the record
- `day_of_week`: Day of the week (0=Monday, 6=Sunday)
- `hour`: Hour of the day (0-23)
- `is_weekend`: Binary flag indicating weekend (1) or weekday (0)
- `is_holiday`: Binary flag indicating holidays
- `ambient_temperature`: Temperature in Celsius with seasonal patterns
- `occupancy`: Target variable - number of occupied charging slots (0-10)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sriCuriosity/EV-Traffic-and-charging-demand-predictor.git
cd EV-Traffic-and-charging-demand-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Demo

Run a quick demonstration with 60 days of data:

```bash
python demo.py
```

This will:
- Generate 60 days of synthetic data
- Train an XGBoost model
- Show evaluation metrics (MAE and RMSE)
- Create a visualization plot

### Full Pipeline

Run the main script to execute the complete prediction pipeline with one year of data:

```bash
python ev_occupancy_predictor.py
```

The script will:
1. Generate synthetic occupancy data for one year
2. Split data into training, validation, and test sets
3. Scale features using MinMaxScaler
4. Create sliding window sequences for time-series analysis
5. Train LSTM and XGBoost models (if available)
6. Evaluate models using MAE and RMSE metrics
7. Generate visualization plots for the last 7 days

## Project Structure

```
EV-Traffic-and-charging-demand-predictor/
├── ev_occupancy_predictor.py   # Main prediction pipeline
├── demo.py                      # Quick demo script (60 days)
├── test_xgboost.py             # Unit test for XGBoost (30 days)
├── test_full_pipeline.py       # Full integration test (365 days)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

## Models

### LSTM (Long Short-Term Memory)
- Two LSTM layers with 50 units each
- Dropout layers for regularization (0.2)
- Dense layers for final prediction
- Adam optimizer with learning rate 0.001

### XGBoost Regressor
- 100 estimators
- Learning rate: 0.1
- Max depth: 5
- Regression objective with squared error

## Results

The models generate:
- Performance metrics (MAE and RMSE) printed to console
- Prediction plots saved as PNG files:
  - `lstm_model_prediction_plot.png`
  - `xgboost_model_prediction_plot.png`

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- TensorFlow >= 2.8.0
- XGBoost >= 1.5.0

## License

This project is open source and available under the MIT License.

## Author

sriCuriosity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
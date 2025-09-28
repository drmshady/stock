from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import gspread
import google.auth
import gspread_dataframe as gd
import warnings
import os # Import os for path handling and checking for saved models
import joblib # Import joblib for saving and loading scikit-learn models
import tensorflow as tf # Import tensorflow for saving and loading Keras models

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow warnings


# Define paths for saving models
RF_MODEL_DIR = 'trained_rf_models'
LSTM_MODEL_DIR = 'trained_lstm_models'
UNIFIED_RF_MODEL_PATH = 'unified_rf_model.pkl'
STOCK_LIST_FILE = '/content/filtered_sp500_stocks.csv' # Define the path to the stock list file
LOOKBACK_PERIOD = 30 # Define the lookback period for LSTM sequences


# Ensure model directories exist (for loading models)
os.makedirs(RF_MODEL_DIR, exist_ok=True)
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)


# --- Include the necessary functions from the prediction script ---

def load_stock_data(file_path):
    """Loads stock ticker data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None


def retrieve_historical_data(tickers, period='5y', interval='1d'):
    """Retrieves historical price data for a list of tickers."""
    historical_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False) # progress=False to reduce output
            if not data.empty:
                historical_data[ticker] = data
            else:
                print(f"Warning: No data downloaded for {ticker}")
        except Exception as e:
            print(f"Could not download data for {ticker}: {e}")
    return historical_data


def engineer_features(historical_data, target_period=20, ma_periods=[10, 50], volatility_period=20, rsi_period=14, macd_periods=(12, 26, 9)):
    """Engineers features from historical data and calculates the target variable."""
    processed_data = {}
    for ticker, data in historical_data.items():
        if not data.empty:
            processed_data[ticker] = data.copy()
            # Calculate the target variable: next month's price change (approx target_period trading days)
            processed_data[ticker]['Price_Change_Next_Month'] = processed_data[ticker]['Close'].pct_change(periods=-target_period).shift(target_period)

            # Calculate moving averages
            for period in ma_periods:
                processed_data[ticker][f'MA_{period}'] = processed_data[ticker]['Close'].rolling(window=period).mean()

            # Calculate Volatility (Standard Deviation of Close Price)
            processed_data[ticker]['Volatility'] = processed_data[ticker]['Close'].rolling(window=volatility_period).std()

            # Calculate Relative Strength Index (RSI)
            delta = processed_data[ticker]['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            processed_data[ticker]['RSI'] = 100 - (100 / (1 + rs))

            # Calculate Moving Average Convergence Divergence (MACD)
            exp1 = processed_data[ticker]['Close'].ewm(span=macd_periods[0], adjust=False).mean()
            exp2 = processed_data[ticker]['Close'].ewm(span=macd_periods[1], adjust=False).mean()
            processed_data[ticker]['MACD'] = exp1 - exp2
            processed_data[ticker]['MACD_Signal'] = processed_data[ticker]['MACD'].ewm(span=macd_periods[2], adjust=False).mean()

            # Drop rows with NaN values resulting from feature calculation
            processed_data[ticker].dropna(inplace=True)

    return processed_data


def create_sequences(data, lookback_period):
    """
    Creates time-series sequences for training advanced models.
    This version is simplified for prediction using the latest data.
    """
    latest_sequence = None
    feature_data = data.drop('Price_Change_Next_Month', axis=1)
    feature_names = feature_data.columns.tolist()

    if len(data) >= lookback_period:
         latest_sequence = feature_data.iloc[-lookback_period:][feature_names].values
         latest_sequence = latest_sequence.reshape(1, lookback_period, latest_sequence.shape[1])

    return {'latest_sequence': latest_sequence}


def load_models(tickers):
    """Loads trained models from disk."""
    loaded_models = {}
    loaded_lstm_models = {}
    loaded_unified_model = None

    # Load individual RF models
    for ticker in tickers:
        model_filename = os.path.join(RF_MODEL_DIR, f'rf_model_{ticker}.pkl')
        if os.path.exists(model_filename):
            try:
                loaded_models[ticker] = joblib.load(model_filename)
            except Exception as e:
                print(f"Error loading RF model for {ticker}: {e}")


    # Load unified RF model
    if os.path.exists(UNIFIED_RF_MODEL_PATH):
        try:
            loaded_unified_model = joblib.load(UNIFIED_RF_MODEL_PATH)
        except Exception as e:
             print(f"Error loading unified RF model: {e}")


    # Load individual LSTM models
    for ticker in tickers:
        model_filename = os.path.join(LSTM_MODEL_DIR, f'lstm_model_{ticker}.keras')
        if os.path.exists(model_filename):
            try:
                # Custom objects might be needed if custom layers were used
                loaded_lstm_models[ticker] = tf.keras.models.load_model(model_filename)
            except Exception as e:
                print(f"Error loading LSTM model for {ticker}: {e}")

    return loaded_models, loaded_lstm_models, loaded_unified_model

# Define weighted averaging ensemble prediction logic (single prediction)
def weighted_average_ensemble_predict(rf_prediction, lstm_prediction, rf_mse, lstm_mse):
    """
    Combines single predictions using weighted averaging based on inverse MSE.
    """
    # Avoid division by zero if MSE is 0 (unlikely but good practice)
    rf_weight = 1 / rf_mse if rf_mse > 0 else 1
    lstm_weight = 1 / lstm_mse if lstm_mse > 0 else 1

    total_weight = rf_weight + lstm_weight

    weighted_prediction = (rf_prediction * rf_weight + lstm_prediction * lstm_weight) / total_weight

    return weighted_prediction


def get_stock_predictions(ticker_list, trained_models, trained_lstm_models, individual_rf_models_mse, lstm_models_mse, lookback_period):
    """
    Retrieves latest data for a list of tickers, makes predictions using the weighted ensemble, and returns predictions.
    """
    latest_predictions = {}

    # Retrieve and process latest historical data for the specified tickers
    historical_data_latest = retrieve_historical_data(ticker_list)
    processed_data_latest = engineer_features(historical_data_latest)

    if processed_data_latest and trained_models and trained_lstm_models and individual_rf_models_mse and lstm_models_mse:
        for ticker in ticker_list:
            if ticker in processed_data_latest and ticker in trained_models and ticker in trained_lstm_models and ticker in individual_rf_models_mse and ticker in lstm_models_mse:
                 try:
                    latest_data = processed_data_latest[ticker].tail(1).copy()

                    # --- Get latest predictions from Individual Random Forest model ---
                    rf_model = trained_models[ticker]
                    X_individual_latest = latest_data.drop('Price_Change_Next_Month', axis=1)

                    # Get the column names from processed_data[ticker], excluding the target column tuple
                    # Need to handle potential MultiIndex here if feature engineering created one
                    individual_model_features = [col for col in processed_data[ticker].columns if col != ('Price_Change_Next_Month', '')] # Assuming MultiIndex target
                    # Fallback if not MultiIndex
                    if ('Price_Change_Next_Month', '') not in processed_data[ticker].columns:
                         individual_model_features = [col for col in processed_data[ticker].columns if col != 'Price_Change_Next_Month']


                    X_individual_latest = X_individual_latest[individual_model_features]
                    rf_predictions_latest = rf_model.predict(X_individual_latest)
                    rf_latest_prediction = rf_predictions_latest[0]

                    # --- Get latest predictions from Individual LSTM model ---
                    lstm_model = trained_lstm_models[ticker]
                    # Prepare the latest data for LSTM prediction (needs to be a sequence)
                    lstm_sequence_data = create_sequences(processed_data[ticker], lookback_period) # Use full processed data to create the latest sequence
                    lstm_latest_data_sequence = lstm_sequence_data['latest_sequence']


                    if lstm_latest_data_sequence is not None:
                         lstm_predictions_latest = lstm_model.predict(lstm_latest_data_sequence, verbose=0).flatten() # verbose=0 to reduce output
                         lstm_latest_prediction = lstm_predictions_latest[0]
                    else:
                         print(f"Warning: Not enough data to create LSTM sequence for latest prediction for {ticker}. Skipping LSTM prediction.")
                         lstm_latest_prediction = np.nan


                    # --- Apply the weighted averaging ensemble ---
                    if not np.isnan(rf_latest_prediction) and not np.isnan(lstm_latest_prediction):
                        # Get the MSEs for weighting (using test set performance)
                        rf_mse = individual_rf_models_mse[ticker]
                        lstm_mse = lstm_models_mse[ticker]

                        # Apply the weighted averaging ensemble
                        weighted_prediction = weighted_average_ensemble_predict(
                            rf_latest_prediction,
                            lstm_latest_prediction,
                            rf_mse,
                            lstm_mse
                        )
                        latest_predictions[ticker] = weighted_prediction

                    else:
                         latest_predictions[ticker] = np.nan # Append NaN if either individual prediction failed


                 except Exception as e:
                     print(f"Error making prediction for {ticker}: {e}")
                     latest_predictions[ticker] = np.nan # Append NaN if error occurs
            else:
                print(f"Warning: Missing models, MSEs, or data for ticker {ticker}. Skipping prediction.")
                latest_predictions[ticker] = np.nan # Append NaN if components are missing

    # Return predictions as a dictionary
    return latest_predictions


def rank_stocks(predictions):
    """Ranks stocks based on their predicted price changes."""
    # Filter out tickers with NaN predictions before ranking
    valid_predictions = {k: v for k, v in predictions.items() if not np.isnan(v)}

    if not valid_predictions:
        return pd.DataFrame(columns=['Predicted_Price_Change']) # Return empty DataFrame if no valid predictions

    # Convert the valid predictions dictionary into a pandas DataFrame
    ranked_stocks = pd.DataFrame.from_dict(valid_predictions, orient='index', columns=['Predicted_Price_Change'])

    # Name the index of the DataFrame 'Ticker'
    ranked_stocks.index.name = 'Ticker'

    # Sort the DataFrame in descending order based on the 'Predicted_Price_Change' column
    ranked_stocks = ranked_stocks.sort_values(by='Predicted_Price_Change', ascending=False)

    return ranked_stocks

# --- Initial Loading and Evaluation (needed for Flask app to have models and MSEs) ---
stock_df = load_stock_data(STOCK_LIST_FILE)
tickers = []
if stock_df is not None and not stock_df.empty:
    tickers = stock_df['Ticker'].tolist()

# Attempt to load trained models
trained_models = {}
trained_lstm_models = {}
unified_model = None
if tickers:
    trained_models, trained_lstm_models, unified_model = load_models(tickers)

# Load processed data and time_series_data for initial evaluation and prediction
processed_data = {}
time_series_data_sequences = {} # Use a different name to avoid conflict
if tickers:
    historical_data = retrieve_historical_data(tickers)
    processed_data = engineer_features(historical_data)
    for ticker, data in processed_data.items():
        if not data.empty and len(data) > LOOKBACK_PERIOD:
            # Use the full create_sequences logic here to get sequences for evaluation
            sequences_and_targets = create_sequences(data, LOOKBACK_PERIOD)
            time_series_data_sequences[ticker] = sequences_and_targets


# Prepare unified data for evaluation test set
unified_data = pd.concat([processed_data[ticker].copy().assign(Ticker=ticker) for ticker in processed_data if not processed_data[ticker].empty])
X_unified = unified_data.drop(['Price_Change_Next_Month', 'Ticker'], axis=1)
y_unified = unified_data['Price_Change_Next_Month']

# Split the data into training and testing sets for the unified model (used for evaluation)
# This split defines the test set dates used for evaluating all models
X_train_unified, X_test_unified, y_train_unified, y_test_unified = train_test_split(X_unified, y_unified, test_size=0.2, random_state=42)
test_sets = {'unified': (X_test_unified, y_test_unified)}


# Evaluate models to get MSEs for weighted ensemble
print("Performing initial model evaluation to get MSEs for weighted ensemble weights...")
unified_mse, individual_rf_models_mse, lstm_models_mse, ensemble_models_mse, weighted_ensemble_models_mse = evaluate_models(
    trained_models, trained_lstm_models, unified_model, test_sets, processed_data, time_series_data_sequences # Pass sequences data for LSTM eval
    )
print("Initial model evaluation complete.")


# --- Flask Application ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = None
    tickers_input = ''
    if request.method == 'POST':
        tickers_input = request.form.get('tickers')
        if tickers_input:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()] # Convert to uppercase
            # Call the prediction function with the list of tickers and loaded models/MSEs
            predictions = get_stock_predictions(
                tickers,
                trained_models,
                trained_lstm_models,
                individual_rf_models_mse,
                lstm_models_mse,
                LOOKBACK_PERIOD
                )

    # Rank the predictions before passing to the template
    ranked_predictions_df = None
    if predictions:
        ranked_predictions_df = rank_stocks(predictions)


    return render_template('predict.html', predictions=ranked_predictions_df.to_dict('index') if ranked_predictions_df is not None else None, tickers=tickers_input if request.method == 'POST' else '')


if __name__ == '__main__':
    # For running in a local environment:
    # app.run(debug=True)
    pass # For Colab deployment, we'll use a different entry point or method

from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
LOOKBACK_PERIOD = 30 # Define the lookback period for LSTM sequences


# Ensure model directories exist (for loading models)
os.makedirs(RF_MODEL_DIR, exist_ok=True)
os.makedirs(LSTM_MODEL_DIR, exist_ok=True)


# --- Include the necessary functions from the prediction script ---

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
            exp2 = data['Close'].ewm(span=macd_periods[1], adjust=False).mean() # Changed from processed_data[ticker]['Close'] to data['Close']
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


def get_stock_predictions(ticker_list, trained_models, trained_lstm_models, individual_rf_models_mse, lstm_models_mse, unified_model, unified_model_features, lookback_period):
    """
    Retrieves latest data for a list of tickers, makes predictions using the weighted ensemble or unified model, and returns predictions.
    """
    latest_predictions = {}

    # Retrieve and process latest historical data for the specified tickers
    historical_data_latest = retrieve_historical_data(ticker_list)
    processed_data_latest = engineer_features(historical_data_latest)

    if processed_data_latest:
        for ticker in ticker_list:
            prediction_source = "N/A" # Track which model provided the prediction
            predicted_price_change = np.nan

            if ticker in processed_data_latest:
                 try:
                    latest_data = processed_data_latest[ticker].tail(1).copy()

                    # Try to use the weighted ensemble if models and MSEs are available for this ticker
                    if ticker in trained_models and ticker in trained_lstm_models and ticker in individual_rf_models_mse and lstm_models_mse is not None and ticker in lstm_models_mse: # Added check for lstm_models_mse being None
                        try:
                           # --- Get latest predictions from Individual Random Forest model ---
                           rf_model = trained_models[ticker]
                           X_individual_latest = latest_data.drop('Price_Change_Next_Month', axis=1)

                           # Get the column names from processed_data[ticker], excluding the target column tuple
                           # Need to handle potential MultiIndex here if feature engineering created one
                           individual_model_features = [col for col in processed_data_latest[ticker].columns if col != ('Price_Change_Next_Month', '')] # Assuming MultiIndex target
                           # Fallback if not MultiIndex
                           if ('Price_Change_Next_Month', '') not in processed_data_latest[ticker].columns:
                                individual_model_features = [col for col in processed_data_latest[ticker].columns if col != 'Price_Change_Next_Month']


                           X_individual_latest = X_individual_latest[individual_model_features]
                           rf_predictions_latest = rf_model.predict(X_individual_latest)
                           rf_latest_prediction = rf_predictions_latest[0]


                           # --- Get latest predictions from Individual LSTM model ---
                           lstm_model = trained_lstm_models[ticker]
                           # Prepare the latest data for LSTM prediction (needs to be a sequence)
                           # Use full processed data to create the latest sequence
                           if len(processed_data_latest[ticker]) >= lookback_period:
                               lstm_sequence_data = create_sequences(processed_data_latest[ticker], lookback_period)
                               lstm_latest_data_sequence = lstm_sequence_data['latest_sequence']

                               if lstm_latest_data_sequence is not None:
                                    lstm_predictions_latest = lstm_model.predict(lstm_latest_data_sequence, verbose=0).flatten() # verbose=0 to reduce output
                                    lstm_latest_prediction = lstm_predictions_latest[0]
                               else:
                                    print(f"Warning: Not enough data to create LSTM sequence for latest prediction for {ticker}. Skipping LSTM prediction.")
                                    lstm_latest_prediction = np.nan
                           else:
                                print(f"Warning: Not enough historical data to create LSTM sequence for latest prediction for {ticker}. Skipping LSTM prediction.")
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
                               predicted_price_change = weighted_prediction
                               prediction_source = "Weighted Ensemble"

                           else:
                                print(f"Warning: Individual RF or LSTM prediction failed for {ticker}. Cannot use Weighted Ensemble.")


                        except Exception as e:
                             print(f"Error during Weighted Ensemble prediction for {ticker}: {e}")


                    # If weighted ensemble failed or not available, try the unified model
                    if prediction_source == "N/A" and unified_model is not None and unified_model_features is not None:
                         try:
                            # Prepare latest data for unified model prediction
                            # Need to flatten columns and ensure they match unified model features
                            latest_data_unified_format = latest_data.copy()
                            # Flatten the MultiIndex columns to strings
                            latest_data_unified_format.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col.strip() for col in latest_data_unified_format.columns]

                            # Drop the target variable from the prediction data
                            X_latest_unified = latest_data_unified_format.drop('Price_Change_Next_Month_', axis=1, errors='ignore')

                            # Reindex the latest data to match the unified training features and fill missing values if any
                            # Important: Ensure the feature names exactly match what the unified model was trained on
                            X_latest_unified = X_latest_unified.reindex(columns=unified_model_features, fill_value=0)


                            # Predict using the unified model
                            unified_predictions_latest = unified_model.predict(X_latest_unified)
                            predicted_price_change = unified_predictions_latest[0]
                            prediction_source = "Unified Model"

                         except Exception as e:
                             print(f"Error during Unified Model prediction for {ticker}: {e}")


                    latest_predictions[ticker] = {
                        'Predicted_Price_Change': predicted_price_change,
                        'Source': prediction_source
                        }

                 except Exception as e:
                     print(f"Error processing data or making prediction for {ticker}: {e}")
                     latest_predictions[ticker] = {
                         'Predicted_Price_Change': np.nan,
                         'Source': "Error"
                         }

            else:
                print(f"Warning: No historical data available for ticker {ticker}. Cannot make prediction.")
                latest_predictions[ticker] = {
                    'Predicted_Price_Change': np.nan,
                    'Source': "No Data"
                    }


    # Return predictions as a dictionary
    return latest_predictions


def rank_stocks(predictions):
    """Ranks stocks based on their predicted price changes."""
    # Filter out tickers with NaN predictions before ranking
    valid_predictions = {k: v for k, v in predictions.items() if not np.isnan(v['Predicted_Price_Change'])}

    if not valid_predictions:
        return pd.DataFrame(columns=['Predicted_Price_Change', 'Source']) # Return empty DataFrame if no valid predictions

    # Convert the valid predictions dictionary into a pandas DataFrame
    ranked_stocks = pd.DataFrame.from_dict(valid_predictions, orient='index')

    # Name the index of the DataFrame 'Ticker'
    ranked_stocks.index.name = 'Ticker'

    # Sort the DataFrame in descending order based on the 'Predicted_Price_Change' column
    ranked_stocks = ranked_stocks.sort_values(by='Predicted_Price_Change', ascending=False)

    return ranked_stocks

# --- Initial Loading and Evaluation (needed for Flask app to have models and MSEs) ---
# We load ALL tickers initially to train/load models and evaluate them.
# This is needed to have the models and their performance metrics (MSEs) available
# for the weighted ensemble prediction logic when a user requests predictions for a subset of tickers.
# In a production environment, training/evaluation might be a separate process.

# For simplicity in this example, we load all tickers from the file initially to ensure
# the models and MSEs are populated for the prediction function.
# The prediction function itself will then fetch data only for the requested tickers.

# Define the path to the stock list file
STOCK_LIST_FILE = 'filtered_sp500_stocks.csv' # Use local path in container

# Load stock ticker data from the file for initial model loading/training
stock_df = None
all_tickers = []
try:
    stock_df = pd.read_csv(STOCK_LIST_FILE)
    if stock_df is not None and not stock_df.empty:
        all_tickers = stock_df['Ticker'].tolist()
except FileNotFoundError:
    print(f"Error: Initial stock list file not found at {STOCK_LIST_FILE}. Cannot load/train models.")
    all_tickers = [] # Set empty list if file not found

# Attempt to load trained models for all tickers
trained_models = {}
trained_lstm_models = {}
unified_model = None
if all_tickers:
    trained_models, trained_lstm_models, unified_model = load_models(all_tickers)

# Check if all models are loaded for all initial tickers
all_initial_models_loaded = True
if unified_model is None:
    all_initial_models_loaded = False
for ticker in all_tickers:
    if ticker not in trained_models or ticker not in trained_lstm_models:
        all_initial_models_loaded = False
        break

# Get unified model features if unified model is available (needed for prediction)
unified_model_features = None
# We need to re-create X_unified from the training data to get feature names
# This assumes the processed_data for the initial tickers is available
try:
    # Retrieve and process historical data for all initial tickers for evaluation purposes if models are not loaded
    # or to get feature names if models were loaded but processed_data wasn't kept
    if 'processed_data_initial' not in locals() or not processed_data_initial:
         print("\nRetrieving and processing historical data for initial setup (features and evaluation)...")
         historical_data_initial = retrieve_historical_data(all_tickers)
         processed_data_initial = engineer_features(historical_data_initial)
         time_series_data_sequences_initial = {}
         for ticker, data in processed_data_initial.items():
              if not data.empty and len(data) > LOOKBACK_PERIOD:
                   sequences_and_targets = create_sequences(data, LOOKBACK_PERIOD)
                   time_series_data_sequences_initial[ticker] = sequences_and_targets

    all_stocks_data_unified_features = []
    if processed_data_initial:
        for ticker, data in processed_data_initial.items():
            data_with_ticker = data.copy()
            data_with_ticker['Ticker'] = ticker
            all_stocks_data_unified_features.append(data_with_ticker)

        if all_stocks_data_unified_features:
            unified_data_features = pd.concat(all_stocks_data_unified_features)
            # Drop the 'Ticker' column for training as it's not a numerical feature for the model
            X_unified_features = unified_data_features.drop(['Price_Change_Next_Month', 'Ticker'], axis=1)
            unified_model_features = X_unified_features.columns.tolist()
            print("Unified model feature names extracted.")
        else:
            print("No data available to extract unified model feature names.")

except Exception as e:
    print(f"Error getting unified model feature names: {e}")
    unified_model_features = None


if all_initial_models_loaded:
    print("\nSuccessfully loaded all initial models. Skipping training.")

    # We still need processed_data and time_series_data for evaluation
    # Retrieve and process data for all initial tickers for evaluation purposes
    # This was done above to get features, so reuse processed_data_initial and time_series_data_sequences_initial

    # Prepare unified data for evaluation test set
    unified_data_eval = pd.concat([processed_data_initial[ticker].copy().assign(Ticker=ticker) for ticker in processed_data_initial if not processed_data_initial[ticker].empty])
    X_unified_eval = unified_data_eval.drop(['Price_Change_Next_Month', 'Ticker'], axis=1)
    y_unified_eval = unified_data_eval['Price_Change_Next_Month']

    # Split the data into training and testing sets for the unified model (used for evaluation)
    # This split defines the test set dates used for evaluating all models
    X_train_unified, X_test_unified, y_train_unified, y_test_unified = train_test_split(X_unified_eval, y_unified_eval, test_size=0.2, random_state=42)
    test_sets_initial = {'unified': (X_test_unified, y_test_unified)}

    # Evaluate loaded models to get MSEs for weighted ensemble
    print("\nEvaluating loaded models to get MSEs for weighted ensemble weights...")
    unified_mse, individual_rf_models_mse, lstm_models_mse, ensemble_models_mse, weighted_ensemble_models_mse = evaluate_models(
        trained_models, trained_lstm_models, unified_model, test_sets_initial, processed_data_initial, time_series_data_sequences_initial # Pass sequences data for LSTM eval
        )
    print("Initial model evaluation complete.")


else:
    print("\nOne or more initial models not found. Training models for all tickers...")
    # Retrieve and process historical data for training
    # This was done above to get features, so reuse processed_data_initial and time_series_data_sequences_initial

    # Train the models for all initial tickers
    trained_models, trained_lstm_models, unified_model, test_sets_initial = train_models(processed_data_initial, time_series_data_sequences_initial, LOOKBACK_PERIOD)

    # Get unified model features after training (needed for prediction)
    if unified_model and test_sets_initial and 'unified' in test_sets_initial and test_sets_initial['unified'][0] is not None:
        unified_model_features = test_sets_initial['unified'][0].columns.tolist() # Use training data features


    # Evaluate the trained models to get MSEs for weighted ensemble
    unified_mse, individual_rf_models_mse, lstm_models_mse, ensemble_models_mse, weighted_ensemble_models_mse = evaluate_models(
        trained_models, trained_lstm_models, unified_model, test_sets_initial, processed_data_initial, time_series_data_sequences_initial
        )

    # Save the trained models (already done in train_models, but explicitly here for clarity)
    # print("\nSaving trained models...")
    # for ticker, model in trained_models.items():
    #      model_filename = os.path.join(RF_MODEL_DIR, f'rf_model_{ticker}.pkl')
    #      joblib.dump(model, model_filename)

    # if unified_model:
    #      joblib.dump(unified_model, UNIFIED_RF_MODEL_PATH)

    # for ticker, model in trained_lstm_models.items():
    #      model_filename = os.path.join(LSTM_MODEL_DIR, f'lstm_model_{ticker}.keras')
    #      model.save(model_filename)
    # print("Models saved.")


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

            # Make predictions using the weighted averaging ensemble or unified model for the user-provided tickers
            predictions_dict = get_stock_predictions(
                tickers,
                trained_models,
                trained_lstm_models,
                individual_rf_models_mse,
                lstm_models_mse,
                unified_model,
                unified_model_features,
                LOOKBACK_PERIOD
                )

            # Rank the predictions before passing to the template
            ranked_predictions_df = None
            if predictions_dict:
                ranked_predictions_df = rank_stocks(predictions_dict)


    return render_template('predict.html', predictions=ranked_predictions_df.to_dict('index') if ranked_predictions_df is not None else None, tickers=tickers_input if request.method == 'POST' else '')


if __name__ == '__main__':
    # For running in a local environment:
    # app.run(debug=True)
    pass # For deployment, the web server (like Gunicorn or Flask's built-in in this case) will run the app

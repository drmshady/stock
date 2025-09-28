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


def load_all_models():
    """Loads all trained models from disk based on available files."""
    loaded_models = {}
    loaded_lstm_models = {}
    loaded_unified_model = None

    # Get list of available individual RF models
    rf_model_files = [f for f in os.listdir(RF_MODEL_DIR) if f.endswith('.pkl')]
    for model_file in rf_model_files:
        ticker = model_file.replace('rf_model_', '').replace('.pkl', '')
        model_filename = os.path.join(RF_MODEL_DIR, model_file)
        try:
            loaded_models[ticker] = joblib.load(model_filename)
        except Exception as e:
            print(f"Error loading RF model for {ticker} from {model_file}: {e}")


    # Load unified RF model
    if os.path.exists(UNIFIED_RF_MODEL_PATH):
        try:
            loaded_unified_model = joblib.load(UNIFIED_RF_MODEL_PATH)
        except Exception as e:
             print(f"Error loading unified RF model from {UNIFIED_RF_MODEL_PATH}: {e}")


    # Get list of available individual LSTM models
    lstm_model_files = [f for f in os.listdir(LSTM_MODEL_DIR) if f.endswith('.keras')]
    for model_file in lstm_model_files:
        ticker = model_file.replace('lstm_model_', '').replace('.keras', '')
        model_filename = os.path.join(LSTM_MODEL_DIR, model_file)
        try:
            # Custom objects might be needed if custom layers were used
            loaded_lstm_models[ticker] = tf.keras.models.load_model(model_filename)
        except Exception as e:
            print(f"Error loading LSTM model for {ticker} from {model_file}: {e}")

    print(f"\nLoaded {len(loaded_models)} individual RF models.")
    print(f"Loaded {len(loaded_lstm_models)} individual LSTM models.")
    if loaded_unified_model:
        print("Loaded unified RF model.")
    else:
        print("Unified RF model not found.")

    return loaded_models, loaded_lstm_models, loaded_unified_model


def evaluate_models_on_tickers(tickers, trained_models, trained_lstm_models, unified_model, lookback_period):
    """
    Evaluates the performance of trained models on the test set for specified tickers
    to get MSEs needed for weighted ensemble calculation.
    This requires retrieving and processing historical data for the tickers.
    """
    individual_rf_models_mse = {}
    lstm_models_mse = {}
    weighted_ensemble_models_mse = {}


    # Retrieve and process historical data for the specified tickers for evaluation
    historical_data_eval = retrieve_historical_data(tickers)
    processed_data_eval = engineer_features(historical_data_eval)

    time_series_data_sequences_eval = {}
    for ticker, data in processed_data_eval.items():
         if not data.empty and len(data) > lookback_period:
              sequences_and_targets = create_sequences_for_evaluation(data, lookback_period) # Use full create_sequences for eval
              time_series_data_sequences_eval[ticker] = sequences_and_targets


    # Prepare unified data for evaluation test set for the specified tickers
    all_stocks_data_eval = []
    for ticker, data in processed_data_eval.items():
        data_with_ticker = data.copy()
        data_with_ticker['Ticker'] = ticker # Add ticker back for unified model
        all_stocks_data_eval.append(data_with_ticker)

    if all_stocks_data_eval:
        unified_data_eval = pd.concat(all_stocks_data_eval)
        # Drop the 'Ticker' column for training as it's not a numerical feature for the model
        X_unified_eval = unified_data_eval.drop(['Price_Change_Next_Month', 'Ticker'], axis=1)
        y_unified_eval = unified_data_eval['Price_Change_Next_Month']

        if len(unified_data_eval) > 1:
            # Split the data into training and testing sets for the unified model (used for evaluation)
            # Use a consistent random_state for reproducibility
            X_train_unified, X_test_unified, y_train_unified, y_test_unified = train_test_split(X_unified_eval, y_unified_eval, test_size=0.2, random_state=42)
            test_sets_eval = {'unified': (X_test_unified, y_test_unified)}
        else:
             print("Not enough data to create unified test set for evaluation.")
             test_sets_eval = {'unified': (None, None)}
    else:
         print("No data available to create unified data for evaluation.")
         test_sets_eval = {'unified': (None, None)}


    X_test_unified, y_test_unified = test_sets_eval['unified']


    # Evaluate individual RandomForestRegressor models
    print("\nEvaluating individual Random Forest models for requested tickers...")
    for ticker, model in trained_models.items():
        if ticker in processed_data_eval and not processed_data_eval[ticker].empty and X_test_unified is not None: # Check if data and unified test set are available
            # Get the test data for the current ticker
            ticker_data = processed_data_eval[ticker]

            # Find the indices that are in both the ticker data's index and the unified X_test index
            test_indices = X_test_unified.index.intersection(ticker_data.index)

            if not test_indices.empty:
                try:
                    X_individual_test = ticker_data.loc[test_indices].drop('Price_Change_Next_Month', axis=1)
                    y_individual_test = ticker_data.loc[test_indices, 'Price_Change_Next_Month']

                    # Get the column names from processed_data_eval[ticker], excluding the target column tuple
                    individual_model_features = [col for col in processed_data_eval[ticker].columns if col != ('Price_Change_Next_Month', '')]
                    if ('Price_Change_Next_Month', '') not in processed_data_eval[ticker].columns: # Fallback if not MultiIndex
                         individual_model_features = [col for col in processed_data_eval[ticker].columns if col != 'Price_Change_Next_Month']


                    # Select only the feature columns for the latest data using the individual model's feature names
                    X_individual_test = X_individual_test[individual_model_features]

                    # Predict on the individual ticker's test data
                    individual_predictions = model.predict(X_individual_test)

                    # Calculate MSE for the individual model
                    mse = mean_squared_error(y_individual_test, individual_predictions)
                    individual_rf_models_mse[ticker] = mse
                except Exception as e:
                    print(f"Error evaluating individual RF model for {ticker}: {e}")
            # else:
            #     print(f"No test data available in the unified test set for {ticker} for individual RF evaluation.")

    print("Individual Random Forest model evaluation complete.")


    # Evaluate individual LSTM models
    print("\nEvaluating individual LSTM models for requested tickers...")
    for ticker, model in trained_lstm_models.items():
        if ticker in time_series_data_sequences_eval:
            X_lstm, y_lstm = time_series_data_sequences_eval[ticker]['sequences'], time_series_data_sequences_eval[ticker]['targets']
            # Split the data again to get the same test set as used during training (if applicable)
            # Use the same random_state and test_size as the unified split for consistency
            if X_lstm.shape[0] > 0:
                _, X_test_lstm, _, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)


                if X_test_lstm.shape[0] > 0:
                    try:
                        # Evaluate the model on the test data
                        loss = model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
                        lstm_models_mse[ticker] = loss
                        # print(f"{ticker} LSTM Model MSE: {loss:.6f}")
                    except Exception as e:
                         print(f"Error evaluating individual LSTM model for {ticker}: {e}")
                # else:
                #     print(f"No test data available for {ticker} for evaluation.")
            # else:
            #     print(f"No time-series sequences available for {ticker} for evaluation.")

    print("Individual LSTM model evaluation complete.")

    # --- Evaluate Weighted Averaging Ensemble ---
    def weighted_average_ensemble_eval(rf_predictions, lstm_predictions, rf_mse, lstm_mse):
      """
      Combines predictions using weighted averaging based on inverse MSE.
      """
      # Avoid division by zero if MSE is 0 (unlikely but good practice)
      rf_weight = 1 / rf_mse if rf_mse > 0 else 1
      lstm_weight = 1 / lstm_mse if lstm_mse > 0 else 1

      total_weight = rf_weight + lstm_weight

      # Calculate weighted average
      weighted_predictions = (rf_predictions * rf_weight + lstm_predictions * lstm_weight) / total_weight

      return weighted_predictions

    print("\nEvaluating Weighted Averaging Ensemble for requested tickers...")
    for ticker in tickers: # Iterate through requested tickers
        if ticker in individual_rf_models_mse and ticker in lstm_models_mse and ticker in processed_data_eval: # Check if MSEs and processed data exist
            # Get the test data and true values for the current ticker
            ticker_data = processed_data_eval[ticker]
            if X_test_unified is not None:
                test_indices = X_test_unified.index.intersection(ticker_data.index)

                if not test_indices.empty:
                    try:
                        y_true_test = ticker_data.loc[test_indices, 'Price_Change_Next_Month']

                        # Get predictions from the individual Random Forest model on the test set
                        rf_model = trained_models[ticker]
                        X_individual = ticker_data.drop('Price_Change_Next_Month', axis=1)
                        individual_model_features = [col for col in processed_data_eval[ticker].columns if col != ('Price_Change_Next_Month', '')]
                        if ('Price_Change_Next_Month', '') not in processed_data_eval[ticker].columns: # Fallback if not MultiIndex
                             individual_model_features = [col for col in processed_data_eval[ticker].columns if col != 'Price_Change_Next_Month']

                        X_individual_test = X_individual.loc[test_indices][individual_model_features]
                        rf_predictions_test = trained_models[ticker].predict(X_individual_test) # Use model from trained_models dict

                        # Get predictions from the individual LSTM model on the test set
                        lstm_model = trained_lstm_models[ticker]
                        if ticker in time_series_data_sequences_eval:
                             X_lstm, y_lstm = time_series_data_sequences_eval[ticker]['sequences'], time_series_data_sequences_eval[ticker]['targets']
                             if X_lstm.shape[0] > 0:
                                _, X_test_lstm, _, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
                                lstm_predictions_test = trained_lstm_models[ticker].predict(X_test_lstm).flatten() # Use model from trained_lstm_models dict
                             else:
                                 lstm_predictions_test = np.array([]) # Empty array if no sequences
                        else:
                            lstm_predictions_test = np.array([]) # Empty array if no sequences


                        # Get the MSEs for weighting
                        rf_mse = individual_rf_models_mse[ticker]
                        lstm_mse = lstm_models_mse[ticker]

                        # Apply the weighted averaging ensemble
                        min_len_pred = min(len(rf_predictions_test), len(lstm_predictions_test))
                        if min_len_pred > 0:
                             weighted_predictions = weighted_average_ensemble_eval(
                                 rf_predictions_test[:min_len_pred],
                                 lstm_predictions_test[:min_len_pred],
                                 rf_mse,
                                 lstm_mse
                             )

                             # Store the weighted ensemble predictions for the test set
                             # weighted_ensemble_predictions_test[ticker] = weighted_predictions # This dictionary is local

                             # Evaluate the weighted ensemble predictions using MSE
                             min_len_eval = min(len(y_true_test), len(weighted_predictions))
                             weighted_ensemble_models_mse[ticker] = mean_squared_error(y_true_test[:min_len_eval], weighted_predictions[:min_len_eval])
                        else:
                            print(f"Not enough test predictions for weighted ensemble evaluation for {ticker}.")


                    except Exception as e:
                         print(f"Error evaluating weighted ensemble for {ticker}: {e}")

            # else:
            #     print(f"No test data available in the unified test set for {ticker} for individual RF evaluation.")
        # else:
        #      print(f"Missing MSEs or processed data for weighted ensemble evaluation for {ticker}.")


    print("Weighted Averaging Ensemble evaluation complete for requested tickers.")

    return individual_rf_models_mse, lstm_models_mse, weighted_ensemble_models_mse


# Helper function for create_sequences specifically for evaluation (needs targets)
def create_sequences_for_evaluation(data, lookback_period):
    """
    Creates time-series sequences and targets for evaluation.
    """
    X, y = [], []
    feature_data = data.drop('Price_Change_Next_Month', axis=1)
    feature_names = feature_data.columns.tolist()

    for i in range(len(data) - lookback_period):
        x_sequence = feature_data.iloc[i:(i + lookback_period)][feature_names].values
        y_target = data['Price_Change_Next_Month'].iloc[i + lookback_period]

        X.append(x_sequence)
        y.append(y_target)

    return {'sequences': np.array(X), 'targets': np.array(y)}


def get_stock_predictions(ticker_list, trained_models, trained_lstm_models, unified_model, unified_model_features, lookback_period):
    """
    Retrieves latest data for a list of tickers, makes predictions using the weighted ensemble or unified model, and returns predictions.
    This version dynamically calculates MSEs for weighted ensemble for the requested tickers.
    """
    latest_predictions = {}

    # Evaluate models for the requested tickers to get the necessary MSEs for the weighted ensemble
    # This needs to happen each time predictions are requested
    individual_rf_models_mse, lstm_models_mse, weighted_ensemble_models_mse = evaluate_models_on_tickers(
        ticker_list, trained_models, trained_lstm_models, unified_model, lookback_period
        )


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
                    if ticker in trained_models and ticker in trained_lstm_models and ticker in individual_rf_models_mse and ticker in lstm_models_mse:
                        try:
                           # --- Get latest predictions from Individual Random Forest model ---
                           rf_model = trained_models[ticker]
                           X_individual_latest = latest_data.drop('Price_Change_Next_Month', axis=1)

                           # Get the column names from processed_data_latest[ticker]
                           individual_model_features = [col for col in processed_data_latest[ticker].columns if col != ('Price_Change_Next_Month', '')] # Assuming MultiIndex target
                           if ('Price_Change_Next_Month', '') not in processed_data_latest[ticker].columns: # Fallback if not MultiIndex
                                individual_model_features = [col for col in processed_data_latest[ticker].columns if col != 'Price_Change_Next_Month']


                           X_individual_latest = X_individual_latest[individual_model_features]
                           rf_predictions_latest = rf_model.predict(X_individual_latest)
                           rf_latest_prediction = rf_predictions_latest[0]


                           # --- Get latest predictions from Individual LSTM model ---
                           lstm_model = trained_lstm_models[ticker]
                           # Prepare the latest data for LSTM prediction (needs to be a sequence)
                           # Use full processed data to create the latest sequence
                           if len(processed_data_latest[ticker]) >= lookback_period:
                               lstm_sequence_data = create_sequences(processed_data_latest[ticker], lookback_period) # Use simplified create_sequences
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
# Load all available models at startup
print("Loading all available models at application startup...")
trained_models, trained_lstm_models, unified_model = load_all_models()

# Get unified model features if unified model is available (needed for prediction)
unified_model_features = None
# We need to get the feature names that the unified model was trained on.
# A robust solution would save/load feature names with the model.
# For this example, we'll attempt to infer them if the unified model is loaded,
# assuming a consistent feature set from the training data structure.
if unified_model:
    try:
        # Create a dummy DataFrame with the expected feature columns
        # This requires knowing the feature names without loading historical data
        # A better approach would be to save/load feature names with the model
        # For simplicity, let's assume a fixed set of feature names based on engineer_features function
        dummy_data = pd.DataFrame(0, index=[0], columns=['Open', 'High', 'Low', 'Close', 'Volume',
                                                        'MA_10', 'MA_50', 'Volatility', 'RSI', 'MACD', 'MACD_Signal'])
        # Engineer features on dummy data to get expected column names (including MultiIndex if applicable)
        dummy_processed_data = engineer_features({'dummy_ticker': dummy_data}, target_period=20)
        if 'dummy_ticker' in dummy_processed_data and not dummy_processed_data['dummy_ticker'].empty:
             # Get column names, flattening MultiIndex if it exists
             feature_columns = dummy_processed_data['dummy_ticker'].drop('Price_Change_Next_Month', axis=1).columns
             unified_model_features = ['_'.join(col).strip() if isinstance(col, tuple) else col.strip() for col in feature_columns]
             print("Unified model feature names inferred.")
        else:
            print("Could not infer unified model feature names from dummy data.")


    except Exception as e:
        print(f"Error inferring unified model feature names: {e}")
        unified_model_features = None


# We no longer need to load initial processed data or perform initial evaluation here,
# as data fetching and evaluation for weighted ensemble MSEs will happen dynamically
# within get_stock_predictions for the tickers requested by the user.


# --- Flask Application ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predictions = None
    tickers_input = ''
    # Initialize ranked_predictions_df to None for GET requests
    ranked_predictions_df = None
    if request.method == 'POST':
        tickers_input = request.form.get('tickers')
        if tickers_input:
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()] # Convert to uppercase

            # Make predictions using the weighted averaging ensemble or unified model for the user-provided tickers
            predictions_dict = get_stock_predictions(
                tickers,
                trained_models, # Pass loaded models
                trained_lstm_models, # Pass loaded LSTM models
                None, # individual_rf_models_mse will be calculated dynamically
                None, # lstm_models_mse will be calculated dynamically
                unified_model, # Pass loaded unified model
                unified_model_features, # Pass unified model features
                LOOKBACK_PERIOD
                )

            # Rank the predictions before passing to the template
            if predictions_dict:
                # The predictions_dict now contains {'Predicted_Price_Change': value, 'Source': source}
                ranked_predictions_df = rank_stocks(predictions_dict)


    return render_template('predict.html', predictions=ranked_predictions_df.to_dict('index') if ranked_predictions_df is not None else None, tickers=tickers_input if request.method == 'POST' else '')


if __name__ == '__main__':
    # For running in a local environment:
    # app.run(debug=True)
    pass # For deployment, the web server (like Gunicorn or Flask's built-in in this case) will run the app

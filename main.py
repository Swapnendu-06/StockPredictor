# Import necessary libraries for data manipulation, machine learning, and visualization
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import yfinance as yf  # Yahoo Finance API for stock data
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.ensemble import RandomForestRegressor  # Random forest regression model
from sklearn.preprocessing import MinMaxScaler  # Feature scaling
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Model evaluation metrics
import matplotlib.pyplot as plt  # Plotting library
import requests  # HTTP requests
import json  # JSON data handling
from datetime import datetime, timedelta  # Date and time manipulation
import os  # Operating system interface
import warnings  # Warning control
import logging  # Logging functionality

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import optional libraries with fallback
try:
    import nltk  # Natural language processing
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Sentiment analysis
    # Download NLTK data for sentiment analysis (runs once)
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available. Sentiment analysis will be skipped.")
    NLTK_AVAILABLE = False

try:
    import talib  # Technical analysis library
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available. Using simple technical indicators.")
    TALIB_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential  # Sequential neural network model
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM and dense layers
    import tensorflow as tf  # TensorFlow for deep learning
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. LSTM model will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Custom class to integrate with Kite MCP (Market Control Protocol)
class KiteMCP:
    def __init__(self, mcp_url="https://mcp.kite.trade/mcp"):
        """
        Initialize KiteMCP class with default MCP URL
        Args:
            mcp_url (str): Base URL for Kite MCP API
        """
        self.mcp_url = mcp_url  # Store the MCP URL
        self.access_token = None  # Initialize access token as None
        self.session = requests.Session()  # Create a session for connection pooling

    def authenticate(self, api_key=None, api_secret=None, request_token=None):
        """
        Authenticate with Zerodha Kite API using OAuth2
        Args:
            api_key (str): API key from Zerodha (can be None to use env var)
            api_secret (str): API secret from Zerodha (can be None to use env var)
            request_token (str): Request token obtained from OAuth flow (can be None to use env var)
        """
        # Get credentials from environment variables if not provided
        api_key = api_key or os.getenv('KITE_API_KEY')
        api_secret = api_secret or os.getenv('KITE_API_SECRET')
        request_token = request_token or os.getenv('KITE_REQUEST_TOKEN')
        
        # Check if all required credentials are available
        if not all([api_key, api_secret, request_token]):
            logger.warning("Kite API credentials not found. Will use yfinance fallback.")
            return False
        
        # Prepare authentication payload
        payload = {
            'api_key': api_key,
            'api_secret': api_secret,
            'request_token': request_token
        }
        
        try:
            # Send POST request to authenticate with timeout
            response = self.session.post(f"{self.mcp_url}/login", json=payload, timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            # Extract access token from response
            self.access_token = response.json().get('access_token')
            logger.info("Successfully authenticated with Kite API")
            return True
        except requests.exceptions.RequestException as e:
            # Log error if authentication fails
            logger.error(f"Authentication failed: {e}")
            return False

    def fetch_market_data(self, symbol, days=365):
        """
        Fetch market data for a given symbol
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            days (int): Number of days of historical data
        Returns:
            pandas.DataFrame: OHLC data with Date as index
        """
        # Try to use Kite API if authenticated
        if self.access_token:
            try:
                # Prepare headers with authorization token
                headers = {'Authorization': f'Bearer {self.access_token}'}
                # Prepare parameters for API call
                params = {'symbol': symbol, 'days': days}
                
                # Make GET request to fetch market data with timeout
                response = self.session.get(f"{self.mcp_url}/market_data", 
                                          headers=headers, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                
                # Parse JSON response
                data = response.json()
                # Create DataFrame from OHLC data
                df = pd.DataFrame(data['ohlc'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                # Set Date as index and return
                logger.info(f"Successfully fetched data from Kite API for {symbol}")
                return df.set_index('Date')
                
            except requests.exceptions.RequestException as e:
                # Log error and fallback to yfinance
                logger.error(f"Error fetching data from Kite API: {e}")
        
        # Use yfinance as fallback data source
        try:
            logger.info(f"Using yfinance fallback for {symbol}")
            # Calculate start date for yfinance
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Download data using yfinance
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Check if data was successfully downloaded
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            logger.info(f"Successfully fetched data using yfinance for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            raise

# Simple technical indicators (fallback when TA-Lib is not available)
def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

# Function to preprocess and engineer features from raw market data
def preprocess_data(df):
    """
    Preprocess market data and add technical indicators
    Args:
        df (pandas.DataFrame): Raw OHLC data
    Returns:
        tuple: (scaled_df, scaler) - Scaled DataFrame and fitted scaler
    """
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Handle missing values using updated pandas methods
        df = df.ffill().bfill()  # Forward fill then backward fill
        
        # Add technical indicators
        if TALIB_AVAILABLE:
            # Use TA-Lib for technical indicators
            df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            # Use simple implementations
            df['SMA20'] = calculate_sma(df['Close'], 20)
            df['RSI'] = calculate_rsi(df['Close'], 14)
            df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
        
        # Add simple price-based features
        df['Price_Change'] = df['Close'].pct_change()  # Daily returns
        df['Volatility'] = df['Close'].rolling(window=20).std()  # 20-day volatility
        
        # Add sentiment analysis (improved approach)
        if NLTK_AVAILABLE:
            sia = SentimentIntensityAnalyzer()
            # Generate sentiment based on price movements (more meaningful than raw prices)
            df['Sentiment'] = df['Price_Change'].apply(
                lambda x: sia.polarity_scores(f"Price {'increased' if x > 0 else 'decreased'} by {abs(x):.2%}")['compound']
            )
        else:
            # Simple sentiment based on price momentum
            df['Sentiment'] = np.where(df['Price_Change'] > 0, 0.1, -0.1)
        
        # Remove rows with NaN values created by technical indicators
        df = df.dropna()
        
        # Check if we have enough data after preprocessing
        if len(df) < 100:
            raise ValueError("Insufficient data after preprocessing. Need at least 100 data points.")
        
        # Select features for scaling
        features_to_scale = ['Close', 'SMA20', 'RSI', 'MACD', 'MACD_signal', 'Price_Change', 'Volatility', 'Sentiment']
        
        # Scale all features to range [0, 1] for better model performance
        scaler = MinMaxScaler()
        # Fit scaler and transform data
        scaled_data = scaler.fit_transform(df[features_to_scale])
        # Create new DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale, index=df.index)
        
        logger.info(f"Successfully preprocessed data. Shape: {scaled_df.shape}")
        return scaled_df, scaler
        
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        raise

# Function to prepare data for LSTM model (creates sequences)
def prepare_lstm_data(df, look_back=60):
    """
    Create sequences for LSTM training
    Args:
        df (pandas.DataFrame): Scaled data
        look_back (int): Number of previous time steps to use for prediction
    Returns:
        tuple: (X, y) - Input sequences and target values
    """
    try:
        X, y = [], []  # Initialize lists for features and targets
        
        # Check if we have enough data for the specified look_back period
        if len(df) <= look_back:
            raise ValueError(f"Not enough data for look_back period of {look_back}")
        
        # Create sequences of look_back length
        for i in range(look_back, len(df)):
            # Add sequence of previous look_back time steps as features
            X.append(df.iloc[i-look_back:i].values)
            # Add current Close price as target
            y.append(df.iloc[i]['Close'])
        
        # Convert lists to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        logger.info(f"LSTM data prepared. X shape: {X_array.shape}, y shape: {y_array.shape}")
        return X_array, y_array
        
    except Exception as e:
        logger.error(f"Error preparing LSTM data: {e}")
        raise

# Function to train Linear Regression model
def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model
    Args:
        X_train: Training features
        y_train: Training targets
    Returns:
        sklearn.LinearRegression: Trained model
    """
    try:
        model = LinearRegression()  # Initialize Linear Regression model
        model.fit(X_train, y_train)  # Train the model
        logger.info("Linear Regression model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Linear Regression model: {e}")
        raise

# Function to train Random Forest model
def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor
    Args:
        X_train: Training features
        y_train: Training targets
    Returns:
        sklearn.RandomForestRegressor: Trained model
    """
    try:
        # Initialize Random Forest with 100 trees and fixed random state
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)  # Train the model
        logger.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        raise

# Function to train LSTM neural network (only if TensorFlow is available)
def train_lstm(X_train, y_train, look_back=60):
    """
    Train an LSTM neural network
    Args:
        X_train: Training sequences
        y_train: Training targets
        look_back (int): Sequence length
    Returns:
        tensorflow.keras.Model: Trained LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available. Skipping LSTM training.")
        return None
    
    try:
        model = Sequential()  # Initialize sequential model
        
        # Add first LSTM layer with 50 units, return sequences for next LSTM layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
        # Add dropout layer to prevent overfitting
        model.add(Dropout(0.2))
        
        # Add second LSTM layer with 50 units
        model.add(LSTM(units=50))
        # Add dropout layer
        model.add(Dropout(0.2))
        
        # Add dense output layer with 1 unit (for price prediction)
        model.add(Dense(units=1))
        
        # Compile model with Adam optimizer and MSE loss
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model with early stopping callback
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        logger.info("LSTM model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance using various metrics
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name (str): Name of the model
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Calculate evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Create metrics dictionary
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        # Log metrics
        logger.info(f"{model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {e}")
        return {}

# Function to plot predictions vs actual values
def plot_predictions(y_test, predictions, model_name, symbol, metrics=None):
    """
    Plot actual vs predicted prices
    Args:
        y_test: Actual test values
        predictions: Model predictions
        model_name (str): Name of the model for title
        symbol (str): Stock symbol
        metrics (dict): Optional metrics to display on plot
    """
    try:
        plt.figure(figsize=(12, 8))  # Create figure with specified size
        
        # Plot actual prices
        plt.plot(y_test.index, y_test, label='Actual Price', linewidth=2, alpha=0.8)
        # Plot predicted prices
        plt.plot(y_test.index, predictions, label=f'Predicted Price ({model_name})', linewidth=2, alpha=0.8)
        
        # Set title, labels, and legend
        plt.title(f'{symbol} Stock Price Prediction - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Scaled Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot if provided
        if metrics:
            textstr = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # Adjust layout and save plot
        plt.tight_layout()
        filename = f'{symbol}_{model_name.replace(" ", "_")}_prediction.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        logger.info(f"Plot saved as {filename}")
        
    except Exception as e:
        logger.error(f"Error plotting predictions for {model_name}: {e}")

# Main execution function
def main(symbol='AAPL', days=365):
    """
    Main function to run the complete stock prediction pipeline
    Args:
        symbol (str): Stock symbol to analyze
        days (int): Number of days of historical data
    """
    try:
        logger.info(f"Starting stock prediction analysis for {symbol}")
        
        # Initialize Kite MCP client
        kite_mcp = KiteMCP()
        
        # Try to authenticate with Kite API
        kite_mcp.authenticate()
        
        # Fetch market data for the specified symbol
        df = kite_mcp.fetch_market_data(symbol, days)
        logger.info(f"Fetched {len(df)} days of data for {symbol}")
        
        # Preprocess data and add technical indicators
        scaled_df, scaler = preprocess_data(df)
        
        # Split data into training and testing sets (80-20 split)
        train_test_split = 0.8
        train_size = int(len(scaled_df) * train_test_split)  # Calculate training size
        train_df = scaled_df.iloc[:train_size]  # Training data
        test_df = scaled_df.iloc[train_size:]   # Testing data
        
        logger.info(f"Training size: {len(train_df)}, Testing size: {len(test_df)}")

        # Define features and target variable
        features = ['SMA20', 'RSI', 'MACD', 'MACD_signal', 'Price_Change', 'Volatility', 'Sentiment']
        X_train = train_df[features]  # Training features
        y_train = train_df['Close']   # Training target (Close price)
        X_test = test_df[features]    # Testing features
        y_test = test_df['Close']     # Testing target

        # Dictionary to store all predictions and metrics
        all_predictions = {}
        all_metrics = {}

        # Train and evaluate Linear Regression model
        try:
            lr_model = train_linear_regression(X_train, y_train)  # Train model
            lr_predictions = lr_model.predict(X_test)             # Make predictions
            lr_metrics = evaluate_model(y_test, lr_predictions, 'Linear Regression')
            all_predictions['Linear Regression'] = lr_predictions
            all_metrics['Linear Regression'] = lr_metrics
            plot_predictions(y_test, lr_predictions, 'Linear Regression', symbol, lr_metrics)
        except Exception as e:
            logger.error(f"Linear Regression failed: {e}")

        # Train and evaluate Random Forest model
        try:
            rf_model = train_random_forest(X_train, y_train)      # Train model
            rf_predictions = rf_model.predict(X_test)             # Make predictions
            rf_metrics = evaluate_model(y_test, rf_predictions, 'Random Forest')
            all_predictions['Random Forest'] = rf_predictions
            all_metrics['Random Forest'] = rf_metrics
            plot_predictions(y_test, rf_predictions, 'Random Forest', symbol, rf_metrics)
        except Exception as e:
            logger.error(f"Random Forest failed: {e}")

        # Train and evaluate LSTM model (if TensorFlow is available)
        lstm_predictions = None
        if TENSORFLOW_AVAILABLE:
            try:
                # Prepare data for LSTM model
                look_back = min(60, len(train_df) // 4)  # Adjust look_back based on data size
                logger.info(f"Using look_back period of {look_back} for LSTM")
                
                # Create training sequences for LSTM
                X_train_lstm, y_train_lstm = prepare_lstm_data(train_df, look_back)
                # Create testing sequences (include some training data for context)
                X_test_lstm, y_test_lstm = prepare_lstm_data(scaled_df.iloc[train_size-look_back:], look_back)

                # Train LSTM model
                lstm_model = train_lstm(X_train_lstm, y_train_lstm, look_back)
                
                if lstm_model is not None:
                    # Make predictions
                    lstm_predictions = lstm_model.predict(X_test_lstm, verbose=0)
                    lstm_predictions = lstm_predictions.flatten()
                    
                    # Evaluate LSTM model
                    lstm_y_test = y_test.iloc[-len(lstm_predictions):]
                    lstm_metrics = evaluate_model(lstm_y_test, lstm_predictions, 'LSTM')
                    all_predictions['LSTM'] = lstm_predictions
                    all_metrics['LSTM'] = lstm_metrics
                    
                    # Plot LSTM results
                    plot_predictions(lstm_y_test, lstm_predictions, 'LSTM', symbol, lstm_metrics)
                    
            except Exception as e:
                logger.error(f"LSTM training failed: {e}")

        # Create ensemble prediction if we have multiple models
        if len(all_predictions) > 1:
            try:
                # Find the minimum length among all predictions
                min_length = min(len(pred) for pred in all_predictions.values())
                
                # Truncate all predictions to the same length
                truncated_predictions = []
                for name, pred in all_predictions.items():
                    if len(pred) > min_length:
                        truncated_predictions.append(pred[-min_length:])
                    else:
                        truncated_predictions.append(pred)
                
                # Calculate ensemble as average of all models
                ensemble_predictions = np.mean(truncated_predictions, axis=0)
                
                # Evaluate ensemble model
                ensemble_y_test = y_test.iloc[-min_length:]
                ensemble_metrics = evaluate_model(ensemble_y_test, ensemble_predictions, 'Ensemble')
                all_predictions['Ensemble'] = ensemble_predictions
                all_metrics['Ensemble'] = ensemble_metrics
                
                # Plot ensemble results
                plot_predictions(ensemble_y_test, ensemble_predictions, 'Ensemble', symbol, ensemble_metrics)
                
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")

        # Save all results to CSV file
        try:
            # Create results DataFrame
            results_data = {'Actual': y_test}
            
            # Add predictions (pad shorter arrays with NaN)
            max_length = len(y_test)
            for name, pred in all_predictions.items():
                if len(pred) < max_length:
                    # Pad with NaN at the beginning
                    padded_pred = np.full(max_length, np.nan)
                    padded_pred[-len(pred):] = pred
                    results_data[name] = padded_pred
                else:
                    results_data[name] = pred[-max_length:]
            
            # Create DataFrame and save
            results = pd.DataFrame(results_data, index=y_test.index)
            results_filename = f'{symbol}_predictions.csv'
            results.to_csv(results_filename)
            logger.info(f"Results saved to {results_filename}")
            
            # Save metrics to separate file
            metrics_filename = f'{symbol}_metrics.csv'
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df.to_csv(metrics_filename)
            logger.info(f"Metrics saved to {metrics_filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        logger.info("Stock prediction analysis completed successfully!")
        
        # Return summary
        return {
            'symbol': symbol,
            'data_points': len(scaled_df),
            'models_trained': list(all_predictions.keys()),
            'metrics': all_metrics
        }
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

# Run the main function if script is executed directly
if __name__ == "__main__":
    try:
        # Run with Apple stock as default
        result = main(symbol='AAPL')
        print(f"\nAnalysis Summary:")
        print(f"Symbol: {result['symbol']}")
        print(f"Data Points: {result['data_points']}")
        print(f"Models Trained: {', '.join(result['models_trained'])}")
        
        # Print best performing model
        if result['metrics']:
            best_model = min(result['metrics'].items(), key=lambda x: x[1].get('MSE', float('inf')))
            print(f"Best Model (lowest MSE): {best_model[0]} (MSE: {best_model[1]['MSE']:.6f})")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(f"Error: {e}")